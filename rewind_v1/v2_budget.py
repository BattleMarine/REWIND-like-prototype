"""Durable invocation history; unknown cumulative caps never authorize a retry loop."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid

from rewind_archive.contracts import canonical,digest
from rewind_prepare.temporal_cache import _atomic,_locked
from rewind_prepare.temporal_selection import seal,check_pin
from rewind_m1.local_rgb import _regular_read


class ResourceLimit(TimeoutError):pass


class Ledger:
    def __init__(self,root,binding,cfg,clock=time.monotonic):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        states=sorted(self.root.glob('ledger-????????.json'))
        if len(states)>16:raise ValueError('v2_ledger_snapshot_count_explicit_cleanup_required')
        self.generation=int(states[-1].stem.split('-')[1]) if states else 0
        self.path=states[-1] if states else self.root/'ledger-00000000.json'
        self.binding,self.cfg,self.clock=binding,cfg,clock;self.current=None
        if cfg['cumulative_limits'] is not None:raise ValueError('v2_cumulative_limit_policy_pending')

    def read(self,*,snapshot=False):
        states=sorted(self.root.glob('ledger-????????.json'))
        if len(states)>16:raise ValueError('v2_ledger_snapshot_count_explicit_cleanup_required')
        if states:
            newest=int(states[-1].stem.split('-')[1])
            if newest<self.generation:raise ValueError('v2_ledger_generation_rollback')
            self.generation=newest;self.path=states[-1]
        if not self.path.exists():
            if snapshot:raise ValueError('v2_ledger_not_published')
            return {'binding':self.binding,'lineage_id':uuid.uuid4().hex,'imports':{},'invocations':[],
            'io_scope':'read/write reserved and observed count external source/copy and produced payload; local logical reads separate; metadata/ledger filesystem writes and physical page-cache IO excluded'}
        raw=_regular_read(self.path,8*1024**2)
        doc=json.loads(raw);check_pin(doc,'rewind_v2_usage_ledger_v1',doc.get('content_sha256'))
        if doc['generation']!=self.generation:raise ValueError('v2_ledger_generation_pin')
        if doc['binding']!=self.binding:raise ValueError('v2_ledger_binding')
        for row in doc['invocations']:
            for name in ('active_seconds_observed','read_bytes_reserved','write_bytes_reserved','read_bytes_observed','write_bytes_observed'):
                value=row[name]
                if type(value) not in (int,float) or not 0<=value<float('inf'):raise ValueError('v2_ledger_usage_invalid')
            if row['read_bytes_observed']>row['read_bytes_reserved'] or row['write_bytes_observed']>row['write_bytes_reserved']:raise ValueError('v2_ledger_observed_exceeds_reservation')
        return doc if snapshot else {k:v for k,v in doc.items() if k not in ('schema','content_sha256','generation')}

    def persist(self):
        if self.current is not None:self.current['active_seconds_observed']=max(self.current['active_seconds_observed'],self.clock()-self.started)
        if self.generation>=99999999:raise ResourceLimit('v2_ledger_eight_digit_generation_exhausted')
        next_path=self.root/f'ledger-{self.generation+1:08d}.json'
        if next_path.exists():raise ValueError('v2_ledger_generation_conflict')
        raw=canonical(seal({'schema':'rewind_v2_usage_ledger_v1',**self.doc,'generation':self.generation+1}))
        if len(raw)>8*1024**2:raise ResourceLimit('v2_ledger_snapshot_bytes')
        _atomic(next_path,raw)
        self.generation+=1;self.path=next_path
        # No full enumeration per publication. Keep four complete generations.
        obsolete=self.root/f'ledger-{self.generation-4:08d}.json'
        if self.generation>4 and obsolete.exists():
            if obsolete.is_symlink() or obsolete.resolve().parent!=self.root.resolve():raise ValueError('v2_ledger_cleanup_path')
            obsolete.unlink()

    def snapshot(self):
        return self.read(snapshot=True)  # One bounded read, validated and returned unchanged.

    @contextmanager
    def invocation(self,phase):
        with _locked(self.root):
            self.doc=self.read()
            for old in self.doc['invocations']:
                if old['status']=='running':
                    old['status']='interrupted_unknown'
                    old['active_seconds_estimated_upper_bound']=max(old['active_seconds_observed'],self.cfg['external_timeout_seconds']+90)
                    old['upper_bound_scope']='estimate assuming notebook hard timeout + SIGINT60/kill30; not observed wall'
            self.started=self.clock();self.current={'number':len(self.doc['invocations'])+1,'phase':phase,'status':'running',
                'active_seconds_observed':0.,'read_bytes_reserved':0,'write_bytes_reserved':0,'read_bytes_observed':0,'write_bytes_observed':0,'local_read_logical_bytes':0,'local_write_payload_bytes':0}
            self.doc['invocations'].append(self.current);self.persist()
            try:yield self
            except BaseException as e:
                self.current['reason']={'type':type(e).__name__,'message':str(e)[:256]}
                self.current['status']='stopped' if isinstance(e,KeyboardInterrupt) else 'paused_resource' if isinstance(e,ResourceLimit) else 'paused_time' if isinstance(e,TimeoutError) else 'failed';raise
            else:self.current['status']='complete'
            finally:
                self.persist();self.current=None

    def remaining(self,terminal=False):
        remaining=self.cfg['seconds_per_invocation']-(self.clock()-self.started)-(0 if terminal else self.cfg['terminal_reserve_seconds'])
        if remaining<=0:raise TimeoutError('v2_invocation_time_limit')
        return remaining

    def check(self):
        self.remaining()
        try:
            import resource
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.cfg['resources']['rss_bytes']:raise MemoryError('v2_rss_limit')
        except ImportError:pass

    def reserve(self,kind,amount):
        self.remaining(terminal=True)
        key=kind+'_bytes_reserved'
        if type(amount) is not int or amount<0:raise ValueError('v2_invalid_reservation')
        if self.current[key]+amount>self.cfg['resources'][kind+'_bytes']:raise ResourceLimit('v2_'+kind+'_limit')
        self.current[key]+=amount;self.persist()

    def observed(self,kind,amount,*,persist=True):
        if type(amount) is not int or amount<0 or self.current[kind+'_bytes_observed']+amount>self.current[kind+'_bytes_reserved']:raise ValueError('v2_invalid_observed_usage')
        self.current[kind+'_bytes_observed']+=amount
        if persist:self.persist()

    def local_io(self,kind,amount):
        if type(amount) is not int or amount<0:raise ValueError('v2_local_io_count')
        key='local_read_logical_bytes' if kind=='read' else 'local_write_payload_bytes'
        self.current[key]=self.current.get(key,0)+amount

    def import_usage(self,usage):
        check_pin(usage,'rewind_v2_usage_ledger_v1',usage.get('content_sha256'))
        if usage['binding']!=self.binding:raise ValueError('v2_imported_usage_binding')
        names=('active_seconds_observed','read_bytes_reserved','write_bytes_reserved','read_bytes_observed','write_bytes_observed','local_read_logical_bytes','local_write_payload_bytes')
        incoming={**usage.get('imports',{}),usage['lineage_id']:{k:sum(r.get(k,0) for r in usage['invocations']) for k in names}}
        for lineage,values in incoming.items():
            if any(type(v) not in (int,float) or not 0<=v<float('inf') for v in values.values()):raise ValueError('v2_imported_usage_values')
            if lineage==self.doc['lineage_id']:continue
            previous=self.doc['imports'].get(lineage,{})
            self.doc['imports'][lineage]={k:max(previous.get(k,0),values.get(k,0)) for k in names}
        self.persist()

    def totals(self):
        return {k:sum(r.get(k,0) for r in self.doc['invocations'])+sum(v.get(k,0) for v in self.doc['imports'].values()) for k in ('active_seconds_observed','read_bytes_reserved','write_bytes_reserved','read_bytes_observed','write_bytes_observed','local_read_logical_bytes','local_write_payload_bytes')}
