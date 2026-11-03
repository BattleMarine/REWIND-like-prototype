"""Fleet inventory→runtime pin→cost plan→bounded run. 네트워크는 pin/run에서만 명시 실행."""
import argparse
import json
from pathlib import Path
from rewind_archive.contracts import canonical
from . import planning
from .storage import sha,Store,status
from .runner import run,recover_pending,release_stale_lease
from .control import coverage,request_stop

def load(p):
    p=Path(p)
    if p.stat().st_size>4*1024**2:raise ValueError('fleet_cli_input_size')
    return json.loads(p.read_bytes())
def save(p,doc):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(canonical(doc)+b'\n')

def authorized_factory():
    import google.auth,requests
    from google.auth.transport.requests import AuthorizedSession,Request
    from urllib3.util.retry import Retry
    from rewind_archive.request_policy import validate_no_hidden_retries
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
    def factory():
        auth_http=requests.Session()
        try:
            auth_http.trust_env=False
            for adapter in auth_http.adapters.values():adapter.max_retries=Retry(total=0,connect=0,read=0,redirect=0,status=0)
            session=AuthorizedSession(credentials,max_refresh_attempts=0,auth_request=Request(session=auth_http))
        except BaseException:auth_http.close();raise
        close_outer=session.close
        def close_all():
            try:close_outer()
            finally:auth_http.close()
        session.close=close_all
        try:
            session.trust_env=False
            for adapter in session.adapters.values():adapter.max_retries=Retry(total=0,connect=0,read=0,redirect=0,status=0)
            validate_no_hidden_retries(session)
        except BaseException:session.close();raise
        return session
    return factory

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('inventory');p.add_argument('--observed',required=True);p.add_argument('--official');p.add_argument('--official-file-sha256');p.add_argument('--output',required=True)
    p=commands.add_parser('pin');p.add_argument('--inventory',required=True);p.add_argument('--select',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--max-requests',type=int,default=64);p.add_argument('--max-seconds',type=float,default=300)
    p=commands.add_parser('plan');p.add_argument('--inventory',required=True);p.add_argument('--estimates',required=True);p.add_argument('--select',nargs='+',required=True);p.add_argument('--scope',choices=('full','calibration'),default='full');p.add_argument('--verification',choices=('catalog_only','full_bounded'),default='catalog_only');p.add_argument('--storage-backend',choices=('local','mounted_remote'),default='mounted_remote');p.add_argument('--execution-profile',choices=('legacy',planning.AUTO_PROFILE),default='legacy');p.add_argument('--output',required=True)
    p=commands.add_parser('calibration-plan');p.add_argument('--inventory',required=True);p.add_argument('--select',required=True);p.add_argument('--output',required=True)
    p=commands.add_parser('survey-plan');p.add_argument('--inventory',required=True);p.add_argument('--select',nargs='+',required=True);p.add_argument('--output',required=True)
    for command in ('status','coverage'):
        p=commands.add_parser(command);p.add_argument('--store',required=True)
    p=commands.add_parser('stop');p.add_argument('--store',required=True);p.add_argument('--plan-sha256',required=True)
    for command in ('run','auto-run','resume'):
        p=commands.add_parser(command);p.add_argument('--inventory',required=True);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--store',required=True);p.add_argument('--workspace',required=True);p.add_argument('--owner',required=True);p.add_argument('--single-writer',action='store_true');p.add_argument('--max-segments',type=int,required=command=='run',default=1000000);p.add_argument('--session-seconds',type=lambda v:None if v=='none' else float(v),required=command=='run',default=None);p.add_argument('--output',required=True)
    p=commands.add_parser('recover-pending');p.add_argument('--store',required=True);p.add_argument('--head',required=True);p.add_argument('--token',required=True);p.add_argument('--owner-confirmed-stopped',action='store_true')
    p=commands.add_parser('release-stale-lease');p.add_argument('--store',required=True);p.add_argument('--token',required=True);p.add_argument('--owner-confirmed-stopped',action='store_true')
    args=parser.parse_args(argv)
    if getattr(args,'output',None) and (Path(args.output).exists() or Path(args.output).is_symlink()):raise ValueError('fleet_new_output_required')
    if args.command=='inventory':
        doc=planning.inventory(load(args.observed))
        if args.official:
            registry=load(args.official)
            if sha(Path(args.official).read_bytes())!=args.official_file_sha256:raise ValueError('official_file_pin')
            doc=planning.official(doc,registry,sha(canonical(registry)))
        save(args.output,doc)
    elif args.command=='pin':
        inv=planning.validate(load(args.inventory),'fleet_inventory_v1')
        if not set(args.select)<={a['key'] for a in inv['archives']}:raise ValueError('fleet_unknown_archive')
        session=authorized_factory()()
        try:doc=planning.pin(inv,session,args.select,max_requests=args.max_requests,max_seconds=args.max_seconds)
        finally:session.close()
        save(args.output,doc)
    elif args.command=='plan':
        doc=planning.plan(load(args.inventory),load(args.estimates),args.select,scope=args.scope,verification=args.verification,storage_backend=args.storage_backend,execution_profile=args.execution_profile);save(args.output,doc)
    elif args.command=='calibration-plan':doc=planning.calibration_plan(load(args.inventory),args.select);save(args.output,doc)
    elif args.command=='survey-plan':doc=planning.survey_plan(load(args.inventory),args.select);save(args.output,doc)
    elif args.command in ('status','coverage'):doc=coverage(args.store)
    elif args.command=='stop':doc=request_stop(args.store,args.plan_sha256)
    elif args.command in ('run','auto-run','resume'):
        inv,plan=load(args.inventory),load(args.plan)
        if plan.get('content_sha256')!=args.plan_sha256:raise ValueError('fleet_external_plan_pin')
        # Incomplete/tampered plan is rejected before a credential/session is constructed.
        planning.validate(plan,'fleet_cost_plan_v1')
        if not plan['complete_estimate']:raise ValueError('fleet_incomplete_plan')
        if args.command!='run' and plan.get('execution_profile')!=planning.AUTO_PROFILE:raise ValueError('fleet_auto_profile_required')
        doc=run(inv,plan,args.store,args.workspace,owner=args.owner,single_writer=args.single_writer,max_segments=args.max_segments,
            session_seconds=args.session_seconds,session_factory=lambda:authorized_factory()(),resume_stopped=args.command=='resume');save(args.output,doc)
    elif args.command=='recover-pending':doc={'head':recover_pending(Store(args.store),args.head,args.token,owner_confirmed_stopped=args.owner_confirmed_stopped)}
    else:release_stale_lease(args.store,args.token,owner_confirmed_stopped=args.owner_confirmed_stopped);doc={'lease_released':True}
    print(json.dumps(doc if args.command not in ('inventory','plan','pin') else {k:doc[k] for k in ('schema','content_sha256')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
