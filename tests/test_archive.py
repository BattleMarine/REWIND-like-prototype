"""Tiny 합성 archive·loopback HTTP만 사용한 준비 도구 검사."""

import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import random
import tarfile
import tempfile
import threading
import time
import unittest
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rewind_archive.contracts import Budget, Limits, PreparationError, canonical, digest, validate_manifest
from rewind_archive.extraction import prepare, report_reserve_bytes
from rewind_archive.fixture import SEQUENCE, create_fixture, local_manifest, tar_bytes
from rewind_archive.matching import identify, match_extractions, verify_extraction
from rewind_archive.sources import DriveSource, HttpSource, LocalSource, SessionResponse, SplitStream
from rewind_archive.__main__ import main


def member(frame=0, view="pose", sequence=SEQUENCE):
    return f"{sequence}/json/frame_{frame}.json" if view=="pose" else f"{sequence}/fisheye_final_image/camera_{view}/final_{frame}.png"


class ArchiveTestCase(unittest.TestCase):
    def setUp(self):
        temporary_root=Path(__file__).resolve().parents[1]/"outputs"/"archive_test_tmp"
        temporary_root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.limits=Limits(io_chunk_bytes=512,media_range_bytes=512,max_ranges=128)
        self.counter=0

    def manifest(self, raw=None, entries=None, kind="pose", split=None, total=None):
        self.counter+=1
        directory=self.root/f"source{self.counter}"
        if raw is None:
            raw=gzip.compress(tar_bytes(entries or [(member(),b"abc")]),mtime=0)
        return local_manifest(directory,raw,kind,split_bytes=split,total_parts=total),directory

    def run_extract(self, manifest, base, limits=None, includes=None, **kwargs):
        self.counter+=1
        out=self.root/f"extracted{self.counter}"
        report=prepare(manifest,out,includes or ["**"],limits or self.limits,base=base,**kwargs)
        return report,out


class ArchiveTests(ArchiveTestCase):
    def test_manifest_contiguous_identity_and_credentials(self):
        manifest,base=self.manifest(split=17)
        validate_manifest(manifest)
        for change in (lambda d:d['parts'][0].update(index=1),lambda d:d['parts'][1].update(index=3),
                       lambda d:d['parts'][0]['source'].update(token='must_not_store'),lambda d:d['parts'][0].update(size_bytes=0)):
            bad=copy.deepcopy(manifest);change(bad)
            with self.assertRaises(PreparationError): validate_manifest(bad)
        bad=copy.deepcopy(manifest);bad['parts'][0]['source']={'type':'http','url':'https://example.com/a?token=secret'}
        bad['parts'][0]['identity']={'etag':'"a"'}
        with self.assertRaises(PreparationError): validate_manifest(bad)
        bad['parts'][0]['source']['url']='https://example.com/a'
        bad['parts'][0]['identity']={'etag':'W/"a"'}
        with self.assertRaises(PreparationError): validate_manifest(bad)
        bad['parts'][0]['identity']={}
        with self.assertRaises(PreparationError): validate_manifest(bad)

    def test_split_boundaries_complete_file_and_gzip_crc(self):
        expected=random.Random(12).randbytes(12000)
        manifest,base=self.manifest(entries=[(member(),expected)],split=337)
        report,out=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'complete')
        self.assertTrue(report['gzip_eof_crc_valid']);self.assertTrue(report['tar_end_blocks_seen'])
        self.assertTrue(report['whole_archive_integrity_verified']);self.assertFalse(report['train_ready'])
        self.assertEqual((out/'files'/member()).read_bytes(),expected)
        self.assertEqual(sum(a['bytes_received'] for a in report['range_audit']),report['stats']['transfer_bytes'])
        self.assertLessEqual(report['stats']['peak_input_chunk_bytes'],512)
        self.assertLessEqual(report['stats']['peak_output_chunk_bytes'],512)
        verify_extraction(out,manifest)

    def test_gnu_longname_directory_and_pax_path(self):
        long_sequence=SEQUENCE+'/'+('X'*120)
        long_name=member(sequence=long_sequence)
        for format in (tarfile.GNU_FORMAT,tarfile.PAX_FORMAT):
            raw=gzip.compress(tar_bytes([(long_name,b"long")],format=format),mtime=0)
            manifest,base=self.manifest(raw=raw)
            report,out=self.run_extract(manifest,base)
            self.assertEqual(report['stop_reason'],'complete')
            self.assertGreater(report['stats']['metadata_bytes'],0)
            self.assertEqual((out/'files'/long_name).read_bytes(),b'long')
        # GNU LongName의 directory를 지원하고 write 대상에는 추가하지 않는다.
        directory=tarfile.TarInfo(long_sequence+'/');directory.type=tarfile.DIRTYPE
        raw=directory.tobuf(format=tarfile.GNU_FORMAT)+tar_bytes([(long_name,b'long')])
        manifest,base=self.manifest(raw=gzip.compress(raw,mtime=0))
        report,_=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'complete')
        self.assertEqual(len(report['files']),1)

    def test_metadata_size_and_malformed_pax_are_bounded(self):
        info=tarfile.TarInfo('././@LongLink');info.type=tarfile.GNUTYPE_LONGNAME;info.size=10**12
        manifest,base=self.manifest(raw=gzip.compress(info.tobuf(format=tarfile.GNU_FORMAT),mtime=0))
        report,out=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'metadata_bytes_limit')
        self.assertEqual(report['stats']['metadata_bytes'],0)
        self.assertFalse(list(out.rglob('*.partial')))
        info=tarfile.TarInfo('pax');info.type=tarfile.XHDTYPE;info.size=10
        raw=info.tobuf()+b'99 path=a\n'+bytes(502)
        manifest,base=self.manifest(raw=gzip.compress(raw,mtime=0))
        report,_=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'invalid_pax')
        self.assertFalse(report['inspection_candidates_usable'])

    def test_incomplete_prefix_retains_only_complete_regular_files(self):
        a=random.Random(1).randbytes(3000);b=random.Random(2).randbytes(6000)
        compressor=__import__('zlib').compressobj(wbits=31)
        tar=tar_bytes([(member(0),a),(member(1),b)])
        prefix=compressor.compress(tar[:512+3072+512+900])+compressor.flush(__import__('zlib').Z_SYNC_FLUSH)
        manifest,base=self.manifest(raw=prefix,total=9)
        report,out=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'incomplete_payload')
        self.assertEqual([f['member'] for f in report['files']],[member(0)])
        self.assertEqual(report['active_incomplete_member']['written_bytes'],900)
        self.assertFalse((out/'files'/member(1)).exists())
        self.assertFalse(list(out.rglob('*.partial')))
        self.assertFalse(report['gzip_eof_crc_valid']);self.assertFalse(report['whole_archive_integrity_verified'])
        self.assertTrue(report['inspection_candidates_usable'])
        verify_extraction(out,manifest)

    def test_complete_tar_does_not_imply_gzip_integrity(self):
        compressor=__import__('zlib').compressobj(wbits=31)
        prefix=compressor.compress(tar_bytes([(member(),b'abc')]))+compressor.flush(__import__('zlib').Z_SYNC_FLUSH)
        manifest,base=self.manifest(raw=prefix)
        report,_=self.run_extract(manifest,base)
        self.assertTrue(report['tar_end_blocks_seen'])
        self.assertFalse(report['gzip_eof_crc_valid']);self.assertFalse(report['whole_archive_integrity_verified'])
        self.assertEqual(report['stop_reason'],'prefix_exhausted')

    def test_truncated_header_and_padding_do_not_publish(self):
        info=tarfile.TarInfo(member());info.size=3
        for raw,reason in ((info.tobuf()[:100],'incomplete_header'),(info.tobuf()+b'abc'+bytes(10),'incomplete_padding')):
            manifest,base=self.manifest(raw=gzip.compress(raw,mtime=0))
            report,out=self.run_extract(manifest,base)
            self.assertEqual(report['stop_reason'],reason)
            self.assertEqual(report['files'],[])
            self.assertFalse(list(out.rglob('*.partial')))

    def test_crc_error_invalidates_retained_candidates_and_reuse_match(self):
        raw=bytearray(gzip.compress(tar_bytes([(member(),random.Random(42).randbytes(9000))]),mtime=0))
        raw[-8]^=1
        manifest,base=self.manifest(raw=bytes(raw))
        report,out=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'gzip_corrupt')
        self.assertFalse(report['inspection_candidates_usable'])
        self.assertGreaterEqual(len(report['files']),1)
        with self.assertRaisesRegex(PreparationError,'unusable'): verify_extraction(out)
        with self.assertRaisesRegex(PreparationError,'unusable'): match_extractions([out])

    def test_trailing_gzip_and_bad_header_fail_closed(self):
        for raw in (gzip.compress(tar_bytes([(member(),b'abc')]),mtime=0)+gzip.compress(b'other',mtime=0),gzip.compress(bytes([1])*512,mtime=0)):
            manifest,base=self.manifest(raw=raw)
            report,_=self.run_extract(manifest,base)
            self.assertFalse(report['inspection_candidates_usable'])
            self.assertIn(report['stop_reason'],('trailing_or_concatenated_gzip','invalid_tar_header'))

    def test_paths_links_special_files_and_duplicate_never_overwrite(self):
        for name in ('../escape','/absolute','C:/drive','a\\b','CON.txt','a./b','a/../b'):
            manifest,base=self.manifest(entries=[(name,b'x')])
            report,_=self.run_extract(manifest,base)
            self.assertEqual(report['stop_reason'],'unsafe_path')
            self.assertFalse(report['inspection_candidates_usable'])
        for type_ in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE,tarfile.GNUTYPE_SPARSE):
            info=tarfile.TarInfo('unsafe');info.type=type_;info.linkname='../escape'
            manifest,base=self.manifest(raw=gzip.compress(info.tobuf(),mtime=0))
            report,_=self.run_extract(manifest,base)
            self.assertEqual(report['stop_reason'],'links_or_special_file')
        manifest,base=self.manifest(entries=[(member(),b'first'),(member(),b'second')])
        report,out=self.run_extract(manifest,base)
        self.assertEqual(report['stop_reason'],'duplicate_tar_path')
        self.assertEqual((out/'files'/member()).read_bytes(),b'first')
        self.assertFalse(report['inspection_candidates_usable'])

    def test_all_budgets_stop_with_measurements(self):
        manifest,base=self.manifest(entries=[(member(i),random.Random(i).randbytes(1500)) for i in range(3)],split=700)
        cases=[({'max_transfer_bytes':600},'transfer_bytes_limit'),({'max_decompressed_bytes':1024},'decompressed_bytes_limit'),
               ({'max_file_bytes':1000},'file_bytes_limit'),({'max_files':1},'files_limit'),
               ({'max_members':1},'members_limit'),({'max_ranges':1},'ranges_limit')]
        for change,reason in cases:
            with self.subTest(change=change):
                limits=replace(self.limits,**change)
                report,out=self.run_extract(manifest,base,limits)
                self.assertEqual(report['stop_reason'],reason)
                self.assertLessEqual(report['stats']['transfer_bytes'],limits.max_transfer_bytes)
                self.assertLessEqual(report['stats']['decompressed_bytes'],limits.max_decompressed_bytes)
                self.assertFalse(list(out.rglob('*.partial')))
        reserve=report_reserve_bytes(manifest,['**'],self.limits)
        report,out=self.run_extract(manifest,base,replace(self.limits,max_disk_bytes=reserve+1600))
        self.assertEqual(report['stop_reason'],'payload_disk_bytes_limit')
        logical=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        self.assertLessEqual(logical,reserve+1600)

    def test_elapsed_budget_and_explicit_limits(self):
        manifest,base=self.manifest()
        ticks=iter(i*.2 for i in range(10000))
        report,_=self.run_extract(manifest,base,replace(self.limits,max_elapsed_seconds=.5,request_timeout_seconds=.1),clock=lambda:next(ticks))
        self.assertEqual(report['stop_reason'],'elapsed_limit')
        for bad in ({'max_files':True},{'io_chunk_bytes':1},{'request_timeout_seconds':100}):
            with self.assertRaises(PreparationError): replace(self.limits,**bad)
        with self.assertRaises(PreparationError): Limits.from_dict({'max_files':1})
        with self.assertRaises(PreparationError): prepare(manifest,self.root/'empty',[],self.limits,base=base)

    def test_local_source_change_between_ranges(self):
        manifest,base=self.manifest(entries=[(member(),random.Random(0).randbytes(2000))],split=5000)
        stream=SplitStream(manifest,base,Budget(self.limits))
        stream.read()
        path=base/manifest['parts'][0]['source']['path']
        os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+100000))
        with self.assertRaisesRegex(PreparationError,'source_changed'): stream.read()
        self.assertEqual(stream.audit[-1]['bytes_received'],0)

    def test_reuse_verifies_manifest_every_file_hash_and_code(self):
        manifest,base=self.manifest()
        report,out=self.run_extract(manifest,base)
        verify_extraction(out,manifest)
        bad=copy.deepcopy(manifest);bad['parts'][0]['version']='different'
        with self.assertRaisesRegex(PreparationError,'manifest_mismatch'): verify_extraction(out,bad)
        with patch('rewind_archive.matching.code_hashes',return_value={}):
            with self.assertRaisesRegex(PreparationError,'code_changed'): verify_extraction(out)
        file=out/'files'/member();file.write_bytes(b'xyz')
        with self.assertRaisesRegex(PreparationError,'file_hash'): verify_extraction(out)
        file.write_bytes(b'changedsize')
        with self.assertRaisesRegex(PreparationError,'file_size'): verify_extraction(out)

    def test_matching_numeric_order_missing_views_gaps_and_unknown_split(self):
        rgb,base=self.manifest(kind='rgb',entries=[(member(10,'right'),b'r'),(member(2,'left'),b'l'),(member(10,'left'),b'l'),(member(1,'left'),b'l'),(member(1,'right'),b'r')])
        _,rout=self.run_extract(rgb,base)
        pose,base=self.manifest(entries=[(member(i),b'{}') for i in (2,10,1)])
        _,pout=self.run_extract(pose,base)
        report=match_extractions([pout,rout])
        seq=report['sequences'][0]
        self.assertEqual(seq['observed_frames'],[1,2,10])
        self.assertEqual(seq['matched_triplet_frames'],[1,10])
        self.assertEqual(seq['missing_frame_ranges_between_observations'],[[3,9]])
        self.assertEqual(seq['missing_views'],[{'frame_index':2,'missing_views':['right']}])
        self.assertEqual(seq['split'],'unknown');self.assertFalse(report['train_ready'])
        self.assertEqual(seq['original_motion_verification'],'unverified')

    def test_matching_duplicate_unrecognized_and_version_errors(self):
        manifest,base=self.manifest(entries=[(member(),b'{}'),('other/file.txt',b'x')])
        _,out=self.run_extract(manifest,base)
        report=match_extractions([out,out])
        self.assertFalse(report['matching_valid'])
        self.assertIn('duplicate_frame_view',[i['kind'] for i in report['issues']])
        self.assertIn('unrecognized_member_name',[i['kind'] for i in report['issues']])
        with self.assertRaises(PreparationError): identify('x/json/frame_1234567890123.json','pose')
        other,base=self.manifest();other['dataset_version']='another'
        _,otherout=self.run_extract(other,base)
        self.assertIn('dataset_version_mismatch',[i['kind'] for i in match_extractions([out,otherout])['issues']])

    def test_full_crc_local_official_prefix_still_not_whole_archive(self):
        manifest,base=self.manifest();manifest['source_scope']='official_prefix_local'
        report,_=self.run_extract(manifest,base)
        self.assertTrue(report['gzip_eof_crc_valid']);self.assertFalse(report['whole_archive_integrity_verified'])

    def test_cli_fixture_extract_verify_match_and_no_overwrite(self):
        fixture=create_fixture(self.root/'fixture')
        outputs=[]
        with contextlib.redirect_stdout(io.StringIO()):
            for kind in ('rgb','pose'):
                out=self.root/kind;outputs.append(out)
                self.assertEqual(main(['extract','--manifest',str(fixture/f'{kind}_manifest.json'),'--limits',str(fixture/'limits.json'),
                    '--include','**','--output',str(out)]),0)
                self.assertEqual(main(['verify','--manifest',str(fixture/f'{kind}_manifest.json'),'--extraction',str(out)]),0)
            self.assertEqual(main(['match','--extraction',str(outputs[0]),'--extraction',str(outputs[1]),'--output',str(self.root/'matching.json')]),0)
        report=json.loads((self.root/'matching.json').read_text())
        self.assertEqual(report['sequences'][0]['matched_triplet_frames'],[0,1,2,10])
        self.assertFalse(report['train_ready'])
        with self.assertRaises(FileExistsError): prepare(json.loads((fixture/'pose_manifest.json').read_text()),outputs[1],['**'],self.limits,base=fixture)


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def do_GET(self):
        self.server.calls.append(dict(self.headers))
        mode=self.server.mode
        if mode=='redirect':
            self.send_response(302);self.send_header('Location','http://127.0.0.1:1/redirect');self.end_headers();return
        start,end=map(int,self.headers['Range'].split('=')[1].split('-'))
        raw=self.server.data[start:end+1]
        self.send_response(200 if mode=='ignore' else 206)
        self.send_header('Content-Range',f'bytes {start+1 if mode=="range" else start}-{end}/{len(self.server.data)}')
        self.send_header('Content-Length',str(len(raw)+(1 if mode=='length' else 0)))
        self.send_header('ETag','"changed"' if mode=='etag' or (mode=='change_late' and len(self.server.calls)>5) else '"fixture"')
        self.end_headers()
        try:
            if mode=='short':
                self.wfile.write(raw[:max(1,len(raw)//2)]);self.wfile.flush();self.close_connection=True
            elif mode=='slow':
                for byte in raw:
                    if self.server.stop.is_set(): break
                    self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.02)
            elif mode=='stall':
                self.wfile.write(raw[:2]);self.wfile.flush();self.server.stop.wait(.4)
            else:
                self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):
            pass


class HttpArchiveTests(ArchiveTestCase):
    @contextlib.contextmanager
    def server(self,data,mode='normal'):
        server=ThreadingHTTPServer(('127.0.0.1',0),RangeHandler)
        server.data,server.mode,server.calls,server.stop=data,mode,[],threading.Event()
        server.daemon_threads=True
        server.handle_error=lambda *args:None
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            yield server,f'http://127.0.0.1:{server.server_port}/part'
        finally:
            server.stop.set();server.shutdown();server.server_close();thread.join()

    def http_manifest(self,data,url):
        return {'schema_version':'archive_manifest_v1','archive_id':'http_fixture','dataset_version':'synthetic_archive_v1','kind':'pose',
            'source_scope':'synthetic_fixture','total_parts':1,'parts':[{'index':0,'name':'pose.tar.gz000','size_bytes':len(data),'version':'v1',
            'source':{'type':'http','url':url},'identity':{'etag':'"fixture"'}}]}

    def test_http_actual_206_and_ranges(self):
        data=gzip.compress(tar_bytes([(member(),random.Random(3).randbytes(2000))]),mtime=0)
        with self.server(data) as (server,url):
            report,out=self.run_extract(self.http_manifest(data,url),self.root)
            self.assertEqual(report['stop_reason'],'complete')
            self.assertTrue(all(c['If-Match']=='"fixture"' for c in server.calls))
            self.assertEqual(report['stats']['transfer_bytes'],len(data))
            self.assertEqual(len(report['range_audit']),len(server.calls))

    def test_http_rejects_status_range_length_change_redirect_without_body(self):
        data=gzip.compress(tar_bytes([(member(),b'abc')]),mtime=0)
        for mode,reason in (('ignore','range_status_not_206'),('range','content_range_mismatch'),('length','content_length_mismatch'),('etag','source_changed'),('redirect','redirect_rejected')):
            with self.subTest(mode=mode), self.server(data,mode) as (server,url):
                report,_=self.run_extract(self.http_manifest(data,url),self.root)
                self.assertEqual(report['stop_reason'],reason)
                self.assertEqual(report['stats']['transfer_bytes'],0)
                self.assertFalse(report['inspection_candidates_usable'])
                self.assertEqual(len(server.calls),1)

    def test_http_partial_body_error_keeps_actual_consumption(self):
        data=gzip.compress(tar_bytes([(member(),random.Random(3).randbytes(2000))]),mtime=0)
        with self.server(data,'short') as (server,url):
            report,_=self.run_extract(self.http_manifest(data,url),self.root)
            self.assertEqual(report['stop_reason'],'short_http_body')
            self.assertEqual(report['stats']['transfer_bytes'],256)
            self.assertEqual(report['range_audit'][0]['bytes_received'],256)
            self.assertEqual(report['range_audit'][0]['status'],'failed')
            self.assertEqual(len(server.calls),1)

    def test_http_late_source_change_rejects_retained_files(self):
        data=gzip.compress(tar_bytes([(member(),random.Random(3).randbytes(1200)),(member(1),random.Random(9).randbytes(9000))]),mtime=0)
        with self.server(data,'change_late') as (server,url):
            report,out=self.run_extract(self.http_manifest(data,url),self.root)
            self.assertEqual(report['stop_reason'],'source_changed')
            self.assertGreater(len(report['files']),0)
            with self.assertRaisesRegex(PreparationError,'unusable'): verify_extraction(out)

    def test_http_slow_trickle_and_stall_deadline(self):
        data=gzip.compress(tar_bytes([(member(),b'abc')]),mtime=0)
        for mode in ('slow','stall'):
            with self.subTest(mode=mode), self.server(data,mode) as (server,url):
                limits=replace(self.limits,max_elapsed_seconds=.16,request_timeout_seconds=.08)
                report,_=self.run_extract(self.http_manifest(data,url),self.root,limits)
                self.assertIn(report['stop_reason'],('elapsed_limit','http_body_failed'))
                self.assertGreater(report['stats']['transfer_bytes'],0)
                self.assertLess(report['stats']['transfer_bytes'],len(data))
                self.assertLess(report['stats']['elapsed_seconds'],.35)
                self.assertEqual(report['range_audit'][0]['bytes_received'],report['stats']['transfer_bytes'])

    def test_drive_adapter_injected_session_metadata_and_read1(self):
        raw=b'0123456789'
        part={'index':0,'name':'pose.tar.gz000','size_bytes':len(raw),'version':'v1','source':{'type':'drive','file_id':'fake_id'},
            'identity':{'version':'7','modified_time':'2026-10-14T00:00:00.000Z'}}
        class Raw(io.BytesIO):
            def read1(self,size,decode_content=False):
                return super().read1(size)
        class Session:
            def __init__(self): self.calls=[];self.version='7';self.with_length=True;self.malformed=None;self.bad_metadata=False
            def get(self,url,**kwargs):
                self.calls.append((url,kwargs))
                if 'fields=' in url:
                    body=canonical(self.malformed if self.bad_metadata else {'size':str(len(raw)),'version':self.version,'modifiedTime':part['identity']['modified_time']})
                    status,headers=200,({'Content-Length':str(len(body))} if self.with_length else {})
                else:
                    body=raw[:5];status,headers=206,{'Content-Length':'5','Content-Range':'bytes 0-4/10'}
                response=SimpleNamespace(status_code=status,headers=headers,raw=Raw(body))
                response.close=response.raw.close
                return response
        session=Session();source=DriveSource(part,session);budget=Budget(self.limits);received=[]
        self.assertEqual(source.read_range(0,4,budget,received.append),b'01234')
        self.assertEqual(len(session.calls),3)
        self.assertTrue(all(k['stream'] and not k['allow_redirects'] for _,k in session.calls))
        self.assertGreater(budget.transfer_bytes,5)
        session.version='8'
        with self.assertRaisesRegex(PreparationError,'source_changed'): source.read_range(0,4,budget,received.append)
        session.with_length=False
        session.version='7'
        self.assertEqual(source.read_range(0,4,budget,received.append),b'01234')
        session.with_length=True
        session.bad_metadata=True
        for bad in (None,[],"text",False,23):
            session.malformed=bad
            with self.assertRaisesRegex(PreparationError,'metadata_response'): source.read_range(0,4,budget,received.append)
        fake=SessionResponse(SimpleNamespace(status_code=200,headers={},raw=SimpleNamespace(read=lambda n:b'a')))
        with self.assertRaisesRegex(PreparationError,'read1_required'): fake.read1(1)


if __name__=='__main__':
    unittest.main()
