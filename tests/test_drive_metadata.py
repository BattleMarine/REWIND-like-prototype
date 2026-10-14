"""실제 Drive/auth 없이 metadata→manifest→bounded pilot 계약을 검증한다."""

import copy
import gzip
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from rewind_archive.contracts import Limits, PreparationError, canonical, digest
from rewind_archive.drive_metadata import MetadataLimits, build_drive_manifest, save_metadata_bundle, validate_metadata_result, validate_selection
from rewind_archive.fixture import SEQUENCE, tar_bytes
from rewind_archive.pilot import run_metadata_pilot, run_pilot


class Raw(io.BytesIO):
    def read1(self, size, decode_content=False):
        return super().read1(size)


class FakeDrive:
    _max_refresh_attempts=0

    def __init__(self, selection, data=None):
        self.calls=[];self.responses=[];self.docs={};self.data=data or {}
        self.protocol={};self.raw_override={};self.error_on=None
        for p in selection['parts']:
            self.docs[p['file_id']]={'id':p['file_id'],'name':p['name'],'size':str(len(self.data.get(p['file_id'],b'x'*100))),
                'version':'9','modifiedTime':'2026-10-14T00:00:00.000Z','mimeType':'application/octet-stream','trashed':False,
                'capabilities':{'canDownload':True}}

    def get(self, url, **kwargs):
        self.calls.append((url,kwargs))
        file_id=urlsplit(url).path.split('/')[-1]
        if self.error_on==file_id:
            raise RuntimeError('sk-proj-secret https://private.example/?token=secret')
        query=parse_qs(urlsplit(url).query)
        if query.get('alt')==['media']:
            data=self.data[file_id]
            start,end=map(int,kwargs['headers']['Range'].split('=')[1].split('-'))
            body=data[start:end+1];status=206
            headers={'Content-Length':str(len(body)),'Content-Range':f'bytes {start}-{end}/{len(data)}'}
        else:
            body=self.raw_override.get(file_id,canonical(self.docs[file_id]));status=200
            headers={'Content-Length':str(len(body))}
        status,headers=self.protocol.get(file_id,(status,headers))
        raw=Raw(body)
        response=SimpleNamespace(status_code=status,headers=headers,raw=raw,close=raw.close)
        self.responses.append(response)
        return response


class DriveMetadataTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'outputs/metadata_test_tmp';root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.selection={'schema_version':'drive_part_selection_v1','archive_id':'selected_pose','dataset_version':'unverified_test',
            'kind':'pose','total_parts':2,'parts':[{'index':i,'name':f'pose.tar.gz{i:03d}','file_id':f'fake_id_{i}'} for i in range(2)]}
        self.limits=MetadataLimits()

    def build(self, session=None, limits=None, selection=None, **kwargs):
        return build_drive_manifest(selection or self.selection,session if session is not None else FakeDrive(self.selection),limits or self.limits,**kwargs)

    def test_success_fetches_only_selected_metadata_and_exact_identity(self):
        session=FakeDrive(self.selection)
        session.docs['fake_id_0']['ignored_debug']='sk-proj-secret'
        result=self.build(session)
        report=result['report'];manifest=validate_metadata_result(result)
        self.assertTrue(report['manifest_ready']);self.assertEqual(report['stats']['requests'],2)
        self.assertEqual(report['stats']['validated_files'],2)
        self.assertEqual(manifest['parts'][0]['version'],'9')
        self.assertEqual(manifest['parts'][0]['identity']['version'],'9')
        self.assertEqual(manifest['parts'][0]['size_bytes'],100)
        self.assertFalse(report['media_requested_by_builder']);self.assertFalse(report['train_ready'])
        self.assertEqual(report['selection_sha256'],digest(self.selection))
        self.assertNotIn('secret',json.dumps(result))
        for url,kwargs in session.calls:
            self.assertNotIn('alt=media',url);self.assertNotIn('/files?',url)
            self.assertEqual(set(parse_qs(urlsplit(url).query)),{'fields'})
            self.assertIn('capabilities(canDownload)',url)
            self.assertTrue(kwargs['stream']);self.assertFalse(kwargs['allow_redirects'])
        self.assertTrue(all(r.raw.closed for r in session.responses))

    def test_selection_rejects_duplicates_gaps_names_and_extra_before_network(self):
        changes=[lambda d:d['parts'][0].update(index=1),lambda d:d['parts'][1].update(index=3),
            lambda d:d['parts'][1].update(file_id='fake_id_0'),lambda d:d['parts'][1].update(name='pose.tar.gz000'),
            lambda d:d['parts'][0].update(name='../pose.tar.gz000'),lambda d:d['parts'][1].update(name='another.tar.gz001'),lambda d:d.update(token='secret'),
            lambda d:d.update(parts=[]),lambda d:d.update(total_parts=1)]
        for mutate in changes:
            bad=copy.deepcopy(self.selection);mutate(bad);session=FakeDrive(self.selection)
            result=self.build(session,selection=bad)
            self.assertIsNone(result['manifest']);self.assertFalse(result['report']['manifest_ready'])
            self.assertEqual(session.calls,[])
        with self.assertRaises(PreparationError): validate_selection(None)

    def test_metadata_fields_types_version_size_and_identity_rejected(self):
        changes=[lambda d:d.update(id='different'),lambda d:d.update(name='wrong000'),lambda d:d.pop('version'),
            lambda d:d.update(version=''),lambda d:d.update(version=None),lambda d:d.update(version=9),lambda d:d.update(version='0'),
            lambda d:d.update(size='0'),lambda d:d.update(size=100),lambda d:d.update(size='01'),lambda d:d.update(size='-1'),lambda d:d.update(size=str(2**63)),
            lambda d:d.update(modifiedTime='unknown'),lambda d:d.update(modifiedTime='2026-02-31T00:00:00Z'),
            lambda d:d.update(trashed=True),lambda d:d.update(trashed=0),lambda d:d.update(capabilities={}),
            lambda d:d.update(capabilities={'canDownload':False}),lambda d:d.update(capabilities={'canDownload':1})]
        for mutate in changes:
            session=FakeDrive(self.selection);mutate(session.docs['fake_id_0'])
            result=self.build(session)
            self.assertIsNone(result['manifest'])
            self.assertEqual(result['report']['stats']['validated_files'],0)
            self.assertEqual(len(session.calls),1)

    def test_native_docs_shortcut_folder_and_missing_mime_fail(self):
        for mime in ('application/vnd.google-apps.document','Application/Vnd.Google-Apps.Document','application/vnd.google-apps.shortcut','application/vnd.google-apps.folder',None,'not/mime?token'):
            session=FakeDrive(self.selection);session.docs['fake_id_0']['mimeType']=mime
            result=self.build(session)
            self.assertIsNone(result['manifest'])
            self.assertEqual(result['report']['stats']['requests'],1)

    def test_protocol_status_length_encoding_and_metadata_json(self):
        protocols=[(206,{'Content-Length':'100'}),(302,{'Content-Length':'100'}),(200,{}),
            (200,{'Content-Length':'65537'}),(200,{'Content-Length':'0'}),
            (200,{'Content-Length':'100','Content-Encoding':'gzip'})]
        for protocol in protocols:
            session=FakeDrive(self.selection);session.protocol['fake_id_0']=protocol
            result=self.build(session)
            self.assertIsNone(result['manifest']);self.assertEqual(result['report']['stats']['transfer_bytes'],0)
        for raw in (b'null',b'[]',b'"string"',b'{"id":1,"id":2}',b'{"debug":"sk-proj-secret"}'):
            session=FakeDrive(self.selection);session.raw_override['fake_id_0']=raw
            result=self.build(session)
            self.assertIsNone(result['manifest']);self.assertNotIn('sk-proj-secret',json.dumps(result))

    def test_each_budget_and_missing_session_fail_without_manifest(self):
        cases=[({'max_files':1},0),({'max_requests':1},1),({'max_transfer_bytes':10},1),({'max_response_bytes':10},1)]
        for changes,calls in cases:
            session=FakeDrive(self.selection);result=self.build(session,replace(self.limits,**changes))
            self.assertIsNone(result['manifest']);self.assertEqual(len(session.calls),calls)
            self.assertLessEqual(result['report']['stats']['transfer_bytes'],result['report']['limits']['max_transfer_bytes'])
        ticks=iter(i*.1 for i in range(1000))
        result=self.build(limits=replace(self.limits,max_elapsed_seconds=.2,request_timeout_seconds=.1),clock=lambda:next(ticks))
        self.assertIsNone(result['manifest']);self.assertEqual(result['report']['stop_reason'],'elapsed_limit')
        result=build_drive_manifest(self.selection,None,self.limits)
        self.assertEqual(result['report']['stop_reason'],'drive_session_required')
        for bad in ({'max_files':True},{'max_elapsed_seconds':float('nan')},{'max_requests':1000}):
            with self.assertRaises(PreparationError): replace(self.limits,**bad)
        with self.assertRaises(PreparationError): MetadataLimits.from_dict({'max_files':1})

    def test_intermediate_failure_is_sanitized_and_bundle_has_no_manifest(self):
        session=FakeDrive(self.selection);session.error_on='fake_id_1'
        result=self.build(session)
        self.assertIsNone(result['manifest']);self.assertEqual(result['report']['stats']['validated_files'],1)
        self.assertEqual(result['report']['stats']['requests'],2)
        self.assertNotIn('secret',json.dumps(result));self.assertNotIn('private.example',json.dumps(result))
        output=save_metadata_bundle(result,self.root/'failed')
        self.assertFalse((output/'archive_manifest.json').exists())
        self.assertTrue((output/'metadata_report.json').exists())
        with self.assertRaises(PreparationError): validate_metadata_result(result)

    def test_bundle_no_overwrite_report_hash_and_manifest_unchanged(self):
        result=self.build();saved=copy.deepcopy(result)
        out=save_metadata_bundle(result,self.root/'success')
        before={p.name:p.read_bytes() for p in out.iterdir()}
        with self.assertRaises(PreparationError): save_metadata_bundle(result,out)
        self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
        self.assertEqual(result,saved)
        altered=copy.deepcopy(result);altered['manifest']['parts'][0]['identity']['version']='10'
        with self.assertRaises(PreparationError): validate_metadata_result(altered)
        altered=copy.deepcopy(result);altered['report']['stats']['requests']=999
        with self.assertRaises(PreparationError): validate_metadata_result(altered)
        with patch('rewind_archive.drive_metadata.code_hashes',return_value={}):
            with self.assertRaises(PreparationError): validate_metadata_result(result)

    def test_metadata_then_explicit_pilot_and_verify_report_persisted(self):
        data=gzip.compress(tar_bytes([(f'{SEQUENCE}/json/frame_0.json',b'{}')]),mtime=0)
        selection=copy.deepcopy(self.selection);selection['parts']=selection['parts'][:1];selection['total_parts']=1
        session=FakeDrive(selection,{'fake_id_0':data})
        result=build_drive_manifest(selection,session,self.limits)
        self.assertTrue(result['report']['manifest_ready']);self.assertEqual(len(session.calls),1)
        report=run_metadata_pilot(result,self.root/'pilot',[f'{SEQUENCE}/json/*'],Limits(),session)
        self.assertEqual(report['complete_files'],1);self.assertEqual(report['verification']['status'],'passed')
        self.assertFalse(report['train_ready']);self.assertEqual(report['scope'],'single_archive_preparation_only')
        self.assertEqual(json.loads((self.root/'pilot/pilot_report.json').read_text()),report)
        self.assertEqual(report['metadata_report_content_sha256'],result['report']['content_sha256'])
        self.assertEqual(report['metadata_stage_stats'],result['report']['stats'])
        self.assertEqual(report['metadata_stage_limits'],result['report']['limits'])
        self.assertTrue(any('alt=media' in u for u,_ in session.calls))

    def test_changed_identity_blocks_media_and_failed_metadata_blocks_pilot(self):
        session=FakeDrive(self.selection);result=self.build(session)
        session.docs['fake_id_0']['version']='10'
        report=run_metadata_pilot(result,self.root/'changed',['**'],Limits(),session)
        self.assertEqual(report['stop_reason'],'source_changed')
        self.assertEqual(report['complete_files'],0);self.assertFalse(report['inspection_candidates_usable'])
        self.assertFalse(any('alt=media' in u for u,_ in session.calls))
        failed=build_drive_manifest(self.selection,None,self.limits)
        calls=len(session.calls)
        with self.assertRaises(PreparationError): run_metadata_pilot(failed,self.root/'never',['**'],Limits(),session)
        self.assertEqual(len(session.calls),calls);self.assertFalse((self.root/'never').exists())


if __name__=='__main__':
    unittest.main()
