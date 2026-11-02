"""실제 notebook cell을 mock upload로 실행한다. 설치/Google/scan 호출 없음."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile,ZIP_STORED

ROOT=Path(__file__).resolve().parents[1]
NB=json.loads((ROOT/'notebooks/archvis_rgb_discovery.ipynb').read_bytes())
PREP=''.join(NB['cells'][2]['source'])
RECOVERY=''.join(NB['cells'][4]['source'])


def sha(raw):return hashlib.sha256(raw).hexdigest()


def archive(files):
    buffer=io.BytesIO()
    with ZipFile(buffer,'w',ZIP_STORED) as z:
        for name,data in files.items():z.writestr(name,data)
    return buffer.getvalue()


class NotebookUploadTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.revision='a'*40
        self.source=archive({'SOURCE_REVISION.txt':self.revision+'\n','package/data.py':'marker_payload = 123\n'})
        selection={'fixture':'exact_selection'}
        self.selection_sha=sha(json.dumps(selection,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
        self.input=archive({'config.json':json.dumps({'selection':selection}),'pose_catalog.json':'{}'})
        previous_path=list(sys.path);previous_bytecode=sys.dont_write_bytecode
        self.addCleanup(lambda:sys.path.__setitem__(slice(None),previous_path))
        self.addCleanup(setattr,sys,'dont_write_bytecode',previous_bytecode)

    def context(self,prefix=''):
        return {'SOURCE_ZIP_SHA256':sha(self.source),'INPUT_ZIP_SHA256':sha(self.input),
            'SELECTION_SHA256':self.selection_sha,'SOURCE_REVISION':self.revision,'NOTEBOOK_REVISION':'b'*40,
            'SOURCE_ROOT':str(self.base/(prefix+'source')),'INPUT_ROOT':str(self.base/(prefix+'inputs')),
            'WORKSPACE':str(self.base/(prefix+'work')),'SNAPSHOT_DIRECTORY':str(self.base/(prefix+'snapshots')),
            'INPUT_SNAPSHOT_DIRECTORY':None,'JOB_ID':'fixed_original_job','RECOVERY_ZIP':str(self.base/(prefix+'recovery.zip'))}

    def execute(self,code,context,uploads=()):
        google=ModuleType('google');google.__path__=[]
        colab=ModuleType('google.colab');google.colab=colab
        uploads=iter(uploads);self.upload_count=0;self.downloads=[]
        def upload():self.upload_count+=1;return next(uploads)
        colab.files=SimpleNamespace(upload=upload,download=self.downloads.append)
        indexed=ModuleType('indexed_gzip');indexed.__version__='1.10.3'
        with patch.dict(sys.modules,{'google':google,'google.colab':colab,'indexed_gzip':indexed}), \
             patch.object(subprocess,'check_call',return_value=0) as install:
            try:exec(compile(code,'actual_notebook_cell','exec'),context)
            finally:self.install_calls=install.call_count

    def normal(self,context):
        self.execute(PREP,context,[{'source.zip':self.source},{'input.zip':self.input}])

    def test_source_only_upload_creates_no_preparation_roots(self):
        context=self.context()
        with self.assertRaisesRegex(AssertionError,'input ZIP'):
            self.execute(PREP,context,[{'source.zip':self.source},{}])
        self.assertFalse(Path(context['SOURCE_ROOT']).exists());self.assertFalse(Path(context['INPUT_ROOT']).exists())
        self.assertFalse(Path(context['WORKSPACE']).exists());self.assertEqual(self.install_calls,0)

    def test_normal_both_and_exact_existing_source_reuse(self):
        context=self.context()
        # 이전 677bc7f 준비 실패로 source만 정상 추출된 상태를 재현한다.
        with ZipFile(io.BytesIO(self.source)) as z:z.extractall(context['SOURCE_ROOT'])
        p=Path(context['SOURCE_ROOT'])/'package/data.py';before=(p.read_bytes(),p.stat().st_mtime_ns)
        self.normal(context)
        self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),before)
        self.assertTrue((Path(context['INPUT_ROOT'])/'config.json').is_file())
        self.normal(context) # 양쪽 모두 동일한 경우에도 덮어쓰지 않는다.
        self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),before)

    def test_fresh_normal_two_zip_preparation(self):
        context=self.context();self.normal(context)
        self.assertEqual((Path(context['SOURCE_ROOT'])/'SOURCE_REVISION.txt').read_text().strip(),self.revision)
        self.assertEqual(self.upload_count,2);self.assertEqual(self.install_calls,1)
        self.assertFalse(Path(context['WORKSPACE']).exists())

    def test_existing_tamper_incomplete_or_extra_rejected_before_other_root_creation(self):
        for case in ('tamper','missing','extra','input_tamper'):
            context=self.context(case)
            if case=='input_tamper':
                with ZipFile(io.BytesIO(self.input)) as z:z.extractall(context['INPUT_ROOT'])
                (Path(context['INPUT_ROOT'])/'config.json').write_text('{}')
                untouched=Path(context['SOURCE_ROOT'])
            else:
                with ZipFile(io.BytesIO(self.source)) as z:z.extractall(context['SOURCE_ROOT'])
                p=Path(context['SOURCE_ROOT'])/'package/data.py'
                if case=='tamper':p.write_text('tampered')
                elif case=='missing':p.unlink()
                else:(p.parent/'extra.txt').write_text('extra')
                untouched=Path(context['INPUT_ROOT'])
            with self.assertRaises(AssertionError):self.normal(context)
            self.assertFalse(untouched.exists());self.assertEqual(self.install_calls,0)

    def test_crc_and_unsafe_path_checked_before_any_extraction(self):
        corrupted=bytearray(self.source);offset=corrupted.find(b'marker_payload');self.assertGreater(offset,0);corrupted[offset]^=1
        for prefix,source in [('crc',bytes(corrupted)),('path',archive({'../escape.txt':'unsafe'}))]:
            context=self.context(prefix);context['SOURCE_ZIP_SHA256']=sha(source)
            with self.assertRaises(Exception):
                self.execute(PREP,context,[{'source.zip':source},{'input.zip':self.input}])
            self.assertFalse(Path(context['SOURCE_ROOT']).exists());self.assertFalse(Path(context['INPUT_ROOT']).exists())
            self.assertEqual(self.install_calls,0)

    def test_consumed_workspace_blocks_before_upload(self):
        context=self.context();Path(context['WORKSPACE']).mkdir()
        with self.assertRaisesRegex(AssertionError,'소비된 job'):self.execute(PREP,context)
        self.assertEqual(self.upload_count,0);self.assertEqual(self.install_calls,0)

    def test_config_and_source_absent_preparation_recovery_and_reuse(self):
        context=self.context()
        self.execute(RECOVERY,context)
        original_target=Path(context['RECOVERY_ZIP'])
        target=original_target.with_name(original_target.stem.removesuffix('_result')+'_preparation_result'+original_target.suffix)
        before=sha(target.read_bytes())
        self.assertFalse(original_target.exists())
        with ZipFile(target) as z:
            info=json.loads(z.read('recovery.json'))
            self.assertFalse(info['input_root_present']);self.assertFalse(info['report_present'])
            self.assertEqual(info['selection_sha256'],self.selection_sha)
            self.assertEqual(info['source_revision'],self.revision)
        self.execute(RECOVERY,context)
        self.assertEqual(sha(target.read_bytes()),before);self.assertEqual(self.downloads,[str(target)])
        self.assertEqual(self.upload_count,0);self.assertEqual(self.install_calls,0)
        self.assertFalse(Path(context['INPUT_ROOT']).exists())
        # 준비실패 ZIP이 있어도 WORKSPACE 생성 후에는 원래 실제 결과 ZIP으로 회수한다.
        Path(context['WORKSPACE']).mkdir();Path(context['SOURCE_ROOT']).mkdir()
        (Path(context['SOURCE_ROOT'])/'SOURCE_REVISION.txt').write_text(self.revision+'\n')
        self.execute(RECOVERY,context)
        with ZipFile(original_target) as z:
            info=json.loads(z.read('recovery.json'))
            self.assertFalse(info['report_present'])
            self.assertEqual(info['execution_status'],'report_missing_execution_state_unknown')
        self.assertEqual(sha(target.read_bytes()),before)
        self.assertFalse(Path(context['INPUT_ROOT']).exists())
