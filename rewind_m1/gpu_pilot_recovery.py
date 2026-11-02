"""학습/torch import 없는 제한된 pilot 결과 ZIP 회수. 원본 소스는 포함하지 않는다."""
import hashlib
import json
from pathlib import Path
import stat
import zipfile

MIB=1024**2
MAX_INPUT=128*MIB
MAX_ZIP=128*MIB
MAX_TOTAL=256*MIB
NAME='synthetic_gpu_pilot_result.zip'


def _zip_check(path):
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        if len(infos)>2048 or sum(i.file_size for i in infos)>MAX_INPUT:
            raise ValueError('pilot_recovery_zip_expansion_limit')
        if any(i.flag_bits&1 for i in infos) or archive.testzip() is not None:
            raise ValueError('pilot_recovery_zip_crc')


def _regular(path):
    for part in (path,*path.parents):
        info=part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('pilot_recovery_link')
    if not path.is_file(): raise ValueError('pilot_recovery_file')


def recover_bundle(root):
    """기존 결과만 읽는다. 이미 완성된 ZIP은 CRC/hash 확인 후 재사용한다."""
    root=Path(root).absolute()
    if not root.is_dir(): raise ValueError('pilot_recovery_root_missing')
    destination=root/NAME; receipt=root/'export_report.json'
    if destination.exists():
        _regular(destination)
        if destination.stat().st_size>MAX_ZIP: raise ValueError('pilot_recovery_zip_limit')
        if not receipt.is_file(): raise ValueError('pilot_recovery_zip_without_completion_receipt')
        _zip_check(destination)
        actual=hashlib.sha256(destination.read_bytes()).hexdigest()
        _regular(receipt)
        if receipt.stat().st_size>65536: raise ValueError('pilot_recovery_receipt_limit')
        previous=json.loads(receipt.read_text(encoding='utf8'))
        if previous['sha256']!=actual: raise ValueError('pilot_recovery_existing_zip_changed')
        return previous
    else:
        files=[]; total=0
        for path in sorted(root.rglob('*')):
            relative=path.relative_to(root)
            if relative.parts[0]=='source' or path in (destination,receipt): continue
            if path.is_dir(): continue
            _regular(path)
            size=path.stat().st_size; total+=size
            if len(files)>=2048 or total>MAX_INPUT: raise ValueError('pilot_recovery_input_limit')
            files.append((path,relative.as_posix(),size,path.stat().st_mtime_ns))
        if not files: raise ValueError('pilot_recovery_no_artifacts')
        # 실패한 ZIP도 보존한다. 이후 재회수는 검증된 기존 ZIP만 재사용한다.
        with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path,name,size,mtime in files:
                archive.write(path,name)
                if (path.stat().st_size,path.stat().st_mtime_ns)!=(size,mtime):
                    raise ValueError('pilot_recovery_artifact_changed')
                if destination.stat().st_size>MAX_ZIP or total+destination.stat().st_size>MAX_TOTAL:
                    raise ValueError('pilot_recovery_output_limit')
        if destination.stat().st_size>MAX_ZIP: raise ValueError('pilot_recovery_zip_limit')
        _zip_check(destination)
        result={'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                'bytes':destination.stat().st_size,'input_bytes':total,'entries':len(files),
                'training_reexecuted':False,'source_excluded':True}
    with receipt.open('x',encoding='utf8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    return result
