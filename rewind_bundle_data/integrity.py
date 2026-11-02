"""Torch/학습 import 없이 완료 bundle 전체 byte와 고정 pin을 검사한다."""
import hashlib
from pathlib import Path
import re
import time

from rewind_archive.contracts import digest, safe_path
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _no_link, _regular_read

MIB = 1024**2


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(path):
    path = Path(path).absolute()
    require(not str(path).replace('\\','/').startswith(('//','/content/drive/')), 'reader_local_only')
    for p in (path,*path.parents):
        _no_link(p)
    return path.resolve(strict=True)


class Integrity:
    def __init__(self, root, expected):
        import psutil
        self.root = local(root)
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.peak_rss = 0
        self.parsed_reads = {}
        require(isinstance(expected,str) and re.fullmatch('[a-f0-9]{64}',expected), 'reader_completion_pin_required')
        raw = _regular_read(local(self.root/'COMPLETE.json'),MIB)
        marker = read_json(raw)
        require(marker.get('schema_version') == 'prepared_bundle_completion_v1' and
                marker.get('content_sha256') == expected == digest({k:v for k,v in marker.items() if k != 'content_sha256'}) and
                marker.get('status') == 'prepared_not_training_release' and marker.get('train_ready') is False,
                'reader_completion_pin_or_state')
        self.marker_sha256, self.marker_file_sha256 = expected, sha(raw)
        self.entries = {}
        require(isinstance(marker.get('files'),list) and 1 <= len(marker['files']) <= 512, 'reader_inventory_limit')
        for row in marker['files']:
            require(isinstance(row,dict) and set(row) == {'path','size_bytes','sha256'}, 'reader_inventory_row')
            name = safe_path(row['path'])
            require(name == row['path'] and name != 'COMPLETE.json' and name.casefold() not in {n.casefold() for n in self.entries}
                    and type(row['size_bytes']) is int and 0 <= row['size_bytes'] <= 64*MIB and
                    isinstance(row['sha256'],str) and re.fullmatch('[a-f0-9]{64}',row['sha256']), 'reader_inventory_row')
            self.entries[name] = row
        require(list(self.entries) == sorted(self.entries), 'reader_inventory_order')
        total = sum(r['size_bytes'] for r in self.entries.values())
        require(type(marker.get('total_bytes')) is int and total == marker['total_bytes'] <= 128*MIB-8192, 'reader_inventory_bytes')
        actual = set()
        for count,path in enumerate(self.root.rglob('*')):
            self.check()
            require(count < 4096, 'reader_inventory_limit')
            path = local(path)
            require(path.is_relative_to(self.root), 'reader_inventory_escape')
            if path.is_dir():
                continue
            name = path.relative_to(self.root).as_posix()
            if name != 'COMPLETE.json':
                actual.add(name)
        require(actual == set(self.entries), 'reader_inventory_mismatch')
        # 생성 시 전체 I/O 검사다. JSON parse 및 PNG decode/cache는 하지 않는다.
        for name in self.entries:
            self.verify_file(name)
        pins = marker.get('input_sha256')
        require(isinstance(pins,dict) and set(pins) == {'plan','state','matching','selection','spec','calibration','motion_map','train','validation','test'}, 'reader_metadata_pins')
        for name,expected_sha in pins.items():
            relative = f'metadata/split/{name}.txt' if name in ('train','validation','test') else f'metadata/{name}.json'
            require(self.entries.get(relative,{}).get('sha256') == expected_sha, 'reader_metadata_pin')
        report = self.doc('report.json',MIB)
        require(report.get('status') == marker['status'] and report.get('train_ready') is False and
                report.get('input_sha256') == pins and report.get('training_release') == 'not_implemented', 'reader_report_gate')
        self.check_marker()

    def check(self):
        self.peak_rss = max(self.peak_rss,self.process.memory_info().rss)
        require(self.peak_rss <= 4096*MIB, 'reader_rss_limit')
        require(time.monotonic()-self.started <= 120, 'reader_elapsed_limit')

    def begin(self):
        # API 호출별 협조적 예산. 세션 전체의 벽시계 제한은 아니다.
        self.started = time.monotonic()
        self.check_marker()

    def path(self,name):
        require(isinstance(name,str) and safe_path(name) == name and name in self.entries, 'reader_unlisted_path')
        result = local(self.root/name)
        require(result.is_relative_to(self.root) and result.is_file(), 'reader_file_escape')
        return result

    def verify_file(self,name):
        self.check()
        path = self.path(name); row = self.entries[name]
        require(path.stat().st_size == row['size_bytes'], 'reader_file_size')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(65536):
                self.check(); h.update(chunk)
        require(path.stat().st_size == row['size_bytes'] and h.hexdigest() == row['sha256'], 'reader_file_hash')

    def read(self,name,maximum):
        self.check()
        require(name in self.entries and self.entries[name]['size_bytes'] <= maximum, 'reader_document_limit')
        row = self.entries[name]
        raw = _regular_read(self.path(name),maximum,row['size_bytes'])
        require(sha(raw) == row['sha256'], 'reader_file_hash')
        self.check()
        return raw

    def doc(self,name,maximum=MIB):
        self.parsed_reads[name] = self.parsed_reads.get(name,0)+1
        return read_json(self.read(name,maximum))

    def check_marker(self):
        self.check()
        require(sha(_regular_read(local(self.root/'COMPLETE.json'),MIB)) == self.marker_file_sha256, 'reader_completion_changed')
