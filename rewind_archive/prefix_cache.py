"""단일 Drive part의 검증된 연속 압축 prefix. gzip 상태/seek index는 보관하지 않는다."""

import hashlib
import json
import os
import re
import stat
import struct
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, BadZipFile

from .contracts import PreparationError,canonical,digest,validate_manifest

MIB=1024**2
CACHE_DISK_BYTES=128*MIB
LOGICAL_BYTES=96*MIB
LOCAL_IO_BYTES=196*MIB  # import96 + replay96 + ZIP/manifest 여유4
MANIFEST_BYTES=MIB
MAX_CHUNKS=512
EXPORT_SECONDS=60.
EXPORT_IO_BYTES=384*MIB  # 복사·내용검증·ZIP hash의 세 번 읽기 + 여유


def linked(path):
    return path.is_symlink() or bool(getattr(path,"is_junction",lambda:False)())


def validate_input_spec(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={"path","manifest_sha256"}:
        raise PreparationError("cache_input_fields")
    if not isinstance(value["path"],str) or not value["path"] or len(value["path"].encode("utf-8"))>4096:
        raise PreparationError("cache_input_path")
    if not isinstance(value["manifest_sha256"],str) or not re.fullmatch("[0-9a-f]{64}",value["manifest_sha256"]):
        raise PreparationError("cache_input_hash")
    return dict(value)


def io_read(budget,stream,size):
    remaining=LOCAL_IO_BYTES-budget.local_io_bytes
    budget.check_time()
    if remaining<=0:raise PreparationError("cache_local_io_limit")
    data=stream.read(min(size,remaining))
    budget.local_io_bytes+=len(data)
    budget.check_time()
    return data


class CountedFile:
    def __init__(self,stream,budget=None,read_callback=None):self.stream=stream;self.budget=budget;self.read_callback=read_callback
    def read(self,size=-1):
        if self.read_callback:return self.read_callback(self.stream,size)
        return io_read(self.budget,self.stream,LOCAL_IO_BYTES if size<0 else size)
    def seek(self,*args):return self.stream.seek(*args)
    def tell(self):return self.stream.tell()
    def seekable(self):return True


def open_bundle(path,budget=None,read_callback=None):
    path=Path(path)
    if linked(path) or any(linked(p) for p in path.parents) or not path.is_file() or path.stat().st_size>CACHE_DISK_BYTES:
        raise PreparationError("cache_bundle_path_or_size")
    stream=path.open("rb")
    try:
        counted=CountedFile(stream,budget,read_callback) if budget or read_callback else stream
        # 고정 두 member/STORED export 형식만 받는다. ZipFile의 대량 central-directory 할당 전에 거부한다.
        counted.seek(-22,2);tail=counted.read(22)
        if len(tail)!=22:raise PreparationError("cache_bundle_directory")
        magic,disk,central_disk,count_disk,count,total,offset,comment=struct.unpack("<4s4H2LH",tail)
        if magic!=b"PK\x05\x06" or disk or central_disk or count_disk!=2 or count!=2 or total>4096 or comment or offset+total!=path.stat().st_size-22:
            raise PreparationError("cache_bundle_directory")
        counted.seek(0);bundle=ZipFile(counted)
        infos=bundle.infolist()
        if len(infos)!=2 or {i.filename for i in infos}!={"manifest.json","prefix.bin"}:
            raise PreparationError("cache_bundle_members")
        for info in infos:
            mode=(info.external_attr>>16)&0xffff
            if info.compress_type!=ZIP_STORED or info.flag_bits&1 or info.extra or info.comment or (stat.S_IFMT(mode) not in (0,stat.S_IFREG)):
                raise PreparationError("cache_bundle_encoding_or_link")
        if bundle.getinfo("manifest.json").file_size>MANIFEST_BYTES or sum(i.file_size for i in infos)>CACHE_DISK_BYTES:
            raise PreparationError("cache_bundle_size")
        return stream,bundle
    except BaseException:
        stream.close();raise


def read_manifest(bundle):
    with bundle.open("manifest.json") as f:raw=f.read(MANIFEST_BYTES+1)
    if len(raw)>MANIFEST_BYTES:raise PreparationError("cache_manifest_size")
    try:doc=json.loads(raw)
    except (ValueError,UnicodeError):raise PreparationError("cache_manifest_json") from None
    fields={"schema_version","archive_manifest","archive_manifest_sha256","prefix_bytes","prefix_sha256","chunks","status","producer_code_sha256","content_sha256"}
    if not isinstance(doc,dict) or set(doc)!=fields or doc["schema_version"]!="bounded_prefix_cache_v1":
        raise PreparationError("cache_manifest_fields")
    if doc["content_sha256"]!=digest({k:v for k,v in doc.items() if k!="content_sha256"}):raise PreparationError("cache_manifest_hash")
    validate_manifest(doc["archive_manifest"])
    if doc["archive_manifest_sha256"]!=digest(doc["archive_manifest"]):raise PreparationError("cache_source_hash")
    if len(doc["archive_manifest"]["parts"])!=1 or doc["archive_manifest"]["parts"][0]["source"]["type"]!="drive":
        raise PreparationError("cache_single_drive_part")
    if type(doc["prefix_bytes"]) is not int or not 0<=doc["prefix_bytes"]<=LOGICAL_BYTES or not re.fullmatch("[0-9a-f]{64}",str(doc["prefix_sha256"])):
        raise PreparationError("cache_prefix_size_or_hash")
    if not isinstance(doc["chunks"],list) or len(doc["chunks"])>MAX_CHUNKS:raise PreparationError("cache_chunks")
    offset=0
    for row in doc["chunks"]:
        if not isinstance(row,dict) or set(row)!={"start","end","sha256"}:raise PreparationError("cache_chunk_fields")
        if type(row["start"]) is not int or type(row["end"]) is not int or row["start"]!=offset or not 1<=row["end"]-row["start"]+1<=256*1024:
            raise PreparationError("cache_gap_or_chunk_size")
        if not isinstance(row["sha256"],str) or not re.fullmatch("[0-9a-f]{64}",row["sha256"]):raise PreparationError("cache_chunk_hash")
        offset=row["end"]+1
    if offset!=doc["prefix_bytes"] or offset>doc["archive_manifest"]["parts"][0]["size_bytes"]:raise PreparationError("cache_prefix_size")
    producer=doc["producer_code_sha256"]
    if not isinstance(producer,dict) or len(producer)>64 or any(not re.fullmatch(r"[A-Za-z0-9_]{1,64}\.py",k) or not isinstance(v,str) or not re.fullmatch("[0-9a-f]{64}",v) for k,v in producer.items()):
        raise PreparationError("cache_producer_hashes")
    if doc["status"] not in ("usable_prefix","unusable"):raise PreparationError("cache_status")
    if bundle.getinfo("prefix.bin").file_size!=offset:raise PreparationError("cache_prefix_size")
    return doc


def input_spec(path):
    stream,bundle=open_bundle(path)
    try:doc=read_manifest(bundle);return {"path":str(Path(path).resolve()),"manifest_sha256":doc["content_sha256"]}
    finally:bundle.close();stream.close()


def save_uploaded_cache(uploaded,directory):
    """파일 선택 UI 이후의 local 단계. 업로드 bytes는 최대128MiB, 처리 deadline60초."""
    if not isinstance(uploaded,dict) or len(uploaded)!=1:raise PreparationError("cache_upload_one_zip")
    data=next(iter(uploaded.values()))
    if not isinstance(data,bytes) or len(data)>CACHE_DISK_BYTES:raise PreparationError("cache_upload_size")
    started=time.monotonic();directory=Path(directory)
    if linked(directory) or any(linked(p) for p in directory.parents):raise PreparationError("cache_upload_path")
    directory.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(data).hexdigest();target=directory/(sha+".zip");partial=directory/(sha+".partial")
    if target.exists() or partial.exists():raise PreparationError("cache_upload_exists")
    try:
        with partial.open("xb") as f:f.write(data)
        spec=input_spec(partial)
        if time.monotonic()-started>60:raise PreparationError("cache_upload_elapsed_limit")
        os.rename(partial,target);spec["path"]=str(target.resolve())
        return spec,{"schema_version":"cache_upload_v1","zip_sha256":sha,"upload_bytes":len(data),
            "local_written_bytes":len(data),"metadata_read_upper_bound_bytes":2*MANIFEST_BYTES,
            "elapsed_seconds":time.monotonic()-started,"max_seconds":60,"user_file_selection_and_upload_excluded":True,
            "manifest_sha256":spec["manifest_sha256"],"network_source_requested":False}
    finally:
        if partial.exists():partial.unlink()


class PrefixCache:
    def __init__(self,output,manifest,budget):
        self.directory=output/"cache";self.directory.mkdir()
        self.path=self.directory/"prefix.bin";self.manifest=manifest;self.budget=budget
        self.chunks=[];self.size=0;self.input_bytes=0;self.hash=hashlib.sha256()
        self.stream=None;self.import_status="cold";self.final=None

    def load(self,spec):
        if spec is None:
            self.stream=self.path.open("x+b");return
        self.import_status="validating"
        source=bundle=None;partial=self.directory/"prefix.partial"
        try:
            source,bundle=open_bundle(spec["path"],self.budget);doc=read_manifest(bundle)
            if doc["content_sha256"]!=spec["manifest_sha256"]:raise PreparationError("cache_input_changed")
            if doc["status"]!="usable_prefix":raise PreparationError("cache_unusable_input")
            if doc["archive_manifest"]!=self.manifest:raise PreparationError("cache_source_changed")
            with partial.open("xb") as target,bundle.open("prefix.bin") as data:
                for row in doc["chunks"]:
                    chunk=bytearray();need=row["end"]-row["start"]+1
                    while len(chunk)<need:
                        # ZIP backing file의 모든 실제 read는 CountedFile에서 계수한다.
                        self.budget.check_time();piece=data.read(min(65536,need-len(chunk)))
                        if not piece:raise PreparationError("cache_short_prefix")
                        chunk.extend(piece)
                    if hashlib.sha256(chunk).hexdigest()!=row["sha256"]:raise PreparationError("cache_chunk_hash_mismatch")
                    self._write(target,chunk);self.hash.update(chunk)
                if data.read(1):raise PreparationError("cache_long_prefix")
            if self.hash.hexdigest()!=doc["prefix_sha256"]:raise PreparationError("cache_prefix_hash_mismatch")
            os.rename(partial,self.path)
            self.stream=self.path.open("r+b");self.chunks=[dict(r) for r in doc["chunks"]]
            self.size=self.input_bytes=doc["prefix_bytes"];self.import_status="verified"
        except (BadZipFile,OSError,ValueError) as exc:
            self.import_status="failed"
            if isinstance(exc,PreparationError):raise
            raise PreparationError("cache_import_failed") from None
        finally:
            if bundle is not None:bundle.close()
            if source is not None:source.close()
            if partial.exists():partial.unlink()

    def _write(self,target,chunk):
        self.budget.check_time()
        if target.tell()+len(chunk)+MANIFEST_BYTES>CACHE_DISK_BYTES:raise PreparationError("cache_disk_limit")
        written=target.write(chunk);self.budget.cache_write_bytes+=written
        if written!=len(chunk):raise PreparationError("cache_short_write")

    def cap(self,start,size):
        if start>=self.input_bytes:return size
        row=next((r for r in self.chunks if r["start"]==start),None)
        if row is None:raise PreparationError("cache_replay_gap")
        return min(size,row["end"]-start+1)

    def read(self,start,end):
        self.stream.seek(start)
        chunk=io_read(self.budget,self.stream,end-start+1)
        row=next((r for r in self.chunks if r["start"]==start and r["end"]==end),None)
        if row is None or len(chunk)!=end-start+1 or hashlib.sha256(chunk).hexdigest()!=row["sha256"]:
            raise PreparationError("cache_replay_hash_mismatch")
        self.budget.local_reads+=1
        return chunk

    def append(self,start,chunk):
        if start!=self.size or len(self.chunks)>=MAX_CHUNKS:raise PreparationError("cache_append_gap_or_cap")
        if start+len(chunk)>LOGICAL_BYTES:raise PreparationError("logical_input_limit")
        self.stream.seek(start)
        try:self._write(self.stream,chunk);self.stream.flush()
        except BaseException:
            self.stream.truncate(start);raise
        self.chunks.append({"start":start,"end":start+len(chunk)-1,"sha256":hashlib.sha256(chunk).hexdigest()})
        self.hash.update(chunk);self.size+=len(chunk)

    def finish(self,usable,producer):
        if self.stream is not None:self.stream.close()
        if not self.path.exists():self.path.write_bytes(b"");self.size=0;self.chunks=[];self.hash=hashlib.sha256()
        body={"schema_version":"bounded_prefix_cache_v1","archive_manifest":self.manifest,
            "archive_manifest_sha256":digest(self.manifest),"prefix_bytes":self.size,"prefix_sha256":self.hash.hexdigest(),
            "chunks":self.chunks,"status":"usable_prefix" if usable else "unusable","producer_code_sha256":producer}
        doc={**body,"content_sha256":digest(body)};raw=canonical(doc)+b"\n"
        if len(raw)>MANIFEST_BYTES or self.size+len(raw)>CACHE_DISK_BYTES:raise PreparationError("cache_manifest_size")
        (self.directory/"manifest.json").write_bytes(raw)
        self.final={"relative_path":"cache/manifest.json","content_sha256":doc["content_sha256"],"status":doc["status"],
            "ready_for_reuse":usable and self.size>0,
            "prefix_bytes":self.size,"input_prefix_bytes":self.input_bytes,"chunks":len(self.chunks),
            "import_status":self.import_status,"cache_disk_bytes":self.size+len(raw),"producer_code_sha256":producer}
        return self.final


def export_cache(directory):
    directory=Path(directory);manifest_path=directory/"manifest.json";prefix=directory/"prefix.bin"
    if linked(directory) or any(linked(p) for p in directory.parents) or not directory.is_dir() or any(linked(p) for p in (manifest_path,prefix)):
        raise PreparationError("cache_export_path")
    if {p.name for p in directory.iterdir()}!={"manifest.json","prefix.bin"}:raise PreparationError("cache_export_members")
    if manifest_path.stat().st_size>MANIFEST_BYTES or sum(p.stat().st_size for p in (manifest_path,prefix))>CACHE_DISK_BYTES-4096:
        raise PreparationError("cache_export_size")
    target=directory.parent/"prefix_cache.zip";partial=directory.parent/"prefix_cache.zip.partial"
    if partial.exists():raise PreparationError("cache_export_exists")
    if target.exists():
        if linked(target) or target.stat().st_size>CACHE_DISK_BYTES:raise PreparationError("cache_export_path")
        report_path=directory.parent/"cache_export_report.json"
        if report_path.is_symlink() or not report_path.is_file() or report_path.stat().st_size>4096:
            raise PreparationError("cache_export_report_missing")
        try:report=json.loads(report_path.read_bytes())
        except (ValueError,UnicodeError):raise PreparationError("cache_export_report_invalid") from None
        if report.get("content_sha256")!=digest({k:v for k,v in report.items() if k!="content_sha256"}):
            raise PreparationError("cache_export_report_hash")
        began=time.monotonic();h=hashlib.sha256();read=0
        with target.open("rb") as f:
            while chunk:=f.read(65536):
                read+=len(chunk);h.update(chunk)
                if time.monotonic()-began>EXPORT_SECONDS:raise PreparationError("cache_export_elapsed_limit")
        if h.hexdigest()!=report.get("zip_sha256"):raise PreparationError("cache_export_zip_changed")
        return target,{**report,"existing_export_reused":True,"reuse_verification_local_bytes":read,
            "reuse_verification_seconds":time.monotonic()-began}
    started=time.monotonic();io_bytes=0
    def check():
        if time.monotonic()-started>EXPORT_SECONDS:raise PreparationError("cache_export_elapsed_limit")
        if io_bytes>EXPORT_IO_BYTES:raise PreparationError("cache_export_io_limit")
    def read_counted(stream,size=-1):
        nonlocal io_bytes
        check();remaining=EXPORT_IO_BYTES-io_bytes
        if remaining<=0:raise PreparationError("cache_export_io_limit")
        data=stream.read(min(remaining,size) if size>=0 else remaining)
        io_bytes+=len(data);check();return data
    try:
        with ZipFile(partial,"x",ZIP_STORED) as z:
            for p in (manifest_path,prefix):
                with p.open("rb") as source,z.open(p.name,"w") as output:
                    while chunk:=read_counted(source,65536):output.write(chunk)
        # 만들어진 archive를 같은 reader로 CRC/manifest/범위/hash까지 검사한다.
        stream,bundle=open_bundle(partial,read_callback=read_counted)
        try:
            doc=read_manifest(bundle);h=hashlib.sha256()
            with bundle.open("prefix.bin") as data:
                for row in doc["chunks"]:
                    chunk=data.read(row["end"]-row["start"]+1);check()
                    if hashlib.sha256(chunk).hexdigest()!=row["sha256"]:raise PreparationError("cache_export_hash")
                    h.update(chunk)
            if h.hexdigest()!=doc["prefix_sha256"]:raise PreparationError("cache_export_hash")
        finally:bundle.close();stream.close()
        # ZIP 자체 hash까지 별도 단계의 IO 상한에 포함한다.
        h=hashlib.sha256()
        with partial.open("rb") as f:
            while chunk:=read_counted(f,65536):h.update(chunk)
        os.rename(partial,target)
        report={"schema_version":"cache_export_v1","manifest_sha256":doc["content_sha256"],"cache_status":doc["status"],
            "zip_sha256":h.hexdigest(),"zip_bytes":target.stat().st_size,"local_read_bytes":io_bytes,
            "elapsed_seconds":time.monotonic()-started,"max_seconds":EXPORT_SECONDS,"max_local_read_bytes":EXPORT_IO_BYTES,
            "network_requested":False}
        report["content_sha256"]=digest(report)
        (directory.parent/"cache_export_report.json").write_bytes(canonical(report)+b"\n")
        return target,report
    finally:
        if partial.exists():partial.unlink()
