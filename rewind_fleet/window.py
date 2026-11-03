"""동일 원본의 GZIDX1.10.3 bounded pack. 독립 index를 합치지 않는다."""
import io
import struct
import sys
from rewind_archive_index.source import IndexFailure

MAX_PACK=4*1024**2
def check(ok,reason):
    if not ok:raise IndexFailure(reason)

def decode(raw,*,allow_empty=False):
    check(sys.byteorder=='little' and 35<=len(raw)<=MAX_PACK and raw[:7]==b'GZIDX\x01\x00','fleet_window_format')
    c,u,spacing,size,n=struct.unpack_from('<QQIII',raw,7)
    check(size==32768 and 65536<=spacing<=16*1024**2 and (0 if allow_empty else 1)<=n<=120,'fleet_window_capacity')
    at=35;rows=[];last=(-1,-1)
    for i in range(n):
        check(at+18<=len(raw),'fleet_window_truncated')
        point=struct.unpack_from('<QQBB',raw,at);at+=18
        check(point[0]>=last[0] and point[1]>=last[1] and point[0]<=c and point[2]<=7 and point[3] in (0,1),'fleet_window_points')
        rows.append([point,None]);last=point[:2]
    for row in rows:
        amount=size if row[0][3] else 0;check(at+amount<=len(raw),'fleet_window_dictionary')
        row[1]=raw[at:at+amount];at+=amount
    check(at==len(raw) and (not rows or rows[0][0][1]==0),'fleet_window_length_or_origin')
    return (c,u,spacing,size),rows

def encode(header,rows):
    return b'GZIDX\x01\x00'+struct.pack('<QQIII',*header,len(rows))+b''.join(struct.pack('<QQBB',*p) for p,d in rows)+b''.join(d for p,d in rows)

def resume(raw,cursor):
    header,rows=decode(raw)
    eligible=[i for i,r in enumerate(rows) if r[0][1]<=cursor]
    last=eligible[-1]
    # zran npoints<=1은 원점에서 재확장한다. 시작 부근 이외에는 정확히2point를 유지한다.
    selected=[rows[0]] if last==0 else [rows[0],rows[last]]
    return encode(header,selected),selected[-1][0][1]

def export(gz):
    check(len(list(gz.seek_points()))<=120,'fleet_window_capacity')
    f=io.BytesIO();gz.export_index(fileobj=f);raw=f.getvalue();decode(raw,allow_empty=True);return raw

def open_engine(source,spacing,pack=None,cursor=0,*,buffer_size=65536):
    import indexed_gzip
    check(indexed_gzip.__version__=='1.10.3','fleet_engine_version')
    # 작은 read도 C handle을 진행시켜 큰 BufferedReader의 lazy index0 상태를 줄인다.
    check(buffer_size in (65536,1024**2),'fleet_engine_buffer')
    gz=indexed_gzip.IndexedGzipFile(fileobj=source,spacing=spacing,buffer_size=buffer_size,readbuf_size=65536,drop_handles=False)
    try:
        if pack:
            header,rows=decode(pack,allow_empty=True)
            if rows:part,start=resume(pack,cursor)
            else:check(cursor<4*spacing,'fleet_empty_index_unbounded_replay');part,start=pack,0
            gz.import_index(fileobj=io.BytesIO(part))
            check(cursor>=start,'fleet_window_before_coverage')
        else:check(cursor==0,'fleet_window_cold_fallback_forbidden')
        gz.seek(cursor);return gz
    except BaseException:gz.close();raise
