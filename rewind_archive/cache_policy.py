"""코드에서 고정한 cache 판별 상한. 입력 manifest의 임의 숫자를 예산으로 쓰지 않는다."""

from dataclasses import dataclass,asdict
from .contracts import PreparationError

MIB=1024**2


@dataclass(frozen=True)
class CachePolicy:
    name:str
    schema:str
    logical_bytes:int
    disk_bytes:int
    local_io_bytes:int
    max_chunks:int
    export_io_bytes:int
    sequences:int
    audit_rows:int
    min_input_bytes:int=0


WARM128_POLICY=CachePolicy("pose_prefix_warm128_v2","bounded_prefix_cache_v2",128*MIB,160*MIB,
    260*MIB,640,512*MIB,256,1024,383*256*1024)
WARM192_POLICY=CachePolicy("pose_prefix_warm192_v3","bounded_prefix_cache_v3",192*MIB,224*MIB,
    388*MIB,896,640*MIB,256,897,128*MIB)
WARM_POLICIES=(WARM128_POLICY,WARM192_POLICY)


def for_profile(profile):
    if not isinstance(profile,str):return None
    from .profiles import WARM128_PROFILE,WARM192_PROFILE
    return {WARM128_PROFILE:WARM128_POLICY,WARM192_PROFILE:WARM192_POLICY}.get(profile)


def validate_expectation(value,required=False,policy=WARM128_POLICY):
    if value is None:
        if required:raise PreparationError("warm_cache_expectation_required")
        return None
    import re
    fields={"manifest_sha256","prefix_sha256","prefix_bytes"}
    if not isinstance(value,dict) or set(value)!=fields:raise PreparationError("cache_expectation_fields")
    if any(not isinstance(value[k],str) or not re.fullmatch("[0-9a-f]{64}",value[k]) for k in ("manifest_sha256","prefix_sha256")):
        raise PreparationError("cache_expectation_hash")
    if policy not in WARM_POLICIES:raise PreparationError("unsupported_cache_policy")
    if type(value["prefix_bytes"]) is not int or not policy.min_input_bytes<=value["prefix_bytes"]<=policy.logical_bytes:
        raise PreparationError("warm_cache_prefix_length")
    return dict(value)


def summary(policy):
    return asdict(policy)
