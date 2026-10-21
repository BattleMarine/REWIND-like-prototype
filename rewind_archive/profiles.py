"""검토한 고정 준비 예산. 변경 가능한 사용자 dict로 preset을 저장하지 않는다."""

from types import MappingProxyType

from .contracts import Limits, PreparationError


SMOKE_PROFILE = "smoke8MiB_v1"
COUNTERPART_PROFILE = "counterpart64MiB_v1"
POSE_COUNTERPART_PROFILE = "poseCounterpart64MiB_v1"
CACHE_PROFILE = "posePrefixCache96MiB_v1"
WARM128_PROFILE = "posePrefixWarm128MiB_v2"
CACHE_PROFILES = (CACHE_PROFILE,WARM128_PROFILE)
POSE_PROFILES = (POSE_COUNTERPART_PROFILE,*CACHE_PROFILES)
TARGET_PAIR_PROFILES = (COUNTERPART_PROFILE, *POSE_PROFILES)
PROFILES = MappingProxyType({
    SMOKE_PROFILE:Limits(),
    COUNTERPART_PROFILE:Limits(max_transfer_bytes=64*1024**2,max_elapsed_seconds=300.0,
        max_ranges=768,max_decompressed_bytes=128*1024**2,max_files=2),
    POSE_COUNTERPART_PROFILE:Limits(max_transfer_bytes=64*1024**2,max_elapsed_seconds=300.0,
        max_ranges=768,max_decompressed_bytes=512*1024**2,max_files=2,
        max_members=10000,max_metadata_bytes=1024**2),
    CACHE_PROFILE:Limits(max_transfer_bytes=96*1024**2,max_elapsed_seconds=420.0,
        max_ranges=1157,max_decompressed_bytes=640*1024**2,max_files=2,
        max_members=16000,max_metadata_bytes=2*1024**2),
    WARM128_PROFILE:Limits(max_transfer_bytes=40*1024**2,max_elapsed_seconds=420.0,
        max_ranges=392,max_decompressed_bytes=800*1024**2,max_files=2,
        max_members=20000,max_metadata_bytes=2*1024**2),
})


def profile_limits(name):
    if not isinstance(name,str) or name not in PROFILES:
        raise PreparationError("unknown_budget_profile")
    return PROFILES[name]


def validate_profile_limits(name, limits):
    preset=profile_limits(name)
    if not isinstance(limits,Limits):
        raise PreparationError("limits_required")
    if limits!=preset:
        raise PreparationError("budget_profile_requires_exact_limits")
    return limits
