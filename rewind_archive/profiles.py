"""검토한 고정 준비 예산. 변경 가능한 사용자 dict로 preset을 저장하지 않는다."""

from types import MappingProxyType

from .contracts import Limits, PreparationError


SMOKE_PROFILE = "smoke8MiB_v1"
COUNTERPART_PROFILE = "counterpart64MiB_v1"
POSE_COUNTERPART_PROFILE = "poseCounterpart64MiB_v1"
CACHE_PROFILE = "posePrefixCache96MiB_v1"
WARM128_PROFILE = "posePrefixWarm128MiB_v2"
WARM192_PROFILE = "posePrefixWarm192MiB_v3"
TAIL352_PROFILE = "posePrefixTail352MiB_v4"
TARGET67_PROFILE = "posePrefixTargets67_v1"
TARGET67_CACHE_EXPECTATION = MappingProxyType({
    "manifest_sha256":"b7bcb31af4be74514615c48f551ef7f0a0449428c8260b79ce9abe0df88c05ab",
    "prefix_sha256":"dd3452a2b1a159419b8e0a247f38010752caeb1a29c92c4e24024cda5ab48306",
    "prefix_bytes":245366784,
})
CACHE_PROFILES = (CACHE_PROFILE,WARM128_PROFILE,WARM192_PROFILE,TAIL352_PROFILE,TARGET67_PROFILE)
POSE_PROFILES = (POSE_COUNTERPART_PROFILE,*CACHE_PROFILES)
TARGET_PAIR_PROFILES = (COUNTERPART_PROFILE, *(p for p in POSE_PROFILES if p!=TARGET67_PROFILE))


def target67_names():
    prefix="ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01"
    return [f"{prefix}/{seq}/json/frame_{frame}.json" for seq,frames in
            (("003/Getting_Up__1_",(5,12)),("010/Football_Hike__4_",range(9)),
             ("025/Leg_Sweep__2_",range(24)),("028/Dribble__4_",range(32))) for frame in frames]


def validate_target67(profile,targets,mode,purpose="targeted"):
    if profile==TARGET67_PROFILE and (mode not in ("extract","review") or purpose not in ("targeted","review") or
            len(targets)!=67 or set(targets)!=set(target67_names())):
        raise PreparationError("pose67_requires_fixed_target_set")


def validate_target67_cache(profile,expectation):
    if profile==TARGET67_PROFILE and expectation!=dict(TARGET67_CACHE_EXPECTATION):
        raise PreparationError("pose67_requires_pinned_234MiB_cache")


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
    WARM192_PROFILE:Limits(max_transfer_bytes=80*1024**2,max_elapsed_seconds=420.0,
        max_ranges=773,max_decompressed_bytes=1200*1024**2,max_files=2,
        max_members=30000,max_metadata_bytes=2*1024**2),
    TAIL352_PROFILE:Limits(max_transfer_bytes=192*1024**2,max_elapsed_seconds=1200.0,
        max_ranges=1916,max_decompressed_bytes=2400*1024**2,max_files=2,
        max_members=60000,max_metadata_bytes=4*1024**2,max_disk_bytes=32*1024**2),
    TARGET67_PROFILE:Limits(max_transfer_bytes=192*1024**2,max_elapsed_seconds=1200.0,
        max_ranges=1916,max_decompressed_bytes=2400*1024**2,max_files=67,
        max_members=60000,max_metadata_bytes=4*1024**2,max_disk_bytes=32*1024**2),
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
