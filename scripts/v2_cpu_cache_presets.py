"""Fixed invocation-only settings for v2 bulk CPU caching."""


STANDARD = "standard"
NINETY_MINUTES = "90min"
OVERNIGHT = "8h"
ACCELERATED = (NINETY_MINUTES, OVERNIGHT)
PRESETS = (STANDARD, *ACCELERATED)
SESSION_90 = 5_400
READ_90 = 112 * 1024**3
WRITE_90 = 24 * 1024**3
FINAL_WORK = {"terminal_reserve_seconds": 30, "package_seconds": 120, "copy_seconds": 120}
PREFETCH_ACCELERATED = 8
READ_8H = 640 * 1024**3
WRITE_8H = 136 * 1024**3


def invocation_config(cfg, preset):
    if preset == STANDARD:
        return cfg
    if preset not in ACCELERATED:
        raise ValueError("v2_quota_invocation_preset")
    read, write = (READ_90, WRITE_90) if preset == NINETY_MINUTES else (READ_8H, WRITE_8H)
    limits = {**cfg["resources"], "read_bytes": read, "write_bytes": write}
    return {**cfg, **FINAL_WORK, "resources": limits}
