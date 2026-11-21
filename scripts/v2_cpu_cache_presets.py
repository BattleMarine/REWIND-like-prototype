"""Fixed invocation-only settings for v2 bulk CPU caching."""


STANDARD = "standard"
NINETY_MINUTES = "90min"
PRESETS = (STANDARD, NINETY_MINUTES)
SESSION_90 = 5_400
READ_90 = 112 * 1024**3
WRITE_90 = 24 * 1024**3
FINAL_WORK_90 = {"terminal_reserve_seconds": 30, "package_seconds": 120, "copy_seconds": 120}
PREFETCH_90 = 8


def invocation_config(cfg, preset):
    if preset == STANDARD:
        return cfg
    if preset != NINETY_MINUTES:
        raise ValueError("v2_quota_invocation_preset")
    limits = {**cfg["resources"], "read_bytes": READ_90, "write_bytes": WRITE_90}
    return {**cfg, **FINAL_WORK_90, "resources": limits}
