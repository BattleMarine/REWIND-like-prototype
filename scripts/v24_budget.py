"""Single-call v2.4 R2 reservation and publication limits."""

MIB = 1024**2
GIB = 1024**3
PILOT_SECONDS = 2700
NUMERICAL_SECONDS = 300
WORKER_RESERVE_SECONDS = 480
PUBLISH_RESERVE_SECONDS = 360
NUMERICAL_PUBLISH_RESERVE_SECONDS = 30
READ_CAP = 2 * GIB
WRITE_CAP = 4 * GIB
STATE_CAP = 192 * MIB
REPORT_CAP = 2 * MIB
REPORT_COUNT = 6  # numerical, plan, two arms, summary and worker receipt
MARKER_CAP = 64 * 1024
LOG_CAP = 32 * 1024

LOCAL_REPORT_RESERVE = REPORT_COUNT * REPORT_CAP
DRIVE_WRITE_RESERVE = 2 * STATE_CAP + LOCAL_REPORT_RESERVE + 2 * MARKER_CAP + LOG_CAP
WORKER_WRITE_CAP = WRITE_CAP - LOCAL_REPORT_RESERVE - DRIVE_WRITE_RESERVE


def write_budget(checkpoint_bytes, extraction_cap, observed_state_bytes):
    worker = {"parent_archive_copy": checkpoint_bytes, "parent_extraction": extraction_cap,
              "two_local_pilot_states": 2 * STATE_CAP}
    planned = sum(worker.values()) + LOCAL_REPORT_RESERVE + DRIVE_WRITE_RESERVE
    if (not 0 < checkpoint_bytes <= extraction_cap or len(observed_state_bytes) != 2
            or any(not 0 < size <= STATE_CAP for size in observed_state_bytes.values())
            or sum(worker.values()) > WORKER_WRITE_CAP):
        raise ValueError("v24_write_reservation_exceeds_allocation")
    return {"worker_reservations_bytes": worker, "local_reports_reserve_bytes": LOCAL_REPORT_RESERVE,
            "drive_publication_reserve_bytes": DRIVE_WRITE_RESERVE,
            "drive_max_state_readback_bytes": 2 * STATE_CAP, "total_planned_reservation_bytes": planned,
            "total_logical_write_cap_bytes": WRITE_CAP, "unallocated_headroom_bytes": WRITE_CAP - planned,
            "observed_cpu_state_bytes": observed_state_bytes}


def limits():
    return {"pilot_seconds": PILOT_SECONDS, "numerical_seconds": NUMERICAL_SECONDS,
            "pilot_worker_terminal_reserve_seconds": WORKER_RESERVE_SECONDS,
            "pilot_publish_reserve_seconds": PUBLISH_RESERVE_SECONDS,
            "numerical_publish_reserve_seconds": NUMERICAL_PUBLISH_RESERVE_SECONDS,
            "pilot_read_bytes": READ_CAP, "pilot_logical_write_bytes": WRITE_CAP,
            "pilot_worker_logical_write_bytes": WORKER_WRITE_CAP, "report_bytes": REPORT_CAP,
            "drive_publication_write_reserve_bytes": DRIVE_WRITE_RESERVE, "state_bytes": STATE_CAP,
            "pilot_scratch_free_bytes": 4 * GIB, "numerical_scratch_free_bytes": 256 * MIB}
