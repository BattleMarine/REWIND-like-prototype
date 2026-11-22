"""One manual CUDA invocation of the fixed v2.1.3 full A-body trials."""

from scripts import v211_run, v213_body


DESTINATION = "reports/v213_body_full128_a_v1"
SCHEMA = "rewind_v213_invocation_v1"


def run(args, pins, report):
    report["fit_policy"] = {"max_checkpoint_bytes": v213_body.CHECKPOINT_CAP,
                            "periodic_checkpoint_updates": v213_body.CHECKPOINT_INTERVAL,
                            "max_fit_copy_seconds": v213_body.CHECKPOINT_COPY_SECONDS}
    v211_run.run(args, pins, report, strategy=v213_body, destination=DESTINATION, run_schema=SCHEMA)


if __name__ == "__main__":
    v211_run.main(run, __doc__)
