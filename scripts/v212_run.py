"""One manual CUDA invocation of the fixed v2.1.2 A LoRA trials."""

from scripts import v211_run, v212_lora


DESTINATION = "reports/v212_body_lora128_a_v1"
SCHEMA = "rewind_v212_invocation_v1"


def run(args, pins, report):
    v211_run.run(args, pins, report, strategy=v212_lora, destination=DESTINATION, run_schema=SCHEMA)


if __name__ == "__main__":
    v211_run.main(run, __doc__)
