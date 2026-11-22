"""Execute only the two fixed finalist seeds on one CUDA device."""

import re

from scripts import v211_run, v211_trial, v212_lora


SEEDS = (137, 181)
STRATEGIES = {strategy.METHOD: strategy for strategy in (v211_trial, v212_lora)}
DESTINATIONS = {v211_trial.METHOD: "reports/v211_body_head128_a_final_v1",
                v212_lora.METHOD: "reports/v212_body_lora128_a_final_v1"}
SCHEMA = "rewind_v2_tuning_final_invocation_v1"


def scope(pins):
    final = pins["final"]
    method = final["method"]
    if (method not in STRATEGIES or final["seeds"] != list(SEEDS) or len(final["shortlist"]) != len(STRATEGIES)
            or set(final["shortlist"]) != set(STRATEGIES)
            or pins["trial_plan"]["conditional_additional_seeds"] != list(SEEDS)
            or {seed for seed in pins["trial_plan"]["initial_seeds"]}.intersection(SEEDS)
            or not re.fullmatch("[0-9a-f]{64}", final["screening_sha256"])):
        raise ValueError("v2_tuning_final_trial_scope")
    return method


def run(args, pins, report):
    report["phase"] = "final_trial_scope"
    method = scope(pins)
    report["final"] = pins["final"]
    v211_run.run(args, pins, report, strategy=STRATEGIES[method], destination=DESTINATIONS[method],
                 run_schema=SCHEMA, seeds=SEEDS)


if __name__ == "__main__":
    v211_run.main(run, __doc__)
