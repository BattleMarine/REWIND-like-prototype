"""Fixed v1.1 epoch expansion using the existing nominal-clock evidence."""

from copy import deepcopy

from rewind_prepare.temporal_selection import check_pin, seal
from .diagnostic import DiagnosticData, PILOT_CONFIG, _doc


RUN_ID = "V1.1-EPOCH-EXPANSION-001"
DECISION_SCHEMA = "archvis_v1_epoch_expansion_decision_v1"
MILESTONE_EPOCHS = (1, 5, 10)
HARD_SECONDS = 7200
RECOVERY_BYTES = 1024**3


def validate_expansion(config):
    expected = deepcopy(_doc(PILOT_CONFIG))
    expected["epochs"] = dict.fromkeys(expected["epochs"], 10)
    expected["limits"].update(max_updates=5400, output_bytes=2*1024**3)
    if config != expected:
        raise ValueError("rewind_v1_epoch_expansion_config")


def expansion_decision(diagnostic_sha256, config_sha256):
    return seal({"schema": DECISION_SCHEMA, "run_id": RUN_ID,
                 "decision_date": "2026-11-10", "authorized_by": "user",
                 "purpose": "same_1000_train_pairs_epoch_expansion",
                 "diagnostic_decision_sha256": diagnostic_sha256, "config_sha256": config_sha256,
                 "epochs": {"frontend": 10, "body": 10, "hand": 10},
                 "milestone_epochs": list(MILESTONE_EPOCHS), "invocation_hard_seconds": HARD_SECONDS})


class ExpansionData(DiagnosticData):
    def _time_identity(self, args):
        decision = _doc(args.expansion_decision)
        check_pin(decision, DECISION_SCHEMA, args.expansion_decision_sha256)
        if decision != expansion_decision(args.decision_sha256, args.config_sha256):
            raise ValueError("rewind_v1_epoch_expansion_decision")
        identity = super()._time_identity(args)
        return {**identity, "run_id": RUN_ID, "purpose": decision["purpose"],
                "diagnostic_run_id": identity["run_id"], "expansion_decision_sha256": decision["content_sha256"]}
