"""V1.2 runner adaptation: disk features, fixed update comparisons and backup cadence."""

from copy import deepcopy
import json
from pathlib import Path
import time

from .frontend import frontend_fingerprint
from .full_data import FeatureStore
from .run_stages import CascadeStages, accumulate, preserve_rng
from .runner import Pipeline, _publish, validate_config


FORMAT = "rewind_v12_full_pipeline_v1"
CONFIG = Path(__file__).resolve().parents[1] / "configs/rewind_v12_full_training.json"
UPDATE_MILESTONES = {"frontend": (500, 5000), "body": (20, 200), "hand": (20, 200)}


def validate_full_config(config):
    validate_config(config)
    expected = json.loads(CONFIG.read_bytes())
    if config != expected:
        raise ValueError("v12_training_fixed_config")
    return config


class FullPipeline(Pipeline):
    def __init__(self, config, config_sha256, data, device, output, budget, feature_root, backup=None):
        super().__init__(config, config_sha256, data, device, output, budget)
        self.binding["format"] = FORMAT
        self.binding["comparison_update_milestones"] = UPDATE_MILESTONES
        # Existing staged runner keeps the same initialization seed71 + stage index (72/73).
        self.epoch_expansion = True
        self.progress.update(milestones={}, update_milestones={})
        self.feature_root, self.backup = Path(feature_root), backup
        self.backup_last_time, self.backup_calls = time.monotonic(), 0

    def _features(self):
        with preserve_rng():
            frontend = self._make_frontend()
            self._load_best("frontend", frontend)
            pin = frontend_fingerprint(frontend.model)
            store = FeatureStore(self.feature_root, self.data, pin)
            clips, stats = store.build(frontend.model, self.frontend_size)
            self.timing.update({key:value for key,value in stats.items() if type(value) in (int,float)})
            self.feature_store_identity = stats["feature_identity"]
            self.cascade = CascadeStages(self.data, clips, self.cfg, self.device, 0)

    def _after_update(self, role, stage):
        step = self._step()
        if step not in UPDATE_MILESTONES[role]:
            return
        name = f"update-{role}-u{step:06d}.pt"
        dependencies = self._dependencies(role)
        pin = _publish(self.root / name, {"binding": self.binding, "stage": role, "stage_update": step,
                                        "global_update": self.progress["updates"], "model": stage.model.state_dict(),
                                        "dependencies": dependencies}, self.budget)
        self.progress["update_milestones"].setdefault(role, {})[str(step)] = {**pin, "dependencies": dependencies}

    def _save_last(self, stage):
        super()._save_last(stage)
        # Remote and local cadence are deliberately independent. The callback must be bounded.
        if self.backup and self.backup_calls < 1 and time.monotonic()-self.backup_last_time >= 3600:
            started = time.monotonic()
            self.backup(self.root, self.last["sha256"], periodic=True)
            self.backup_calls += 1
            self.backup_last_time = time.monotonic()
            accumulate(self.timing, {"periodic_backup_wall_seconds": time.monotonic()-started})

    def run(self, resume_sha256=None, stop_after_updates=None):
        from .full_recovery import RemoteBudgetExhausted
        try:
            return super().run(resume_sha256, stop_after_updates)
        except RemoteBudgetExhausted as exc:
            self.progress = deepcopy(self.durable_progress)
            return self._report("paused", {"type": type(exc).__name__, "reason": str(exc)})
