"""Add ordered prefetch to the frozen 256 training worker and its original CLI."""

from itertools import islice
from types import SimpleNamespace

from scripts import run_v2_gpu_training as common
from scripts import run_v2_gpu_training_256 as original
from scripts.v2_gpu_training_cache256_parallel import ParallelRangeCache256
from scripts.v2_gpu_training_pipeline import Training


class PrefetchData256(original.FullData256):
    validation_prefetch = False
    validation_chunk = None

    def frame(self, key, split, size):
        if split == "validation" and self.validation_prefetch:
            chunk = self.frames[split]
            if self.validation_chunk is not chunk:
                self.cache.set_order(chunk, split)
                self.validation_chunk = chunk
        return super().frame(key, split, size)

    def clip(self, key, split, frontend, size, input_sha):
        sequence, start = key
        keys = ((sequence, frame) for frame in range(start, start+self.real_frames(key, split)))
        self.cache.set_order(keys, split)
        try:
            return super().clip(key, split, frontend, size, input_sha)
        finally:
            self.cache.drain()


class PrefetchTraining256(Training):
    def _epoch(self, role, stage, stop_at):
        self.frontend_stop_at = stop_at
        try:
            return super()._epoch(role, stage, stop_at)
        finally:
            self.data.cache.drain()

    def _order(self, role, epoch):
        keys = super()._order(role, epoch)
        if role == "frontend":
            size = self.batch_sizes[role]
            start = self.progress["batch"]*size
            end = (self.frontend_stop_at-self.progress["updates"])*size+start
            self.data.cache.set_order(islice(keys, start, max(start, end)), "train")
        return keys

    def _validate_epoch(self, role, stage):
        if role != "frontend":
            return super()._validate_epoch(role, stage)
        self.data.cache.drain()
        self.data.validation_prefetch = True
        try:
            return super()._validate_epoch(role, stage)
        finally:
            self.data.validation_prefetch = False
            self.data.validation_chunk = None
            self.data.cache.drain()


PROFILE = SimpleNamespace(**{**vars(original.PROFILE), "data": PrefetchData256})


def main(argv=None):
    # The frozen runner owns the CLI, Drive writer, model binding and saves.
    # Only these process-local factories vary for this worker invocation.
    old_training, old_cache = common.Training, original.RangeCache256
    common.Training, original.RangeCache256 = PrefetchTraining256, ParallelRangeCache256
    try:
        common.main(argv, profile=PROFILE)
    finally:
        common.Training, original.RangeCache256 = old_training, old_cache


if __name__ == "__main__":
    main()
