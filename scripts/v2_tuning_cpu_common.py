"""Fixed evidence and bounded I/O for the independent v2 tuning CPU checks."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from scripts.v2_gpu128_a_bundle_format import (INDEX, FINAL_BOUNDARY, SOURCE, SOURCE_SHA, SOURCE_RECEIPT,
                                               MIB, GIB, BundleView, canonical, check_json_pin, metadata, sha)


ROOT_INDEX_SHA = "4a4e1c119e08f0026ef4af9a87c8edbb757934fbcee7ace1b61c2e424e080f40"
DESIGN = "08cc82a"
JOBS = {
    "inputs": {"seconds": 1200, "read_bytes": 512 * MIB, "scratch_bytes": 2 * GIB},
    "reference": {"seconds": 420, "read_bytes": 32 * MIB, "scratch_bytes": GIB},
    "trials": {"seconds": 1200, "read_bytes": 512 * MIB, "scratch_bytes": 3 * GIB},
}
REPORT_CAP = 2 * MIB


def require(condition, error):
    if not condition:
        raise ValueError(error)


def write_json(path, value):
    raw = canonical(value)
    require(len(raw) <= REPORT_CAP, f"report_size: {path.name}")
    temp = path.with_suffix(".partial")
    temp.write_bytes(raw)
    temp.replace(path)


class Meter:
    def __init__(self, job, workspace):
        self.limits = JOBS[job]
        self.started = time.monotonic()
        self.workspace = workspace
        self.drive_read_bytes = self.local_read_bytes = 0
        self.phase = "startup"

    def check(self):
        if time.monotonic() - self.started >= self.limits["seconds"] - 30:
            raise TimeoutError(f"CPU worker deadline: {self.phase}")

    def mark(self, phase):
        self.phase = phase
        self.persist()
        self.check()

    def persist(self):
        write_json(self.workspace / "usage.json",
                   {"phase": self.phase, "elapsed_seconds": time.monotonic() - self.started,
                    "project_read_observed_bytes": self.drive_read_bytes,
                    "local_cache_receipt_read_bytes": self.local_read_bytes})

    def read(self, handle, size):
        self.check()
        remaining = self.limits["read_bytes"] - self.drive_read_bytes
        if size > remaining:
            raise OSError("CPU project read cap")
        raw = handle.read(size)
        self.drive_read_bytes += len(raw)
        return raw

    def local_io(self, mode, size):
        require(mode == "read", "unexpected local cache operation")
        self.local_read_bytes += size


class Project(BundleView):
    """Read the live project tree directly and meter only external payloads."""

    def __init__(self, project, meter):
        require(project.is_dir(), "expected a project directory")
        super().__init__(project)
        self.meter = meter

    @contextmanager
    def open(self, name, size=None):
        with super().open(name, size) as source:
            yield SimpleNamespace(read=lambda size: self.meter.read(source, size))

    def assert_size(self, name, pin):
        self.meter.check()
        path = self.file(name)
        require(path.stat().st_size == pin["bytes"], f"project_file_size: {name}")

    def stage(self, name, pin, destination):
        self.assert_size(name, pin)
        self.copy(name, destination, pin, self.meter.check)
        return destination


def base_index(kit):
    raw = (kit / INDEX).read_bytes()
    require(sha(raw) == ROOT_INDEX_SHA, "A index SHA mismatch")
    index = json.loads(raw)
    check_json_pin(index, "rewind_v2_gpu128_a_portable_v1")
    require(index["identity"]["seed"] == 42 and index["identity"]["source"] == SOURCE_SHA
            and index["identity"]["cache_boundary"] == index["cache_boundary_sha256"] == FINAL_BOUNDARY,
            "A index identity mismatch")
    return index


class EvidenceView:
    """Use the kit for source receipt metadata, and the project for A's raw receipts."""

    def __init__(self, kit, project):
        self.kit, self.project = kit, project

    def read(self, name, cap, pin=None):
        if name in (SOURCE_RECEIPT, SOURCE):
            with BundleView(self.kit) as view:
                return view.read(name, cap, pin)
        return self.project.read(name, cap, pin)


def native_evidence(kit, project, index):
    source, diagnostic, step, comparison = metadata(EvidenceView(kit, project), index["members"])
    require(index["identity"] == diagnostic["identity"], "A index and diagnostic disagree")
    return source, diagnostic, step, comparison


def staged_archive(project, name, pin, workspace):
    """Check peak scratch allowance before copying an exact outer ZIP."""
    require(shutil.disk_usage(workspace).free >= project.meter.limits["scratch_bytes"],
            "CPU scratch free space below reservation")
    return project.stage(name, pin, workspace / Path(name).name)


def load_scope(root, pins):
    from scripts.run_v2_gpu_training import load_inputs

    return load_inputs(SimpleNamespace(invocation=root, workspace=root, slot="A"), pins)
