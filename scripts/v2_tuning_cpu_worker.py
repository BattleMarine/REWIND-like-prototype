"""Run one fixed CPU check against the read-only project tree."""

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from scripts.v2_tuning_cpu_common import DESIGN, JOBS, Meter, Project, base_index, require, write_json


def run(job, kit, project, workspace):
    project = project.resolve(strict=True)
    require(workspace.parent.is_dir() and not workspace.resolve().is_relative_to(project),
            "CPU scratch must be outside the project")
    workspace.mkdir()
    meter = Meter(job, workspace)
    report = {"schema": "rewind_v2_tuning_cpu_report_v1", "job": job,
              "design_revision": DESIGN, "status": "failed", "limits": JOBS[job]}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        require(shutil.disk_usage(workspace).free >= JOBS[job]["scratch_bytes"], "CPU scratch preflight")
        import numpy
        import PIL
        import torch

        torch.set_num_threads(1)
        require(not torch.cuda.is_available(), "CPU check requires CUDA hidden")
        report["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                                 "numpy": numpy.__version__, "pillow": PIL.__version__, "device": "cpu"}
        index = base_index(kit)
        report["parent"] = {"identity": index["identity"],
                            "checkpoint_receipt_sha256": index["checkpoint_receipt_sha256"]}
        lane = importlib.import_module(f"scripts.v2_tuning_cpu_{job}")
        with Project(project, meter) as sources:
            report["checks"] = lane.run(kit, sources, index, workspace)
        meter.check()
        report["status"] = "passed"
        meter.phase = "finished"
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1600], "phase": meter.phase}
        if isinstance(exc, (KeyboardInterrupt, TimeoutError)):
            report["status"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "timed_out"
        raise
    finally:
        meter.persist()
        report["usage"] = json.loads((workspace / "usage.json").read_bytes())
        write_json(workspace / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=JOBS)
    for name in ("kit", "project", "workspace"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.job, args.kit, args.project, args.workspace)
    print(json.dumps({"job": args.job, "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
