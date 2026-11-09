"""Offline plan, CPU raw pack preparation and cache restore for ArchVis v1."""

import argparse
from pathlib import Path

from rewind_archive.contracts import canonical
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _regular_read
from .extracted import MAX_MANIFEST_BYTES
from .temporal_cache import Limits, prepare, restore
from .temporal_selection import plan_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in ("inventory", "spec-path", "split-dir", "previous-manifest", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    plan.add_argument("--previous-sha256", required=True)
    plan.add_argument("--seed", type=int, default=71)

    pack = commands.add_parser("pack")
    for name in ("plan", "rgb-root", "pose-root", "spec-path", "calibration-path", "output"):
        pack.add_argument("--" + name, type=Path, required=True)
    pack.add_argument("--expected-sha256", required=True)
    for name in ("max-read-bytes", "max-disk-bytes", "max-buffer-bytes"):
        pack.add_argument("--" + name, type=int, required=True)
    pack.add_argument("--max-seconds", type=float, required=True)
    pack.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=1)
    pack.add_argument("--max-parts", type=int)

    reuse = commands.add_parser("restore")
    reuse.add_argument("--cache", type=Path, required=True)
    reuse.add_argument("--output", type=Path, required=True)
    reuse.add_argument("--expected-sha256", required=True)
    reuse.add_argument("--max-disk-bytes", type=int, required=True)
    reuse.add_argument("--max-seconds", type=float, required=True)

    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "plan":
        output = args.pop("output")
        result = plan_selection(**args)
        with output.open("xb") as handle:
            handle.write(canonical(result) + b"\n")
        result = {key: value for key, value in result.items() if key != "records"}
    elif command == "pack":
        limits = Limits(**{key: args.pop(key) for key in Limits.__dataclass_fields__})
        plan_doc = read_json(_regular_read(args.pop("plan"), MAX_MANIFEST_BYTES))
        result = prepare(plan_doc, limits=limits, **args)
        result = {key: value for key, value in result.items() if key != "parts"}
    else:
        result = restore(**args)
    print(canonical(result).decode("utf-8"))


if __name__ == "__main__":
    main()
