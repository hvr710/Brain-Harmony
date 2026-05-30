from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import RoiDownstreamDataset, load_roi
from .tasks import SKIPPED_TASKS, filter_tasks


def inspect_task(task, check_arrays: bool, max_array_checks: int) -> dict:
    result = {
        "task_id": task.task_id,
        "name": task.name,
        "benchmark": task.benchmark,
        "problem": task.problem,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = RoiDownstreamDataset(task, split, max_samples=None)
        split_info = {
            "usable": len(dataset),
            "raw_total": dataset.stats["total"],
            "missing_fmri": dataset.stats.get("missing_fmri", 0),
            "missing_label": dataset.stats["missing_label"],
            "bad_label": dataset.stats["bad_label"],
        }
        if check_arrays:
            shapes = []
            for sample in dataset.samples[:max_array_checks]:
                try:
                    shapes.append(list(load_roi(sample.path).shape))
                except Exception as exc:  # noqa: BLE001
                    shapes.append(f"ERROR: {exc}")
            split_info["array_shapes"] = shapes
        result["splits"][split] = split_info
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["origin", "omni", "all"], default="all")
    parser.add_argument("--data_profile", choices=["auto", "nas", "hs"], default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument("--check_arrays", action="store_true")
    parser.add_argument("--max_array_checks", type=int, default=2)
    args = parser.parse_args()

    benchmarks = ["origin", "omni"] if args.benchmark == "all" else [args.benchmark]
    report = {"data_profile": args.data_profile, "tasks": [], "skipped": SKIPPED_TASKS}
    for benchmark in benchmarks:
        for task in filter_tasks(benchmark, data_profile=args.data_profile):
            try:
                report["tasks"].append(inspect_task(task, args.check_arrays, args.max_array_checks))
            except Exception as exc:  # noqa: BLE001
                report["tasks"].append(
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "benchmark": task.benchmark,
                        "error": str(exc),
                    }
                )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
