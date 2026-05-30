from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import RoiDownstreamDataset, load_roi
from .paths import get_data_profile
from .tasks import filter_tasks


def inspect_task(task, check_arrays: int) -> dict:
    info = {
        "task_id": task.task_id,
        "benchmark": task.benchmark,
        "name": task.name,
        "problem": task.problem,
        "roi_dir": task.roi_dir,
        "label_csv": task.label_csv,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        ds = RoiDownstreamDataset(task, split)
        targets = np.asarray([sample.target for sample in ds.samples], dtype=np.float64)
        split_info = {
            "split_file": task.split_files[split],
            "raw_total": ds.stats["total"],
            "usable": len(ds),
            "missing_fmri": ds.stats.get("missing_fmri", 0),
            "missing_label": ds.stats.get("missing_label", 0),
            "bad_label": ds.stats.get("bad_label", 0),
            "target_min": float(targets.min()) if len(targets) else None,
            "target_max": float(targets.max()) if len(targets) else None,
            "target_mean": float(targets.mean()) if len(targets) else None,
            "target_std": float(targets.std()) if len(targets) else None,
            "examples": [
                {"sample_id": sample.sample_id, "target": sample.target, "path": sample.path}
                for sample in ds.samples[:3]
            ],
        }
        if check_arrays:
            shapes = []
            for sample in ds.samples[:check_arrays]:
                try:
                    shapes.append(list(load_roi(sample.path).shape))
                except Exception as exc:  # noqa: BLE001
                    shapes.append(f"ERROR: {exc}")
            split_info["array_shapes"] = shapes
        info["splits"][split] = split_info
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_profile", choices=["auto", "nas", "hs"], default="auto")
    parser.add_argument("--benchmark", choices=["all", "origin", "omni"], default="all")
    parser.add_argument("--check_arrays", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    profile = get_data_profile(args.data_profile)
    benchmarks = ["origin", "omni"] if args.benchmark == "all" else [args.benchmark]
    report = {"profile": profile.__dict__, "tasks": []}
    for benchmark in benchmarks:
        for task in filter_tasks(benchmark, data_profile=args.data_profile):
            try:
                report["tasks"].append(inspect_task(task, args.check_arrays))
            except Exception as exc:  # noqa: BLE001
                report["tasks"].append({"task_id": task.task_id, "benchmark": task.benchmark, "error": str(exc)})

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

