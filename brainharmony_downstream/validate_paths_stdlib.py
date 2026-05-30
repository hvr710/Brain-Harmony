from __future__ import annotations

import argparse
import csv
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .tasks import filter_tasks


MISSING = {"", "nan", "NaN", "NA", "N/A", "None", "null", "-999", "-999.0"}


def norm_key(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("sub-"):
        text = text[4:]
    if re.fullmatch(r"\d+(\.0+)?", text):
        return str(int(float(text)))
    return text


def base_name(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").split("/")[-1]


def subject_candidates(path: str, parser: str) -> list[str]:
    name = base_name(path)
    stem = re.sub(r"\.(npy|npz)$", "", name, flags=re.IGNORECASE)
    out: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        for item in (value, value.lstrip("0") or "0"):
            key = norm_key(item)
            if key not in out:
                out.append(key)

    if parser == "abide":
        match = re.search(r"(\d{7})", stem)
        add(match.group(1) if match else None)
    elif parser == "adhd":
        add(stem.split("_")[0])
    elif parser == "ppmi":
        match = re.search(r"sub-(\d+)", stem)
        add(match.group(1) if match else None)
    elif parser == "adni":
        match = re.search(r"sub-?([A-Za-z0-9]+)", stem) or re.search(r"(\d{3}S\d+)", stem, flags=re.IGNORECASE)
        add(match.group(1) if match else stem)
    elif parser == "nki":
        match = re.search(r"sub-([^_]+)", stem)
        if match:
            raw = match.group(1)
            add(raw)
            add(raw.replace("A", ""))
        add(stem.split("_")[0].replace("sub-", ""))
    elif parser == "sald":
        add(stem.split("_")[0])
    elif parser == "abcd":
        match = re.search(r"(sub-NDARINV[^_]+)", stem)
        add(match.group(1) if match else stem.split("_")[0])
    elif parser == "hcp":
        match = re.search(r"^(\d+)", stem)
        add(match.group(1) if match else stem.split("__")[0])
    elif parser == "bhrc":
        match = re.search(r"sub-(\d+)", stem)
        add(match.group(1) if match else stem)
    else:
        add(stem)
    return out


def split_candidates(value: str, parser: str) -> list[str]:
    text = value.strip()
    if "/" in text or "\\" in text or text.lower().endswith((".npy", ".npz")):
        return subject_candidates(text, parser)
    out = []
    for item in (text, text.lstrip("0") or "0"):
        key = norm_key(item)
        if key not in out:
            out.append(key)
    if parser == "nki" and text.startswith("A"):
        key = norm_key(text[1:])
        if key not in out:
            out.append(key)
    return out


@lru_cache(maxsize=64)
def roi_index(roi_dir: str, parser: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for root, _, files in os.walk(roi_dir):
        for name in files:
            if not name.lower().endswith((".npy", ".npz")):
                continue
            path = os.path.join(root, name)
            for key in subject_candidates(path, parser):
                index.setdefault(key, path)
    return index


def label_rows(task) -> dict[str, dict[str, str]]:
    if not task.label_csv or not task.label_key:
        return {}
    rows = {}
    with open(task.label_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = norm_key(row.get(task.label_key, ""))
            if key and key not in rows:
                rows[key] = row
    return rows


def coerce_label(value: Any, task):
    text = str(value).strip()
    if text in MISSING:
        return None
    if task.is_regression or task.label_mode == "float":
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return int(float(text))
    except ValueError:
        return None


def path_label(path: str, task):
    if task.path_label_mode is None:
        return None
    parts = [part.upper() for part in path.replace("\\", "/").split("/")]
    if task.path_label_mode == "adni_mci":
        if "CN" in parts:
            return 0
        if "MCI" in parts:
            return 1
    if task.path_label_mode == "adni_ad":
        if "CN" in parts:
            return 0
        if "AD" in parts:
            return 1
    return None


def inspect_task(task) -> dict:
    labels = label_rows(task)
    index = roi_index(task.roi_dir, task.subject_parser)
    result = {"task_id": task.task_id, "benchmark": task.benchmark, "splits": {}}
    for split, split_file in task.split_files.items():
        total = missing_fmri = missing_label = bad_label = 0
        targets = []
        with open(split_file, "r", encoding="utf-8") as handle:
            entries = [line.strip() for line in handle if line.strip()]
        for entry in entries:
            total += 1
            path = None
            candidates = split_candidates(entry, task.subject_parser)
            for candidate in candidates:
                path = index.get(candidate)
                if path:
                    break
            if path is None:
                missing_fmri += 1
                continue
            target = path_label(path, task)
            if target is None:
                row = None
                for candidate in candidates:
                    row = labels.get(candidate)
                    if row:
                        break
                if row is None:
                    missing_label += 1
                    continue
                target = coerce_label(row.get(task.label_column, ""), task)
                if target is None:
                    bad_label += 1
                    continue
            targets.append(float(target))
        result["splits"][split] = {
            "total": total,
            "usable": len(targets),
            "missing_fmri": missing_fmri,
            "missing_label": missing_label,
            "bad_label": bad_label,
            "target_min": min(targets) if targets else None,
            "target_max": max(targets) if targets else None,
            "target_mean": mean(targets) if targets else None,
            "target_std": pstdev(targets) if len(targets) > 1 else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_profile", choices=["nas", "hs"], required=True)
    parser.add_argument("--benchmark", choices=["all", "origin", "omni"], default="all")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    benchmarks = ["origin", "omni"] if args.benchmark == "all" else [args.benchmark]
    report = {"data_profile": args.data_profile, "tasks": []}
    for benchmark in benchmarks:
        for task in filter_tasks(benchmark, data_profile=args.data_profile):
            report["tasks"].append(inspect_task(task))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

