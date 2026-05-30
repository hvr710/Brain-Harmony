from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .tasks import TaskSpec


WINDOWS_MOUNTS = {
    "/mnt/dataset3": r"\\10.16.93.90\dataset3",
    "/mnt/dataset4": r"\\10.20.33.82\dataset4",
}

MISSING = {"", "nan", "NaN", "NA", "N/A", "None", "null", "-999", "-999.0"}


@dataclass
class Sample:
    path: str
    sample_id: str
    target: float | int
    split_key: str


def resolve_path(path: str) -> str:
    if os.name != "nt":
        return path
    normalized = path.replace("\\", "/")
    for mount, unc in WINDOWS_MOUNTS.items():
        if normalized.startswith(mount):
            suffix = normalized[len(mount) :].lstrip("/")
            return str(Path(unc) / Path(suffix))
    return path


def _base_name(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").split("/")[-1]


def _norm_key(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("sub-"):
        text = text[4:]
    if re.fullmatch(r"\d+(\.0+)?", text):
        return str(int(float(text)))
    return text


def _subject_candidates(path: str, parser: str) -> list[str]:
    name = _base_name(path)
    stem = re.sub(r"\.(npy|npz)$", "", name, flags=re.IGNORECASE)
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        for item in (value, value.lstrip("0") or "0"):
            key = _norm_key(item)
            if key not in candidates:
                candidates.append(key)

    if parser == "abide":
        match = re.search(r"(\d{7})", stem)
        add(match.group(1) if match else None)
    elif parser == "adhd":
        add(stem.split("_")[0])
    elif parser == "ppmi":
        match = re.search(r"sub-(\d+)", stem)
        add(match.group(1) if match else None)
    elif parser == "adni":
        match = re.search(r"sub-?([A-Za-z0-9]+)", stem)
        if not match:
            match = re.search(r"(\d{3}S\d+)", stem, flags=re.IGNORECASE)
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
    return candidates


def _split_key_candidates(value: str, parser: str) -> list[str]:
    text = value.strip()
    if "/" in text or "\\" in text or text.lower().endswith((".npy", ".npz")):
        return _subject_candidates(text, parser)
    candidates: list[str] = []
    for item in (text, text.lstrip("0") or "0"):
        key = _norm_key(item)
        if key not in candidates:
            candidates.append(key)
    if parser == "nki" and text.startswith("A"):
        key = _norm_key(text[1:])
        if key not in candidates:
            candidates.append(key)
    return candidates


def _read_split(path: str) -> list[str]:
    local = resolve_path(path)
    with open(local, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


@lru_cache(maxsize=64)
def _roi_file_index(roi_dir: str, parser: str) -> dict[str, str]:
    local = resolve_path(roi_dir)
    root = Path(local)
    index: dict[str, str] = {}
    if not root.exists():
        return index
    for suffix in ("*.npy", "*.npz"):
        for path in root.rglob(suffix):
            path_text = str(path)
            for key in _subject_candidates(path_text, parser):
                index.setdefault(key, path_text)
    return index


def _resolve_roi_entry(entry: str, spec: TaskSpec) -> tuple[str | None, str]:
    if "/" in entry or "\\" in entry or entry.lower().endswith((".npy", ".npz")):
        return entry, (_subject_candidates(entry, spec.subject_parser) or [entry])[0]

    index = _roi_file_index(spec.roi_dir, spec.subject_parser)
    for candidate in _split_key_candidates(entry, spec.subject_parser):
        path = index.get(candidate)
        if path is not None:
            return path, candidate
    return None, (_split_key_candidates(entry, spec.subject_parser) or [entry])[0]


def _load_label_rows(spec: TaskSpec) -> dict[str, dict[str, str]]:
    if not spec.label_csv or not spec.label_key:
        return {}
    lookup: dict[str, dict[str, str]] = {}
    with open(resolve_path(spec.label_csv), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _norm_key(row.get(spec.label_key, ""))
            if key and key not in lookup:
                lookup[key] = row
    return lookup


def _coerce_label(value: Any, spec: TaskSpec) -> float | int | None:
    text = str(value).strip()
    if text in MISSING:
        return None
    if spec.is_regression or spec.label_mode == "float":
        try:
            return float(text)
        except ValueError:
            return None
    try:
        numeric = int(float(text))
    except ValueError:
        return None
    if spec.label_mode == "nonzero_positive":
        return 0 if numeric == 0 else 1
    return numeric


def _path_label(path: str, spec: TaskSpec) -> int | None:
    if spec.path_label_mode is None:
        return None
    parts = [part.upper() for part in path.replace("\\", "/").split("/")]
    if spec.path_label_mode == "adni_mci":
        if "CN" in parts:
            return 0
        if "MCI" in parts:
            return 1
    if spec.path_label_mode == "adni_ad":
        if "CN" in parts:
            return 0
        if "AD" in parts:
            return 1
    return None


def build_samples(spec: TaskSpec, split: str, max_samples: int | None = None) -> tuple[list[Sample], dict[str, int]]:
    label_rows = _load_label_rows(spec)
    raw_entries = _read_split(spec.split_files[split])
    samples: list[Sample] = []
    stats = {"total": 0, "missing_fmri": 0, "missing_label": 0, "bad_label": 0}

    for raw_entry in raw_entries:
        stats["total"] += 1
        raw_path, sample_id = _resolve_roi_entry(raw_entry, spec)
        if raw_path is None:
            stats["missing_fmri"] += 1
            continue

        target = _path_label(raw_path, spec)
        candidates = _split_key_candidates(raw_entry, spec.subject_parser)
        if sample_id not in candidates:
            candidates = [sample_id, *candidates]

        if target is None:
            row = None
            for candidate in candidates:
                row = label_rows.get(candidate)
                if row is not None:
                    sample_id = candidate
                    break
            if row is None:
                stats["missing_label"] += 1
                continue
            target = _coerce_label(row.get(spec.label_column, ""), spec)
            if target is None:
                stats["bad_label"] += 1
                continue

        samples.append(Sample(path=raw_path, sample_id=sample_id, target=target, split_key=raw_entry))
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples, stats


def cache_sample_name(sample: Sample) -> str:
    digest = hashlib.sha1(sample.path.encode("utf-8")).hexdigest()[:10]
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)
    return f"{safe_id}_{digest}.npz"


def load_roi(path: str) -> np.ndarray:
    local = resolve_path(path)
    if local.lower().endswith(".npz"):
        with np.load(local, allow_pickle=False) as arr:
            if "roi" in arr:
                data = arr["roi"]
            elif "data" in arr:
                data = arr["data"]
            else:
                first_key = list(arr.keys())[0]
                data = arr[first_key]
    else:
        data = np.load(local, allow_pickle=False)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D ROI array from {path}, got shape={data.shape}")
    if data.shape[0] == 400:
        roi = data
    elif data.shape[1] == 400:
        roi = data.T
    else:
        raise ValueError(f"Expected one ROI dimension to be 400 from {path}, got shape={data.shape}")
    return np.nan_to_num(roi, copy=False)


class RoiDownstreamDataset(Dataset):
    def __init__(
        self,
        spec: TaskSpec,
        split: str,
        patch_size: int = 48,
        target_num_patches: int = 18,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.split = split
        self.patch_size = patch_size
        self.target_num_patches = target_num_patches
        self.target_length = patch_size * target_num_patches
        self.seed = seed
        self.samples, self.stats = build_samples(spec, split, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, roi: np.ndarray, idx: int) -> tuple[np.ndarray, int]:
        length = roi.shape[1]
        if length > self.target_length:
            if self.split == "train":
                rng = random.Random(self.seed + idx + length)
                start = rng.randint(0, length - self.target_length)
            else:
                start = (length - self.target_length) // 2
            roi = roi[:, start : start + self.target_length]
            length = self.target_length

        valid_length = length
        if length < self.target_length:
            padded = np.zeros((400, self.target_length), dtype=np.float32)
            padded[:, :length] = roi[:, :length]
            roi = padded
        return roi, valid_length

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        roi = load_roi(sample.path)
        roi, valid_length = self._crop_or_pad(roi, idx)
        valid_patches = min(self.target_num_patches, int(math.ceil(valid_length / self.patch_size)))
        mask2d = np.zeros((400, self.target_num_patches), dtype=np.int64)
        mask2d[:, :valid_patches] = 1
        target_dtype = torch.long if self.spec.is_classification else torch.float32
        return {
            "x": torch.from_numpy(roi).unsqueeze(0),
            "patch_size": torch.tensor(self.patch_size, dtype=torch.long),
            "attention_mask": torch.from_numpy(mask2d.reshape(-1)),
            "target": torch.tensor(sample.target, dtype=target_dtype),
            "sample_id": sample.sample_id,
            "path": sample.path,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "patch_size": torch.stack([item["patch_size"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "path": [item["path"] for item in batch],
    }
