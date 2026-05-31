from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path


@dataclass(frozen=True)
class DataProfile:
    name: str
    fmri_roi_root: str
    label_root: str
    split_root: str


WINDOWS_NAS = DataProfile(
    name="nas",
    fmri_roi_root=r"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi",
    label_root=r"\\10.16.93.90\dataset3\nzh\fmri_FM\data\data_csv",
    split_root=r"\\10.16.93.90\dataset3\nzh\fmri_FM\data\split",
)

LINUX_NAS = DataProfile(
    name="nas",
    fmri_roi_root="/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi",
    label_root="/mnt/dataset3/nzh/fmri_FM/data/data_csv",
    split_root="/mnt/dataset3/nzh/fmri_FM/data/split",
)

HS = DataProfile(
    name="hs",
    fmri_roi_root="/vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi",
    label_root="/vePFS-0x0d/nzh/data/fmri/data_csv",
    split_root="/vePFS-0x0d/nzh/data/fmri/split",
)


def _exists(profile: DataProfile) -> bool:
    return Path(profile.fmri_roi_root).exists() and Path(profile.label_root).exists() and Path(profile.split_root).exists()


def _with_env_overrides(profile: DataProfile) -> DataProfile:
    prefix = profile.name.upper()
    return replace(
        profile,
        fmri_roi_root=os.environ.get(f"BH_{prefix}_FMRI_ROI_ROOT", os.environ.get("BH_FMRI_ROI_ROOT", profile.fmri_roi_root)),
        label_root=os.environ.get(f"BH_{prefix}_LABEL_ROOT", os.environ.get("BH_LABEL_ROOT", profile.label_root)),
        split_root=os.environ.get(f"BH_{prefix}_SPLIT_ROOT", os.environ.get("BH_SPLIT_ROOT", profile.split_root)),
    )


def get_data_profile(name: str = "auto") -> DataProfile:
    requested = (name or "auto").lower()
    if requested == "auto":
        env_name = os.environ.get("BH_DATA_PROFILE", "").lower()
        if env_name:
            requested = env_name

    if requested == "nas":
        return _with_env_overrides(WINDOWS_NAS if os.name == "nt" else LINUX_NAS)
    if requested == "hs":
        return _with_env_overrides(HS)
    if requested not in {"auto", ""}:
        raise ValueError(f"Unknown data profile: {name}. Use auto, nas, or hs.")

    candidates = [WINDOWS_NAS] if os.name == "nt" else [LINUX_NAS, HS]
    for profile in candidates:
        if _exists(profile):
            return _with_env_overrides(profile)
    return _with_env_overrides(candidates[0])
