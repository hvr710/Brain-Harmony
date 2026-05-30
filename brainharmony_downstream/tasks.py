from __future__ import annotations

from dataclasses import dataclass

from .paths import DataProfile, get_data_profile


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    benchmark: str
    name: str
    problem: str
    metric: str
    split_files: dict[str, str]
    roi_dir: str
    subject_parser: str
    label_csv: str | None = None
    label_key: str | None = None
    label_column: str | None = None
    num_classes: int | None = None
    label_mode: str = "auto"
    path_label_mode: str | None = None
    data_profile: str = "auto"
    skip_reason: str | None = None

    @property
    def is_classification(self) -> bool:
        return self.problem == "classification"

    @property
    def is_regression(self) -> bool:
        return self.problem == "regression"


def _join(*parts: str) -> str:
    cleaned = [part for part in parts if part]
    if not cleaned:
        return ""
    first = cleaned[0].rstrip("/\\")
    rest = [part.strip("/\\") for part in cleaned[1:]]
    sep = "\\" if first.startswith("\\\\") else "/"
    return sep.join([first, *rest])


def _split(profile: DataProfile, dataset: str, version: str = "v1") -> dict[str, str]:
    root = _join(profile.split_root, dataset)
    return {
        "train": _join(root, f"train_{version}.txt"),
        "val": _join(root, f"val_{version}.txt"),
        "test": _join(root, f"test_{version}.txt"),
    }


def _roi(profile: DataProfile, dataset: str, subdir: str = "Schaefer2018_400") -> str:
    return _join(profile.fmri_roi_root, dataset, subdir)


def build_tasks(profile_name: str = "auto") -> dict[str, TaskSpec]:
    profile = get_data_profile(profile_name)
    csv = profile.label_root

    abide_splits = _split(profile, "ABIDE")
    ppmi_splits = _split(profile, "PPMI")
    adhd_splits = _split(profile, "ADHD")
    adni_mci_splits = _split(profile, "ADNI_MCI")
    adni_ad_splits = _split(profile, "ADNI_AD")
    nki_splits = _split(profile, "NKI")
    sald_splits = _split(profile, "SALD")
    abcd_splits = _split(profile, "ABCD")
    hcp_splits = _split(profile, "HCP")
    bhrc_splits = _split(profile, "BHRC")

    tasks: dict[str, TaskSpec] = {
        "origin_abide_i_asd": TaskSpec(
            task_id="origin_abide_i_asd",
            benchmark="origin",
            name="ABIDE-I ASD vs Control",
            problem="classification",
            metric="f1",
            split_files=abide_splits,
            roi_dir=_roi(profile, "ABIDE"),
            subject_parser="abide",
            label_csv=_join(csv, "ABIDE.csv"),
            label_key="Subject",
            label_column="DX_GROUP",
            num_classes=2,
            label_mode="int",
            data_profile=profile.name,
        ),
        "origin_adhd200": TaskSpec(
            task_id="origin_adhd200",
            benchmark="origin",
            name="ADHD-200 ADHD vs Control",
            problem="classification",
            metric="f1",
            split_files=adhd_splits,
            roi_dir=_roi(profile, "ADHD"),
            subject_parser="adhd",
            label_csv=_join(csv, "ADHD.csv"),
            label_key="Subject",
            label_column="DX_GROUP",
            num_classes=2,
            label_mode="int",
            data_profile=profile.name,
        ),
        "origin_ppmi": TaskSpec(
            task_id="origin_ppmi",
            benchmark="origin",
            name="PPMI Control / PD / Prodromal diagnosis",
            problem="classification",
            metric="f1",
            split_files=ppmi_splits,
            roi_dir=_roi(profile, "PPMI"),
            subject_parser="ppmi",
            label_csv=_join(csv, "PPMI.csv"),
            label_key="Subject",
            label_column="DX_GROUP",
            num_classes=3,
            label_mode="int",
            data_profile=profile.name,
        ),
        "origin_adni_mci": TaskSpec(
            task_id="origin_adni_mci",
            benchmark="origin",
            name="ADNI Control vs MCI",
            problem="classification",
            metric="f1",
            split_files=adni_mci_splits,
            roi_dir=_roi(profile, "ADNI"),
            subject_parser="adni",
            num_classes=2,
            label_mode="int",
            path_label_mode="adni_mci",
            data_profile=profile.name,
        ),
        "omni_abide_age": TaskSpec(
            task_id="omni_abide_age",
            benchmark="omni",
            name="ABIDE age regression",
            problem="regression",
            metric="pearson",
            split_files=abide_splits,
            roi_dir=_roi(profile, "ABIDE"),
            subject_parser="abide",
            label_csv=_join(csv, "ABIDE.csv"),
            label_key="Subject",
            label_column="age",
            label_mode="float",
            data_profile=profile.name,
        ),
        "omni_nki_age": TaskSpec(
            task_id="omni_nki_age",
            benchmark="omni",
            name="NKI age regression",
            problem="regression",
            metric="pearson",
            split_files=nki_splits,
            roi_dir=_roi(profile, "NKI"),
            subject_parser="nki",
            label_csv=_join(csv, "NKI.csv"),
            label_key="Subject",
            label_column="age",
            label_mode="float",
            data_profile=profile.name,
        ),
        "omni_sald_age": TaskSpec(
            task_id="omni_sald_age",
            benchmark="omni",
            name="SALD age regression",
            problem="regression",
            metric="pearson",
            split_files=sald_splits,
            roi_dir=_roi(profile, "SALD"),
            subject_parser="sald",
            label_csv=_join(csv, "SALD.csv"),
            label_key="Subject",
            label_column="age",
            label_mode="float",
            data_profile=profile.name,
        ),
        "omni_abcd_sex": TaskSpec(
            task_id="omni_abcd_sex",
            benchmark="omni",
            name="ABCD sex classification",
            problem="classification",
            metric="f1",
            split_files=abcd_splits,
            roi_dir=_roi(profile, "ABCD"),
            subject_parser="abcd",
            label_csv=_join(csv, "ABCD.csv"),
            label_key="Subject",
            label_column="Gender",
            num_classes=2,
            label_mode="int",
            data_profile=profile.name,
        ),
        "omni_hcp_sex": TaskSpec(
            task_id="omni_hcp_sex",
            benchmark="omni",
            name="HCP sex classification",
            problem="classification",
            metric="f1",
            split_files=hcp_splits,
            roi_dir=_roi(profile, "HCP"),
            subject_parser="hcp",
            label_csv=_join(csv, "HCP.csv"),
            label_key="Subject",
            label_column="Gender",
            num_classes=2,
            label_mode="int",
            data_profile=profile.name,
        ),
        "omni_bhrc_sex": TaskSpec(
            task_id="omni_bhrc_sex",
            benchmark="omni",
            name="BHRC sex classification",
            problem="classification",
            metric="f1",
            split_files=bhrc_splits,
            roi_dir=_roi(profile, "BHRC"),
            subject_parser="bhrc",
            label_csv=_join(csv, "BHRC.csv"),
            label_key="Subject",
            label_column="Gender",
            num_classes=2,
            label_mode="int",
            data_profile=profile.name,
        ),
        "omni_ppmi_pd": TaskSpec(
            task_id="omni_ppmi_pd",
            benchmark="omni",
            name="PPMI Control / PD / Prodromal diagnosis",
            problem="classification",
            metric="f1",
            split_files=ppmi_splits,
            roi_dir=_roi(profile, "PPMI"),
            subject_parser="ppmi",
            label_csv=_join(csv, "PPMI.csv"),
            label_key="Subject",
            label_column="DX_GROUP",
            num_classes=3,
            label_mode="int",
            data_profile=profile.name,
        ),
        "omni_adni_mci": TaskSpec(
            task_id="omni_adni_mci",
            benchmark="omni",
            name="ADNI Control vs MCI",
            problem="classification",
            metric="f1",
            split_files=adni_mci_splits,
            roi_dir=_roi(profile, "ADNI"),
            subject_parser="adni",
            num_classes=2,
            label_mode="int",
            path_label_mode="adni_mci",
            data_profile=profile.name,
        ),
        "omni_adni_ad": TaskSpec(
            task_id="omni_adni_ad",
            benchmark="omni",
            name="ADNI Control vs AD",
            problem="classification",
            metric="f1",
            split_files=adni_ad_splits,
            roi_dir=_roi(profile, "ADNI"),
            subject_parser="adni",
            num_classes=2,
            label_mode="int",
            path_label_mode="adni_ad",
            data_profile=profile.name,
        ),
        "omni_nki_edu": TaskSpec(
            task_id="omni_nki_edu",
            benchmark="omni",
            name="NKI education classification",
            problem="classification",
            metric="f1",
            split_files=nki_splits,
            roi_dir=_roi(profile, "NKI"),
            subject_parser="nki",
            label_csv=_join(csv, "NKI.csv"),
            label_key="Subject",
            label_column="education_group",
            num_classes=3,
            label_mode="int",
            data_profile=profile.name,
        ),
    }
    return tasks


TASKS: dict[str, TaskSpec] = build_tasks("auto")

SKIPPED_TASKS: dict[str, str] = {
    "origin_abide_ii_asd": "ABIDE-II has processed ROI files but no confirmed label/test split in the current origin benchmark.",
    "origin_hcpa_flanker": "HCP-A fMRI is unavailable in the provided data.",
}


def task_ids(benchmark: str | None = None, data_profile: str = "auto") -> list[str]:
    tasks = build_tasks(data_profile)
    ids = list(tasks)
    if benchmark:
        ids = [task_id for task_id in ids if tasks[task_id].benchmark == benchmark]
    return ids


def get_task(task_id: str, data_profile: str = "auto") -> TaskSpec:
    tasks = build_tasks(data_profile)
    try:
        return tasks[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(tasks))
        raise KeyError(f"Unknown task_id={task_id}. Known tasks: {known}") from exc


def filter_tasks(benchmark: str | None = None, data_profile: str = "auto") -> list[TaskSpec]:
    tasks = build_tasks(data_profile)
    selected = list(tasks.values())
    if benchmark:
        selected = [task for task in selected if task.benchmark == benchmark]
    return selected
