# BrainHarmonix-F v3 fMRI-only downstream protocol

This branch is the v3 replacement for the earlier quick downstream probe.

## Data profiles

Use `--data_profile nas` on servers with NAS mounts such as `l40` and `4090`.

- fMRI ROI root: `/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi`
- labels: `/mnt/dataset3/nzh/fmri_FM/data/data_csv`
- splits: `/mnt/dataset3/nzh/fmri_FM/data/split`

Use `--data_profile hs` on the Volcengine host.

- fMRI ROI root: `/vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi`
- labels: `/vePFS-0x0d/nzh/data/fmri/data_csv`
- splits: `/vePFS-0x0d/nzh/data/fmri/split`

The split files are `train_v1.txt`, `val_v1.txt`, and `test_v1.txt` under each dataset folder.

## What changed from v2

- Split files now contain subject ids, and the loader resolves each id to the matching `.npy` or `.npz` ROI file.
- Labels now come from the new `data_csv` roots, not the old dataset parent folders.
- Regression labels are standardized using train split mean/std for training.
- Regression output reports both standardized `mse` / `mse_z` and raw-scale `mse_raw`, plus Pearson.
- `ft` is no longer raw end-to-end encoder finetuning. It now follows the original repository structure more closely: stage0 token extraction with frozen Harmonix-F, then a stage2 latent-token downstream transformer.
- fMRI-only is used. T1w is not required or loaded.

## Validate data

```bash
python -m brainharmony_downstream.validate_data --data_profile nas --benchmark all --check_arrays 1 --output v3_data_check_nas.json
python -m brainharmony_downstream.validate_data --data_profile hs --benchmark all --check_arrays 1 --output v3_data_check_hs.json
```

## Run one smoke test

```bash
python -m brainharmony_downstream.train --task omni_abide_age --mode lp --data_profile nas --output_dir outputs/smoke_lp --smoke
python -m brainharmony_downstream.train --task omni_abide_age --mode ft --data_profile nas --output_dir outputs/smoke_ft --token_cache_dir outputs/token_cache --smoke
```

## Generate v3 formal scripts

```bash
python -m brainharmony_downstream.v3_formal_launch --benchmark origin --root 1_origin_ds/outputs --data_profile nas --cuda 0 --shards 1
python -m brainharmony_downstream.v3_formal_launch --benchmark omni --root 2_omni_ds/outputs --data_profile nas --cuda 5,6,7 --shards 3
```

