from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .data import RoiDownstreamDataset, build_samples, cache_sample_name, collate_batch
from .modeling import DownstreamHead, FMRITokenStage2Model, build_harmonix_f_encoder, pool_tokens
from .tasks import get_task


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _out_dim(task) -> int:
    return int(task.num_classes or 1) if task.is_classification else 1


def _loss(task) -> nn.Module:
    return nn.CrossEntropyLoss() if task.is_classification else nn.MSELoss()


@dataclass
class TargetTransform:
    is_regression: bool
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def from_task(cls, task) -> "TargetTransform":
        if not task.is_regression:
            return cls(False)
        samples, _ = build_samples(task, "train")
        values = np.asarray([sample.target for sample in samples], dtype=np.float32)
        std = float(values.std())
        return cls(True, mean=float(values.mean()), std=std if std > 1e-6 else 1.0)

    def encode(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not self.is_regression:
            return y
        return (y - self.mean) / self.std

    def decode(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not self.is_regression:
            return y
        return y * self.std + self.mean

    def to_json(self) -> dict[str, float | bool]:
        return {"is_regression": self.is_regression, "mean": self.mean, "std": self.std}


def _metrics(task, y_true_raw: np.ndarray, y_pred: np.ndarray, loss: float, transform: TargetTransform) -> dict[str, float]:
    if task.is_classification:
        pred = y_pred.argmax(axis=1)
        average = "binary" if (task.num_classes or 2) == 2 else "weighted"
        return {
            "loss": float(loss),
            "acc": float(accuracy_score(y_true_raw, pred)),
            "f1": float(f1_score(y_true_raw, pred, average=average, zero_division=0)),
        }

    pred_z = y_pred.reshape(-1).astype(np.float64)
    true_raw = y_true_raw.reshape(-1).astype(np.float64)
    true_z = transform.encode(true_raw)
    pred_raw = transform.decode(pred_z)
    mse_z = float(np.mean((pred_z - true_z) ** 2))
    mse_raw = float(np.mean((pred_raw - true_raw) ** 2))
    if len(true_raw) > 1 and np.std(true_raw) > 0 and np.std(pred_raw) > 0:
        pearson = float(np.corrcoef(true_raw, pred_raw)[0, 1])
    else:
        pearson = 0.0
    return {"loss": float(loss), "mse": mse_z, "mse_z": mse_z, "mse_raw": mse_raw, "pearson": pearson}


def _score(task, metrics: dict[str, float]) -> float:
    return metrics["f1"] if task.is_classification else metrics["pearson"]


def _make_raw_loader(task, split: str, args, shuffle: bool, batch_size: int) -> DataLoader:
    max_samples = args.smoke_samples if args.smoke else None
    dataset = RoiDownstreamDataset(task, split, max_samples=max_samples, seed=args.seed)
    if len(dataset) == 0:
        raise RuntimeError(f"No usable samples for {task.task_id}/{split}; stats={dataset.stats}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=False,
    )


@torch.no_grad()
def extract_split_features(task, split: str, args, device: torch.device, cache_path: Path, encoder=None) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loader = _make_raw_loader(task, split, args, shuffle=False, batch_size=args.extract_batch_size)
    if encoder is None:
        encoder = build_harmonix_f_encoder(args.checkpoint, device=device, attn_mode=args.attn_mode)
    encoder.eval()
    xs, ys, ids = [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        patch_size = batch["patch_size"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            tokens = encoder(x, int(patch_size[0].item()), attention_mask=mask)
            features = pool_tokens(tokens, mask)
        xs.append(features.float().cpu().numpy())
        ys.append(batch["target"].cpu().numpy())
        ids.extend(batch["sample_id"])
    np.savez(cache_path, x=np.concatenate(xs), y_raw=np.concatenate(ys), sample_id=np.asarray(ids))


def load_or_extract_features(task, args, device: torch.device) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache_root = Path(args.feature_cache_dir or Path(args.output_dir) / "feature_cache")
    missing: list[tuple[str, Path]] = []
    for split in ("train", "val", "test"):
        cache_path = cache_root / task.task_id / f"{split}.npz"
        if not cache_path.exists() or args.rebuild_cache:
            missing.append((split, cache_path))
    if missing:
        encoder = build_harmonix_f_encoder(args.checkpoint, device=device, attn_mode=args.attn_mode)
        for split, cache_path in missing:
            extract_split_features(task, split, args, device, cache_path, encoder=encoder)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    data = {}
    for split in ("train", "val", "test"):
        arr = np.load(cache_root / task.task_id / f"{split}.npz", allow_pickle=False)
        y_key = "y_raw" if "y_raw" in arr else "y"
        data[split] = (arr["x"].astype("float32"), arr[y_key])
    return data


@torch.no_grad()
def extract_split_tokens(task, split: str, args, device: torch.device, cache_dir: Path, encoder=None) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = RoiDownstreamDataset(task, split, max_samples=args.smoke_samples if args.smoke else None, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    if encoder is None:
        encoder = build_harmonix_f_encoder(args.checkpoint, device=device, attn_mode=args.attn_mode)
    encoder.eval()
    manifest: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        patch_size = batch["patch_size"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            tokens = encoder(x, int(patch_size[0].item()), attention_mask=mask)
        tokens_np = tokens.detach().cpu().to(torch.float16).numpy()
        mask_np = batch["attention_mask"].cpu().numpy().astype(np.int64)
        y_np = batch["target"].cpu().numpy()
        for i, sample_id in enumerate(batch["sample_id"]):
            sample = dataset.samples[len(manifest)]
            name = cache_sample_name(sample)
            path = cache_dir / name
            np.savez_compressed(path, tokens=tokens_np[i], attention_mask=mask_np[i], y_raw=y_np[i], sample_id=sample_id)
            manifest.append({"file": name, "sample_id": sample_id, "y_raw": float(y_np[i]) if task.is_regression else int(y_np[i])})
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def ensure_token_cache(task, args, device: torch.device) -> dict[str, list[dict[str, Any]]]:
    cache_root = Path(args.token_cache_dir or Path(args.output_dir) / "token_cache")
    manifests: dict[str, list[dict[str, Any]]] = {}
    missing: list[tuple[str, Path]] = []
    for split in ("train", "val", "test"):
        cache_dir = cache_root / task.task_id / split
        manifest_path = cache_dir / "manifest.json"
        if args.rebuild_cache or not manifest_path.exists():
            missing.append((split, cache_dir))
        else:
            manifests[split] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if missing:
        encoder = build_harmonix_f_encoder(args.checkpoint, device=device, attn_mode=args.attn_mode)
        for split, cache_dir in missing:
            manifests[split] = extract_split_tokens(task, split, args, device, cache_dir, encoder=encoder)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return manifests


class TokenCacheDataset(Dataset):
    def __init__(self, cache_dir: Path, manifest: list[dict[str, Any]], task, transform: TargetTransform) -> None:
        self.cache_dir = cache_dir
        self.manifest = manifest
        self.task = task
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.manifest[idx]
        arr = np.load(self.cache_dir / item["file"], allow_pickle=False)
        tokens = torch.from_numpy(arr["tokens"].astype("float32"))
        mask = torch.from_numpy(arr["attention_mask"].astype("int64"))
        y_raw = float(item["y_raw"]) if self.task.is_regression else int(item["y_raw"])
        if self.task.is_regression:
            target = torch.tensor((y_raw - self.transform.mean) / self.transform.std, dtype=torch.float32)
        else:
            target = torch.tensor(y_raw, dtype=torch.long)
        return {"tokens": tokens, "attention_mask": mask, "target": target, "target_raw": torch.tensor(y_raw), "sample_id": item["sample_id"]}


def collate_tokens(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "target_raw": torch.stack([item["target_raw"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
    }


def train_feature_head(task, arrays, params, args, device: torch.device, transform: TargetTransform) -> dict[str, Any]:
    head = DownstreamHead(768, _out_dim(task), mode=args.mode, hidden_dim=params["hidden_dim"], dropout=params["dropout"]).to(device)
    criterion = _loss(task)
    optimizer = torch.optim.AdamW(head.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best = {"score": -math.inf, "epoch": -1, "val": {}, "test": {}}

    x_train, y_train_raw = arrays["train"]
    y_train = transform.encode(y_train_raw.astype("float32")) if task.is_regression else y_train_raw
    target_dtype = torch.long if task.is_classification else torch.float32
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train, dtype=target_dtype), torch.tensor(y_train_raw))
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    stale_epochs = 0
    for epoch in range(params["epochs"]):
        head.train()
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = head(x)
            loss = criterion(out, y if task.is_classification else y.view(-1, 1))
            loss.backward()
            optimizer.step()
        val_metrics = eval_feature_head(task, head, arrays["val"], criterion, device, transform)
        score = _score(task, val_metrics)
        if score >= best["score"]:
            stale_epochs = 0
            best = {"score": score, "epoch": epoch, "val": val_metrics, "test": eval_feature_head(task, head, arrays["test"], criterion, device, transform)}
            torch.save({"model": head.state_dict(), "params": params, "epoch": epoch, "target_transform": transform.to_json()}, Path(args.output_dir) / "checkpoint-best.pth")
        else:
            stale_epochs += 1
            if stale_epochs >= params["patience"]:
                break
    return best


@torch.no_grad()
def eval_feature_head(task, head, arrays, criterion, device: torch.device, transform: TargetTransform) -> dict[str, float]:
    head.eval()
    x, y_raw = arrays
    y = transform.encode(y_raw.astype("float32")) if task.is_regression else y_raw
    target_dtype = torch.long if task.is_classification else torch.float32
    ds = TensorDataset(torch.from_numpy(x), torch.tensor(y, dtype=target_dtype), torch.tensor(y_raw))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    losses, preds, targets_raw = [], [], []
    for xb, yb, yrb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = head(xb)
        loss = criterion(out, yb if task.is_classification else yb.view(-1, 1))
        losses.append(float(loss.item()) * len(xb))
        preds.append(out.detach().cpu().numpy())
        targets_raw.append(yrb.numpy())
    return _metrics(task, np.concatenate(targets_raw), np.concatenate(preds), sum(losses) / len(ds), transform)


def _adjust_lr(optimizer, epoch: int, step: int, steps_per_epoch: int, params: dict[str, Any]) -> None:
    progress = epoch + step / max(1, steps_per_epoch)
    warmup = params.get("warmup_epochs", 5)
    if progress < warmup:
        lr = params["lr"] * progress / max(1e-6, warmup)
    else:
        total = max(warmup + 1, params["epochs"])
        ratio = (progress - warmup) / max(1e-6, total - warmup)
        lr = params.get("min_lr", 1e-6) + (params["lr"] - params.get("min_lr", 1e-6)) * 0.5 * (1.0 + math.cos(math.pi * ratio))
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_stage2_tokens(task, params, args, device: torch.device, transform: TargetTransform) -> dict[str, Any]:
    manifests = ensure_token_cache(task, args, device)
    cache_root = Path(args.token_cache_dir or Path(args.output_dir) / "token_cache") / task.task_id
    datasets = {
        split: TokenCacheDataset(cache_root / split, manifests[split], task, transform)
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=params["batch_size"], shuffle=True, num_workers=args.num_workers, collate_fn=collate_tokens, pin_memory=torch.cuda.is_available()),
        "val": DataLoader(datasets["val"], batch_size=params["batch_size"], shuffle=False, num_workers=args.num_workers, collate_fn=collate_tokens, pin_memory=torch.cuda.is_available()),
        "test": DataLoader(datasets["test"], batch_size=params["batch_size"], shuffle=False, num_workers=args.num_workers, collate_fn=collate_tokens, pin_memory=torch.cuda.is_available()),
    }
    model = FMRITokenStage2Model(
        out_dim=_out_dim(task),
        num_latent_tokens=params["num_latent_tokens"],
        drop_path_rate=params["drop_path"],
        head_type=params["stage2_head"],
        head_dropout=params["dropout"],
        attn_mode=args.attn_mode,
    ).to(device)
    criterion = _loss(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = {"score": -math.inf, "epoch": -1, "val": {}, "test": {}}
    stale_epochs = 0
    for epoch in range(params["epochs"]):
        model.train()
        for step, batch in enumerate(loaders["train"]):
            _adjust_lr(optimizer, epoch, step, len(loaders["train"]), params)
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(tokens, mask)
                loss = criterion(out, target if task.is_classification else target.view(-1, 1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        val_metrics = eval_stage2_tokens(task, model, loaders["val"], criterion, device, transform)
        score = _score(task, val_metrics)
        if score >= best["score"]:
            stale_epochs = 0
            best = {"score": score, "epoch": epoch, "val": val_metrics, "test": eval_stage2_tokens(task, model, loaders["test"], criterion, device, transform)}
            torch.save({"model": model.state_dict(), "params": params, "epoch": epoch, "target_transform": transform.to_json()}, Path(args.output_dir) / "checkpoint-best.pth")
        else:
            stale_epochs += 1
            if stale_epochs >= params["patience"]:
                break
    return best


@torch.no_grad()
def eval_stage2_tokens(task, model, loader, criterion, device: torch.device, transform: TargetTransform) -> dict[str, float]:
    model.eval()
    losses, preds, targets_raw = [], [], []
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(tokens, mask)
            loss = criterion(out, target if task.is_classification else target.view(-1, 1))
        losses.append(float(loss.item()) * tokens.shape[0])
        preds.append(out.float().cpu().numpy())
        targets_raw.append(batch["target_raw"].cpu().numpy())
    return _metrics(task, np.concatenate(targets_raw), np.concatenate(preds), sum(losses) / len(loader.dataset), transform)


def suggest_params(trial, args) -> dict[str, Any]:
    mode = args.mode
    if args.smoke:
        return {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "hidden_dim": 256, "batch_size": 2, "epochs": 1, "patience": 1, "num_latent_tokens": 16, "drop_path": 0.0, "stage2_head": "linear", "warmup_epochs": 0, "min_lr": 1e-6}
    if trial is None:
        defaults = {
            "lp": {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "hidden_dim": 512, "batch_size": 64, "epochs": 80, "patience": 12},
            "mlp": {"lr": 5e-4, "weight_decay": 1e-4, "dropout": 0.2, "hidden_dim": 512, "batch_size": 64, "epochs": 100, "patience": 15},
            "ft": {"lr": 5e-4, "weight_decay": 0.05, "dropout": 0.0, "hidden_dim": 512, "batch_size": 8, "epochs": 50, "patience": 8, "num_latent_tokens": 128, "drop_path": 0.1, "stage2_head": "linear", "warmup_epochs": 5, "min_lr": 1e-6},
        }[mode]
        keys = ["lr", "weight_decay", "dropout", "hidden_dim", "batch_size", "epochs", "patience", "num_latent_tokens", "drop_path", "stage2_head", "warmup_epochs", "min_lr"]
        return {key: getattr(args, key, None) if getattr(args, key, None) is not None else defaults.get(key) for key in keys if key in defaults or getattr(args, key, None) is not None}
    return {
        "lr": trial.suggest_float("lr", 1e-5 if mode == "ft" else 1e-4, 5e-4 if mode == "ft" else 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2 if mode != "ft" else 0.1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16] if mode == "ft" else [16, 32, 64]),
        "epochs": 30 if mode == "ft" else 50,
        "patience": 8 if mode == "ft" else 10,
        "num_latent_tokens": trial.suggest_categorical("num_latent_tokens", [64, 128]),
        "drop_path": trial.suggest_float("drop_path", 0.0, 0.2),
        "stage2_head": trial.suggest_categorical("stage2_head", ["linear", "mlp"]),
        "warmup_epochs": 5,
        "min_lr": 1e-6,
    }


def run_trial(task, args, device: torch.device, trial=None, arrays=None) -> dict[str, Any]:
    params = suggest_params(trial, args)
    transform = TargetTransform.from_task(task)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode in {"lp", "mlp"}:
        if arrays is None:
            arrays = load_or_extract_features(task, args, device)
        result = train_feature_head(task, arrays, params, args, device, transform)
    else:
        result = train_stage2_tokens(task, params, args, device, transform)
    result["params"] = params
    result["seed"] = args.seed
    result["target_transform"] = transform.to_json()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["lp", "mlp", "ft"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_profile", default="auto", choices=["auto", "nas", "hs"])
    parser.add_argument("--checkpoint", default="checkpoints/harmonix-f/model.pth")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--token_cache_dir", default="")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn_mode", default="flash_attention_2")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--extract_batch_size", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_samples", type=int, default=8)
    parser.add_argument("--storage", default="")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_latent_tokens", type=int, default=None)
    parser.add_argument("--drop_path", type=float, default=None)
    parser.add_argument("--stage2_head", choices=["linear", "mlp"], default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    args = parser.parse_args()

    start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    task = get_task(args.task, data_profile=args.data_profile)
    device = _device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    arrays = load_or_extract_features(task, args, device) if args.mode in {"lp", "mlp"} else None

    if args.trials > 1 and not args.smoke:
        import optuna

        study = optuna.create_study(direction="maximize", study_name=f"{task.task_id}_{args.mode}", storage=args.storage or None, load_if_exists=True)

        def objective(trial):
            trial_dir = Path(args.output_dir) / f"trial_{trial.number:04d}"
            trial_args = argparse.Namespace(**vars(args))
            trial_args.output_dir = str(trial_dir)
            result = run_trial(task, trial_args, device, trial=trial, arrays=arrays)
            trial.set_user_attr("result", result)
            return result["score"]

        study.optimize(objective, n_trials=args.trials)
        result = {"best_value": study.best_value, "best_params": study.best_params, "best_trial": study.best_trial.number}
    else:
        result = run_trial(task, args, device, trial=None, arrays=arrays)

    result["elapsed_sec"] = time.time() - start
    result["task"] = task.task_id
    result["mode"] = args.mode
    result["data_profile"] = task.data_profile
    (Path(args.output_dir) / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

