from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .launch import shard_tasks
from .tasks import filter_tasks


PYTHON_BIN = "/data/ningzh/envs/harmony/bin/python"
DEFAULT_REPO = {
    "nas": "/mnt/dataset3/nzh/fmri_FM/Brain-Harmony",
    "hs": "/vePFS-0x0d/nzh/fmri FM/Brain-Harmony_v3",
}


def shquote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"

PARAMS = {
    "lp": {
        "seeds": [0, 1, 2, 3, 4],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.0,
        "hidden_dim": 512,
        "batch_size": 64,
        "epochs": 80,
        "patience": 12,
    },
    "mlp": {
        "seeds": [0, 1, 2, 3, 4],
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "hidden_dim": 512,
        "batch_size": 64,
        "epochs": 100,
        "patience": 15,
    },
    "ft": {
        "seeds": [0, 1, 2],
        "lr": 5e-4,
        "weight_decay": 0.05,
        "dropout": 0.0,
        "hidden_dim": 512,
        "batch_size": 8,
        "epochs": 50,
        "patience": 8,
        "num_latent_tokens": 128,
        "drop_path": 0.1,
        "stage2_head": "linear",
        "warmup_epochs": 5,
        "min_lr": 1e-6,
    },
}


def command(task_id: str, mode: str, seed: int, output_root: Path, data_profile: str, python_bin: str) -> str:
    params = PARAMS[mode]
    out = output_root / task_id / mode / f"seed_{seed}"
    log = out / "train.log"
    out_q = shquote(out.as_posix())
    feat_q = shquote((output_root / "feature_cache").as_posix())
    tok_q = shquote((output_root / "token_cache").as_posix())
    extras = ""
    if mode == "ft":
        extras = (
            f"--num_latent_tokens {params['num_latent_tokens']} --drop_path {params['drop_path']} "
            f"--stage2_head {params['stage2_head']} --warmup_epochs {params['warmup_epochs']} "
            f"--min_lr {params['min_lr']} "
        )
    return (
        f"mkdir -p {out_q}\n"
        f"echo RUN task={task_id} mode={mode} seed={seed} $(date)\n"
        f"{shquote(python_bin)} -m brainharmony_downstream.train "
        f"--task {task_id} --mode {mode} --trials 1 --seed {seed} "
        f"--data_profile {data_profile} --output_dir {out_q} "
        f"--feature_cache_dir {feat_q} "
        f"--token_cache_dir {tok_q} "
        f"--lr {params['lr']} --weight_decay {params['weight_decay']} "
        f"--dropout {params['dropout']} --hidden_dim {params['hidden_dim']} "
        f"--batch_size {params['batch_size']} --epochs {params['epochs']} "
        f"--patience {params['patience']} {extras}"
        f"2>&1 | tee {shquote(log.as_posix())}"
    )


def write_script(path: Path, output_root: Path, tasks: list[str], cuda: str, data_profile: str, repo_root: str, python_bin: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {shquote(repo_root)}",
        "export PYTHONPATH=$PWD:${PYTHONPATH:-}",
        f"export BH_DATA_PROFILE={data_profile}",
        f"export CUDA_VISIBLE_DEVICES={cuda}",
        "echo START $(date)",
        "nvidia-smi || true",
    ]
    for task_id in tasks:
        for mode in ("lp", "mlp", "ft"):
            for seed in PARAMS[mode]["seeds"]:
                lines.append(command(task_id, mode, seed, output_root, data_profile, python_bin))
    lines.extend(["echo END $(date)", "exec bash"])
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def write_run_md(output_root: Path, benchmark: str, tasks: list[str], sessions: list[str], data_profile: str) -> None:
    text = f"""# v3_formal BrainHarmonix-F {benchmark}

- Created: {datetime.now().isoformat(timespec="seconds")}
- Data profile: `{data_profile}`
- Purpose: fMRI-only v3 formal run with train-standardized regression labels.
- Stage2 `ft`: frozen Harmonix-F token extraction plus original-repo-style latent-token downstream transformer.
- Tasks: {", ".join(tasks)}
- Sessions: {", ".join(f"`{s}`" for s in sessions)}
- Checkpoint policy: each seed saves only `checkpoint-best.pth`.

## Monitor
- `tmux ls`
- `watch -n 5 nvidia-smi`
- `tail -f {output_root.as_posix()}/*/*/seed_*/train.log`
"""
    (output_root / "run.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["origin", "omni"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data_profile", choices=["nas", "hs"], required=True)
    parser.add_argument("--repo_linux", default="")
    parser.add_argument("--python_bin", default=PYTHON_BIN)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.root) / "v3_formal"
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [task.task_id for task in filter_tasks(args.benchmark, data_profile=args.data_profile)]
    cuda_ids = [item.strip() for item in args.cuda.split(",") if item.strip()]
    sessions: list[str] = []
    for idx, shard in enumerate(shard_tasks(tasks, args.shards)):
        suffix = "" if args.shards == 1 else f"_s{idx + 1}"
        session = f"bh_v3_{args.benchmark}{suffix}"
        script = output_root / f"run_{args.benchmark}{suffix}.sh"
        write_script(script, output_root, shard, cuda_ids[idx % len(cuda_ids)], args.data_profile, repo_root, args.python_bin)
        sessions.append(session)
        print(f"tmux new -d -s {session} 'bash {script.as_posix()}'")
    write_run_md(output_root, args.benchmark, tasks, sessions, args.data_profile)
    print(output_root)


if __name__ == "__main__":
    main()
    repo_root = args.repo_linux or DEFAULT_REPO[args.data_profile]
