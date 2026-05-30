from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .launch import REPO_LINUX, shard_tasks
from .tasks import filter_tasks


PYTHON_BIN = "/data/ningzh/envs/harmony/bin/python"

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
        "lr": 5e-5,
        "weight_decay": 0.05,
        "dropout": 0.0,
        "hidden_dim": 512,
        "batch_size": 4,
        "epochs": 40,
        "patience": 6,
    },
}


def command(task_id: str, mode: str, seed: int, output_root: Path) -> str:
    params = PARAMS[mode]
    out = output_root / task_id / mode / f"seed_{seed}"
    log = out / "train.log"
    return (
        f"mkdir -p {out.as_posix()}\n"
        f"echo RUN task={task_id} mode={mode} seed={seed} $(date)\n"
        f"{PYTHON_BIN} -m brainharmony_downstream.train "
        f"--task {task_id} --mode {mode} --trials 1 --seed {seed} "
        f"--output_dir {out.as_posix()} "
        f"--feature_cache_dir {(output_root / 'feature_cache').as_posix()} "
        f"--lr {params['lr']} --weight_decay {params['weight_decay']} "
        f"--dropout {params['dropout']} --hidden_dim {params['hidden_dim']} "
        f"--batch_size {params['batch_size']} --epochs {params['epochs']} "
        f"--patience {params['patience']} "
        f"2>&1 | tee {log.as_posix()}"
    )


def write_script(path: Path, output_root: Path, tasks: list[str], cuda: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {REPO_LINUX}",
        "export PYTHONPATH=$PWD:${PYTHONPATH:-}",
        f"export CUDA_VISIBLE_DEVICES={cuda}",
        "echo START $(date)",
        "nvidia-smi || true",
    ]
    for task_id in tasks:
        for mode in ("lp", "mlp", "ft"):
            for seed in PARAMS[mode]["seeds"]:
                lines.append(command(task_id, mode, seed, output_root))
    lines.extend(["echo END $(date)", "exec bash"])
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def write_run_md(output_root: Path, benchmark: str, tasks: list[str], sessions: list[str]) -> None:
    params = "\n".join(
        f"- `{mode}`: seeds={cfg['seeds']}, lr={cfg['lr']}, wd={cfg['weight_decay']}, "
        f"batch={cfg['batch_size']}, epochs={cfg['epochs']}, patience={cfg['patience']}"
        for mode, cfg in PARAMS.items()
    )
    text = f"""# v2_formal BrainHarmonix-F {benchmark}

- Created: {datetime.now().isoformat(timespec="seconds")}
- Purpose: fixed-hyperparameter formal run, no Optuna
- Tasks: {", ".join(tasks)}
- Sessions: {", ".join(f"`{s}`" for s in sessions)}
- Checkpoint policy: each seed saves only `checkpoint-best.pth`

## Fixed Hyperparameters
{params}

## Monitor
- `tmux ls`
- `tmux attach -t {sessions[0] if sessions else ''}`
- `watch -n 5 nvidia-smi`
- `tail -f {output_root.as_posix()}/*/*/seed_*/train.log`
"""
    (output_root / "run.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["origin", "omni"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.root) / "v2_formal"
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [task.task_id for task in filter_tasks(args.benchmark)]
    cuda_ids = [item.strip() for item in args.cuda.split(",") if item.strip()]
    sessions: list[str] = []
    for idx, shard in enumerate(shard_tasks(tasks, args.shards)):
        suffix = "" if args.shards == 1 else f"_s{idx + 1}"
        session = f"bh_v2f_{args.benchmark}{suffix}"
        script = output_root / f"run_{args.benchmark}{suffix}.sh"
        write_script(script, output_root, shard, cuda_ids[idx % len(cuda_ids)])
        sessions.append(session)
        print(f"tmux new -d -s {session} 'bash {script.as_posix()}'")
    write_run_md(output_root, args.benchmark, tasks, sessions)
    print(output_root)


if __name__ == "__main__":
    main()
