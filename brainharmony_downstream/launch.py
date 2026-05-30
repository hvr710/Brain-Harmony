from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

from .tasks import SKIPPED_TASKS, filter_tasks


REPO_LINUX = "/mnt/dataset3/nzh/fmri_FM/Brain-Harmony"


def next_version(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
            used.append(int(child.name[1:]))
    return root / f"v{(max(used) + 1) if used else 1}"


def write_script(path: Path, task_ids: list[str], modes: list[str], trials: int, cuda: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {REPO_LINUX}",
        "export PYTHONPATH=$PWD:${PYTHONPATH:-}",
        "PYTHON_BIN=/data/ningzh/envs/harmony/bin/python",
        "if [ -f /data/ningzh/envs/harmony/bin/activate ]; then source /data/ningzh/envs/harmony/bin/activate; fi",
        "if command -v conda >/dev/null 2>&1; then conda activate /data/ningzh/envs/harmony 2>/dev/null || true; fi",
        f"export CUDA_VISIBLE_DEVICES={cuda}",
        "echo START $(date)",
        "nvidia-smi || true",
    ]
    for task_id in task_ids:
        for mode in modes:
            out = f"{path.parent.as_posix()}/{task_id}/{mode}"
            log = f"{out}/train.log"
            lines.extend(
                [
                    f"mkdir -p {out}",
                    f"echo RUN task={task_id} mode={mode} $(date)",
                    (
                        "$PYTHON_BIN -m brainharmony_downstream.train "
                        f"--task {task_id} --mode {mode} --trials {trials if mode != 'ft' else max(1, min(trials, 15))} "
                        f"--output_dir {out} --feature_cache_dir {path.parent.as_posix()}/feature_cache "
                        f"2>&1 | tee {log}"
                    ),
                ]
            )
    lines.extend(["echo END $(date)", "exec bash"])
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def write_run_md(
    path: Path,
    benchmark: str,
    task_ids: list[str],
    modes: list[str],
    trials: int,
    note: str,
    sessions: list[str],
) -> None:
    skipped = "\n".join(f"- `{k}`: {v}" for k, v in SKIPPED_TASKS.items())
    text = f"""# {path.name} BrainHarmonix-F {benchmark}

- Created: {datetime.now().isoformat(timespec="seconds")}
- Model: BrainHarmonix-F, checkpoint `checkpoints/harmonix-f/model.pth`
- Benchmark: `{benchmark}`
- Tasks: {", ".join(task_ids)}
- Modes: {", ".join(modes)}
- Optuna trials: {trials} for lp/mlp, up to 15 for ft
- Note: {note or "initial run"}

## Skipped / gated tasks
{skipped}

## Monitor
- `tmux ls`
- Sessions: {", ".join(f"`{session}`" for session in sessions)}
- `tmux attach -t {sessions[0] if sessions else f"bh_{path.name}_{benchmark}"}`
- `watch -n 5 nvidia-smi`
- `tail -f {path.as_posix()}/*/*/train.log`

## Difference From Previous Version
- First generated version for this output folder unless noted above.
"""
    (path / "run.md").write_text(text, encoding="utf-8")


def start_tmux(session: str, script: Path) -> None:
    cmd = ["tmux", "new", "-d", "-s", session, f"bash {script.as_posix()}"]
    subprocess.run(cmd, check=True)


def shard_tasks(task_ids: list[str], shards: int) -> list[list[str]]:
    shards = max(1, shards)
    result = [[] for _ in range(shards)]
    for idx, task_id in enumerate(task_ids):
        result[idx % shards].append(task_id)
    return [items for items in result if items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["origin", "omni"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--modes", default="lp,mlp,ft")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--note", default="")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--tasks", default="")
    args = parser.parse_args()

    root = Path(args.root)
    version = next_version(root)
    version.mkdir(parents=True, exist_ok=False)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    requested = [item.strip() for item in args.tasks.split(",") if item.strip()] or None
    task_ids = [task.task_id for task in filter_tasks(args.benchmark, requested)]
    cuda_ids = [item.strip() for item in args.cuda.split(",") if item.strip()]
    scripts: list[tuple[str, Path]] = []
    task_shards = shard_tasks(task_ids, args.shards)
    for idx, shard in enumerate(task_shards):
        suffix = "" if len(task_shards) == 1 else f"_s{idx + 1}"
        script = version / f"run_{args.benchmark}{suffix}.sh"
        cuda = cuda_ids[idx % len(cuda_ids)] if cuda_ids else "0"
        write_script(script, shard, modes, args.trials, cuda)
        scripts.append((f"bh_{version.name}_{args.benchmark}{suffix}", script))
    write_run_md(
        version,
        args.benchmark,
        task_ids,
        modes,
        args.trials,
        args.note,
        [session for session, _ in scripts],
    )
    print(version)
    for session, script in scripts:
        print(f"tmux new -d -s {session} 'bash {script.as_posix()}'")
    if args.start:
        for session, script in scripts:
            start_tmux(session, script)


if __name__ == "__main__":
    main()
