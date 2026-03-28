#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(max(value, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, required=True)
    parser.add_argument("--tag-prefix", type=str, required=True)
    parser.add_argument("--total-shards", type=int, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=-1)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    args = parser.parse_args()

    if args.total_shards <= 0:
        raise ValueError("total_shards must be positive")
    if not args.gpus:
        raise ValueError("At least one GPU id is required")

    log_dir = ROOT / "results" / "rich_features" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    start_shard = max(args.start_shard, 0)
    end_shard = args.total_shards - 1 if args.end_shard < 0 else min(args.end_shard, args.total_shards - 1)
    pending = list(range(start_shard, end_shard + 1))
    active: dict[int, tuple[int, int, subprocess.Popen[str], float, Path]] = {}
    completed_durations: list[float] = []
    start_time = time.time()

    while pending or active:
        while pending and len(active) < len(args.gpus):
            shard_index = pending.pop(0)
            busy_gpus = {gpu_id for _shard_idx, gpu_id, _proc, _t0, _log in active.values()}
            gpu_id = next(gpu for gpu in args.gpus if gpu not in busy_gpus)
            tag = f"{args.tag_prefix}_s{shard_index:02d}"
            log_path = log_dir / f"{tag}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")
            cmd = [
                args.python_bin,
                str(ROOT / "scripts" / "extract_prompt_hidden_features.py"),
                "--checkpoint",
                args.checkpoint,
                "--manifest-path",
                args.manifest_path,
                "--prompt-len",
                str(args.prompt_len),
                "--decode-len",
                str(args.decode_len),
                "--num-shards",
                str(args.total_shards),
                "--shard-index",
                str(shard_index),
                "--num-sequences",
                "-1",
                "--tag",
                tag,
            ]
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            active[process.pid] = (shard_index, gpu_id, process, time.time(), log_path)
            print(f"[launch] shard={shard_index} gpu={gpu_id} tag={tag}", flush=True)

        time.sleep(5.0)
        finished_pids = []
        for pid, (shard_index, gpu_id, process, shard_start, log_path) in active.items():
            status = process.poll()
            if status is None:
                continue
            duration = time.time() - shard_start
            completed_durations.append(duration)
            finished_pids.append(pid)
            if status != 0:
                raise RuntimeError(f"Shard {shard_index} failed with exit code {status}. See {log_path}")
            mean_duration = sum(completed_durations) / max(len(completed_durations), 1)
            remaining = len(pending) + (len(active) - 1)
            eta_seconds = mean_duration * remaining / max(len(args.gpus), 1)
            print(
                f"[done] shard={shard_index} gpu={gpu_id} duration={format_seconds(duration)} "
                f"completed={len(completed_durations)}/{args.total_shards} eta={format_seconds(eta_seconds)}",
                flush=True,
            )
        for pid in finished_pids:
            active.pop(pid, None)

    print(
        f"[complete] tag_prefix={args.tag_prefix} total_elapsed={format_seconds(time.time() - start_time)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
