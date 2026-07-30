#!/usr/bin/env python
"""Validate scripted experts for every registered MetaWorld task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from metaworld_pi05_adapter import MetaworldPi05Adapter
from shared.experts.metaworld_expert import MetaworldScriptedExpert
from shared.task_configs import TASK_CONFIGS


def probe_task(task_key: str, episodes: int, base_seed: int) -> dict:
    """Run fixed-seed expert-only episodes for one task."""
    task = TASK_CONFIGS[task_key]
    env = MetaworldPi05Adapter(
        env_name=task["env_id"],
        seed=base_seed,
        camera_name=task["camera_name"],
        resolution=task["resolution"],
    )
    expert = MetaworldScriptedExpert(task["env_id"])
    results = []
    try:
        for episode in range(episodes):
            episode_seed = base_seed + episode
            env.reset(seed=episode_seed)
            expert.reset(env)
            success = False
            steps = task["max_timesteps"]
            for step in range(task["max_timesteps"]):
                _, reward, _, _ = env.step(expert.act(env))
                if reward >= 1.0:
                    success = True
                    steps = step + 1
                    break
            results.append(
                {"episode": episode, "seed": episode_seed, "success": success, "steps": steps}
            )
    finally:
        env.close()
    successes = sum(int(item["success"]) for item in results)
    return {
        "task_key": task_key,
        "env_id": task["env_id"],
        "successes": successes,
        "episodes": episodes,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42000)
    parser.add_argument("--minimum-successes", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0 <= args.minimum_successes <= args.episodes:
        parser.error("--minimum-successes must be between 0 and --episodes")

    os.environ.setdefault("MUJOCO_GL", "egl")
    tasks = [
        probe_task(task_key, args.episodes, args.base_seed + index * 100)
        for index, task_key in enumerate(TASK_CONFIGS)
    ]
    summary = {
        "schema_version": 1,
        "episodes_per_task": args.episodes,
        "minimum_successes": args.minimum_successes,
        "passed": all(item["successes"] >= args.minimum_successes for item in tasks),
        "tasks": tasks,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
