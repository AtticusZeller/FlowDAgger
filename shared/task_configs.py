"""Per-task configuration registry for the paper's MetaWorld-12 suite."""

import os

_COMMON = {
    "suite": "metaworld",
    "expert_class": "MetaworldScriptedExpert",
    "openpi_config": "pi05_metaworld",
    "hf_checkpoint": "mmurray-ms/pi05-metaworld",
    "openpi_checkpoint": os.environ.get("METAWORLD_CHECKPOINT", ""),
    "max_timesteps": 300,
    "resolution": 256,
    "camera_name": "corner3",
    "video_camera_name": "corner",
    "video_width": 640,
    "video_height": 480,
    "video_rotate_180": True,
    "expert_kwargs": {},
}


def _task(name: str, env_id: str, prompt: str) -> dict:
    """Build one task entry with the shared MT50 policy settings."""
    return {**_COMMON, "name": name, "env_id": env_id, "prompt": prompt}


# Prompts match mmurz/metaworld_mt50_v3/meta/tasks.jsonl exactly.
TASK_CONFIGS = {
    "metaworld_assembly": _task(
        "assembly", "assembly-v3", "Pick up a nut and place it onto a peg"
    ),
    "metaworld_bin_picking": _task(
        "bin_picking",
        "bin-picking-v3",
        "Grasp the puck from one bin and place it into another bin",
    ),
    "metaworld_box_close": _task(
        "box_close", "box-close-v3", "Grasp the cover and close the box with it"
    ),
    "metaworld_coffee_pull": _task(
        "coffee_pull", "coffee-pull-v3", "Pull a mug from a coffee machine"
    ),
    "metaworld_dial_turn": _task(
        "dial_turn", "dial-turn-v3", "Rotate a dial 180 degrees"
    ),
    "metaworld_door_lock": _task(
        "door_lock",
        "door-lock-v3",
        "Lock the door by rotating the lock clockwise",
    ),
    "metaworld_hammer": _task(
        "hammer", "hammer-v3", "Hammer a screw on the wall"
    ),
    "metaworld_hand_insert": _task(
        "hand_insert", "hand-insert-v3", "Insert the gripper into a hole"
    ),
    "metaworld_lever_pull": _task(
        "lever_pull", "lever-pull-v3", "Pull a lever down 90 degrees"
    ),
    "metaworld_pick_place": _task(
        "pick_place", "pick-place-v3", "Pick and place a puck to a goal"
    ),
    "metaworld_soccer": _task(
        "soccer", "soccer-v3", "Kick a soccer into the goal"
    ),
    "metaworld_stick_push": _task(
        "stick_push",
        "stick-push-v3",
        "Grasp a stick and push a box using the stick",
    ),
}


def get_task_config(task_id):
    """Return the config dict for task_id, or raise if it is not registered."""
    if task_id in TASK_CONFIGS:
        return TASK_CONFIGS[task_id]
    raise KeyError(
        f"Unknown task_id {task_id!r}. Available: {sorted(TASK_CONFIGS)}. "
        f"Add an entry to shared/task_configs.py::TASK_CONFIGS."
    )
