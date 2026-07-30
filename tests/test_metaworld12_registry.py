"""MetaWorld-12 task and scripted-expert registry tests."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from shared.experts.metaworld_expert import _POLICY_BY_ENV
from shared.task_configs import TASK_CONFIGS
from flowdagger_pi05.metaworld_pi05_adapter import MetaworldPi05Adapter


PAPER_TASKS = {
    "metaworld_assembly": "assembly-v3",
    "metaworld_bin_picking": "bin-picking-v3",
    "metaworld_box_close": "box-close-v3",
    "metaworld_coffee_pull": "coffee-pull-v3",
    "metaworld_dial_turn": "dial-turn-v3",
    "metaworld_door_lock": "door-lock-v3",
    "metaworld_hammer": "hammer-v3",
    "metaworld_hand_insert": "hand-insert-v3",
    "metaworld_lever_pull": "lever-pull-v3",
    "metaworld_pick_place": "pick-place-v3",
    "metaworld_soccer": "soccer-v3",
    "metaworld_stick_push": "stick-push-v3",
}


def test_paper_tasks_have_mt50_prompts_and_scripted_experts() -> None:
    assert set(TASK_CONFIGS) == set(PAPER_TASKS)
    for task_key, env_id in PAPER_TASKS.items():
        task = TASK_CONFIGS[task_key]
        assert task["env_id"] == env_id
        assert task["prompt"]
        assert task["openpi_config"] == "pi05_metaworld"
        assert task["video_camera_name"] == "corner"
        assert (task["video_width"], task["video_height"]) == (640, 480)
        assert env_id in _POLICY_BY_ENV


def test_registered_policy_classes_exist_in_installed_metaworld() -> None:
    from metaworld import ALL_V3_ENVIRONMENTS, policies

    for env_id, class_name in _POLICY_BY_ENV.items():
        assert env_id in ALL_V3_ENVIRONMENTS
        assert hasattr(policies, class_name)


def test_video_renderer_is_independent_from_policy_observation() -> None:
    env = MetaworldPi05Adapter(
        "assembly-v3",
        camera_name="corner3",
        resolution=64,
        video_camera_name="corner",
        video_width=320,
        video_height=240,
    )
    try:
        observation = env.reset(seed=42)
        video_frame = env.render_video()
    finally:
        env.close()

    assert observation["image"].shape == (64, 64, 3)
    assert video_frame.shape == (240, 320, 3)
