import numpy as np
from pathlib import Path
import pytest
import yaml

from hardcoded_gait import HardcodedGaitPlayer
from joint_mapping import POLICY_JOINT_ORDER
from motor_command_layer import MotorCommandLayer


ROOT = Path(__file__).resolve().parents[1]


def config_path():
    return ROOT / "config" / "hardcoded_gait_july13_minimal.yaml"


def test_templates_are_finite_and_periodic():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    template = player.templates["forward"]
    assert np.allclose(template.sample(0.0), template.sample(1.0), atol=1.0e-6)
    assert np.all(np.isfinite(template.sample(0.37)))


def test_default_replay_is_reduced_from_simulation():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    samples = np.asarray([player.templates["forward"].sample(x / 500) for x in range(500)])
    replay = player.amplitude_scale * samples
    assert np.max(np.abs(replay)) < 0.16


def test_minimal_replay_moves_every_leg_and_limits_hip_excursion():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    for direction in ("forward", "backward"):
        samples = np.asarray(
            [player.templates[direction].sample(x / 2000) for x in range(2000)]
        )
        replay = player.amplitude_scale * samples
        spans = np.ptp(replay, axis=0)
        assert np.all(spans[4:12] >= 0.03)
        assert np.max(spans[0:4]) <= 0.015
        assert np.max(spans[4:12]) <= 0.16


def test_minimal_profile_uses_only_the_smooth_fundamental():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    for template in player.templates.values():
        np.testing.assert_array_equal(template.harmonic_weights, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(template.coefficients[3:], np.zeros((4, 12)))


def test_direction_change_blends_without_target_jump():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    for _ in range(200):
        previous = player.update(1.0, 0.02, np.zeros(12, dtype=np.float32))
    changed = player.update(-1.0, 0.02, previous)
    assert np.max(np.abs(changed - previous)) <= player.max_target_step_rad


def test_direction_change_preserves_acceleration_bound():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    dt = 0.02
    for _ in range(200):
        player.update(1.0, dt, player.last_target)
    velocity_before = player.last_velocity.copy()
    player.update(-1.0, dt, player.last_target)
    acceleration = (player.last_velocity - velocity_before) / dt
    assert np.max(np.abs(acceleration)) <= player.max_target_acceleration_rad_s2 + 1e-5


def test_target_step_guard_is_always_active():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    player.direction_blend_seconds = 0.0
    previous = np.full(12, 1.0, dtype=np.float32)
    changed = player.update(1.0, 0.02, previous)
    assert np.max(np.abs(changed - previous)) <= player.max_target_step_rad + 1.0e-7


def test_velocity_and_acceleration_guards_are_always_active():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    previous_velocity = player.last_velocity.copy()
    for _ in range(300):
        player.update(1.0, 0.02, player.last_target)
        assert np.max(np.abs(player.last_velocity)) <= player.max_target_velocity_rad_s + 1e-6
        acceleration = (player.last_velocity - previous_velocity) / 0.02
        assert np.max(np.abs(acceleration)) <= player.max_target_acceleration_rad_s2 + 1e-5
        previous_velocity = player.last_velocity.copy()


def test_entry_blend_has_zero_endpoint_slope():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    start = np.full(12, 0.1, dtype=np.float32)
    first = player.update(1.0, 1e-4, start)
    assert np.max(np.abs(first - start)) < 1e-6


def test_all_templates_remain_inside_physical_joint_limits_at_full_scale():
    player = HardcodedGaitPlayer.from_yaml(
        config_path(), amplitude_scale=1.0, frequency_scale=1.0
    )
    limits = yaml.safe_load((ROOT / "config" / "joint_limits.yaml").read_text())[
        "joint_limits"
    ]
    samples = np.asarray(
        [
            player.templates[direction].sample(phase / 10000.0)
            for direction in ("forward", "backward")
            for phase in range(10000)
        ]
    )
    q_min = np.asarray([limits[name]["min"] for name in POLICY_JOINT_ORDER])
    q_max = np.asarray([limits[name]["max"] for name in POLICY_JOINT_ORDER])
    assert np.all(samples >= q_min - 1.0e-7)
    assert np.all(samples <= q_max + 1.0e-7)


@pytest.mark.parametrize("direction", [0.0, np.nan, np.inf, -np.inf])
def test_invalid_direction_is_rejected(direction):
    player = HardcodedGaitPlayer.from_yaml(config_path())
    with pytest.raises(ValueError, match="direction"):
        player.update(direction, 0.02)


@pytest.mark.parametrize(
    "current_target",
    [np.zeros(11), np.zeros((1, 12)), np.full(12, np.nan)],
)
def test_invalid_handoff_target_is_rejected(current_target):
    player = HardcodedGaitPlayer.from_yaml(config_path())
    with pytest.raises(ValueError, match="current_target"):
        player.update(1.0, 0.02, current_target)


def test_hermite_substeps_repack_valid_mit_commands():
    joint = "BL_hip_joint"
    layer = MotorCommandLayer(
        POLICY_JOINT_ORDER,
        {name: index + 1 for index, name in enumerate(POLICY_JOINT_ORDER)},
        active_joints=[joint],
        joint_can_bus={joint: "back"},
    )
    base = {
        "joint_name": joint,
        "motor_id": 1,
        "phase": "policy",
        "command_encoding": "official",
        "direction": -1.0,
        "offset": 0.0,
        "kp": 250.0,
        "kd": 4.0,
        "tau_ff": 0.0,
    }
    previous = [{**base, "q_des": 0.0, "joint_v_des": 0.0}]
    current = [{**base, "q_des": 0.1, "joint_v_des": 0.0}]
    samples = [
        layer.interpolate_mit_commands(previous, current, alpha, 0.02)[0]
        for alpha in (0.25, 0.5, 0.75, 1.0)
    ]
    assert [item["q_des"] for item in samples] == pytest.approx(
        [0.015625, 0.05, 0.084375, 0.1]
    )
    assert all(len(item["data"]) == 8 for item in samples)
    assert all(np.isfinite(item["joint_v_des"]) for item in samples)
