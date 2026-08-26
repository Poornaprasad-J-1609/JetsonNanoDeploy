import numpy as np
from pathlib import Path

from hardcoded_gait import HardcodedGaitPlayer


ROOT = Path(__file__).resolve().parents[1]


def config_path():
    return ROOT / "config" / "hardcoded_gait_sim_vx0p5.yaml"


def test_templates_are_finite_and_periodic():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    template = player.templates["forward"]
    assert np.allclose(template.sample(0.0), template.sample(1.0), atol=1.0e-6)
    assert np.all(np.isfinite(template.sample(0.37)))


def test_default_replay_is_reduced_from_simulation():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    samples = np.asarray([player.templates["forward"].sample(x / 500) for x in range(500)])
    replay = player.amplitude_scale * samples
    assert np.max(np.abs(replay)) < 0.15


def test_direction_change_blends_without_target_jump():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    for _ in range(200):
        previous = player.update(1.0, 0.02, np.zeros(12, dtype=np.float32))
    changed = player.update(-1.0, 0.02, previous)
    assert np.max(np.abs(changed - previous)) < 0.002


def test_target_step_guard_is_always_active():
    player = HardcodedGaitPlayer.from_yaml(config_path())
    player.direction_blend_seconds = 0.0
    previous = np.full(12, 1.0, dtype=np.float32)
    changed = player.update(1.0, 0.02, previous)
    assert np.max(np.abs(changed - previous)) <= player.max_target_step_rad + 1.0e-7
