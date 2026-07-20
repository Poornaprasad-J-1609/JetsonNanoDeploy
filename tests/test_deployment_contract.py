import argparse
import contextlib
import io
import py_compile
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from can_topology import (
    add_can_topology_args,
    backend_for_port,
    ports_for_active_joints,
    resolve_joint_can_bus,
    resolve_port_by_bus,
    validate_unique_motor_ids_per_physical_bus,
)
from imu_interface import ImuReading, imu_reading_quality
from joystick_interface import CommandSource
from main_controller import (
    action_equivalent_for_q_target,
    validate_required_policy_imu,
    smoothstep,
)
from motor_command_layer import (
    MotorCommandLayer,
    decode_mit_feedback_frame,
    mit_can_id,
    pack_mit_command,
    signed_offset_to_uint,
)
from policy_runner import (
    EXPECTED_POLICY_JOINT_ORDER,
    PolicyRunner,
)


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_all_yaml_files_parse():
    for path in sorted((ROOT / "config").glob("*.yaml")):
        assert load_yaml(path) is not None, path


def test_python_files_compile():
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            py_compile.compile(str(path), doraise=True)


def test_one_can_socketcan_defaults_and_unique_ids():
    parser = argparse.ArgumentParser()
    add_can_topology_args(parser)
    args = parser.parse_args([])
    assert args.can_count == 1
    assert args.port == "slcan0"
    assert args.can_backend == "socketcan"

    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_can_bus = resolve_joint_can_bus(runner.policy_order, args.can_count)
    assert set(joint_can_bus.values()) == {"can0"}

    port_by_bus = resolve_port_by_bus(args)
    active_port_by_bus = ports_for_active_joints(
        port_by_bus,
        joint_can_bus,
        runner.policy_order,
    )
    assert active_port_by_bus == {"can0": "slcan0"}
    assert len({int(motor_ids[name]) for name in runner.policy_order}) == 12
    validate_unique_motor_ids_per_physical_bus(
        motor_ids=motor_ids,
        joint_can_bus=joint_can_bus,
        active_joints=runner.policy_order,
        port_by_bus=active_port_by_bus,
    )
    assert backend_for_port("slcan0", "auto") == "socketcan"
    assert backend_for_port("/dev/ttyUSB0", "auto") == "serial-at"


def test_policy_contract_observation_and_action():
    runner = PolicyRunner()
    assert runner.policy_order == EXPECTED_POLICY_JOINT_ORDER
    assert runner.observation_dim == 48
    assert runner.action_dim == 12
    assert runner.action_scale == pytest.approx(0.25)

    previous = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    obs = runner.build_observation(
        base_ang_vel_b=np.array([0.1, -0.2, 0.3], dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        command=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        q_current=runner.q_default.copy(),
        qd_current=np.zeros(12, dtype=np.float32),
        previous_action=previous,
    )
    assert obs.shape == (48,)
    np.testing.assert_array_equal(obs[0:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(obs[36:48], previous)

    action = runner.infer_action(obs)
    assert action.shape == (12,)
    assert np.all(np.isfinite(action))
    with pytest.raises(ValueError):
        runner.infer_action(np.zeros(45, dtype=np.float32))


def test_live_imu_populates_policy_slots_without_base_velocity():
    runner = PolicyRunner()
    gyro = np.array([0.11, -0.22, 0.33], dtype=np.float32)
    gravity = np.array([0.05, -0.04, -0.998], dtype=np.float32)
    previous = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    obs = runner.build_observation(
        base_ang_vel_b=gyro,
        projected_gravity_b=gravity,
        command=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        q_current=runner.q_default.copy(),
        qd_current=np.zeros(12, dtype=np.float32),
        previous_action=previous,
    )
    np.testing.assert_array_equal(obs[0:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(obs[3:6], gyro)
    np.testing.assert_allclose(obs[6:9], gravity)
    np.testing.assert_allclose(obs[36:48], previous)


def test_imu_reading_quality_accepts_valid_and_rejects_bad_vectors():
    valid = ImuReading(
        base_ang_vel_b=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        timestamp=time.monotonic(),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rpy_abs_deg=np.zeros(3, dtype=np.float32),
        det_r=1.0,
        cross_err=0.0,
    )
    ok, reason = imu_reading_quality(valid)
    assert ok, reason

    invalid_quat = ImuReading(
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        timestamp=time.monotonic(),
        quaternion_wxyz=np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    ok, reason = imu_reading_quality(invalid_quat)
    assert not ok
    assert "quaternion" in reason


def test_required_stale_imu_blocks_active_motion_guard():
    class RequiredImu:
        required = True
        stale_timeout = 0.01
        source_name = "xsens"

        def read(self):
            return None

    class Estimator:
        imu_sensor = RequiredImu()
        last_imu_reading = None

        def imu_required(self):
            return True

        def refresh_imu(self):
            return False

        def imu_status(self):
            return "xsens:missing"

    reason = validate_required_policy_imu(Estimator())
    assert reason is not None
    assert "required IMU" in reason


def test_exact_policy_entry_blend_preserves_raw_action_after_entry():
    runner = PolicyRunner()
    raw_action = np.linspace(-0.75, 0.75, 12, dtype=np.float32)
    q_actor = runner.action_to_q_target(raw_action)
    q_entry = runner.q_stand + np.linspace(0.0, 0.11, 12, dtype=np.float32)

    alpha = smoothstep(0.5)
    q_blend = ((1.0 - alpha) * q_entry + alpha * q_actor).astype(np.float32)
    sent_during_entry = action_equivalent_for_q_target(runner, q_blend)
    assert not np.allclose(sent_during_entry, raw_action)

    sent_after_entry = action_equivalent_for_q_target(runner, q_actor)
    np.testing.assert_allclose(sent_after_entry, raw_action, atol=1e-6)


def test_command_source_identity_and_keyboard_grace_scope():
    fixed = CommandSource(source="fixed", vx=0.1, vy=0.0, yaw=0.0)
    assert fixed.source_name == "fixed"
    fixed.close()

    with contextlib.redirect_stdout(io.StringIO()):
        keyboard = CommandSource(
            source="keyboard",
            max_vx=1.8,
            max_vy=0.8,
            max_yaw=0.8,
            speed_scale_initial=0.1,
            speed_scale_min=0.05,
            speed_scale_max=0.2,
            speed_scale_step=0.01,
            keyboard_command_timeout=0.1,
        )
    assert keyboard.source_name == "keyboard"
    now = time.monotonic()
    keyboard.impl.movement_key_deadlines["w"] = now + 0.5
    keyboard.impl._update_command_from_active_keys(now)
    np.testing.assert_allclose(keyboard.read(), [0.18, 0.0, 0.0], atol=1e-6)
    keyboard.close()


def test_keyboard_repeat_release_and_emergency_behavior():
    with contextlib.redirect_stdout(io.StringIO()):
        source = CommandSource(
            source="keyboard",
            max_vx=1.8,
            max_vy=0.8,
            max_yaw=0.8,
            speed_scale_initial=0.1,
            speed_scale_min=0.05,
            speed_scale_max=0.2,
            speed_scale_step=0.01,
            keyboard_command_timeout=0.1,
        )
    keyboard = source.impl
    keyboard.key_queue.extend([" ", " ", " "])
    assert keyboard.get_mode_request() == "stand"
    assert keyboard.get_mode_request() is None

    now = time.monotonic()
    keyboard.movement_key_deadlines["w"] = now - 0.01
    keyboard._update_command_from_active_keys(now)
    np.testing.assert_array_equal(source.read(), np.zeros(3, dtype=np.float32))

    keyboard.key_queue.append("x")
    assert "emergency stop" in source.get_emergency_stop_request()
    source.close()


def test_main_controller_safe_defaults_are_pinned():
    source = (ROOT / "src" / "main_controller.py").read_text(encoding="utf-8")
    assert "--auto-push-log" in source
    assert "default=False" in source[source.index("--auto-push-log"):source.index("--log-git-remote")]
    assert "--exact-policy-after-entry" in source
    assert "default=True" in source[source.index("--exact-policy-after-entry"):source.index("--fake-start")]
    assert "if not bool(exact_policy_after_entry):" in source


def test_policy_and_pose_pd_torque_limits_are_separate():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    configured_pose_limits = layer.pose_pd_torque_limits()
    configured_policy_limit = layer.policy_pd_torque_limit_for_joint(runner.policy_order[0])

    layer.set_policy_pd_torque_limit(21.0)
    assert layer.policy_pd_torque_limit_for_joint(runner.policy_order[0]) == pytest.approx(21.0)
    assert layer.pose_pd_torque_limits() == configured_pose_limits

    layer.set_pose_pd_torque_limit(12.0)
    pose_limits = layer.pose_pd_torque_limits()
    assert pose_limits["startup"] == pytest.approx(12.0)
    assert pose_limits["hold"] == pytest.approx(12.0)
    assert configured_policy_limit != pytest.approx(12.0)
    assert layer.policy_pd_torque_limit_for_joint(runner.policy_order[0]) == pytest.approx(21.0)


def test_four_bar_transmission_is_inactive():
    cfg = load_yaml(ROOT / "config" / "four_bar_transmission.yaml")
    assert cfg["four_bar_transmission"]["enabled"] is False


def test_extended_can_id_and_mit_pack_decode_quantization():
    proto = load_yaml(ROOT / "config" / "mit_motor_control.yaml")["mit_protocol"]
    can_id = mit_can_id(0x7F, proto, tau_ff=0.25)
    assert (can_id >> 24) & 0x1F == int(proto["comm_type_mit_control"])
    assert can_id & 0xFF == 0x7F
    assert can_id <= 0x1FFFFFFF

    payload = pack_mit_command(
        p_des=0.25,
        v_des=-0.1,
        kp=12.0,
        kd=0.3,
        proto=proto,
    )
    feedback_id = (
        int(proto["comm_type_feedback"]) << 24
        | ((0x00 << 8) | 0x7F) << 8
        | 0x00
    )
    feedback_payload = (
        signed_offset_to_uint(0.25, proto["p_min"], proto["p_max"]).to_bytes(2, "big")
        + signed_offset_to_uint(-0.1, proto["v_min"], proto["v_max"]).to_bytes(2, "big")
        + signed_offset_to_uint(0.0, proto["tau_min"], proto["tau_max"]).to_bytes(2, "big")
        + int(350).to_bytes(2, "big")
    )
    decoded = decode_mit_feedback_frame(feedback_id, feedback_payload, proto)
    assert len(payload) == 8
    assert decoded["motor_id"] == 0x7F
    assert decoded["position"] == pytest.approx(0.25, abs=8e-4)
    assert decoded["velocity"] == pytest.approx(-0.1, abs=8e-4)
