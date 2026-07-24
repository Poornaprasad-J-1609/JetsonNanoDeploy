import argparse
import contextlib
import io
import queue
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
from imu_interface import ImuReading, XsensBackgroundReader, imu_reading_quality
from joystick_interface import CommandSource
from main_controller import (
    CAN_FEEDBACK_RECEIVE_EVERY_N_CYCLES,
    CsvRunLogger,
    MeasuredTorqueSupervisor,
    PolicyTorqueRamp,
    action_equivalent_for_q_target,
    clip_policy_hip_actions,
    compact_telemetry_record,
    constant_joint_map,
    requires_calf_endpoint_gate,
    runtime_stand_command_phase,
    policy_entry_gain_blend_scale,
    shifted_safety_filter_with_diagnostics,
    stand_recovery_gain_blend_scale,
    stand_ready_for_walking,
    stand_state_ready_for_policy_entry,
    should_validate_stand_state_for_policy_entry,
    torque_ramp_supervision_due,
    validate_required_policy_imu,
    validate_torque_profile,
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
from safety_monitor import SafetyMonitor


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
    assert runner.policy_order == [
        "BL_hip_joint",
        "BR_hip_joint",
        "FL_hip_joint",
        "FR_hip_joint",
        "BL_thigh_joint",
        "BR_thigh_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "BL_calf_joint",
        "BR_calf_joint",
        "FL_calf_joint",
        "FR_calf_joint",
    ]
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


def test_hip_action_clip_preserves_thigh_and_calf_outputs():
    policy_order = list(EXPECTED_POLICY_JOINT_ORDER)
    raw = np.linspace(-6.0, 6.0, len(policy_order), dtype=np.float32)
    clipped = clip_policy_hip_actions(raw, policy_order, hip_clip_abs=1.6)

    for index, joint_name in enumerate(policy_order):
        if "_hip_joint" in joint_name:
            assert abs(float(clipped[index])) <= 1.6 + 1.0e-6
        else:
            assert clipped[index] == pytest.approx(raw[index])
    np.testing.assert_array_equal(
        clip_policy_hip_actions(raw, policy_order, hip_clip_abs=0.0),
        raw,
    )


def test_hip_action_scale_applies_after_clip():
    policy_order = list(EXPECTED_POLICY_JOINT_ORDER)
    raw = np.full(len(policy_order), 2.0, dtype=np.float32)
    conditioned = clip_policy_hip_actions(
        raw,
        policy_order,
        hip_clip_abs=1.6,
        hip_scale=0.65,
    )

    for index, joint_name in enumerate(policy_order):
        expected = 1.04 if "_hip_joint" in joint_name else 2.0
        assert conditioned[index] == pytest.approx(expected)


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


def test_async_xsens_cache_returns_immediately_and_reuses_latest_sample():
    reader = XsensBackgroundReader(
        {"port": "/dev/null", "baud": 115200, "require_live": True},
        stale_timeout=0.20,
    )
    sample = ImuReading(
        base_ang_vel_b=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        timestamp=time.monotonic(),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    with reader._lock:
        reader._latest = sample
        reader._latest_host_time = time.monotonic()
        reader._latest_sensor_timestamp = sample.timestamp
    started = time.monotonic()
    cached = reader.latest_sample()
    elapsed = time.monotonic() - started
    assert elapsed < 0.002
    assert cached is not sample
    np.testing.assert_allclose(cached.base_ang_vel_b, sample.base_ang_vel_b)

    with reader._lock:
        reader._latest_host_time = time.monotonic() - 1.0
    assert reader.latest_sample() is None
    reader.stop()


def test_target_stage_logging_keeps_actor_entry_limits_and_transmitted_separate():
    runner = PolicyRunner()

    class Estimator:
        projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        base_lin_vel_b = np.zeros(3, dtype=np.float32)
        base_ang_vel_b = np.zeros(3, dtype=np.float32)
        last_feedback_by_joint = {}
        imu_sensor = None

    q = np.zeros(12, dtype=np.float32)
    actor = np.linspace(-0.3, 0.3, 12, dtype=np.float32)
    entry = actor * 0.5
    joint_limit = np.clip(entry, -0.1, 0.1)
    rate_limit = joint_limit * 0.8
    transmitted = rate_limit * 0.6
    command = [{"joint_name": runner.policy_order[0], "q_des": float(transmitted[0])}]
    source = CommandSource(source="fixed", vx=0.0, vy=0.0, yaw=0.0)
    record = compact_telemetry_record(
        step=1,
        mode="policy",
        command=np.zeros(3, dtype=np.float32),
        command_source=source,
        commands=command,
        estimator=Estimator(),
        q_current=q,
        qd_current=q,
        q_actor_target=actor,
        q_entry_blended_target=entry,
        q_joint_limit_filtered_target=joint_limit,
        q_rate_limited_target=rate_limit,
        q_safety_target=rate_limit,
        q_target=rate_limit,
        entry_blend_active=True,
        target_joint_limited=np.ones(12, dtype=bool),
        target_rate_limited=np.zeros(12, dtype=bool),
        policy_order=runner.policy_order,
    )
    source.close()
    joint = runner.policy_order[0]
    assert record[f"{joint}_actor_q_target"] == pytest.approx(actor[0])
    assert record[f"{joint}_entry_blended_q_target"] == pytest.approx(entry[0])
    assert record[f"{joint}_joint_limit_filtered_q_target"] == pytest.approx(joint_limit[0])
    assert record[f"{joint}_rate_limited_q_target"] == pytest.approx(rate_limit[0])
    assert record[f"{joint}_q_des_transmitted"] == pytest.approx(transmitted[0])
    assert record[f"{joint}_joint_limit_active"] == 1
    assert record[f"{joint}_entry_blend_active"] == 1


def test_safety_filter_reports_intermediate_limit_and_rate_targets():
    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order)
    q_previous = np.zeros(12, dtype=np.float32)
    q_shift = np.zeros(12, dtype=np.float32)
    q_target = np.full(12, 99.0, dtype=np.float32)
    filtered, diag = shifted_safety_filter_with_diagnostics(
        safety,
        q_target,
        q_previous,
        q_shift,
        apply_rate_limit=True,
        use_policy_limits=True,
    )
    assert filtered.shape == (12,)
    assert diag["joint_limit_filtered_q_target"].shape == (12,)
    assert diag["rate_limited_q_target"].shape == (12,)
    assert np.any(diag["target_joint_limited"])
    assert np.any(diag["target_rate_limited"])


def test_all_hip_hard_and_policy_limits_are_half_radian():
    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order, control_dt=runner.control_dt)

    for joint_name in (
        "BL_hip_joint",
        "BR_hip_joint",
        "FL_hip_joint",
        "FR_hip_joint",
    ):
        index = runner.policy_order.index(joint_name)
        assert safety.q_min[index] == pytest.approx(-0.5)
        assert safety.q_max[index] == pytest.approx(0.5)
        assert safety.policy_q_min[index] == pytest.approx(-0.5)
        assert safety.policy_q_max[index] == pytest.approx(0.5)
        assert safety.dq_max[index] == pytest.approx(0.04)


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


def test_medium_walk_launcher_latches_terminal_movement_commands():
    launcher = (ROOT / "scripts" / "run_medium_walk.sh").read_text(
        encoding="utf-8"
    )
    assert "--keyboard-control-mode latched" in launcher


def test_main_controller_safe_defaults_are_pinned():
    source = (ROOT / "src" / "main_controller.py").read_text(encoding="utf-8")
    assert "--auto-push-log" in source
    assert "default=False" in source[source.index("--auto-push-log"):source.index("--log-git-remote")]
    assert "--exact-policy-after-entry" in source
    assert "default=True" in source[source.index("--exact-policy-after-entry"):source.index("--fake-start")]
    assert "if not bool(exact_policy_after_entry):" in source


def test_loaded_stand_readiness_is_separate_from_zero_calibration():
    # Values measured in the 2026-07-22 hardware log: the loaded back calves
    # settled near 0.20 rad while the complete synchronized stand target was 0.
    assert not stand_ready_for_walking(0.0, 0.1983, 3.0, 3.64, 0.25)
    assert stand_ready_for_walking(0.0, 0.1983, 3.64, 3.64, 0.25)
    assert not stand_ready_for_walking(0.0, 0.251, 3.64, 3.64, 0.25)
    assert not stand_ready_for_walking(0.0, 0.1983, 3.64, 3.64, 0.08)


def test_policy_entry_revalidates_current_stand_position_and_velocity():
    q_stand = np.zeros(12, dtype=np.float32)
    active_indices = list(range(12))

    ready, error, velocity = stand_state_ready_for_policy_entry(
        q_current=np.full(12, 0.047, dtype=np.float32),
        qd_current=np.full(12, 0.006, dtype=np.float32),
        q_stand_target=q_stand,
        active_indices=active_indices,
        error_tolerance_rad=0.25,
        velocity_tolerance_rad_s=0.15,
    )
    assert ready
    assert error == pytest.approx(0.047)
    assert velocity == pytest.approx(0.006)

    ready, error, _ = stand_state_ready_for_policy_entry(
        q_current=np.full(12, 0.436, dtype=np.float32),
        qd_current=np.full(12, 0.006, dtype=np.float32),
        q_stand_target=q_stand,
        active_indices=active_indices,
        error_tolerance_rad=0.25,
        velocity_tolerance_rad_s=0.15,
    )
    assert not ready
    assert error == pytest.approx(0.436)

    ready, _, velocity = stand_state_ready_for_policy_entry(
        q_current=np.full(12, 0.047, dtype=np.float32),
        qd_current=np.full(12, 0.610, dtype=np.float32),
        q_stand_target=q_stand,
        active_indices=active_indices,
        error_tolerance_rad=0.25,
        velocity_tolerance_rad_s=0.15,
    )
    assert not ready
    assert velocity == pytest.approx(0.610)


def test_stand_state_gate_runs_only_before_policy_takeover():
    assert should_validate_stand_state_for_policy_entry(
        walking_armed=True,
        walk_requested=True,
        control_mode="stand",
        previous_walk_requested=False,
    )
    assert not should_validate_stand_state_for_policy_entry(
        walking_armed=True,
        walk_requested=True,
        control_mode="stand",
        previous_walk_requested=True,
    )
    assert not should_validate_stand_state_for_policy_entry(
        walking_armed=False,
        walk_requested=True,
        control_mode="stand",
        previous_walk_requested=False,
    )


def test_torque_ramp_supervision_stays_off_entry_critical_path():
    assert not torque_ramp_supervision_due(0.0, 0)
    assert not torque_ramp_supervision_due(0.998, 100)
    assert torque_ramp_supervision_due(1.0, 100)
    assert not torque_ramp_supervision_due(1.0, 101)
    assert torque_ramp_supervision_due(1.0, 105)


def test_return_from_policy_to_stand_uses_loaded_stand_path():
    assert runtime_stand_command_phase(False, False) == "stand"
    assert runtime_stand_command_phase(False, True) == "stand"
    assert runtime_stand_command_phase(True, False) == "stand"
    assert runtime_stand_command_phase(True, True) == "stand"


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


def test_periodic_commands_update_independent_can_adapters_in_parallel():
    class SlowPeriodicBus:
        def __init__(self):
            self.calls = []

        def update_periodic_sequence(self, frames, period_s):
            time.sleep(0.030)
            self.calls.append((list(frames), float(period_s)))
            return len(frames)

    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    front_joint = "FR_hip_joint"
    back_joint = "BR_hip_joint"
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[front_joint, back_joint],
        joint_can_bus={front_joint: "front", back_joint: "back"},
    )
    front_bus = SlowPeriodicBus()
    back_bus = SlowPeriodicBus()
    commands = [
        {
            "joint_name": front_joint,
            "bus_name": "front",
            "can_id": 1,
            "data": bytes(8),
        },
        {
            "joint_name": back_joint,
            "bus_name": "back",
            "can_id": 7,
            "data": bytes(8),
        },
    ]

    try:
        started = time.monotonic()
        count = layer.update_periodic_commands(
            {"front": front_bus, "back": back_bus},
            commands,
            period_s=0.005,
        )
        elapsed = time.monotonic() - started
    finally:
        layer.close()

    assert count == 2
    assert elapsed < 0.050
    assert front_bus.calls == [([(1, bytes(8))], 0.005)]
    assert back_bus.calls == [([(7, bytes(8))], 0.005)]


def test_pose_torque_limit_preserves_synchronized_target_and_scales_impedance():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FL_thigh_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_pose_pd_torque_limit(12.0)

    targets = (0.30, 0.40, 0.50, 0.60)
    feedback_positions = (0.00, 0.18, 0.12, 0.31)
    sent_targets = []
    for target, feedback_position in zip(targets, feedback_positions):
        q_target = np.zeros(12, dtype=np.float32)
        q_target[joint_index] = target
        command = layer.build_mit_commands(
            q_target,
            phase="sit",
            feedback_by_joint={
                joint_name: {
                    "position_raw": feedback_position,
                    "joint_position": feedback_position,
                    "joint_velocity": 0.0,
                }
            },
        )[0]
        sent_targets.append(command["q_des"])
        assert command["q_des"] == pytest.approx(target, abs=1.0e-6)
        assert command["q_before_torque_limit"] == pytest.approx(target, abs=1.0e-6)
        assert 0.0 < command["impedance_scale"] <= 1.0
        assert abs(command["tau_pd_est"]) <= 12.05

    assert np.all(np.diff(sent_targets) > 0.0)


def test_configured_pose_path_matches_proven_e9a4a13_packet_behavior():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FL_thigh_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    q_target = np.zeros(12, dtype=np.float32)
    q_target[joint_index] = 0.40
    command = layer.build_mit_commands(
        q_target,
        phase="stand",
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]

    assert layer.pose_pd_torque_limits()["stand"] == pytest.approx(0.0)
    assert command["command_encoding"] == "legacy_9b03a77"
    assert command["q_des"] == pytest.approx(0.40)
    assert not command["torque_limited"]
    assert command["kp"] == pytest.approx(50.0)
    assert command["kd"] == pytest.approx(1.8)
    assert command["kp_effective"] == pytest.approx(500.0, abs=0.1)
    assert command["kd_effective"] == pytest.approx(36.0, abs=0.1)


def test_policy_restores_0c17450_gains_without_changing_pose_gains():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FR_calf_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    q_target = np.zeros(12, dtype=np.float32)
    feedback = {
        joint_name: {
            "position_raw": 0.0,
            "joint_position": 0.0,
            "joint_velocity": 0.0,
        }
    }

    policy_command = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint=feedback,
    )[0]
    pose_command = layer.build_mit_commands(
        q_target,
        phase="stand",
        feedback_by_joint=feedback,
    )[0]

    assert policy_command["command_encoding"] == "official"
    assert policy_command["kp_effective"] == pytest.approx(110.0, abs=0.1)
    assert policy_command["kd_effective"] == pytest.approx(6.5, abs=0.01)
    assert pose_command["command_encoding"] == "legacy_9b03a77"
    assert pose_command["kp_effective"] == pytest.approx(750.0, abs=0.1)
    assert pose_command["kd_effective"] == pytest.approx(36.0, abs=0.1)

    back_joint = "BL_calf_joint"
    back_command = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[back_joint],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    ).build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint={
            back_joint: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]
    assert back_command["kp_effective"] == pytest.approx(130.0, abs=0.1)
    assert back_command["kd_effective"] == pytest.approx(8.0, abs=0.01)


def test_policy_entry_blends_effective_pose_gains_without_a_gain_step():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FR_calf_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    q_target = np.zeros(12, dtype=np.float32)
    feedback = {
        joint_name: {
            "position_raw": 0.0,
            "joint_position": 0.0,
            "joint_velocity": 0.0,
        }
    }

    entry_start = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint=feedback,
        gain_blend_from_phase="stand",
        gain_blend_alpha=0.0,
    )[0]
    entry_middle = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint=feedback,
        gain_blend_from_phase="stand",
        gain_blend_alpha=0.5,
    )[0]
    entry_end = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint=feedback,
        gain_blend_from_phase="stand",
        gain_blend_alpha=1.0,
    )[0]

    assert entry_start["command_encoding"] == "official"
    assert entry_start["gain_blend_from_phase"] == "startup"
    assert entry_start["gain_blend_alpha"] == pytest.approx(0.0)
    assert entry_start["kp_effective"] == pytest.approx(750.0, abs=0.2)
    assert entry_start["kd_effective"] == pytest.approx(36.0, abs=0.02)
    assert entry_middle["kp_effective"] == pytest.approx(430.0, abs=0.2)
    assert entry_middle["kd_effective"] == pytest.approx(21.25, abs=0.02)
    assert entry_end["gain_blend_from_phase"] is None
    assert entry_end["kp_effective"] == pytest.approx(110.0, abs=0.2)
    assert entry_end["kd_effective"] == pytest.approx(6.5, abs=0.02)


def test_stand_recovery_blends_effective_policy_gains_without_a_gain_step():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FR_calf_joint"
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    q_target = np.zeros(12, dtype=np.float32)
    feedback = {
        joint_name: {
            "position_raw": 0.0,
            "joint_position": 0.0,
            "joint_velocity": 0.0,
        }
    }

    recovery_start = layer.build_mit_commands(
        q_target,
        phase="stand",
        feedback_by_joint=feedback,
        gain_blend_from_phase="policy",
        gain_blend_alpha=0.0,
    )[0]
    recovery_middle = layer.build_mit_commands(
        q_target,
        phase="stand",
        feedback_by_joint=feedback,
        gain_blend_from_phase="policy",
        gain_blend_alpha=0.5,
    )[0]
    recovery_end = layer.build_mit_commands(
        q_target,
        phase="stand",
        feedback_by_joint=feedback,
        gain_blend_from_phase="policy",
        gain_blend_alpha=1.0,
    )[0]

    assert recovery_start["command_encoding"] == "legacy_9b03a77"
    assert recovery_start["gain_blend_from_phase"] == "policy"
    assert recovery_start["kp_effective"] == pytest.approx(110.0, abs=0.2)
    assert recovery_start["kd_effective"] == pytest.approx(6.5, abs=0.02)
    assert recovery_middle["kp_effective"] == pytest.approx(430.0, abs=0.2)
    assert recovery_middle["kd_effective"] == pytest.approx(21.25, abs=0.02)
    assert recovery_end["gain_blend_from_phase"] is None
    assert recovery_end["kp_effective"] == pytest.approx(750.0, abs=0.2)
    assert recovery_end["kd_effective"] == pytest.approx(36.0, abs=0.02)


def test_policy_entry_gain_blend_remains_inside_policy_torque_limit():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "BL_thigh_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_policy_pd_torque_limit(30.0)
    q_target = np.zeros(12, dtype=np.float32)
    q_target[joint_index] = 0.40
    feedback = {
        joint_name: {
            "position_raw": 0.0,
            "joint_position": 0.0,
            "joint_velocity": 0.0,
        }
    }

    for alpha in np.linspace(0.0, 1.0, 101):
        command = layer.build_mit_commands(
            q_target,
            phase="policy",
            feedback_by_joint=feedback,
            gain_blend_from_phase="stand",
            gain_blend_alpha=float(alpha),
        )[0]
        assert abs(command["tau_pd_est"]) <= 30.05


def test_policy_entry_gain_handoff_spans_complete_target_ramp():
    assert policy_entry_gain_blend_scale(0.0, 2.0) == pytest.approx(0.0)
    assert policy_entry_gain_blend_scale(0.5, 2.0) == pytest.approx(0.15625)
    assert policy_entry_gain_blend_scale(1.0, 2.0) == pytest.approx(0.5)
    assert policy_entry_gain_blend_scale(2.0, 2.0) == pytest.approx(1.0)
    assert policy_entry_gain_blend_scale(0.0, 0.0) == pytest.approx(1.0)


def test_stand_recovery_gain_handoff_is_continuous_from_policy_state():
    assert stand_recovery_gain_blend_scale(1.0, 0.0, 2.0) == pytest.approx(0.0)
    assert stand_recovery_gain_blend_scale(1.0, 1.0, 2.0) == pytest.approx(0.5)
    assert stand_recovery_gain_blend_scale(1.0, 2.0, 2.0) == pytest.approx(1.0)
    assert stand_recovery_gain_blend_scale(0.25, 0.0, 2.0) == pytest.approx(0.75)
    assert stand_recovery_gain_blend_scale(0.25, 1.0, 2.0) == pytest.approx(0.875)
    assert stand_recovery_gain_blend_scale(0.25, 2.0, 2.0) == pytest.approx(1.0)
    assert stand_recovery_gain_blend_scale(1.0, 0.0, 0.0) == pytest.approx(1.0)


def test_final_policy_packet_respects_position_rate_and_torque_limits():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "BL_thigh_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_policy_pd_torque_limit(30.0)
    q_target = np.zeros(12, dtype=np.float32)
    q_target[joint_index] = 0.60
    q_previous = np.zeros(12, dtype=np.float32)
    max_delta = np.full(12, 0.08, dtype=np.float32)
    command = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
        previous_command_q=q_previous,
        max_command_delta=max_delta,
    )[0]

    assert command["command_rate_limited"]
    assert command["q_des"] == pytest.approx(0.08, abs=1.0e-6)
    assert abs(command["tau_pd_est"]) <= 30.05

    extreme_feedback = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint={
            joint_name: {
                "position_raw": -2.0,
                "joint_position": -2.0,
                "joint_velocity": 0.0,
            }
        },
        gain_blend_from_phase="stand",
        gain_blend_alpha=0.0,
        previous_command_q=q_previous,
        max_command_delta=max_delta,
    )[0]
    assert abs(extreme_feedback["q_des"]) <= 0.080001
    assert extreme_feedback["impedance_scale"] < 1.0
    assert abs(extreme_feedback["tau_pd_est"]) <= 30.05


def test_policy_torque_limit_still_limits_position_target():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FL_thigh_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_policy_pd_torque_limit(12.0)
    q_target = np.zeros(12, dtype=np.float32)
    q_target[joint_index] = 0.60
    command = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]

    assert command["q_des"] < 0.60
    assert command["impedance_scale"] == pytest.approx(1.0)
    assert abs(command["tau_pd_est"]) <= 12.05


def test_policy_packet_boundary_uses_physical_not_diagnostic_limits():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "BR_calf_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    assert layer.policy_target_limits[joint_name][1] > 0.0
    assert layer.hard_joint_limits[joint_name][1] == pytest.approx(0.0)
    assert layer.apply_hard_joint_limit(joint_name, 0.50, phase="policy") == pytest.approx(0.0)

    q_target = np.zeros(12, dtype=np.float32)
    q_target[joint_index] = 0.50
    command = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]
    assert command["q_requested"] == pytest.approx(0.50)
    assert command["q_des"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("joint_name", "actor_target", "expected_preload_sign"),
    (
        ("FL_calf_joint", -0.50, -1.0),
        ("BR_calf_joint", +0.50, +1.0),
    ),
)
def test_virtual_joint_stop_preserves_bounded_policy_torque_at_physical_limit(
    joint_name,
    actor_target,
    expected_preload_sign,
):
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_policy_pd_torque_limit(14.0)
    q_safe = np.zeros(12, dtype=np.float32)
    q_actor = q_safe.copy()
    q_actor[joint_index] = actor_target
    command = layer.build_mit_commands(
        q_safe,
        phase="policy",
        prelimit_q_target=q_actor,
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]

    assert command["q_des"] == pytest.approx(0.0)
    assert command["q_prelimit_requested"] == pytest.approx(actor_target)
    assert command["joint_limit_preload_error"] == pytest.approx(actor_target)
    assert abs(command["joint_limit_preload_tau_ff"]) == pytest.approx(8.0)
    assert np.sign(command["joint_limit_preload_tau_ff"]) == expected_preload_sign
    assert abs(command["tau_ff"]) == pytest.approx(8.0)
    assert np.sign(command["tau_ff"]) == (
        expected_preload_sign * layer.joint_directions[joint_name]
    )
    assert abs(command["tau_pd_est"]) <= 14.05


def test_virtual_joint_stop_requires_fresh_feedback_and_policy_phase():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "BR_calf_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_policy_pd_torque_limit(14.0)
    q_safe = np.zeros(12, dtype=np.float32)
    q_actor = q_safe.copy()
    q_actor[joint_index] = 0.50

    missing_feedback = layer.build_mit_commands(
        q_safe,
        phase="policy",
        prelimit_q_target=q_actor,
        feedback_by_joint={},
    )[0]
    pose_command = layer.build_mit_commands(
        q_safe,
        phase="stand",
        prelimit_q_target=q_actor,
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]

    assert missing_feedback["joint_limit_preload_tau_ff"] == pytest.approx(0.0)
    assert pose_command["joint_limit_preload_tau_ff"] == pytest.approx(0.0)


def test_virtual_joint_stop_does_not_turn_rate_limiting_into_feedforward_torque():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    joint_name = "FL_calf_joint"
    joint_index = runner.policy_order.index(joint_name)
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[joint_name],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.set_policy_pd_torque_limit(14.0)

    # Both targets are inside the physical [0, 1.36] calf envelope. Their
    # difference represents target slew limiting, not a mechanical stop.
    q_rate_limited = np.zeros(12, dtype=np.float32)
    q_rate_limited[joint_index] = 0.10
    q_prelimit = q_rate_limited.copy()
    q_prelimit[joint_index] = 0.50
    command = layer.build_mit_commands(
        q_rate_limited,
        phase="policy",
        prelimit_q_target=q_prelimit,
        feedback_by_joint={
            joint_name: {
                "position_raw": 0.0,
                "joint_position": 0.0,
                "joint_velocity": 0.0,
            }
        },
    )[0]

    assert command["q_prelimit_hard_limited"] == pytest.approx(0.50)
    assert command["joint_limit_preload_error"] == pytest.approx(0.0)
    assert command["joint_limit_preload_tau_ff"] == pytest.approx(0.0)


def test_per_joint_policy_torque_limits_are_preserved_in_commands():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    effective = {name: 14.0 + index for index, name in enumerate(runner.policy_order)}
    start = constant_joint_map(runner.policy_order, 14.0)
    final = constant_joint_map(runner.policy_order, 24.0)
    layer.set_policy_pd_torque_limits(
        effective,
        start_limits_by_joint=start,
        final_limits_by_joint=final,
    )
    for index, joint_name in enumerate(runner.policy_order):
        assert layer.policy_pd_torque_limit_for_joint(joint_name) == pytest.approx(14.0 + index)
        assert layer.policy_pd_torque_limit_start[joint_name] == pytest.approx(14.0)
        assert layer.policy_pd_torque_limit_final[joint_name] == pytest.approx(24.0)

    with pytest.raises(KeyError):
        layer.set_policy_pd_torque_limits({"not_a_joint": 1.0})


def test_motor_command_build_does_not_reload_limits_per_joint():
    runner = PolicyRunner()
    motor_ids = load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    layer.control_limit_reload_interval_s = 999.0
    layer.joint_limit_reload_interval_s = 999.0
    calls = {"control": 0, "joint": 0}

    def count_control_reload(force=False):
        calls["control"] += 1
        return False

    def count_joint_reload(force=False):
        calls["joint"] += 1
        return False

    layer.reload_control_limits = count_control_reload
    layer.reload_joint_limits = count_joint_reload
    commands = layer.build_mit_commands(np.zeros(12, dtype=np.float32), phase="policy")
    assert len(commands) == 12
    assert calls["control"] <= 1
    assert calls["joint"] <= 1


def test_policy_torque_profile_validation_rejects_unsafe_values():
    runner = PolicyRunner()
    start = constant_joint_map(runner.policy_order, 14.0)
    final = constant_joint_map(runner.policy_order, 24.0)
    validate_torque_profile(start, final, runner.policy_order, ceiling=40.0)

    bad_final = dict(final)
    bad_final[runner.policy_order[0]] = 41.0
    with pytest.raises(ValueError, match="ceiling"):
        validate_torque_profile(start, bad_final, runner.policy_order, ceiling=40.0)

    bad_start = dict(start)
    bad_start[runner.policy_order[0]] = 25.0
    with pytest.raises(ValueError, match="exceeds final"):
        validate_torque_profile(bad_start, final, runner.policy_order, ceiling=40.0)


def test_calf_endpoint_gate_applies_only_to_active_nonlinear_transmission():
    direct_stage18 = {"FL_calf_joint": 18.0}
    nonlinear_stage14 = {"FL_calf_joint": 14.0}
    nonlinear_stage18 = {"FL_calf_joint": 18.0}

    assert not requires_calf_endpoint_gate(False, direct_stage18)
    assert not requires_calf_endpoint_gate(True, nonlinear_stage14)
    assert requires_calf_endpoint_gate(True, nonlinear_stage18)


def test_policy_torque_ramp_waits_for_clean_entry_and_backs_off_smoothly():
    runner = PolicyRunner()
    start = constant_joint_map(runner.policy_order, 14.0)
    final = constant_joint_map(runner.policy_order, 24.0)
    ramp = PolicyTorqueRamp(
        runner.policy_order,
        start,
        final,
        delay_s=2.0,
        ramp_s=8.0,
        require_clean=True,
        print_interval_s=999.0,
    )

    effective = ramp.update(
        steady_policy_elapsed_s=10.0,
        entry_complete=False,
        feedback_fresh_count=12,
        feedback_count_expected=12,
        feedback_age_max_s=0.01,
        encoder_margin_rad=0.5,
        tracking_error_max=0.0,
        measured_torque_max=0.0,
        cycle_work_s=0.005,
        now=1.0,
    )
    assert ramp.paused
    assert ramp.progress == pytest.approx(0.0)
    assert max(effective.values()) == pytest.approx(14.0)

    effective = ramp.update(
        steady_policy_elapsed_s=6.0,
        entry_complete=True,
        feedback_fresh_count=12,
        feedback_count_expected=12,
        feedback_age_max_s=0.01,
        encoder_margin_rad=0.5,
        tracking_error_max=0.0,
        measured_torque_max=0.0,
        cycle_work_s=0.005,
        now=2.0,
    )
    assert not ramp.paused
    assert ramp.progress == pytest.approx(smoothstep(0.5))
    assert max(effective.values()) == pytest.approx(19.0)

    for step in range(5):
        effective = ramp.update(
            steady_policy_elapsed_s=8.0,
            entry_complete=True,
            feedback_fresh_count=12,
            feedback_count_expected=12,
            feedback_age_max_s=0.01,
            encoder_margin_rad=0.5,
            tracking_error_max=0.0,
            measured_torque_max=35.0,
            cycle_work_s=0.005,
            now=3.0 + step,
        )
    assert ramp.paused
    assert ramp.violation_count == 5
    assert 0.0 < ramp.progress < smoothstep(0.5)
    assert 14.0 < max(effective.values()) < 19.0

    previous_max = max(effective.values())
    effective = ramp.update(
        steady_policy_elapsed_s=8.02,
        entry_complete=True,
        feedback_fresh_count=12,
        feedback_count_expected=12,
        feedback_age_max_s=0.01,
        encoder_margin_rad=0.5,
        tracking_error_max=0.0,
        measured_torque_max=35.0,
        cycle_work_s=0.005,
        now=8.0,
    )
    assert 0.0 < previous_max - max(effective.values()) < 0.1

    backed_off_max = max(effective.values())
    effective = ramp.update(
        steady_policy_elapsed_s=8.04,
        entry_complete=True,
        feedback_fresh_count=12,
        feedback_count_expected=12,
        feedback_age_max_s=0.01,
        encoder_margin_rad=0.5,
        tracking_error_max=0.0,
        measured_torque_max=0.0,
        cycle_work_s=0.005,
        now=9.0,
    )
    assert 0.0 < max(effective.values()) - backed_off_max < 0.1


def test_policy_torque_ramp_holds_instead_of_backing_off_for_tracking_error():
    runner = PolicyRunner()
    start = constant_joint_map(runner.policy_order, 14.0)
    final = constant_joint_map(runner.policy_order, 20.0)
    ramp = PolicyTorqueRamp(runner.policy_order, start, final)

    effective = ramp.update(
        steady_policy_elapsed_s=6.0,
        entry_complete=True,
        feedback_fresh_count=12,
        feedback_count_expected=12,
        feedback_age_max_s=0.01,
        encoder_margin_rad=0.5,
        tracking_error_max=0.0,
        measured_torque_max=0.0,
        cycle_work_s=0.005,
    )
    before = max(effective.values())
    for _ in range(20):
        effective = ramp.update(
            steady_policy_elapsed_s=6.5,
            entry_complete=True,
            feedback_fresh_count=12,
            feedback_count_expected=12,
            feedback_age_max_s=0.01,
            encoder_margin_rad=0.5,
            tracking_error_max=0.35,
            measured_torque_max=0.0,
            cycle_work_s=0.005,
        )

    assert ramp.paused
    assert ramp.violation_count == 0
    assert max(effective.values()) == pytest.approx(before)


def test_policy_torque_ramp_identifies_fixed_stage():
    runner = PolicyRunner()
    fixed = constant_joint_map(runner.policy_order, 14.0)
    staged = constant_joint_map(runner.policy_order, 18.0)

    assert PolicyTorqueRamp(runner.policy_order, fixed, fixed).is_fixed
    assert not PolicyTorqueRamp(runner.policy_order, fixed, staged).is_fixed


def test_measured_torque_supervisor_uses_rolling_window_soft_limit():
    runner = PolicyRunner()

    class Estimator:
        last_feedback_by_joint = {
            joint_name: {"joint_torque": 0.0}
            for joint_name in runner.policy_order
        }

    supervisor = MeasuredTorqueSupervisor(
        runner.policy_order,
        {"hip": 35.0, "thigh": 40.0, "calf": 40.0, "default": 40.0},
        window=3,
    )
    joint_name = "FR_calf_joint"
    for torque in (10.0, 20.0, 45.0):
        Estimator.last_feedback_by_joint[joint_name] = {"joint_torque": torque}
        stats = supervisor.update(Estimator())
    assert stats["average_by_joint"][joint_name] == pytest.approx(25.0)
    assert stats["window_max_by_joint"][joint_name] == pytest.approx(45.0)
    assert stats["soft_limit_active_by_joint"][joint_name] is True


def test_async_csv_logger_writes_without_blocking(tmp_path):
    log_path = tmp_path / "async_log.csv"
    logger = CsvRunLogger(
        enabled=True,
        log_file=str(log_path),
        policy_order=PolicyRunner().policy_order,
        async_enabled=True,
        queue_size=4,
        flush_seconds=0.05,
    )
    record = {
        "phase": "policy",
        "step": 1,
        "mode": "policy",
        "vx": 0.0,
        "vy": 0.0,
        "vxy": 0.0,
        "yaw": 0.0,
        "policy_vx": 0.0,
        "policy_vy": 0.0,
        "policy_yaw": 0.0,
        "speed": 0.1,
        "imu": "fake",
        "act_max": 0.0,
        "tau_cmd": 0.0,
        "tau_cmd_max": 0.0,
        "cmds": 0,
        "bus_counts": "none",
        "tau_fb": None,
        "tau_fb_max": None,
    }
    started = time.monotonic()
    for index in range(12):
        row = dict(record)
        row["step"] = index
        logger.log(row)
    submit_elapsed = time.monotonic() - started
    logger.close()

    assert submit_elapsed < 0.10
    assert log_path.exists()
    assert "compact_line" in log_path.read_text(encoding="utf-8").splitlines()[0]


def test_async_csv_full_queue_increments_dropped_record_count():
    logger = CsvRunLogger(enabled=False)

    class FullQueue:
        def qsize(self):
            return 1

        def put_nowait(self, _record):
            raise queue.Full()

    logger.enabled = True
    logger.async_enabled = True
    logger._writer = object()
    logger._queue = FullQueue()
    logger.submit({"step": 1})
    assert logger.dropped_records == 1


def test_stage40_guard_is_present_in_main_controller_source():
    source = (ROOT / "src" / "main_controller.py").read_text(encoding="utf-8")
    assert "--acknowledge-40nm-suspension-test" in source
    assert 'args.torque_profile_stage == "stage40"' in source
    assert "stage40 requires --acknowledge-40nm-suspension-test" in source


def test_medium_walk_restores_measured_good_0c17450_profile():
    launcher = (ROOT / "scripts" / "run_medium_walk.sh").read_text(
        encoding="utf-8"
    )
    assert "--joint-velocity-source finite-difference" in launcher
    assert "--exact-policy-after-entry" in launcher
    assert "--no-exact-policy-after-entry" not in launcher
    assert "--torque-profile-stage stage20" in launcher
    assert CAN_FEEDBACK_RECEIVE_EVERY_N_CYCLES == 2


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
