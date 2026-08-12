#!/usr/bin/env python3
import argparse
import csv
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor
from state_estimator import FakeStateEstimator, MitFeedbackStateEstimator
from joystick_interface import CommandSource, load_joystick_defaults, load_speed_scale_defaults
from imu_interface import (
    create_imu_sensor,
    imu_reading_quality,
    load_imu_config,
    policy_frame_roll_pitch_from_gravity,
)
from motor_command_layer import MotorCommandLayer, print_mit_commands
from timing_scheduler import DeadlineScheduler, timing_qualification_passed
from can_command_streamer import CanCommandStreamer
from sit_stand_trace_logger import SitStandTraceLogger
from gait_diagnostics import (
    calculate_tracking_errors,
    command_targets_in_policy_order,
    validate_joint_velocity_arrays,
)
from joint_mapping import AuthoritativeJointMapping
from gait_phase_analysis import classify_diagonal_trot
from policy_qualification import (
    calf_calibration_gate,
    replay_policy_csv,
    root_cause_report_lines,
)
from can_topology import (
    add_can_topology_args,
    backend_for_port,
    close_can_buses,
    open_can_buses,
    ports_for_active_joints,
    resolve_joint_can_bus,
    resolve_port_by_bus,
    socketcan_preflight,
    topology_lines,
    validate_unique_motor_ids_per_physical_bus,
)


ROOT = Path(__file__).resolve().parents[1]
CAN_FEEDBACK_RECEIVE_EVERY_N_CYCLES = 2

TELEMETRY_PORT_DEFAULT = 57543


class TelemetrySender:
    """Non-blocking UDP telemetry broadcaster."""

    def __init__(self, port, policy_order, estimator, joint_can_bus=None):
        self._port = int(port)
        self._policy_order = list(policy_order)
        self._estimator = estimator
        self._joint_can_bus = dict(joint_can_bus or {})
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def send(
        self,
        step,
        mode,
        command,
        command_source,
        commands,
        action=None,
        safety_ok=True,
        safety_reason="",
    ):
        try:
            command = np.asarray(command, dtype=np.float32)
            est = self._estimator

            packet = {
                "step": int(step),
                "mode": str(mode),
                "cmd":  [float(command[0]), float(command[1]), float(command[2])],
                "speed": float(command_speed_scale(command_source)),
                "imu":   estimator_imu_status(est),
                "safe":  bool(safety_ok),
                "fault_reason": str(safety_reason or ""),
                "act_max": float(np.max(np.abs(action))) if action is not None else 0.0,
                "base_vel": [float(x) for x in est.base_lin_vel_b],
                "ang_vel":  [float(x) for x in est.base_ang_vel_b],
                "gravity":  [float(x) for x in est.projected_gravity_b],
                "imu_view": self._build_imu_view(),
                "joints": self._build_joints(commands),
                "ts": time.monotonic(),
            }

            data = json.dumps(packet, separators=(",", ":")).encode()
            self._sock.sendto(data, ("127.0.0.1", self._port))
        except Exception:
            pass

    def _build_imu_view(self):
        reading = getattr(self._estimator, "last_imu_reading", None)
        if reading is None:
            return {}

        def vec(value):
            if value is None:
                return None
            return [float(x) for x in np.asarray(value, dtype=np.float32).reshape(-1)]

        axes = getattr(reading, "axes_world", None)
        if axes is not None:
            axes = np.asarray(axes, dtype=np.float32)
            if axes.shape == (3, 3):
                axes = [[float(v) for v in axes[:, i]] for i in range(3)]
            else:
                axes = None

        det_r = getattr(reading, "det_r", None)
        cross_err = getattr(reading, "cross_err", None)

        return {
            "quat_wxyz": vec(getattr(reading, "quaternion_wxyz", None)),
            "rpy_abs_deg": vec(getattr(reading, "rpy_abs_deg", None)),
            "axes_world": axes,
            "projected_gravity": [float(x) for x in self._estimator.projected_gravity_b],
            "det_r": float(det_r) if det_r is not None else None,
            "cross_err": float(cross_err) if cross_err is not None else None,
        }

    def _build_joints(self, commands):
        est = self._estimator
        q_current = getattr(est, "q_current", None)
        qd_current = getattr(est, "qd_current", None)
        feedback_by_joint = getattr(est, "last_feedback_by_joint", {})
        joint_index_by_name = getattr(est, "joint_index_by_name",
                                       {n: i for i, n in enumerate(self._policy_order)})
        joints = []
        for cmd in commands:
            name  = cmd["joint_name"]
            index = joint_index_by_name.get(name)
            fb    = feedback_by_joint.get(name, {})
            joints.append({
                "n":     name,
                "id":    int(cmd["motor_id"]),
                "bus":   self._joint_can_bus.get(name, "front"),
                "qd":    float(cmd["q_des"]),
                "qf":    float(q_current[index])  if q_current  is not None and index is not None else None,
                "vf":    float(qd_current[index]) if qd_current is not None and index is not None else None,
                "tc":    float(cmd["tau_ff"]),
                "tf":    float(fb["torque"])       if "torque"       in fb else None,
                "temp":  float(fb["temperature_c"]) if "temperature_c" in fb else None,
                "fault": int(fb.get("fault_bits", 0)),
                "mode":  int(fb.get("mode_status", 0)),
            })
        return joints

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_motor_ids():
    cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return cfg["motor_ids"]


def load_joint_can_bus():
    cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return cfg.get("joint_can_bus", {})


def load_active_joints():
    cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    active_joints = cfg.get("active_joints", [])
    return list(active_joints or [])


def print_joint_coordinate_contract(mapping, runner, estimator):
    q_current = np.asarray(estimator.q_current, dtype=np.float32)
    print("\nPOLICY/HARDWARE JOINT MAPPING")
    for line in mapping.startup_table_lines():
        print(line)
    print("\nJOINT COORDINATE CONTRACT")
    print(
        "joint training_default training_stand hardware_software_zero "
        "measured_policy_q joint_pos_relative"
    )
    for route in mapping.routes:
        index = route.policy_index
        print(
            f"{route.policy_joint_name:<18s} "
            f"{float(runner.q_default[index]):+9.5f} "
            f"{float(runner.q_stand[index]):+9.5f} "
            f"{route.encoder_offset:+9.5f} "
            f"{float(q_current[index]):+9.5f} "
            f"{float(q_current[index] - runner.q_policy_reference[index]):+9.5f}"
        )


def load_motion_assist_config():
    return load_yaml(ROOT / "config" / "motion_assist.yaml")


def load_policy_deployment_defaults():
    cfg = load_yaml(ROOT / "config" / "control_limits.yaml")
    return cfg.get("policy_deployment", {})


def smoothstep(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def stand_recovery_gain_blend_scale(
    policy_gain_blend_alpha_at_stop,
    elapsed_s,
    recovery_ramp_s,
):
    """Reverse the current stand-to-policy gain blend without a gain step."""
    policy_alpha = float(np.clip(policy_gain_blend_alpha_at_stop, 0.0, 1.0))
    start_stand_alpha = 1.0 - policy_alpha
    recovery_ramp_s = float(recovery_ramp_s)
    if recovery_ramp_s <= 0.0:
        return 1.0
    progress = smoothstep(
        min(1.0, max(0.0, float(elapsed_s)) / recovery_ramp_s)
    )
    return float(
        start_stand_alpha + (1.0 - start_stand_alpha) * progress
    )


def torque_ramp_supervision_due(policy_entry_scale, step, cadence_steps=5):
    """Run gradual authority supervision only after entry and at 10 Hz."""
    cadence_steps = max(1, int(cadence_steps))
    return bool(
        float(policy_entry_scale) >= 0.999
        and int(step) % cadence_steps == 0
    )


def runtime_stand_command_phase(policy_has_started, walking_armed):
    """Stand always uses the loaded pose path; gains are blended on recovery."""
    return "stand"


def synchronized_pose_trajectory(start, target, elapsed_s, duration_s):
    """Interpolate every joint with one smooth phase so they finish together."""
    position, _, alpha = synchronized_pose_trajectory_state(
        start,
        target,
        elapsed_s,
        duration_s,
    )
    return position, alpha


def synchronized_pose_trajectory_state(start, target, elapsed_s, duration_s):
    """Return synchronized smoothstep position and analytic velocity targets."""
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    duration_s = max(float(duration_s), 1.0e-6)
    alpha = float(np.clip(float(elapsed_s) / duration_s, 0.0, 1.0))
    blend = smoothstep(alpha)
    blend_rate = 6.0 * alpha * (1.0 - alpha) / duration_s
    position = (start + blend * (target - start)).astype(np.float32)
    velocity = (blend_rate * (target - start)).astype(np.float32)
    if alpha >= 1.0:
        velocity.fill(0.0)
    return position, velocity, alpha


def fake_start_pose_array(runner, name):
    if name == "stand":
        return runner.q_stand.copy()
    if name == "crouch":
        return runner.q_crouch.copy()
    if name == "random_small":
        rng = np.random.default_rng(7)
        return rng.uniform(-0.25, 0.25, size=len(runner.policy_order)).astype(np.float32)
    raise ValueError(f"Unknown fake start pose: {name}")


def active_feedback_bus_motor_ids(estimator, motor_layer, active_joints=None):
    active_joints = list(active_joints or motor_layer.active_joints)
    if hasattr(estimator, "expected_feedback_bus_motor_ids"):
        return estimator.expected_feedback_bus_motor_ids(active_joints)
    return {
        (
            motor_layer.joint_can_bus.get(joint_name, "front"),
            int(motor_layer.motor_ids[joint_name]),
        )
        for joint_name in active_joints
    }


def refresh_estimator_feedback(estimator, timeout=0.0, expected_bus_motor_ids=None):
    can_feedback_streamer = getattr(estimator, "can_feedback_streamer", None)
    if can_feedback_streamer is not None:
        frames = can_feedback_streamer.drain_received()
        count = 0
        if frames and hasattr(estimator, "update_from_frames"):
            count = estimator.update_from_frames(frames)
        # While command streaming is active, this worker exclusively owns the
        # SocketCAN receive calls. Concurrent recv() calls from the 50 Hz loop
        # previously starved the 200 Hz sender and reduced it to 143-188 Hz.
        if can_feedback_streamer.has_active_commands:
            return count
    if hasattr(estimator, "refresh_from_bus"):
        try:
            return estimator.refresh_from_bus(
                timeout=timeout,
                expected_bus_motor_ids=expected_bus_motor_ids,
            )
        except TypeError:
            return estimator.refresh_from_bus(timeout=timeout)
    return 0


def read_estimator_state(estimator, refresh_imu=True):
    if hasattr(estimator, "read_cached"):
        return estimator.read_cached(refresh_imu=refresh_imu)
    return estimator.read()


def joystick_walk_requested(command, threshold):
    command = np.asarray(command, dtype=np.float32)
    return bool(np.max(np.abs(command)) > float(threshold))


def scaled_policy_command(command, gain=1.0, vx_abs_max=0.0, vy_abs_max=0.0, yaw_abs_max=0.0):
    cmd = np.asarray(command, dtype=np.float32).copy()
    cmd *= max(0.0, float(gain))

    limits = (vx_abs_max, vy_abs_max, yaw_abs_max)
    for i, limit in enumerate(limits):
        limit = float(limit)
        if limit > 0.0:
            cmd[i] = float(np.clip(cmd[i], -limit, limit))
    return cmd.astype(np.float32)


def filtered_policy_action(
    raw_action,
    previous_action,
    clip_abs=0.0,
    smoothing=0.0,
    delta_limit_abs=0.0,
):
    action = np.asarray(raw_action, dtype=np.float32).copy()
    previous_action = np.asarray(previous_action, dtype=np.float32)

    clip_abs = float(clip_abs)
    if clip_abs > 0.0:
        action = np.clip(action, -clip_abs, clip_abs)

    smoothing = float(np.clip(smoothing, 0.0, 0.98))
    if smoothing > 0.0:
        action = (1.0 - smoothing) * action + smoothing * previous_action

    delta_limit_abs = float(delta_limit_abs)
    if delta_limit_abs > 0.0:
        delta = np.clip(
            action - previous_action,
            -delta_limit_abs,
            delta_limit_abs,
        )
        action = previous_action + delta
    return action.astype(np.float32)


def policy_previous_action_observation(
    previous_raw_action,
    previous_sent_action,
    exact_policy_after_entry,
):
    """Return the previous raw actor output required by the training contract.

    Motor-facing conditioning is deliberately excluded. Isaac Lab records the
    actor output in the previous-action observation before deployment clipping,
    smoothing, rate limiting, joint limiting, or torque limiting are applied.
    """
    previous_raw_action = np.asarray(previous_raw_action, dtype=np.float32)
    previous_sent_action = np.asarray(previous_sent_action, dtype=np.float32)
    if previous_raw_action.shape != previous_sent_action.shape:
        raise ValueError(
            "previous raw/sent actions must have identical shapes; got "
            f"{list(previous_raw_action.shape)} and "
            f"{list(previous_sent_action.shape)}"
        )
    selected = previous_raw_action
    if not np.all(np.isfinite(selected)):
        raise ValueError("previous policy action observation contains NaN or Inf")
    return selected.copy()


def clip_policy_hip_actions(
    raw_action,
    policy_order,
    hip_clip_abs=0.0,
    hip_scale=1.0,
):
    """Condition hip motor actions without changing thigh/calf authority."""
    action = np.asarray(raw_action, dtype=np.float32).copy()
    if action.shape != (len(policy_order),):
        raise ValueError(
            f"raw_action has shape {list(action.shape)}; "
            f"expected [{len(policy_order)}]"
        )
    hip_clip_abs = float(hip_clip_abs)
    hip_scale = float(hip_scale)
    if not np.isfinite(hip_scale) or hip_scale < 0.0:
        raise ValueError("hip_scale must be finite and >= 0")
    for index, joint_name in enumerate(policy_order):
        if "_hip_joint" in str(joint_name):
            if hip_clip_abs > 0.0:
                action[index] = float(
                    np.clip(action[index], -hip_clip_abs, hip_clip_abs)
                )
            action[index] *= hip_scale
    return action


def action_equivalent_for_q_target(runner, q_target):
    """Convert a deployment joint target back into actor-action coordinates."""
    q_target = np.asarray(q_target, dtype=np.float32)
    if q_target.shape != (len(runner.policy_order),):
        raise ValueError(
            f"q_target has shape {list(q_target.shape)}; "
            f"expected [{len(runner.policy_order)}]"
        )
    if not np.all(np.isfinite(q_target)):
        raise ValueError("q_target contains NaN or Inf")
    return ((q_target - runner.q_policy_reference) / runner.action_scale).astype(np.float32)


def policy_prelimit_target_for_commands(
    q_policy_target,
    q_safe_target,
    exact_policy_after_entry,
):
    """Prevent exact-policy mode from synthesizing virtual-stop feedforward."""
    q_policy_target = np.asarray(q_policy_target, dtype=np.float32)
    q_safe_target = np.asarray(q_safe_target, dtype=np.float32)
    if q_policy_target.shape != q_safe_target.shape:
        raise ValueError("policy and safe targets must have identical shapes")
    return (
        q_safe_target.copy()
        if bool(exact_policy_after_entry)
        else q_policy_target.copy()
    )


def projected_gravity_to_roll_pitch(projected_gravity_b):
    # projected_gravity_b = R_world_from_body.T @ [0, 0, -1]. Therefore a
    # positive body roll produces negative gravity-Y, and positive pitch
    # produces positive gravity-X. These signs match IsaacLab and the Xsens
    # world-from-body quaternion conversion in imu_interface.py.
    return policy_frame_roll_pitch_from_gravity(projected_gravity_b)


def imu_telemetry_fields(estimator):
    """Return raw and policy-frame IMU telemetry in every controller mode."""
    reading = getattr(estimator, "last_imu_reading", None)

    def vector(value, length):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape != (length,) or not np.all(np.isfinite(arr)):
            return None
        return arr

    policy_gyro = vector(getattr(estimator, "base_ang_vel_b", None), 3)
    raw_gyro = vector(
        None if reading is None else getattr(reading, "base_ang_vel_b", None),
        3,
    )
    quat = vector(
        None if reading is None else getattr(reading, "quaternion_wxyz", None),
        4,
    )
    rpy = vector(
        None if reading is None else getattr(reading, "rpy_abs_deg", None),
        3,
    )

    fields = {}
    for prefix, values, names in (
        ("imu_gyro_raw", raw_gyro, ("x", "y", "z")),
        ("imu_gyro_policy", policy_gyro, ("x", "y", "z")),
        ("imu_quat", quat, ("w", "x", "y", "z")),
        ("imu_abs_rpy", rpy, ("roll_deg", "pitch_deg", "yaw_deg")),
    ):
        for index, name in enumerate(names):
            fields[f"{prefix}_{name}"] = (
                None if values is None else float(values[index])
            )
    fields["imu_yaw_deg"] = None if rpy is None else float(rpy[2])
    return fields


def imu_posture_correction(projected_gravity_b, policy_order, cfg):
    imu_cfg = cfg.get("imu_posture", {})
    if not bool(imu_cfg.get("enabled", False)):
        return np.zeros(len(policy_order), dtype=np.float32)

    roll, pitch = projected_gravity_to_roll_pitch(projected_gravity_b)
    deadband = np.radians(max(0.0, float(imu_cfg.get("deadband_deg", 0.0))))

    def apply_deadband(value):
        magnitude = abs(float(value))
        if magnitude <= deadband:
            return 0.0
        return float(np.sign(value) * (magnitude - deadband))

    roll = apply_deadband(roll)
    pitch = apply_deadband(pitch)
    roll_corr = float(
        np.clip(
            float(imu_cfg.get("roll_kp", 0.0)) * roll,
            -float(imu_cfg.get("max_roll_correction", 0.0)),
            float(imu_cfg.get("max_roll_correction", 0.0)),
        )
    )
    pitch_corr = float(
        np.clip(
            float(imu_cfg.get("pitch_kp", 0.0)) * pitch,
            -float(imu_cfg.get("max_pitch_correction", 0.0)),
            float(imu_cfg.get("max_pitch_correction", 0.0)),
        )
    )

    gains = imu_cfg.get("joint_gains", {})
    correction = np.zeros(len(policy_order), dtype=np.float32)
    for index, joint_name in enumerate(policy_order):
        joint_gain = gains.get(joint_name, {})
        correction[index] = (
            float(joint_gain.get("roll", 0.0)) * roll_corr
            + float(joint_gain.get("pitch", 0.0)) * pitch_corr
        )

    return correction


def apply_imu_posture_stabilization(q_target, projected_gravity_b, policy_order, cfg):
    q_target = np.asarray(q_target, dtype=np.float32).copy()
    q_target += imu_posture_correction(
        projected_gravity_b=projected_gravity_b,
        policy_order=policy_order,
        cfg=cfg,
    )

    return q_target


def stand_policy_imu_correction(
    runner,
    base_ang_vel_b,
    projected_gravity_b,
    q_current,
    qd_current,
    cfg,
):
    """Return only the actor response caused by live IMU state.

    Subtracting an otherwise identical upright policy pass prevents the
    policy's nonzero nominal stand action from moving the legs by itself.
    Joint feedback remains present in both passes and therefore cancels from
    the comparison except for its interaction with the IMU observation.
    """
    previous_action = np.zeros(len(runner.policy_order), dtype=np.float32)
    zero_command = np.zeros(3, dtype=np.float32)
    policy_cfg = cfg.get("stand_policy_imu", {})
    policy_gyro = (
        np.asarray(base_ang_vel_b, dtype=np.float32)
        if bool(policy_cfg.get("use_live_gyro", False))
        else np.zeros(3, dtype=np.float32)
    )
    if bool(policy_cfg.get("use_live_joint_state", False)):
        policy_q = np.asarray(q_current, dtype=np.float32)
        policy_qd = np.asarray(qd_current, dtype=np.float32)
    else:
        policy_q = runner.q_stand
        policy_qd = np.zeros(len(runner.policy_order), dtype=np.float32)
    live_obs = runner.build_observation(
        base_ang_vel_b=policy_gyro,
        projected_gravity_b=projected_gravity_b,
        command=zero_command,
        q_current=policy_q,
        qd_current=policy_qd,
        previous_action=previous_action,
    )
    upright_obs = runner.build_observation(
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        command=zero_command,
        q_current=policy_q,
        qd_current=policy_qd,
        previous_action=previous_action,
    )
    live_action = runner.infer_action(live_obs)
    upright_action = runner.infer_action(upright_obs)
    delta_action = np.asarray(live_action - upright_action, dtype=np.float32)

    gain = max(0.0, float(policy_cfg.get("gain", 1.0)))
    max_correction = max(0.0, float(policy_cfg.get("max_correction", 0.12)))
    correction = runner.action_scale * gain * delta_action
    if max_correction > 0.0:
        correction = np.clip(correction, -max_correction, max_correction)

    return (
        correction.astype(np.float32),
        live_obs,
        live_action.astype(np.float32),
        delta_action,
    )


def apply_motion_assists(q_target, projected_gravity_b, runner, cfg):
    return apply_imu_posture_stabilization(
        q_target=q_target,
        projected_gravity_b=projected_gravity_b,
        policy_order=runner.policy_order,
        cfg=cfg,
    )


def command_speed_scale(command_source):
    if hasattr(command_source, "get_speed_scale"):
        return float(command_source.get_speed_scale())
    return 1.0


def command_torque_stats(commands):
    if not commands:
        return 0.0, 0.0

    tau_values = []
    for cmd in commands:
        value = cmd.get("tau_pd_est")
        if value is None or not np.isfinite(value):
            value = cmd.get("tau_ff", 0.0)
        tau_values.append(float(value))
    tau = np.asarray(tau_values, dtype=np.float32)
    return float(np.abs(tau).mean()), float(np.abs(tau).max())


def command_bus_counts(commands):
    counts = {}
    for cmd in commands:
        bus_name = str(cmd.get("bus_name", "front"))
        counts[bus_name] = counts.get(bus_name, 0) + 1
    return counts


def format_bus_counts(counts):
    if not counts:
        return "none"
    return " ".join(
        f"{bus_name}:{counts[bus_name]:02d}"
        for bus_name in sorted(counts)
    )


def max_can_tx_duration_s(buses):
    """Return the slowest most-recent CAN command batch duration."""
    if buses is None:
        return None
    if not isinstance(buses, dict):
        value = getattr(buses, "last_sequence_duration_s", None)
        return None if value is None else float(value)

    durations = []
    seen_bus_ids = set()
    for bus in buses.values():
        if id(bus) in seen_bus_ids:
            continue
        seen_bus_ids.add(id(bus))
        value = getattr(bus, "last_sequence_duration_s", None)
        if value is not None:
            durations.append(float(value))
    return max(durations) if durations else None


def feedback_age_summary(estimator, active_joints, max_age_s=None):
    """Return (fresh_count, max_age_s) for the active MIT feedback snapshot."""
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", None)
    if not isinstance(feedback_by_joint, dict):
        return None, None

    now = time.monotonic()
    ages = []
    fresh_count = 0
    for joint_name in active_joints:
        feedback = feedback_by_joint.get(joint_name)
        if not isinstance(feedback, dict):
            continue
        timestamp = feedback.get("timestamp")
        if timestamp is None:
            continue
        age = now - float(timestamp)
        if np.isfinite(age):
            ages.append(age)
            if max_age_s is None or age <= float(max_age_s):
                fresh_count += 1
    return fresh_count, (max(ages) if ages else None)


def feedback_torque_stats(estimator):
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {})
    if not feedback_by_joint:
        return None

    tau = np.asarray(
        [float(feedback["torque"]) for feedback in feedback_by_joint.values()],
        dtype=np.float32,
    )
    return float(np.abs(tau).mean()), float(np.abs(tau).max())


def estimator_imu_status(estimator):
    if hasattr(estimator, "imu_status"):
        return estimator.imu_status()
    return "none"


def format_vector(values, precision=4):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(value):+.{precision}f}" for value in arr) + "]"


def validate_required_policy_imu(estimator, max_roll_pitch_deg=60.0):
    if not hasattr(estimator, "imu_required") or not estimator.imu_required():
        return None
    if hasattr(estimator, "refresh_imu"):
        estimator.refresh_imu()
    reading = getattr(estimator, "last_imu_reading", None)
    stale_timeout = float(getattr(getattr(estimator, "imu_sensor", None), "stale_timeout", 0.25))
    ok, reason = imu_reading_quality(
        reading,
        stale_timeout=stale_timeout,
        max_roll_pitch_deg=max_roll_pitch_deg,
    )
    if not ok:
        return f"required IMU invalid/stale ({estimator_imu_status(estimator)}): {reason}"
    return None


def wait_for_live_policy_imu(
    estimator,
    source_name,
    samples=5,
    timeout_s=5.0,
    max_roll_pitch_deg=60.0,
):
    if not hasattr(estimator, "imu_required") or not estimator.imu_required():
        return True

    samples = max(1, int(samples))
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last_counted_timestamp = None
    counted_timestamps = []
    last_reason = "no samples received"
    reading = None
    while time.monotonic() < deadline and len(counted_timestamps) < samples:
        if hasattr(estimator, "refresh_imu"):
            estimator.refresh_imu()
        reading = getattr(estimator, "last_imu_reading", None)
        timestamp = None if reading is None else getattr(reading, "timestamp", None)
        try:
            timestamp = None if timestamp is None else float(timestamp)
        except (TypeError, ValueError):
            timestamp = None
        if timestamp is None or timestamp == last_counted_timestamp:
            time.sleep(0.001)
            continue
        ok, reason = imu_reading_quality(
            reading,
            previous_timestamp=last_counted_timestamp,
            require_timestamp_advance=last_counted_timestamp is not None,
            stale_timeout=float(getattr(getattr(estimator, "imu_sensor", None), "stale_timeout", 0.25)),
            max_roll_pitch_deg=max_roll_pitch_deg,
        )
        if ok:
            counted_timestamps.append(timestamp)
            last_counted_timestamp = timestamp
        else:
            last_reason = reason
            counted_timestamps.clear()
            last_counted_timestamp = None
        time.sleep(0.001)

    if len(counted_timestamps) < samples:
        print(
            "\nERROR: required live IMU did not produce "
            f"{samples} consecutive valid policy-frame sample(s): {last_reason}"
        )
        return False

    if hasattr(estimator, "refresh_imu"):
        estimator.refresh_imu()
    reading = getattr(estimator, "last_imu_reading", reading)
    gyro = getattr(estimator, "base_ang_vel_b", getattr(reading, "base_ang_vel_b", np.zeros(3)))
    gravity = getattr(
        estimator,
        "projected_gravity_b",
        getattr(reading, "projected_gravity_b", np.array([0.0, 0.0, -1.0])),
    )
    roll, pitch = projected_gravity_to_roll_pitch(gravity)
    elapsed = max(1.0e-6, counted_timestamps[-1] - counted_timestamps[0])
    sample_rate = 0.0 if len(counted_timestamps) < 2 else (len(counted_timestamps) - 1) / elapsed
    imu_sensor = getattr(estimator, "imu_sensor", None)
    if hasattr(imu_sensor, "status"):
        sensor_status = imu_sensor.status()
        background_rate = float(sensor_status.get("sample_rate_hz", 0.0) or 0.0)
        if np.isfinite(background_rate) and background_rate > 0.0:
            sample_rate = background_rate
    print(f"[IMU] source={source_name} status=live")
    print(f"[IMU] gyro_body={format_vector(gyro)}")
    print(f"[IMU] projected_gravity={format_vector(gravity)}")
    print(f"[IMU] roll={np.degrees(roll):+.2f}deg pitch={np.degrees(pitch):+.2f}deg")
    print(f"[IMU] sample_rate={sample_rate:.1f}Hz")
    return True


TORQUE_PROFILE_STAGES = {
    "stage14": (14.0, 14.0),
    "stage18": (14.0, 18.0),
    "stage20": (14.0, 20.0),
    "stage24": (14.0, 24.0),
    "stage30": (14.0, 30.0),
    "stage36": (14.0, 36.0),
    "stage40": (14.0, 40.0),
    "stage100": (100.0, 100.0),
}


def requires_calf_endpoint_gate(four_bar_enabled, final_torque_limits):
    """Require linkage calibration only for an active nonlinear transmission."""
    return bool(four_bar_enabled) and max(
        (float(value) for value in final_torque_limits.values()),
        default=0.0,
    ) > 14.0


def constant_joint_map(policy_order, value):
    value = float(value)
    return {joint_name: value for joint_name in policy_order}


def resolve_profile_values(section, policy_order, fallback):
    if section is None:
        return dict(fallback)
    if not isinstance(section, dict):
        raise ValueError("policy torque profile section must be a mapping")
    default = float(section.get("default", np.nan))
    resolved = {}
    for joint_name in policy_order:
        if joint_name in section:
            value = float(section[joint_name])
        elif np.isfinite(default):
            value = default
        else:
            value = float(fallback[joint_name])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{joint_name}: torque profile value must be finite and >= 0")
        resolved[joint_name] = value
    unknown = sorted(set(section) - set(policy_order) - {"default"})
    if unknown:
        raise KeyError("Unknown torque profile joint(s): " + ", ".join(unknown))
    return resolved


def load_policy_torque_profile(path, policy_order, start_fallback, final_fallback):
    cfg = load_yaml(Path(path).expanduser())
    profile = (cfg or {}).get("policy_torque_profile", cfg or {})
    if not isinstance(profile, dict):
        raise ValueError("policy torque profile YAML must contain a mapping")
    start = resolve_profile_values(
        profile.get("start_nm"),
        policy_order,
        start_fallback,
    )
    final = resolve_profile_values(
        profile.get("final_nm"),
        policy_order,
        final_fallback,
    )
    return start, final


def validate_torque_profile(start_by_joint, final_by_joint, policy_order, ceiling):
    ceiling = float(ceiling)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("absolute torque ceiling must be finite and > 0")
    for joint_name in policy_order:
        start = float(start_by_joint[joint_name])
        final = float(final_by_joint[joint_name])
        if not np.isfinite(start) or not np.isfinite(final):
            raise ValueError(f"{joint_name}: torque profile contains NaN/Inf")
        if start < 0.0 or final < 0.0:
            raise ValueError(f"{joint_name}: torque profile values must be >= 0")
        if start > final:
            raise ValueError(f"{joint_name}: torque start {start:.2f} exceeds final {final:.2f}")
        if final > ceiling:
            raise ValueError(
                f"{joint_name}: torque final {final:.2f} exceeds absolute ceiling {ceiling:.2f}"
            )


def joint_group_soft_limit(joint_name, soft_limits):
    group = "calf" if "calf" in joint_name else "thigh" if "thigh" in joint_name else "hip"
    return float(soft_limits.get(group, soft_limits.get("default", 40.0)))


class MeasuredTorqueSupervisor:
    def __init__(self, policy_order, soft_limits, window=12):
        self.policy_order = list(policy_order)
        self.soft_limits = dict(soft_limits)
        self.window = max(1, int(window))
        self.history = {joint_name: [] for joint_name in self.policy_order}

    def update(self, estimator):
        feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
        current_by_joint = {}
        average_by_joint = {}
        window_max_by_joint = {}
        soft_limit_active_by_joint = {}
        max_abs = 0.0
        for joint_name in self.policy_order:
            feedback = feedback_by_joint.get(joint_name, {})
            value = feedback.get("joint_torque", feedback.get("torque"))
            try:
                torque = float(value)
            except (TypeError, ValueError):
                torque = 0.0
            current_by_joint[joint_name] = torque
            max_abs = max(max_abs, abs(torque))

            history = self.history[joint_name]
            history.append(abs(torque))
            if len(history) > self.window:
                del history[:-self.window]
            average = float(np.mean(history)) if history else 0.0
            window_max = float(max(history)) if history else 0.0
            average_by_joint[joint_name] = average
            window_max_by_joint[joint_name] = window_max

            soft_limit = joint_group_soft_limit(joint_name, self.soft_limits)
            soft_limit_active_by_joint[joint_name] = bool(window_max > soft_limit)

        return {
            "current_by_joint": current_by_joint,
            "average_by_joint": average_by_joint,
            "window_max_by_joint": window_max_by_joint,
            "soft_limit_active_by_joint": soft_limit_active_by_joint,
            "max_abs": max_abs,
        }


def encoder_margin_to_policy_limits(safety, q_values, policy_order):
    try:
        q_values = np.asarray(q_values, dtype=np.float32).reshape(-1)
        lower = np.asarray(safety.policy_q_min, dtype=np.float32)
        upper = np.asarray(safety.policy_q_max, dtype=np.float32)
    except Exception:
        return float("inf"), ""
    margins = []
    labels = []
    for index, joint_name in enumerate(policy_order):
        if index >= q_values.size:
            continue
        margin = min(
            float(q_values[index] - lower[index]),
            float(upper[index] - q_values[index]),
        )
        margins.append(margin)
        labels.append((margin, joint_name))
    if not margins:
        return float("inf"), ""
    margin, joint_name = min(labels, key=lambda item: item[0])
    return float(margin), str(joint_name)


def torque_ramp_timing_fault(timing_snapshot):
    """Return only the scheduler's sustained timing-fault state.

    A single cycle can exceed the 20 ms policy period without constituting a
    controller timing fault. The ramp qualifies that cycle independently with
    its configurable max_cycle_work_s gate.
    """
    return bool(getattr(timing_snapshot, "timing_fault", False))


class PolicyTorqueRamp:
    def __init__(
        self,
        policy_order,
        start_by_joint,
        final_by_joint,
        delay_s=2.0,
        ramp_s=8.0,
        require_clean=True,
        max_tracking_error_rad=0.25,
        max_measured_torque=30.0,
        min_encoder_margin_rad=0.08,
        max_feedback_age_s=0.04,
        max_cycle_work_s=0.020,
        print_interval_s=1.0,
    ):
        self.policy_order = list(policy_order)
        self.start_by_joint = {name: float(start_by_joint[name]) for name in self.policy_order}
        self.final_by_joint = {name: float(final_by_joint[name]) for name in self.policy_order}
        self.effective_by_joint = dict(self.start_by_joint)
        self.delay_s = max(0.0, float(delay_s))
        self.ramp_s = max(1.0e-6, float(ramp_s))
        self.require_clean = bool(require_clean)
        self.max_tracking_error_rad = float(max_tracking_error_rad)
        self.max_measured_torque = float(max_measured_torque)
        self.min_encoder_margin_rad = float(min_encoder_margin_rad)
        self.max_feedback_age_s = float(max_feedback_age_s)
        self.max_cycle_work_s = float(max_cycle_work_s)
        self.print_interval_s = max(0.25, float(print_interval_s))
        self.progress = 0.0
        self.paused = False
        self.pause_reason = ""
        self.violation_count = 0
        self.recovery_ceiling = 1.0
        self.backoff_progress_step = 0.005
        self.recovery_progress_step = 0.0025
        self._last_state = None
        self._last_print_time = -1.0e9

    @property
    def start_max(self):
        return max(self.start_by_joint.values(), default=0.0)

    @property
    def final_max(self):
        return max(self.final_by_joint.values(), default=0.0)

    @property
    def effective_max(self):
        return max(self.effective_by_joint.values(), default=0.0)

    @property
    def is_fixed(self):
        return all(
            abs(self.final_by_joint[name] - self.start_by_joint[name]) <= 1.0e-9
            for name in self.policy_order
        )

    def _reason(
        self,
        entry_complete,
        imu_fault,
        feedback_fresh_count,
        feedback_count_expected,
        feedback_age_max_s,
        encoder_margin_rad,
        encoder_margin_joint,
        tracking_error_max,
        measured_torque_max,
        measured_soft_limit_active,
        cycle_work_s,
        motor_fault,
        timing_fault,
    ):
        if not entry_complete:
            return "policy entry blend active"
        if imu_fault:
            return str(imu_fault)
        if feedback_fresh_count < feedback_count_expected:
            return f"fresh feedback {feedback_fresh_count}/{feedback_count_expected}"
        if np.isfinite(feedback_age_max_s) and feedback_age_max_s > self.max_feedback_age_s:
            return f"feedback age {feedback_age_max_s:.3f}s"
        if np.isfinite(encoder_margin_rad) and encoder_margin_rad < self.min_encoder_margin_rad:
            return f"{encoder_margin_joint or 'joint'} encoder margin {encoder_margin_rad:.3f} rad"
        if np.isfinite(tracking_error_max) and tracking_error_max > self.max_tracking_error_rad:
            return f"tracking error {tracking_error_max:.3f} rad"
        if np.isfinite(measured_torque_max) and measured_torque_max > self.max_measured_torque:
            return f"measured torque {measured_torque_max:.2f} Nm"
        if measured_soft_limit_active:
            return "measured torque soft limit active"
        if np.isfinite(cycle_work_s) and cycle_work_s > self.max_cycle_work_s:
            return f"cycle work {1000.0 * cycle_work_s:.1f} ms"
        if motor_fault:
            return str(motor_fault)
        if timing_fault:
            return "timing overrun"
        return ""

    def update(
        self,
        steady_policy_elapsed_s,
        entry_complete,
        imu_fault=None,
        feedback_fresh_count=0,
        feedback_count_expected=12,
        feedback_age_max_s=float("inf"),
        encoder_margin_rad=float("inf"),
        encoder_margin_joint="",
        tracking_error_max=0.0,
        measured_torque_max=0.0,
        measured_soft_limit_active=False,
        cycle_work_s=0.0,
        motor_fault=None,
        timing_fault=False,
        now=None,
        print_fn=None,
    ):
        now = time.monotonic() if now is None else float(now)
        reason = self._reason(
            entry_complete=entry_complete,
            imu_fault=imu_fault,
            feedback_fresh_count=int(feedback_fresh_count or 0),
            feedback_count_expected=int(feedback_count_expected or 0),
            feedback_age_max_s=float(feedback_age_max_s),
            encoder_margin_rad=float(encoder_margin_rad),
            encoder_margin_joint=encoder_margin_joint,
            tracking_error_max=float(tracking_error_max),
            measured_torque_max=float(measured_torque_max),
            measured_soft_limit_active=bool(measured_soft_limit_active),
            cycle_work_s=float(cycle_work_s),
            motor_fault=motor_fault,
            timing_fault=timing_fault,
        )
        clean = reason == ""
        if self.require_clean and not clean:
            self.paused = True
            self.pause_reason = reason
            if not entry_complete:
                self.violation_count = 0
                self.progress = 0.0
                self.recovery_ceiling = 1.0
                self.effective_by_joint = dict(self.start_by_joint)
            elif reason.startswith("tracking error"):
                # Tracking error means the current authority is not producing
                # the requested motion. Lowering torque here creates a
                # positive-feedback stall and a visible step in stiffness.
                # Hold the present limit and resume upward only gradually.
                self.violation_count = 0
                self.recovery_ceiling = min(
                    self.recovery_ceiling,
                    self.progress,
                )
            elif (
                reason.startswith("feedback age")
                or reason.startswith("fresh feedback")
                or reason.startswith("cycle work")
                or reason == "timing overrun"
            ):
                # These are producer/telemetry qualification gates, not
                # evidence that the current motor torque is unsafe. Hold the
                # present stage and wait for a clean supervision sample.
                # Backing off on ordinary previous-cycle feedback (still well
                # inside the runtime freshness limit) trapped stage20 around
                # 14 Nm and made the loaded robot sag as policy took control.
                self.violation_count = 0
            else:
                self.violation_count += 1
                if self.violation_count >= 5:
                    if self.violation_count == 5:
                        self.recovery_ceiling = min(
                            self.recovery_ceiling,
                            self.progress,
                        )
                    self.recovery_ceiling = max(
                        0.0,
                        self.recovery_ceiling - self.backoff_progress_step,
                    )
                    self.progress = min(self.progress, self.recovery_ceiling)
                    self.effective_by_joint = {
                        joint_name: self.start_by_joint[joint_name]
                        + self.progress
                        * (self.final_by_joint[joint_name] - self.start_by_joint[joint_name])
                        for joint_name in self.policy_order
                    }
        else:
            self.paused = False
            self.pause_reason = ""
            self.violation_count = 0
            self.recovery_ceiling = min(
                1.0,
                self.recovery_ceiling + self.recovery_progress_step,
            )
            scheduled_progress = smoothstep(
                np.clip(
                    (float(steady_policy_elapsed_s) - self.delay_s) / self.ramp_s,
                    0.0,
                    1.0,
                )
            )
            self.progress = min(float(scheduled_progress), self.recovery_ceiling)
            self.effective_by_joint = {
                joint_name: self.start_by_joint[joint_name]
                + self.progress
                * (self.final_by_joint[joint_name] - self.start_by_joint[joint_name])
                for joint_name in self.policy_order
            }

        state = "paused" if self.paused else "running"
        if print_fn is not None:
            if state != self._last_state:
                if self.paused:
                    print_fn(f"[TORQUE RAMP PAUSED] {self.pause_reason}")
                elif self._last_state == "paused":
                    print_fn("[TORQUE RAMP RESUMED]")
                self._last_state = state
            if self.violation_count == 5:
                print_fn(
                    "[TORQUE RAMP BACKOFF] "
                    "effective torque will decrease gradually while the "
                    f"violation persists (current={self.effective_max:.1f} Nm)"
                )
            if now - self._last_print_time >= self.print_interval_s and not self.paused:
                self._last_print_time = now
                print_fn(
                    "[TORQUE RAMP] "
                    f"effective={self.effective_max:.1f} Nm "
                    f"start={self.start_max:.1f} final={self.final_max:.1f} "
                    f"progress={100.0 * self.progress:.0f}%"
                )
        return self.effective_by_joint

    def telemetry(self):
        return {
            "policy_torque_limit_start_nm": self.start_max,
            "policy_torque_limit_final_nm": self.final_max,
            "policy_torque_limit_effective_nm": self.effective_max,
            "policy_torque_ramp_progress": self.progress,
            "policy_torque_ramp_paused": int(bool(self.paused)),
            "policy_torque_ramp_pause_reason": self.pause_reason,
        }


def compact_telemetry_record(
    step,
    mode,
    command,
    command_source,
    commands,
    estimator,
    action=None,
    phase="policy",
    policy_command=None,
    observation=None,
    raw_action=None,
    sent_action=None,
    q_current=None,
    qd_current=None,
    q_actor_target=None,
    q_entry_blended_target=None,
    q_joint_limit_filtered_target=None,
    q_rate_limited_target=None,
    q_safety_target=None,
    q_target=None,
    target_joint_limited=None,
    target_rate_limited=None,
    entry_blend_active=False,
    policy_order=None,
    policy_sha256=None,
    policy_entry_scale=0.0,
    policy_entry_elapsed_s=0.0,
    policy_entry_restart_count=0,
    policy_entry_restart_reason="",
    imu_correction_abs_max=0.0,
    loop_dt_s=None,
    loop_period_s=None,
    cycle_work_s=0.0,
    deadline_lateness_s=0.0,
    policy_inference_s=None,
    command_input_s=None,
    observation_build_s=None,
    policy_target_conversion_s=None,
    safety_filter_s=None,
    command_build_s=None,
    can_tx_s=None,
    feedback_read_s=None,
    pre_feedback_read_s=None,
    steady_feedback_read_s=None,
    safety_check_s=None,
    logging_s=None,
    terminal_print_s=None,
    imu_cache_read_s=None,
    feedback_age_max_s=None,
    feedback_fresh_count=None,
    feedback_current_cycle_count=None,
    feedback_previous_cycle_count=None,
    feedback_missing_count=None,
    feedback_stale_count=None,
    torque_ramp_state=None,
    measured_soft_limit_active_by_joint=None,
    measured_torque_average_by_joint=None,
    measured_torque_window_max_by_joint=None,
    missed_deadlines=0,
    consecutive_overruns=0,
    max_overrun_s=0.0,
    missed_deadlines_total=None,
    missed_deadlines_this_cycle=0,
    consecutive_work_overruns=None,
    scheduler_resync_count=0,
    max_cycle_work_s=0.0,
    max_lateness_s=0.0,
    policy_steady_cycles=0,
    policy_target_clip_counts=None,
    policy_torque_clip_counts=None,
):
    command = np.asarray(command, dtype=np.float32)
    policy_command = command if policy_command is None else np.asarray(policy_command, dtype=np.float32)
    tau_cmd_mean, tau_cmd_max = command_torque_stats(commands)
    bus_counts = command_bus_counts(commands)
    tau_fb = feedback_torque_stats(estimator)
    action_abs_max = 0.0 if action is None else float(np.max(np.abs(action)))
    gravity = np.asarray(
        getattr(estimator, "projected_gravity_b", [0.0, 0.0, -1.0]),
        dtype=np.float32,
    )
    imu_roll, imu_pitch = projected_gravity_to_roll_pitch(gravity)

    if loop_period_s is None:
        loop_period_s = loop_dt_s
    if missed_deadlines_total is None:
        missed_deadlines_total = missed_deadlines
    if consecutive_work_overruns is None:
        consecutive_work_overruns = consecutive_overruns

    record = {
        "phase": str(phase),
        "step": int(step),
        "mode": str(mode),
        "vx": float(command[0]),
        "vy": float(command[1]),
        "vxy": float(np.linalg.norm(command[:2])),
        "yaw": float(command[2]),
        "policy_vx": float(policy_command[0]),
        "policy_vy": float(policy_command[1]),
        "policy_vxy": float(np.linalg.norm(policy_command[:2])),
        "policy_yaw": float(policy_command[2]),
        "speed": float(command_speed_scale(command_source)),
        "imu": estimator_imu_status(estimator),
        "gravity_x": float(gravity[0]),
        "gravity_y": float(gravity[1]),
        "gravity_z": float(gravity[2]),
        "imu_roll_deg": float(np.degrees(imu_roll)),
        "imu_pitch_deg": float(np.degrees(imu_pitch)),
        "imu_correction_abs_max": float(imu_correction_abs_max),
        "act_max": action_abs_max,
        "tau_cmd": tau_cmd_mean,
        "tau_cmd_max": tau_cmd_max,
        "cmds": int(len(commands)),
        "bus_counts": format_bus_counts(bus_counts),
        "loop_dt_ms": None if loop_dt_s is None else 1000.0 * float(loop_dt_s),
        "loop_hz": None if not loop_dt_s else 1.0 / float(loop_dt_s),
        "loop_period_ms": (
            None if loop_period_s is None else 1000.0 * float(loop_period_s)
        ),
        "cycle_work_ms": 1000.0 * float(cycle_work_s),
        "deadline_lateness_ms": 1000.0 * float(deadline_lateness_s),
        "policy_inference_ms": (
            None if policy_inference_s is None else 1000.0 * float(policy_inference_s)
        ),
        "command_input_ms": (
            None if command_input_s is None else 1000.0 * float(command_input_s)
        ),
        "observation_build_ms": (
            None if observation_build_s is None else 1000.0 * float(observation_build_s)
        ),
        "policy_target_conversion_ms": (
            None if policy_target_conversion_s is None else 1000.0 * float(policy_target_conversion_s)
        ),
        "safety_filter_ms": (
            None if safety_filter_s is None else 1000.0 * float(safety_filter_s)
        ),
        "command_build_ms": (
            None if command_build_s is None else 1000.0 * float(command_build_s)
        ),
        "mit_command_build_ms": (
            None if command_build_s is None else 1000.0 * float(command_build_s)
        ),
        "can_tx_ms": None if can_tx_s is None else 1000.0 * float(can_tx_s),
        "feedback_read_ms": (
            None if feedback_read_s is None else 1000.0 * float(feedback_read_s)
        ),
        "pre_feedback_read_ms": (
            None if pre_feedback_read_s is None else 1000.0 * float(pre_feedback_read_s)
        ),
        "feedback_pre_read_ms": (
            None if pre_feedback_read_s is None else 1000.0 * float(pre_feedback_read_s)
        ),
        "steady_feedback_read_ms": (
            None if steady_feedback_read_s is None else 1000.0 * float(steady_feedback_read_s)
        ),
        "feedback_post_read_ms": (
            None if steady_feedback_read_s is None else 1000.0 * float(steady_feedback_read_s)
        ),
        "safety_check_ms": (
            None if safety_check_s is None else 1000.0 * float(safety_check_s)
        ),
        "logging_ms": None if logging_s is None else 1000.0 * float(logging_s),
        "csv_logging_ms": None if logging_s is None else 1000.0 * float(logging_s),
        "terminal_print_ms": (
            None if terminal_print_s is None else 1000.0 * float(terminal_print_s)
        ),
        "imu_cache_read_ms": None if imu_cache_read_s is None else 1000.0 * float(imu_cache_read_s),
        "imu_serial_read_ms": "",
        "feedback_age_max_ms": (
            None if feedback_age_max_s is None else 1000.0 * float(feedback_age_max_s)
        ),
        "feedback_fresh": "" if feedback_fresh_count is None else int(feedback_fresh_count),
        "feedback_current_cycle": (
            "" if feedback_current_cycle_count is None else int(feedback_current_cycle_count)
        ),
        "feedback_previous_cycle": (
            "" if feedback_previous_cycle_count is None else int(feedback_previous_cycle_count)
        ),
        "feedback_missing": "" if feedback_missing_count is None else int(feedback_missing_count),
        "feedback_stale": "" if feedback_stale_count is None else int(feedback_stale_count),
        "missed_deadlines": int(missed_deadlines),
        "consecutive_overruns": int(consecutive_overruns),
        "max_overrun_ms": 1000.0 * float(max_overrun_s),
        "missed_deadlines_total": int(missed_deadlines_total),
        "missed_deadlines_this_cycle": int(missed_deadlines_this_cycle),
        "consecutive_work_overruns": int(consecutive_work_overruns),
        "scheduler_resync_count": int(scheduler_resync_count),
        "max_cycle_work_ms": 1000.0 * float(max_cycle_work_s),
        "max_lateness_ms": 1000.0 * float(max_lateness_s),
        "policy_steady_cycles": int(policy_steady_cycles),
        "policy_entry_scale": float(policy_entry_scale),
        "policy_entry_elapsed_s": float(policy_entry_elapsed_s),
        "policy_entry_restart_count": int(policy_entry_restart_count),
        "policy_entry_restart_reason": str(policy_entry_restart_reason or ""),
        "entry_blend_active": int(bool(entry_blend_active)),
        "policy_torque_limit_start_nm": "",
        "policy_torque_limit_final_nm": "",
        "policy_torque_limit_effective_nm": "",
        "policy_torque_ramp_progress": "",
        "policy_torque_ramp_paused": "",
        "policy_torque_ramp_pause_reason": "",
        "tau_fb": None,
        "tau_fb_max": None,
        "fault_reason": "",
        "policy_joint_order": "" if policy_order is None else ",".join(policy_order),
        "policy_sha256": "" if policy_sha256 is None else str(policy_sha256),
        "tracking_error_max": "",
        "policy_authority_loss_max": "",
    }
    record.update(imu_telemetry_fields(estimator))
    if tau_fb is not None:
        record["tau_fb"] = float(tau_fb[0])
        record["tau_fb_max"] = float(tau_fb[1])

    q_target_sent = q_target
    qd_target_sent = None
    if q_target is not None and policy_order is not None and commands:
        q_target_sent = np.asarray(q_target, dtype=np.float32).copy()
        qd_target_sent = np.zeros(len(policy_order), dtype=np.float32)
        index_by_joint = {name: index for index, name in enumerate(policy_order)}
        for command_item in commands:
            index = index_by_joint.get(command_item.get("joint_name"))
            if index is not None and "q_des" in command_item:
                q_target_sent[index] = float(command_item["q_des"])
            if index is not None and "joint_v_des" in command_item:
                qd_target_sent[index] = float(command_item["joint_v_des"])

    arrays = {
        "obs": (observation, 48, 3),
        "action": (raw_action, 12, 2),
        "sent_action": (sent_action, 12, 2),
        "q": (q_current, 12, 2),
        "qd": (qd_current, 12, 2),
        "q_actor_target": (q_actor_target, 12, 2),
        "q_entry_blended_target": (q_entry_blended_target, 12, 2),
        "q_joint_limit_filtered_target": (q_joint_limit_filtered_target, 12, 2),
        "q_rate_limited_target": (q_rate_limited_target, 12, 2),
        "q_safety_target": (q_safety_target, 12, 2),
        "q_target": (q_target_sent, 12, 2),
        "qd_target": (qd_target_sent, 12, 2),
    }
    for prefix, (values, count, width) in arrays.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (count,):
            continue
        for index, value in enumerate(values):
            record[f"{prefix}_{index:0{width}d}"] = float(value)

    clip_arrays = {
        "policy_clip": policy_target_clip_counts,
        "torque_clip": policy_torque_clip_counts,
    }
    denominator = max(1, int(policy_steady_cycles))
    for prefix, values in clip_arrays.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if values.shape != (12,):
            continue
        for index, count in enumerate(values):
            count = int(count)
            record[f"{prefix}_count_{index:02d}"] = count
            record[f"{prefix}_percent_{index:02d}"] = 100.0 * count / denominator

    for prefix, values in (
        ("target_joint_limited", target_joint_limited),
        ("target_rate_limited", target_rate_limited),
    ):
        if values is None:
            continue
        values = np.asarray(values, dtype=bool).reshape(-1)
        if values.shape != (12,):
            continue
        for index, value in enumerate(values):
            record[f"{prefix}_{index:02d}"] = int(bool(value))

    if policy_order is not None:
        measured_soft_limit_active_by_joint = measured_soft_limit_active_by_joint or {}
        measured_torque_average_by_joint = measured_torque_average_by_joint or {}
        measured_torque_window_max_by_joint = measured_torque_window_max_by_joint or {}
        command_by_joint = {
            command_item.get("joint_name"): command_item
            for command_item in (commands or [])
            if command_item.get("joint_name") is not None
        }
        feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
        now = time.monotonic()

        q_values = None if q_current is None else np.asarray(q_current, dtype=np.float32).reshape(-1)
        qd_values = None if qd_current is None else np.asarray(qd_current, dtype=np.float32).reshape(-1)
        q_target_values = (
            None
            if q_target_sent is None
            else np.asarray(q_target_sent, dtype=np.float32).reshape(-1)
        )
        q_actor_target_values = (
            None
            if q_actor_target is None
            else np.asarray(q_actor_target, dtype=np.float32).reshape(-1)
        )
        q_entry_blended_target_values = (
            None
            if q_entry_blended_target is None
            else np.asarray(q_entry_blended_target, dtype=np.float32).reshape(-1)
        )
        q_joint_limit_filtered_target_values = (
            None
            if q_joint_limit_filtered_target is None
            else np.asarray(q_joint_limit_filtered_target, dtype=np.float32).reshape(-1)
        )
        q_rate_limited_target_values = (
            None
            if q_rate_limited_target is None
            else np.asarray(q_rate_limited_target, dtype=np.float32).reshape(-1)
        )
        q_safety_target_values = (
            None
            if q_safety_target is None
            else np.asarray(q_safety_target, dtype=np.float32).reshape(-1)
        )
        qd_target_values = (
            None
            if qd_target_sent is None
            else np.asarray(qd_target_sent, dtype=np.float32).reshape(-1)
        )
        raw_action_values = (
            None if raw_action is None else np.asarray(raw_action, dtype=np.float32).reshape(-1)
        )
        sent_action_values = (
            None if sent_action is None else np.asarray(sent_action, dtype=np.float32).reshape(-1)
        )
        target_joint_limited_values = (
            None if target_joint_limited is None else np.asarray(target_joint_limited, dtype=bool).reshape(-1)
        )
        target_rate_limited_values = (
            None if target_rate_limited is None else np.asarray(target_rate_limited, dtype=bool).reshape(-1)
        )
        obs_values = (
            None if observation is None else np.asarray(observation, dtype=np.float32).reshape(-1)
        )
        tracking_errors = None
        if (
            q_actor_target_values is not None
            and q_target_values is not None
            and q_values is not None
            and q_actor_target_values.shape == q_target_values.shape == q_values.shape
        ):
            tracking_errors = calculate_tracking_errors(
                q_actor_target_values,
                q_target_values,
                q_values,
            )
            record["tracking_error_max"] = tracking_errors.tracking_error_max
            record["policy_authority_loss_max"] = (
                tracking_errors.policy_authority_loss_max
            )

        for index, joint_name in enumerate(policy_order):
            prefix = str(joint_name)
            command_item = command_by_joint.get(joint_name, {})
            feedback = feedback_by_joint.get(joint_name, {})

            record[f"{prefix}_index"] = int(index)
            if q_values is not None and q_values.shape[0] > index:
                record[f"{prefix}_q_fb"] = float(q_values[index])
            if qd_values is not None and qd_values.shape[0] > index:
                record[f"{prefix}_qd_fb"] = float(qd_values[index])
            if q_actor_target_values is not None and q_actor_target_values.shape[0] > index:
                record[f"{prefix}_q_actor_target"] = float(q_actor_target_values[index])
                record[f"{prefix}_actor_q_target"] = float(q_actor_target_values[index])
            if (
                q_entry_blended_target_values is not None
                and q_entry_blended_target_values.shape[0] > index
            ):
                record[f"{prefix}_entry_blended_q_target"] = float(q_entry_blended_target_values[index])
            if (
                q_joint_limit_filtered_target_values is not None
                and q_joint_limit_filtered_target_values.shape[0] > index
            ):
                record[f"{prefix}_joint_limit_filtered_q_target"] = float(
                    q_joint_limit_filtered_target_values[index]
                )
            if (
                q_rate_limited_target_values is not None
                and q_rate_limited_target_values.shape[0] > index
            ):
                record[f"{prefix}_rate_limited_q_target"] = float(
                    q_rate_limited_target_values[index]
                )
            if q_safety_target_values is not None and q_safety_target_values.shape[0] > index:
                record[f"{prefix}_q_safety_target"] = float(q_safety_target_values[index])
                record[f"{prefix}_safety_filtered_q_target"] = float(q_safety_target_values[index])
            if q_target_values is not None and q_target_values.shape[0] > index:
                record[f"{prefix}_q_target"] = float(q_target_values[index])
                record[f"{prefix}_q_des_transmitted"] = float(q_target_values[index])
                if q_values is not None and q_values.shape[0] > index:
                    record[f"{prefix}_q_error"] = float(q_target_values[index] - q_values[index])
                    record[f"{prefix}_q_tracking_error"] = float(q_target_values[index] - q_values[index])
            if tracking_errors is not None:
                record[f"{prefix}_actor_to_feedback_error"] = float(
                    tracking_errors.actor_to_feedback[index]
                )
                record[f"{prefix}_actor_to_transmitted_error"] = float(
                    tracking_errors.actor_to_transmitted[index]
                )
                record[f"{prefix}_transmitted_to_feedback_error"] = float(
                    tracking_errors.transmitted_to_feedback[index]
                )
            if qd_target_values is not None and qd_target_values.shape[0] > index:
                record[f"{prefix}_qd_target"] = float(qd_target_values[index])
            if raw_action_values is not None and raw_action_values.shape[0] > index:
                record[f"{prefix}_action_raw"] = float(raw_action_values[index])
                record[f"{prefix}_raw_actor_action"] = float(raw_action_values[index])
            if sent_action_values is not None and sent_action_values.shape[0] > index:
                record[f"{prefix}_action_sent"] = float(sent_action_values[index])
            if target_joint_limited_values is not None and target_joint_limited_values.shape[0] > index:
                record[f"{prefix}_target_joint_limited"] = int(bool(target_joint_limited_values[index]))
                record[f"{prefix}_joint_limit_active"] = int(bool(target_joint_limited_values[index]))
            if target_rate_limited_values is not None and target_rate_limited_values.shape[0] > index:
                record[f"{prefix}_target_rate_limited"] = int(bool(target_rate_limited_values[index]))
                record[f"{prefix}_rate_limit_active"] = int(bool(target_rate_limited_values[index]))
            record[f"{prefix}_entry_blend_active"] = int(bool(entry_blend_active))
            if obs_values is not None and obs_values.shape[0] >= 48:
                record[f"{prefix}_obs_joint_pos"] = float(obs_values[12 + index])
                record[f"{prefix}_obs_joint_vel"] = float(obs_values[24 + index])
                record[f"{prefix}_obs_prev_action"] = float(obs_values[36 + index])

            if command_item:
                for key in (
                    "motor_id",
                    "bus_name",
                    "phase",
                    "command_encoding",
                    "q_requested",
                    "q_prelimit_requested",
                    "q_prelimit_hard_limited",
                    "q_des",
                    "q_before_torque_limit",
                    "torque_limited",
                    "impedance_scale",
                    "kp_scale",
                    "kd_scale",
                    "torque_limit_effective",
                    "torque_limit_start",
                    "torque_limit_final",
                    "tau_pd_est",
                    "offset",
                    "direction",
                    "p_des",
                    "p_base",
                    "p_limit_adjustment",
                    "joint_v_des",
                    "joint_v_des_requested",
                    "v_des",
                    "kp",
                    "kd",
                    "kp_effective",
                    "kd_effective",
                    "joint_tau_ff",
                    "joint_tau_ff_effective",
                    "joint_limit_preload_error",
                    "joint_limit_preload_tau_ff_requested",
                    "joint_limit_preload_tau_ff",
                    "tau_ff",
                    "can_id",
                ):
                    value = command_item.get(key)
                    if value is None:
                        continue
                    record[f"{prefix}_cmd_{key}"] = value

            if isinstance(feedback, dict) and feedback:
                timestamp = feedback.get("timestamp")
                try:
                    age_ms = 1000.0 * (now - float(timestamp))
                except (TypeError, ValueError):
                    age_ms = None
                if age_ms is not None and np.isfinite(age_ms):
                    record[f"{prefix}_fb_age_ms"] = float(age_ms)
                    record[f"{prefix}_feedback_age_ms"] = float(age_ms)
                for key in (
                    "comm_type",
                    "motor_id",
                    "bus_name",
                    "fault_bits",
                    "mode_status",
                    "position_raw",
                    "velocity_raw",
                    "torque_raw",
                    "joint_position",
                    "joint_velocity",
                    "joint_velocity_mit",
                    "joint_velocity_finite_difference",
                    "joint_velocity_source",
                    "joint_torque",
                    "position",
                    "velocity",
                    "torque",
                    "temperature_c",
                    "joint_direction",
                ):
                    value = feedback.get(key)
                    if value is None:
                        continue
                    record[f"{prefix}_fb_{key}"] = value
                if feedback.get("joint_torque") is not None:
                    record[f"{prefix}_measured_torque"] = feedback.get("joint_torque")
            record[f"{prefix}_measured_torque_avg"] = measured_torque_average_by_joint.get(
                joint_name,
                "",
            )
            record[f"{prefix}_measured_torque_window_max"] = (
                measured_torque_window_max_by_joint.get(joint_name, "")
            )
            if command_item:
                if command_item.get("tau_pd_est") is not None:
                    record[f"{prefix}_estimated_pd_torque"] = command_item.get("tau_pd_est")
                    measured = feedback.get("joint_torque", feedback.get("torque")) if isinstance(feedback, dict) else None
                    try:
                        denom = abs(float(command_item.get("tau_pd_est")))
                        ratio = 0.0 if denom < 1.0e-6 else abs(float(measured)) / denom
                    except (TypeError, ValueError):
                        ratio = ""
                    record[f"{prefix}_measured_to_estimated_ratio"] = ratio
                if command_item.get("q_before_torque_limit") is not None:
                    record[f"{prefix}_q_before_torque_limit"] = command_item.get("q_before_torque_limit")
                if command_item.get("q_des") is not None:
                    record[f"{prefix}_q_after_torque_limit"] = command_item.get("q_des")
                if command_item.get("torque_limited") is not None:
                    record[f"{prefix}_torque_limited"] = int(bool(command_item.get("torque_limited")))
                    record[f"{prefix}_torque_limit_active"] = int(bool(command_item.get("torque_limited")))
                if command_item.get("torque_limit_effective") is not None:
                    record[f"{prefix}_effective_torque_limit"] = command_item.get("torque_limit_effective")
                    record[f"{prefix}_torque_limit_effective"] = command_item.get("torque_limit_effective")
                if command_item.get("torque_limit_start") is not None:
                    record[f"{prefix}_torque_limit_start"] = command_item.get("torque_limit_start")
                if command_item.get("torque_limit_final") is not None:
                    record[f"{prefix}_torque_limit_final"] = command_item.get("torque_limit_final")
            record[f"{prefix}_measured_soft_limit_active"] = int(
                bool(measured_soft_limit_active_by_joint.get(joint_name, False))
            )

    imu_sensor = getattr(estimator, "imu_sensor", None)
    if imu_sensor is not None and hasattr(imu_sensor, "status"):
        status = imu_sensor.status()
        record["imu_serial_read_ms"] = status.get("last_serial_read_ms", "")
        record["imu_sample_age_ms"] = (
            "" if status.get("age_s") is None else 1000.0 * float(status["age_s"])
        )
        record["imu_sample_rate_hz"] = status.get("sample_rate_hz", "")
        record["imu_packet_count"] = status.get("packet_count", "")
        record["imu_valid_count"] = status.get("valid_count", "")
        record["imu_invalid_count"] = status.get("invalid_count", "")
        record["imu_parse_errors"] = status.get("parse_errors", "")
        record["imu_last_quality_reason"] = status.get("last_quality_reason", "")
    if torque_ramp_state:
        record.update(torque_ramp_state)
    return record


def joint_telemetry_fieldnames(policy_order):
    fields = []
    scalar_fields = [
        "index",
        "q_fb",
        "qd_fb",
        "q_actor_target",
        "actor_q_target",
        "entry_blended_q_target",
        "joint_limit_filtered_q_target",
        "rate_limited_q_target",
        "q_safety_target",
        "safety_filtered_q_target",
        "q_target",
        "q_des_transmitted",
        "qd_target",
        "q_error",
        "q_tracking_error",
        "actor_to_feedback_error",
        "actor_to_transmitted_error",
        "transmitted_to_feedback_error",
        "action_raw",
        "raw_actor_action",
        "action_sent",
        "obs_joint_pos",
        "obs_joint_vel",
        "obs_prev_action",
        "target_joint_limited",
        "target_rate_limited",
        "joint_limit_active",
        "rate_limit_active",
        "entry_blend_active",
        "estimated_pd_torque",
        "measured_torque",
        "measured_to_estimated_ratio",
        "measured_torque_avg",
        "measured_torque_window_max",
        "effective_torque_limit",
        "measured_soft_limit_active",
        "q_before_torque_limit",
        "q_after_torque_limit",
        "torque_limit_effective",
        "torque_limit_start",
        "torque_limit_final",
        "torque_limited",
        "torque_limit_active",
        "feedback_age_ms",
    ]
    command_fields = [
        "motor_id",
        "bus_name",
        "phase",
        "command_encoding",
        "q_requested",
        "q_prelimit_requested",
        "q_prelimit_hard_limited",
        "q_des",
        "q_before_torque_limit",
        "torque_limited",
        "impedance_scale",
        "kp_scale",
        "kd_scale",
        "torque_limit_effective",
        "torque_limit_start",
        "torque_limit_final",
        "tau_pd_est",
        "offset",
        "direction",
        "p_des",
        "p_base",
        "p_limit_adjustment",
        "joint_v_des",
        "joint_v_des_requested",
        "v_des",
        "kp",
        "kd",
        "kp_effective",
        "kd_effective",
        "joint_tau_ff",
        "joint_tau_ff_effective",
        "joint_limit_preload_error",
        "joint_limit_preload_tau_ff_requested",
        "joint_limit_preload_tau_ff",
        "tau_ff",
        "can_id",
    ]
    feedback_fields = [
        "age_ms",
        "comm_type",
        "motor_id",
        "bus_name",
        "fault_bits",
        "mode_status",
        "position_raw",
        "velocity_raw",
        "torque_raw",
        "joint_position",
        "joint_velocity",
        "joint_velocity_mit",
        "joint_velocity_finite_difference",
        "joint_velocity_source",
        "joint_torque",
        "position",
        "velocity",
        "torque",
        "temperature_c",
        "joint_direction",
    ]
    for joint_name in policy_order or []:
        prefix = str(joint_name)
        fields.extend(f"{prefix}_{name}" for name in scalar_fields)
        fields.extend(f"{prefix}_cmd_{name}" for name in command_fields)
        fields.extend(f"{prefix}_fb_{name}" for name in feedback_fields)
    return fields


def compact_telemetry_line(record):
    line = (
        f"step={int(record['step']):06d} "
        f"mode={str(record['mode']):6s} "
        f"vx={float(record['vx']): .3f} "
        f"vy={float(record['vy']): .3f} "
        f"vxy={float(record['vxy']): .3f} "
        f"yaw={float(record['yaw']): .3f} "
        f"speed_scale={float(record['speed']):.3f} "
        f"imu={record['imu']} "
        f"tilt_rp=[{float(record.get('imu_roll_deg', 0.0)):+.1f},"
        f"{float(record.get('imu_pitch_deg', 0.0)):+.1f}]deg "
        f"imu_corr={float(record.get('imu_correction_abs_max', 0.0)):.3f} "
        f"act_max={float(record['act_max']): .3f} "
        f"tau_cmd={float(record['tau_cmd']): .3f} "
        f"tau_cmd_max={float(record['tau_cmd_max']): .3f} "
        f"cmds={int(record['cmds']):02d} "
        f"bus={record.get('bus_counts', 'none')}"
    )
    if record.get("loop_hz") not in (None, ""):
        line += f" hz={float(record['loop_hz']):.1f}"
    if record.get("cycle_work_ms") not in (None, ""):
        line += f" work={float(record['cycle_work_ms']):.1f}ms"
    if float(record.get("deadline_lateness_ms") or 0.0) > 0.0:
        line += f" late={float(record['deadline_lateness_ms']):.1f}ms"
    if record.get("policy_torque_limit_effective_nm") not in (None, ""):
        line += (
            f" torque_limit={float(record['policy_torque_limit_effective_nm']):.1f}Nm"
            f" ramp={100.0 * float(record.get('policy_torque_ramp_progress') or 0.0):.0f}%"
        )
        if int(record.get("policy_torque_ramp_paused") or 0):
            line += f" ramp_paused={record.get('policy_torque_ramp_pause_reason', '')}"
    if int(record.get("consecutive_work_overruns") or 0) > 0:
        line += f" work_ovr={int(record['consecutive_work_overruns'])}"
        line += (
            f" infer={float(record.get('policy_inference_ms') or 0.0):.1f}ms"
            f" build={float(record.get('command_build_ms') or 0.0):.1f}ms"
            f" feedback={float(record.get('steady_feedback_read_ms') or 0.0):.1f}ms"
            f" safety={float(record.get('safety_check_ms') or 0.0):.1f}ms"
            f" logging={float(record.get('logging_ms') or 0.0):.1f}ms"
        )
    if record.get("can_tx_ms") not in (None, ""):
        line += f" tx={float(record['can_tx_ms']):.1f}ms"
    if record.get("feedback_age_max_ms") not in (None, ""):
        line += (
            f" fb_age={float(record['feedback_age_max_ms']):.1f}ms"
            f" fb={record.get('feedback_fresh', '')}"
        )
    if (
        abs(float(record.get("policy_vx", record["vx"])) - float(record["vx"])) > 1e-5
        or abs(float(record.get("policy_vy", record["vy"])) - float(record["vy"])) > 1e-5
        or abs(float(record.get("policy_yaw", record["yaw"])) - float(record["yaw"])) > 1e-5
    ):
        line += (
            f" policy_command=[vx={float(record['policy_vx']): .3f},"
            f"vy={float(record['policy_vy']): .3f},"
            f"yaw={float(record['policy_yaw']): .3f}]"
        )
    else:
        line += (
            f" policy_command=[vx={float(record['policy_vx']): .3f},"
            f"vy={float(record['policy_vy']): .3f},"
            f"yaw={float(record['policy_yaw']): .3f}]"
        )

    if record["tau_fb"] is None:
        line += " tau_fb=na tau_fb_max=na"
    else:
        line += (
            f" tau_fb={float(record['tau_fb']): .3f} "
            f"tau_fb_max={float(record['tau_fb_max']): .3f}"
        )
    return line


def timing_breakdown_line(
    cycle_work_s,
    imu_cache_read_s=0.0,
    command_input_s=0.0,
    observation_build_s=0.0,
    policy_inference_s=0.0,
    policy_target_conversion_s=0.0,
    safety_filter_s=0.0,
    command_build_s=0.0,
    can_tx_s=0.0,
    steady_feedback_read_s=0.0,
    safety_check_s=0.0,
    logging_s=0.0,
    terminal_print_s=0.0,
):
    return (
        "[TIMING] "
        f"work={1000.0 * float(cycle_work_s):.1f}ms "
        f"imu_cache={1000.0 * float(imu_cache_read_s):.1f}ms "
        f"input={1000.0 * float(command_input_s):.1f}ms "
        f"obs={1000.0 * float(observation_build_s):.1f}ms "
        f"infer={1000.0 * float(policy_inference_s):.1f}ms "
        f"target={1000.0 * float(policy_target_conversion_s):.1f}ms "
        f"filter={1000.0 * float(safety_filter_s):.1f}ms "
        f"build={1000.0 * float(command_build_s):.1f}ms "
        f"tx={1000.0 * float(can_tx_s):.1f}ms "
        f"feedback={1000.0 * float(steady_feedback_read_s):.1f}ms "
        f"safety={1000.0 * float(safety_check_s):.1f}ms "
        f"logging={1000.0 * float(logging_s):.1f}ms "
        f"print={1000.0 * float(terminal_print_s):.1f}ms"
    )


def print_compact_telemetry(step, mode, command, command_source, commands, estimator, action=None):
    record = compact_telemetry_record(
        step=step,
        mode=mode,
        command=command,
        command_source=command_source,
        commands=commands,
        estimator=estimator,
        action=action,
    )
    print(compact_telemetry_line(record))


class CsvRunLogger:
    BASE_FIELDNAMES = [
        "run_id",
        "wall_time",
        "elapsed_s",
        "phase",
        "step",
        "mode",
        "vx",
        "vy",
        "vxy",
        "yaw",
        "policy_vx",
        "policy_vy",
        "policy_vxy",
        "policy_yaw",
        "speed",
        "imu",
        "gravity_x",
        "gravity_y",
        "gravity_z",
        "imu_roll_deg",
        "imu_pitch_deg",
        "imu_yaw_deg",
        "imu_gyro_raw_x",
        "imu_gyro_raw_y",
        "imu_gyro_raw_z",
        "imu_gyro_policy_x",
        "imu_gyro_policy_y",
        "imu_gyro_policy_z",
        "imu_quat_w",
        "imu_quat_x",
        "imu_quat_y",
        "imu_quat_z",
        "imu_abs_rpy_roll_deg",
        "imu_abs_rpy_pitch_deg",
        "imu_abs_rpy_yaw_deg",
        "imu_correction_abs_max",
        "act_max",
        "tau_cmd",
        "tau_cmd_max",
        "cmds",
        "bus_counts",
        "loop_dt_ms",
        "loop_hz",
        "loop_period_ms",
        "cycle_work_ms",
        "deadline_lateness_ms",
        "policy_inference_ms",
        "command_input_ms",
        "observation_build_ms",
        "policy_target_conversion_ms",
        "safety_filter_ms",
        "command_build_ms",
        "mit_command_build_ms",
        "can_tx_ms",
        "can_command_dt_ms",
        "can_command_hz",
        "can_command_generation",
        "can_command_send_count",
        "can_feedback_receive_count",
        "can_feedback_last_drain_ms",
        "can_feedback_max_drain_ms",
        "can_command_last_batch_ms",
        "can_command_max_batch_ms",
        "can_command_missed_deadlines",
        "can_command_consecutive_overruns",
        "can_command_scheduler_lateness_ms",
        "can_command_max_scheduler_lateness_ms",
        "can_command_stale_events",
        "can_command_target_age_ms",
        "can_command_fault",
        "feedback_read_ms",
        "pre_feedback_read_ms",
        "feedback_pre_read_ms",
        "steady_feedback_read_ms",
        "feedback_post_read_ms",
        "safety_check_ms",
        "logging_ms",
        "csv_logging_ms",
        "csv_queue_depth",
        "csv_dropped_records",
        "terminal_print_ms",
        "imu_cache_read_ms",
        "imu_serial_read_ms",
        "imu_sample_age_ms",
        "imu_sample_rate_hz",
        "imu_packet_count",
        "imu_valid_count",
        "imu_invalid_count",
        "imu_parse_errors",
        "imu_last_quality_reason",
        "feedback_age_max_ms",
        "feedback_fresh",
        "feedback_current_cycle",
        "feedback_previous_cycle",
        "feedback_missing",
        "feedback_stale",
        "missed_deadlines",
        "consecutive_overruns",
        "max_overrun_ms",
        "missed_deadlines_total",
        "missed_deadlines_this_cycle",
        "consecutive_work_overruns",
        "scheduler_resync_count",
        "max_cycle_work_ms",
        "max_lateness_ms",
        "policy_steady_cycles",
        "policy_entry_scale",
        "policy_entry_elapsed_s",
        "policy_entry_restart_count",
        "policy_entry_restart_reason",
        "entry_blend_active",
        "policy_torque_limit_start_nm",
        "policy_torque_limit_final_nm",
        "policy_torque_limit_effective_nm",
        "policy_torque_ramp_progress",
        "policy_torque_ramp_paused",
        "policy_torque_ramp_pause_reason",
        "tau_fb",
        "tau_fb_max",
        "fault_reason",
        "policy_joint_order",
        "policy_sha256",
        "tracking_error_max",
        "policy_authority_loss_max",
        "compact_line",
        "command_line",
        "runtime_control_hz",
        "policy_action_scale",
        "policy_action_formula",
        "policy_frame_origin",
        "exact_policy_after_entry",
        "previous_action_source",
        "can_topology",
        "can_backend",
        "torque_profile_stage",
        "policy_torque_start_max_nm",
        "policy_torque_final_max_nm",
        "policy_torque_ramp_delay_s",
        "policy_torque_ramp_seconds",
        "pose_test_only",
        "pose_gains_config",
    ]
    BASE_FIELDNAMES += [f"obs_{index:03d}" for index in range(48)]
    BASE_FIELDNAMES += [f"action_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"sent_action_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"qd_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_actor_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_entry_blended_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_joint_limit_filtered_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_rate_limited_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_safety_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"qd_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"target_joint_limited_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"target_rate_limited_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"policy_clip_count_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"policy_clip_percent_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"torque_clip_count_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"torque_clip_percent_{index:02d}" for index in range(12)]

    @staticmethod
    def _unique_log_path(directory, stem):
        candidate = directory / f"{stem}.csv"
        if not candidate.exists():
            return candidate

        repeat = 2
        while True:
            candidate = directory / f"{stem}_repeat{repeat:02d}.csv"
            if not candidate.exists():
                return candidate
            repeat += 1

    def __init__(
        self,
        enabled=True,
        log_dir=None,
        log_file=None,
        policy_order=None,
        flush_every=1,
        run_metadata=None,
        log_prefix="grallator_run",
        async_enabled=True,
        queue_size=500,
        flush_seconds=1.0,
    ):
        self.enabled = bool(enabled)
        self.path = None
        self.run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        self.start_time = time.monotonic()
        self._file = None
        self._writer = None
        self.flush_every = max(1, int(flush_every))
        self._rows_since_flush = 0
        self.last_log_duration_s = 0.0
        self.async_enabled = bool(async_enabled)
        self.queue_size = max(1, int(queue_size))
        self.flush_seconds = max(0.05, float(flush_seconds))
        self.dropped_records = 0
        self._queue = None
        self._stop_event = threading.Event()
        self._worker = None
        self._last_flush_time = time.monotonic()
        self.run_metadata = dict(run_metadata or {})
        self.fieldnames = list(self.BASE_FIELDNAMES)
        self.fieldnames.extend(joint_telemetry_fieldnames(policy_order or []))

        if not self.enabled:
            return

        if log_file:
            self.path = Path(log_file).expanduser()
            self.run_id = self.path.stem
        else:
            directory = Path(log_dir).expanduser() if log_dir else ROOT / "logs"
            safe_prefix = "".join(
                char for char in str(log_prefix) if char.isalnum() or char in "_.-"
            ).strip("._-")
            if not safe_prefix:
                raise ValueError("log_prefix must contain a filename-safe character")
            self.path = self._unique_log_path(directory, f"{safe_prefix}_{self.run_id}")
            self.run_id = self.path.stem.removeprefix(f"{safe_prefix}_")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._file.flush()
        if self.async_enabled:
            self._queue = queue.Queue(maxsize=self.queue_size)
            self._worker = threading.Thread(
                target=self._run_worker,
                name="AsyncCsvWriter",
                daemon=True,
            )
            self._worker.start()

    def _row_from_record(self, record):
        record = dict(record)
        captured_monotonic = float(
            record.pop("_csv_capture_monotonic", time.monotonic())
        )
        captured_wall_time = str(
            record.pop(
                "_csv_capture_wall_time",
                datetime.now().isoformat(timespec="milliseconds"),
            )
        )
        row = {field: "" for field in self.fieldnames}
        row.update(self.run_metadata)
        row.update(record)
        if "csv_logging_ms" in row and row["csv_logging_ms"] in ("", None, 0.0):
            row["csv_logging_ms"] = 1000.0 * float(self.last_log_duration_s)
        if row["csv_queue_depth"] in ("", None):
            row["csv_queue_depth"] = self.queue_depth()
        if row["csv_dropped_records"] in ("", None):
            row["csv_dropped_records"] = int(self.dropped_records)
        row["run_id"] = self.run_id
        row["wall_time"] = captured_wall_time
        row["elapsed_s"] = f"{captured_monotonic - self.start_time:.6f}"
        row["compact_line"] = compact_telemetry_line(record)
        if row["tau_fb"] is None:
            row["tau_fb"] = ""
        if row["tau_fb_max"] is None:
            row["tau_fb_max"] = ""
        return row

    def _write_record(self, record):
        started = time.monotonic()
        row = self._row_from_record(record)
        self._writer.writerow(row)
        self._rows_since_flush += 1
        now = time.monotonic()
        if (
            self._rows_since_flush >= self.flush_every
            or now - self._last_flush_time >= self.flush_seconds
        ):
            self._file.flush()
            self._rows_since_flush = 0
            self._last_flush_time = now
        self.last_log_duration_s = time.monotonic() - started
        return self.last_log_duration_s

    def _run_worker(self):
        while not self._stop_event.is_set() or (
            self._queue is not None and not self._queue.empty()
        ):
            try:
                record = self._queue.get(timeout=0.05)
            except queue.Empty:
                if (
                    self._file is not None
                    and time.monotonic() - self._last_flush_time >= self.flush_seconds
                ):
                    self._file.flush()
                    self._rows_since_flush = 0
                    self._last_flush_time = time.monotonic()
                continue
            try:
                self._write_record(record)
            finally:
                self._queue.task_done()

    def queue_depth(self):
        if self._queue is None:
            return 0
        return int(self._queue.qsize())

    def submit(self, record):
        if not self.enabled or self._writer is None:
            return 0.0
        stamped_record = dict(record)
        stamped_record.setdefault("_csv_capture_monotonic", time.monotonic())
        stamped_record.setdefault(
            "_csv_capture_wall_time",
            datetime.now().isoformat(timespec="milliseconds"),
        )
        stamped_record["csv_queue_depth"] = self.queue_depth()
        stamped_record["csv_dropped_records"] = int(self.dropped_records)
        if not self.async_enabled:
            return self._write_record(stamped_record)
        started = time.monotonic()
        try:
            self._queue.put_nowait(stamped_record)
        except queue.Full:
            self.dropped_records += 1
        return time.monotonic() - started

    def log(self, record):
        return self.submit(record)

    def close(self):
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def publish_run_log_to_git(log_path, remote="origin", timeout_s=20.0):
    """Commit and push exactly one completed run log after motor shutdown."""
    if log_path is None:
        return False, "no CSV log was created"

    log_path = Path(log_path).expanduser().resolve()
    log_root = (ROOT / "logs").resolve()
    try:
        relative_path = log_path.relative_to(ROOT.resolve())
        log_path.relative_to(log_root)
    except ValueError:
        return False, f"refusing to publish a log outside {log_root}"

    if not log_path.is_file():
        return False, f"log file does not exist: {log_path}"

    def run_git(*arguments, check=True):
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
            check=check,
        )

    try:
        branch = run_git("branch", "--show-current").stdout.strip()
        if not branch:
            return False, "repository is in detached HEAD state"

        upstream = run_git(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            check=False,
        )
        base_ref = upstream.stdout.strip() if upstream.returncode == 0 else ""
        if not base_ref:
            candidate = f"{remote}/{branch}"
            exists = run_git("rev-parse", "--verify", candidate, check=False)
            if exists.returncode == 0:
                base_ref = candidate
        if not base_ref:
            return False, "no upstream or remote-tracking branch is configured"

        ahead_paths = run_git(
            "diff",
            "--name-only",
            f"{base_ref}..HEAD",
        ).stdout.splitlines()
        non_log_paths = [
            path for path in ahead_paths
            if not path.replace("\\", "/").startswith("logs/")
        ]
        if non_log_paths:
            return False, (
                "branch has unpushed non-log commits; refusing an automatic "
                "push: " + ", ".join(non_log_paths[:4])
            )
        if ahead_paths:
            print(
                "CSV log Git push: retrying existing unpushed log-only "
                f"commit(s): {len(ahead_paths)} file(s)"
            )

        run_git("add", "-f", "--", str(relative_path))
        staged = run_git(
            "diff",
            "--cached",
            "--quiet",
            "--",
            str(relative_path),
            check=False,
        )
        if staged.returncode == 0:
            return False, "log is unchanged"
        if staged.returncode != 1:
            return False, staged.stderr.strip() or "could not inspect staged log"

        run_git(
            "commit",
            "-m",
            f"log: {log_path.name}",
            "--",
            str(relative_path),
        )
        run_git("push", str(remote), f"HEAD:{branch}")
        return True, f"pushed {relative_path} to {remote}/{branch}"
    except subprocess.TimeoutExpired:
        return False, f"git operation timed out after {float(timeout_s):.0f}s"
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = stderr.strip() or str(exc)
        return False, detail


def print_joystick_debug(command_source):
    if not hasattr(command_source, "raw_state"):
        return

    raw = command_source.raw_state()
    if raw is None:
        return

    axes = " ".join(
        f"{axis_id}:{value:+.3f}"
        for axis_id, value in enumerate(raw.get("axes", []))
    )
    buttons = " ".join(
        f"{button_id}:{value}"
        for button_id, value in enumerate(raw.get("buttons", []))
    )
    hats = " ".join(
        f"{hat_id}:{value}"
        for hat_id, value in enumerate(raw.get("hats", []))
    )
    print(f"  raw_axes=[{axes}] raw_buttons=[{buttons}] raw_hats=[{hats}]")


def print_joint_debug(commands, estimator):
    if not commands:
        return

    q_current = getattr(estimator, "q_current", None)
    qd_current = getattr(estimator, "qd_current", None)
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {})

    for cmd in commands:
        joint_name = cmd["joint_name"]
        index = None
        if q_current is not None and hasattr(estimator, "joint_index_by_name"):
            index = estimator.joint_index_by_name.get(joint_name)

        q_fb = None if index is None else float(q_current[index])
        qd_fb = None if index is None or qd_current is None else float(qd_current[index])
        tau_fb = None
        if joint_name in feedback_by_joint:
            tau_fb = float(feedback_by_joint[joint_name]["torque"])

        q_err = None if q_fb is None else float(cmd["q_des"] - q_fb)
        target_motion = abs(float(cmd["q_requested"])) > 0.02 or abs(float(cmd["q_des"])) > 0.02
        status = "no-feedback" if joint_name not in feedback_by_joint else "tracking"
        if not target_motion:
            status = "target-zero" if status != "no-feedback" else "target-zero/no-feedback"
        elif q_err is not None:
            if abs(q_err) <= 0.06:
                status = "tracking"
            elif qd_fb is not None and abs(qd_fb) < 0.02:
                status = "slow/stuck"
            else:
                status = "lagging"
        if abs(float(cmd["q_requested"]) - float(cmd["q_des"])) > 1e-5:
            status += "/clipped"

        line = (
            f"  joint={joint_name:16s} "
            f"id=0x{int(cmd['motor_id']):02X} "
            f"bus={cmd.get('bus_name', 'front'):5s} "
            f"status={status:20s} "
            f"q_req={cmd['q_requested']:+.3f} "
            f"q_des={cmd['q_des']:+.3f}"
        )
        if q_fb is not None:
            line += f" q_fb={q_fb:+.3f} q_err={q_err:+.3f}"
        if qd_fb is not None:
            line += f" qd_fb={qd_fb:+.3f}"
        line += f" qd_des={float(cmd.get('joint_v_des', 0.0)):+.3f}"
        if tau_fb is not None:
            line += f" tau_fb={tau_fb:+.3f}"
        print(line)


def init_policy_joint_summary(policy_order):
    n = len(policy_order)
    inf = np.full(n, np.inf, dtype=np.float64)
    neg_inf = np.full(n, -np.inf, dtype=np.float64)
    return {
        "cycles": np.zeros(n, dtype=np.int64),
        "actor_min": inf.copy(),
        "actor_max": neg_inf.copy(),
        "sent_min": inf.copy(),
        "sent_max": neg_inf.copy(),
        "measured_min": inf.copy(),
        "measured_max": neg_inf.copy(),
        "tracking_error_sq": np.zeros(n, dtype=np.float64),
        "tracking_error_abs_max": np.zeros(n, dtype=np.float64),
        "joint_limited": np.zeros(n, dtype=np.int64),
        "rate_limited": np.zeros(n, dtype=np.int64),
        "torque_limited": np.zeros(n, dtype=np.int64),
        "max_estimated_torque": np.zeros(n, dtype=np.float64),
        "max_measured_torque": np.zeros(n, dtype=np.float64),
        "max_feedback_age_ms": np.zeros(n, dtype=np.float64),
    }


def update_policy_joint_summary(
    summary,
    policy_order,
    commands,
    estimator,
    q_actor_target,
    q_des_transmitted,
    target_joint_limited=None,
    target_rate_limited=None,
):
    if q_actor_target is None or q_des_transmitted is None:
        return

    q_actor_target = np.asarray(q_actor_target, dtype=np.float32).reshape(-1)
    q_des_transmitted = np.asarray(q_des_transmitted, dtype=np.float32).reshape(-1)
    q_current = np.asarray(getattr(estimator, "q_current", []), dtype=np.float32).reshape(-1)
    if q_actor_target.shape[0] != len(policy_order) or q_des_transmitted.shape[0] != len(policy_order):
        return
    if q_current.shape[0] != len(policy_order):
        return

    target_joint_limited = (
        np.zeros(len(policy_order), dtype=bool)
        if target_joint_limited is None
        else np.asarray(target_joint_limited, dtype=bool).reshape(-1)
    )
    target_rate_limited = (
        np.zeros(len(policy_order), dtype=bool)
        if target_rate_limited is None
        else np.asarray(target_rate_limited, dtype=bool).reshape(-1)
    )
    command_by_joint = {
        item.get("joint_name"): item
        for item in (commands or [])
        if item.get("joint_name") is not None
    }
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
    now = time.monotonic()

    for index, joint_name in enumerate(policy_order):
        summary["cycles"][index] += 1
        summary["actor_min"][index] = min(summary["actor_min"][index], float(q_actor_target[index]))
        summary["actor_max"][index] = max(summary["actor_max"][index], float(q_actor_target[index]))
        summary["sent_min"][index] = min(summary["sent_min"][index], float(q_des_transmitted[index]))
        summary["sent_max"][index] = max(summary["sent_max"][index], float(q_des_transmitted[index]))
        summary["measured_min"][index] = min(summary["measured_min"][index], float(q_current[index]))
        summary["measured_max"][index] = max(summary["measured_max"][index], float(q_current[index]))

        error = float(q_des_transmitted[index] - q_current[index])
        summary["tracking_error_sq"][index] += error * error
        summary["tracking_error_abs_max"][index] = max(
            summary["tracking_error_abs_max"][index],
            abs(error),
        )
        if target_joint_limited.shape[0] == len(policy_order) and target_joint_limited[index]:
            summary["joint_limited"][index] += 1
        if target_rate_limited.shape[0] == len(policy_order) and target_rate_limited[index]:
            summary["rate_limited"][index] += 1

        command = command_by_joint.get(joint_name, {})
        if command.get("torque_limited"):
            summary["torque_limited"][index] += 1
        tau_est = command.get("tau_pd_est")
        if tau_est is not None:
            try:
                tau_est = abs(float(tau_est))
                if np.isfinite(tau_est):
                    summary["max_estimated_torque"][index] = max(
                        summary["max_estimated_torque"][index],
                        tau_est,
                    )
            except (TypeError, ValueError):
                pass

        feedback = feedback_by_joint.get(joint_name, {})
        if isinstance(feedback, dict):
            tau_measured = feedback.get("joint_torque", feedback.get("torque"))
            try:
                tau_measured = abs(float(tau_measured))
                if np.isfinite(tau_measured):
                    summary["max_measured_torque"][index] = max(
                        summary["max_measured_torque"][index],
                        tau_measured,
                    )
            except (TypeError, ValueError):
                pass
            timestamp = feedback.get("timestamp")
            try:
                age_ms = 1000.0 * (now - float(timestamp))
                if np.isfinite(age_ms):
                    summary["max_feedback_age_ms"][index] = max(
                        summary["max_feedback_age_ms"][index],
                        age_ms,
                    )
            except (TypeError, ValueError):
                pass


def print_policy_joint_summary(summary, policy_order, steady_cycles=0, steady_torque_counts=None):
    total_cycles = np.asarray(summary["cycles"], dtype=np.int64)
    if not np.any(total_cycles > 0):
        return

    print("\nPOLICY JOINT TRACKING SUMMARY")
    print("-" * 80)
    steady_torque_counts = (
        np.asarray(summary["torque_limited"], dtype=np.int64)
        if steady_torque_counts is None
        else np.asarray(steady_torque_counts, dtype=np.int64)
    )
    steady_denominator = max(1, int(steady_cycles))
    for index, joint_name in enumerate(policy_order):
        cycles = int(total_cycles[index])
        if cycles <= 0:
            continue
        rms = float(np.sqrt(summary["tracking_error_sq"][index] / max(1, cycles)))
        joint_pct = 100.0 * int(summary["joint_limited"][index]) / max(1, cycles)
        rate_pct = 100.0 * int(summary["rate_limited"][index]) / max(1, cycles)
        torque_pct = 100.0 * int(summary["torque_limited"][index]) / max(1, cycles)
        steady_torque_pct = 100.0 * int(steady_torque_counts[index]) / steady_denominator
        actor_amp = float(summary["actor_max"][index] - summary["actor_min"][index])
        transmitted_amp = float(summary["sent_max"][index] - summary["sent_min"][index])
        measured_amp = float(summary["measured_max"][index] - summary["measured_min"][index])
        target_to_transmitted_ratio = (
            transmitted_amp / actor_amp if actor_amp > 1.0e-6 else 1.0
        )
        transmitted_to_measured_ratio = (
            measured_amp / transmitted_amp if transmitted_amp > 1.0e-6 else 1.0
        )
        print(f"{joint_name}:")
        print(f"  policy cycles: {cycles}")
        print(
            "  actor target range: "
            f"[{summary['actor_min'][index]:+.3f}, {summary['actor_max'][index]:+.3f}] rad"
        )
        print(
            "  transmitted range: "
            f"[{summary['sent_min'][index]:+.3f}, {summary['sent_max'][index]:+.3f}] rad"
        )
        print(
            "  measured range: "
            f"[{summary['measured_min'][index]:+.3f}, {summary['measured_max'][index]:+.3f}] rad"
        )
        print(f"  RMS tracking error: {rms:.3f} rad")
        print(f"  maximum tracking error: {summary['tracking_error_abs_max'][index]:.3f} rad")
        print(f"  joint limited: {joint_pct:.1f}%")
        print(f"  rate limited: {rate_pct:.1f}%")
        print(f"  torque limited: {torque_pct:.1f}%")
        print(f"  maximum estimated torque: {summary['max_estimated_torque'][index]:.3f} Nm")
        print(f"  maximum measured torque: {summary['max_measured_torque'][index]:.3f} Nm")
        print(f"  maximum feedback age: {summary['max_feedback_age_ms'][index]:.1f} ms")
        print(f"  actor->transmitted amplitude ratio: {target_to_transmitted_ratio:.2f}")
        print(f"  transmitted->measured amplitude ratio: {transmitted_to_measured_ratio:.2f}")
        if steady_torque_pct > 20.0 or target_to_transmitted_ratio < 0.70 or rms > 0.15:
            print(
                "[POLICY AUTHORITY] "
                f"{joint_name} torque limited in {steady_torque_pct:.1f}% of steady policy cycles. "
                f"Actor amplitude: {actor_amp:.3f} rad. "
                f"Transmitted amplitude: {transmitted_amp:.3f} rad. "
                f"Measured amplitude: {measured_amp:.3f} rad. "
                f"RMS tracking error: {rms:.3f} rad. "
                "Do not ground-test if routing, IMU timing, encoder limits, or tracking are not clean. "
                "Consider only a controlled 18 Nm suspended comparison after the 14 Nm plots pass."
            )


def request_feedback_snapshot(motor_layer, buses, mode):
    """Poll encoder feedback with RobStride management frames.

    RobStride comm_type_stop is also used by the simple encoder reader as a
    poll frame, but it can briefly drop MIT torque. Use this only before active
    control starts, never during sit/stand/walk transitions.
    """
    if mode == "mit-signal" and buses is not None:
        motor_layer.send_raw_commands(buses, motor_layer.build_feedback_poll_commands())
        time.sleep(0.002)
        motor_layer.send_raw_commands(buses, motor_layer.build_enable_commands())


def count_active_feedback(estimator, active_joints):
    feedback = getattr(estimator, "last_feedback_by_joint", {}) or {}
    return sum(1 for joint_name in active_joints if joint_name in feedback)


def count_fresh_active_feedback(estimator, active_joints, max_age_s):
    feedback = getattr(estimator, "last_feedback_by_joint", {}) or {}
    now = time.monotonic()
    count = 0
    for joint_name in active_joints:
        item = feedback.get(joint_name)
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp")
        try:
            age = now - float(timestamp)
        except (TypeError, ValueError):
            continue
        if np.isfinite(age) and age <= float(max_age_s):
            count += 1
    return count


def run_calf_range_check(
    runner,
    safety,
    motor_layer,
    estimator,
    buses,
    mode,
    feedback_timeout=0.10,
    samples=8,
    sample_period_s=0.04,
):
    """Read-only calf encoder/transmission diagnostic.

    This intentionally sends only feedback poll frames. It never enables MIT
    torque and never builds policy or pose commands.
    """
    if mode != "mit-signal" or buses is None:
        print("ERROR: --calf-range-check requires --mode mit-signal with real CAN buses.")
        return False

    calf_joints = [
        joint_name
        for joint_name in runner.policy_order
        if "calf" in joint_name and joint_name in motor_layer.active_joints
    ]
    if not calf_joints:
        print("ERROR: --calf-range-check found no active calf joints.")
        return False

    expected_bus_motor_ids = active_feedback_bus_motor_ids(
        estimator,
        motor_layer,
        active_joints=calf_joints,
    )
    q_samples = {joint_name: [] for joint_name in calf_joints}

    print("\n################################################################################")
    print("CALF RANGE CHECK: READ ONLY")
    print("################################################################################")
    print("No motor enable frames, MIT commands, pose commands, or policy commands are sent.")
    print("This validates the current converted calf feedback and four-bar range.")

    for _ in range(max(1, int(samples))):
        try:
            motor_layer.send_raw_commands(buses, motor_layer.build_feedback_poll_commands())
            refresh_estimator_feedback(
                estimator,
                timeout=float(feedback_timeout),
                expected_bus_motor_ids=expected_bus_motor_ids,
            )
        except Exception as exc:
            print("\nERROR: calf feedback decode failed:", exc)
            print(
                "If this is a four-bar range error, expand or recalibrate the measured "
                "calf lookup before walking. For raw capture, use check_motor_connections "
                "with four-bar disabled."
            )
            return False

        feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
        for joint_name in calf_joints:
            feedback = feedback_by_joint.get(joint_name)
            if isinstance(feedback, dict):
                q_samples[joint_name].append(float(feedback.get("position", 0.0)))
        time.sleep(max(0.0, float(sample_period_s)))

    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
    now = time.monotonic()
    print()
    print(
        "Joint                |   Bus | Motor | State     | Age ms | "
        "Joint rad | Raw rad | Offset | Dir | Motor th | J=dq/dth | "
        "Torque | Hard margin | Policy margin | 4bar | Fault"
    )
    print("-" * 174)

    ok = True
    for joint_name in calf_joints:
        index = runner.policy_order.index(joint_name)
        motor_id = int(motor_layer.motor_ids[joint_name])
        bus_name = motor_layer.joint_can_bus.get(joint_name, "can0")
        feedback = feedback_by_joint.get(joint_name)
        if not isinstance(feedback, dict):
            ok = False
            print(
                f"{joint_name:20s} | {bus_name:>5s} | 0x{motor_id:02X}  | "
                "MISSING   |     na |       na |      na |     na |  na |"
                "       na |       na |     na |          na |           na |   na | na"
            )
            continue

        age_ms = 1000.0 * max(0.0, now - float(feedback.get("timestamp", now)))
        joint_q = float(feedback.get("position", feedback.get("joint_position", 0.0)))
        raw_q = float(feedback.get("position_raw", feedback.get("position", 0.0)))
        offset = float(motor_layer.joint_offsets.get(joint_name, 0.0))
        direction = float(motor_layer.joint_directions.get(joint_name, 1.0))
        motor_q = float(feedback.get("motor_position", joint_q))
        jacobian = float(feedback.get("transmission_jacobian", 1.0))
        torque = float(feedback.get("torque", feedback.get("joint_torque", 0.0)))
        fault = int(feedback.get("fault", feedback.get("fault_bits", 0)) or 0)
        fourbar = bool(feedback.get("transmission_enabled", False))

        hard_margin = min(
            float(joint_q - safety.q_min[index]),
            float(safety.q_max[index] - joint_q),
        )
        policy_margin = min(
            float(joint_q - safety.policy_q_min[index]),
            float(safety.policy_q_max[index] - joint_q),
        )
        if hard_margin < 0.0 or policy_margin < 0.0:
            ok = False
        repeatability = 0.0
        if len(q_samples[joint_name]) >= 2:
            repeatability = max(q_samples[joint_name]) - min(q_samples[joint_name])
        if repeatability > 0.02:
            ok = False

        print(
            f"{joint_name:20s} | {bus_name:>5s} | 0x{motor_id:02X}  | "
            f"CONNECTED | {age_ms:6.1f} | {joint_q:+9.4f} | {raw_q:+7.4f} | "
            f"{offset:+6.3f} | {direction:+3.0f} | {motor_q:+8.4f} | "
            f"{jacobian:+8.4f} | {torque:+6.2f} | {hard_margin:+11.4f} | "
            f"{policy_margin:+13.4f} | "
            f"{'yes' if fourbar else ' no':>4s} | 0x{fault:02X}"
        )
        if repeatability > 0.0:
            print(f"  repeatability window for {joint_name}: {repeatability:.5f} rad")

    print("-" * 174)
    print(
        "Calf range check:",
        "PASS" if ok else "CHECK REQUIRED",
        "- all values above are converted joint radians used by policy/control.",
    )
    return ok


def fresh_feedback_by_joint(estimator, active_joints, max_age_s):
    feedback = getattr(estimator, "last_feedback_by_joint", {}) or {}
    now = time.monotonic()
    fresh = {}
    missing = []
    for joint_name in active_joints:
        item = feedback.get(joint_name)
        if not isinstance(item, dict):
            missing.append(joint_name)
            continue
        timestamp = item.get("timestamp")
        try:
            age = now - float(timestamp)
        except (TypeError, ValueError):
            missing.append(joint_name)
            continue
        if np.isfinite(age) and age <= float(max_age_s):
            fresh[joint_name] = item
        else:
            missing.append(joint_name)
    return fresh, missing


def feedback_recency_summary(estimator, active_joints, max_age_s):
    feedback = getattr(estimator, "last_feedback_by_joint", {}) or {}
    now = time.monotonic()
    command_timestamp = getattr(estimator, "last_command_send_timestamp", None)
    current_cycle = 0
    previous_cycle = 0
    stale = 0
    missing = 0
    for joint_name in active_joints:
        item = feedback.get(joint_name)
        if not isinstance(item, dict):
            missing += 1
            continue
        timestamp = item.get("timestamp")
        try:
            timestamp = float(timestamp)
            age = now - timestamp
        except (TypeError, ValueError):
            missing += 1
            continue
        if not np.isfinite(age) or age > float(max_age_s):
            stale += 1
            continue
        if command_timestamp is not None and timestamp >= float(command_timestamp):
            current_cycle += 1
        else:
            previous_cycle += 1
    return {
        "fresh_current_cycle": int(current_cycle),
        "fresh_previous_cycle": int(previous_cycle),
        "stale": int(stale),
        "missing": int(missing),
    }


def fresh_active_feedback_names(estimator, active_joints, max_age_s):
    feedback = getattr(estimator, "last_feedback_by_joint", {}) or {}
    now = time.monotonic()
    fresh = set()
    stale_or_missing = []
    for joint_name in active_joints:
        item = feedback.get(joint_name)
        if not isinstance(item, dict):
            stale_or_missing.append(joint_name)
            continue
        timestamp = item.get("timestamp")
        try:
            age = now - float(timestamp)
        except (TypeError, ValueError):
            stale_or_missing.append(joint_name)
            continue
        if np.isfinite(age) and age <= float(max_age_s):
            fresh.add(joint_name)
        else:
            stale_or_missing.append(joint_name)
    return fresh, stale_or_missing


def refresh_active_feedback_before_fault(
    estimator,
    motor_layer,
    safety,
    buses,
    mode,
    feedback_timeout,
    q_keepalive=None,
    phase="startup",
    allow_poll_snapshot=False,
    max_wait_s=0.12,
    max_age_s=None,
    can_streamer=None,
):
    """Try to refresh active motor feedback before declaring a stale fault.

    During live MIT control we keep torque on by sending another safe MIT target
    instead of using RobStride stop/poll frames. This avoids false stops from a
    few milliseconds of serial/CAN scheduling jitter while still failing if a
    motor really stops replying.
    """
    if not encoder_feedback_required(mode, estimator):
        refresh_estimator_feedback(estimator, timeout=feedback_timeout)
        active_count = len(motor_layer.active_joints)
        return active_count, active_count

    active_joints = list(motor_layer.active_joints)
    n_active = len(active_joints)
    expected_feedback_ids = active_feedback_bus_motor_ids(
        estimator,
        motor_layer,
        active_joints,
    )
    max_age_s = (
        getattr(safety, "max_feedback_age_s", 0.25)
        if max_age_s is None
        else float(max_age_s)
    )
    deadline = time.monotonic() + max(float(max_wait_s), float(feedback_timeout), 0.02)

    refresh_estimator_feedback(
        estimator,
        timeout=0.0,
        expected_bus_motor_ids=expected_feedback_ids,
    )
    fresh = count_fresh_active_feedback(estimator, active_joints, max_age_s)
    while fresh < n_active and time.monotonic() < deadline:
        if mode == "mit-signal" and buses is not None:
            if q_keepalive is not None:
                feedback_for_keepalive, _ = fresh_feedback_by_joint(
                    estimator,
                    active_joints,
                    max_age_s,
                )
                keepalive_commands = motor_layer.build_mit_commands(
                    q_keepalive,
                    phase=phase,
                    feedback_by_joint=feedback_for_keepalive,
                )
                if can_streamer is not None:
                    can_streamer.submit(keepalive_commands)
                else:
                    motor_layer.send_signal_commands(buses, keepalive_commands)
            elif allow_poll_snapshot:
                request_feedback_snapshot(motor_layer, buses, mode)

        remaining = max(0.0, deadline - time.monotonic())
        refresh_estimator_feedback(
            estimator,
            timeout=min(float(feedback_timeout), remaining),
            expected_bus_motor_ids=expected_feedback_ids,
        )
        fresh = count_fresh_active_feedback(estimator, active_joints, max_age_s)
        if fresh >= n_active:
            break
        time.sleep(0.002)

    return fresh, n_active


def acquire_hold_target_from_feedback(
    estimator,
    motor_layer,
    safety,
    q_previous_target,
    feedback_timeout,
    capture_seconds,
    buses,
    mode,
    allow_poll_snapshot=False,
):
    """Build a hold target from fresh per-joint feedback only.

    If a motor does not provide fresh feedback within the short capture window,
    keep its previous command target instead of copying a stale q_current value.
    """
    active_joints = list(motor_layer.active_joints)
    max_age_s = getattr(safety, "max_feedback_age_s", 0.25)
    capture_seconds = max(float(feedback_timeout), float(capture_seconds), 0.02)
    deadline = time.monotonic() + capture_seconds

    if allow_poll_snapshot:
        request_feedback_snapshot(motor_layer, buses, mode)

    fresh_names = set()
    stale_or_missing = list(active_joints)
    while time.monotonic() < deadline:
        if mode == "mit-signal" and buses is not None:
            feedback_for_keepalive, _ = fresh_feedback_by_joint(
                estimator,
                active_joints,
                max_age_s,
            )
            keepalive_commands = motor_layer.build_mit_commands(
                q_previous_target,
                phase="startup",
                feedback_by_joint=feedback_for_keepalive,
            )
            motor_layer.send_signal_commands(buses, keepalive_commands)

        remaining = max(0.0, deadline - time.monotonic())
        refresh_estimator_feedback(
            estimator,
            timeout=min(float(feedback_timeout), remaining),
        )
        fresh_names, stale_or_missing = fresh_active_feedback_names(
            estimator,
            active_joints,
            max_age_s,
        )
        if len(fresh_names) >= len(active_joints):
            break
        time.sleep(0.002)

    refresh_estimator_feedback(estimator, timeout=0.0)
    q_current, qd_current, base_lin_vel_b, base_ang_vel_b, projected_gravity_b = estimator.read()
    q_hold = np.asarray(q_previous_target, dtype=np.float32).copy()
    index_by_joint = getattr(
        estimator,
        "joint_index_by_name",
        {name: index for index, name in enumerate(motor_layer.policy_order)},
    )
    for joint_name in fresh_names:
        index = index_by_joint.get(joint_name)
        if index is not None:
            q_hold[index] = q_current[index]

    return (
        q_hold,
        len(fresh_names),
        stale_or_missing,
        q_current,
        qd_current,
        base_lin_vel_b,
        base_ang_vel_b,
        projected_gravity_b,
    )


def snapshot_hold_target_from_fresh_feedback(
    estimator,
    motor_layer,
    safety,
    q_previous_target,
    q_current,
):
    """Capture H from the fresh feedback already read for this control cycle.

    The 200 Hz CAN worker must keep streaming while the 50 Hz loop changes
    modes. Waiting for another complete feedback set after clearing the worker
    lets the cached frames age out and can turn a normal H request into an
    encoder-stale fault. Joints without fresh feedback retain their last sent
    target.
    """
    active_joints = list(motor_layer.active_joints)
    max_age_s = getattr(safety, "max_feedback_age_s", 0.25)
    fresh_names, _ = fresh_active_feedback_names(
        estimator,
        active_joints,
        max_age_s,
    )
    q_hold = np.asarray(q_previous_target, dtype=np.float32).copy()
    q_measured = np.asarray(q_current, dtype=np.float32).reshape(-1)
    if q_measured.shape != q_hold.shape:
        raise ValueError("hold feedback and previous target shapes must match")

    index_by_joint = getattr(
        estimator,
        "joint_index_by_name",
        {name: index for index, name in enumerate(motor_layer.policy_order)},
    )
    captured = set()
    for joint_name in fresh_names:
        index = index_by_joint.get(joint_name)
        if index is None or index < 0 or index >= q_measured.size:
            continue
        measured = float(q_measured[index])
        if not np.isfinite(measured):
            continue
        q_hold[index] = measured
        captured.add(joint_name)

    missing = [name for name in active_joints if name not in captured]
    return q_hold, len(captured), missing


def encoder_feedback_required(mode, estimator):
    return mode == "mit-signal" and hasattr(estimator, "last_feedback_by_joint")


def encoder_safety_stop_reason(
    safety,
    estimator,
    active_joints,
    mode,
    require_feedback=None,
    q_shift=None,
    feedback_by_joint=None,
    use_policy_limits=False,
):
    if not encoder_feedback_required(mode, estimator) and feedback_by_joint is None:
        return None

    q_current = getattr(estimator, "q_current", None)
    if q_current is None:
        return "ABNORMAL ENCODER ANGLE: estimator has no joint position vector"
    q_for_safety = np.asarray(q_current, dtype=np.float32)
    if q_shift is not None:
        q_for_safety = q_for_safety - np.asarray(q_shift, dtype=np.float32)

    if require_feedback is None:
        require_feedback = encoder_feedback_required(mode, estimator)

    stop, reason = safety.encoder_sanity_check(
        q_current=q_for_safety,
        active_joints=active_joints,
        feedback_by_joint=(
            getattr(estimator, "last_feedback_by_joint", None)
            if feedback_by_joint is None
            else feedback_by_joint
        ),
        require_feedback=require_feedback,
        use_policy_limits=use_policy_limits,
    )
    return reason if stop else None


def publish_safety_fault(
    telemetry,
    csv_logger,
    step,
    mode,
    command,
    command_source,
    commands,
    estimator,
    reason,
    action=None,
    phase="policy",
):
    record = compact_telemetry_record(
        step=step,
        mode=mode,
        command=command,
        command_source=command_source,
        commands=commands,
        estimator=estimator,
        action=action,
        phase=phase,
    )
    record["fault_reason"] = str(reason)
    if csv_logger is not None:
        csv_logger.log(record)
    if telemetry is not None:
        telemetry.send(
            step=step,
            mode=mode,
            command=command,
            command_source=command_source,
            commands=commands,
            action=action,
            safety_ok=False,
            safety_reason=reason,
        )


def imu_source_name(source, imu_defaults):
    if source == "auto":
        source = imu_defaults.get("source", "fake")
    return str(source).replace("-", "_").lower()


def initialize_hold_target(estimator, feedback_timeout):
    refresh_estimator_feedback(estimator, timeout=feedback_timeout)
    q_current, _, _, _, _ = estimator.read()

    print("\n" + "#" * 80)
    print("STARTUP PHASE: PASSIVE / IDLE")
    print("#" * 80)
    print("No hold, sit, stand, or walking command is sent until controller input.")
    return q_current.copy()


def active_joint_indices(policy_order, active_joints):
    index_by_joint = {name: index for index, name in enumerate(policy_order)}
    return [
        index_by_joint[name]
        for name in active_joints
        if name in index_by_joint
    ]


def max_active_error(q_a, q_b, active_indices):
    if not active_indices:
        return 0.0
    q_a = np.asarray(q_a, dtype=np.float32)
    q_b = np.asarray(q_b, dtype=np.float32)
    return float(np.max(np.abs(q_a[active_indices] - q_b[active_indices])))


def stand_ready_for_walking(
    command_error,
    feedback_error,
    trajectory_elapsed_s,
    trajectory_duration_s,
    error_tolerance_rad,
):
    """Return true only after the complete stand trajectory has settled.

    Walking readiness is deliberately separate from hardware/software-zero
    calibration. A loaded PD-controlled leg needs finite steady-state position
    error to support weight, so requiring the calibration tolerance here can
    permanently block policy entry even though the robot is safely standing.
    """
    values = (
        command_error,
        feedback_error,
        trajectory_elapsed_s,
        trajectory_duration_s,
        error_tolerance_rad,
    )
    if not all(np.isfinite(float(value)) for value in values):
        return False
    tolerance = float(error_tolerance_rad)
    duration = max(0.0, float(trajectory_duration_s))
    trajectory_complete = float(trajectory_elapsed_s) >= duration - 1.0e-6
    return bool(
        tolerance > 0.0
        and trajectory_complete
        and float(command_error) <= tolerance
        and float(feedback_error) <= tolerance
    )


def stand_state_ready_for_policy_entry(
    q_current,
    qd_current,
    q_stand_target,
    active_indices,
    error_tolerance_rad,
    velocity_tolerance_rad_s,
):
    """Validate the measured stand state immediately before policy entry."""
    error_tolerance = float(error_tolerance_rad)
    velocity_tolerance = float(velocity_tolerance_rad_s)
    if (
        not np.isfinite(error_tolerance)
        or not np.isfinite(velocity_tolerance)
        or error_tolerance <= 0.0
        or velocity_tolerance <= 0.0
    ):
        return False, float("inf"), float("inf")
    try:
        q_current = np.asarray(q_current, dtype=np.float32)
        qd_current = np.asarray(qd_current, dtype=np.float32)
        q_stand_target = np.asarray(q_stand_target, dtype=np.float32)
        indices = np.asarray(active_indices, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return False, float("inf"), float("inf")
    if indices.size == 0:
        return True, 0.0, 0.0
    if (
        q_current.ndim != 1
        or qd_current.shape != q_current.shape
        or q_stand_target.shape != q_current.shape
        or np.any(indices < 0)
        or np.any(indices >= q_current.size)
    ):
        return False, float("inf"), float("inf")
    active_q = q_current[indices]
    active_qd = qd_current[indices]
    active_stand = q_stand_target[indices]
    if not (
        np.all(np.isfinite(active_q))
        and np.all(np.isfinite(active_qd))
        and np.all(np.isfinite(active_stand))
    ):
        return False, float("inf"), float("inf")
    position_error = float(np.max(np.abs(active_q - active_stand)))
    velocity = float(np.max(np.abs(active_qd)))
    ready = (
        position_error <= error_tolerance
        and velocity <= velocity_tolerance
    )
    return bool(ready), position_error, velocity


def should_validate_stand_state_for_policy_entry(
    walking_armed,
    walk_requested,
    control_mode,
    previous_walk_requested,
):
    """Run the stand-state gate once, immediately before policy takeover."""
    return bool(
        walking_armed
        and walk_requested
        and control_mode == "stand"
        and not previous_walk_requested
    )


def automatic_policy_takeover_requested(enabled, walking_armed, control_mode):
    """Enter policy mode with a zero command as soon as stand is settled."""
    return bool(enabled and walking_armed and control_mode == "stand")


def policy_pose_support_scale(policy_entry_scale, support_floor):
    """Blend pose support continuously from stand into its policy-mode floor."""
    entry = float(policy_entry_scale)
    floor = float(support_floor)
    if not np.isfinite(entry) or not np.isfinite(floor):
        raise ValueError("policy pose-support scales must be finite")
    if floor < 0.0 or floor > 1.0:
        raise ValueError("pose_support.policy_scale must be within 0.0..1.0")
    entry = float(np.clip(entry, 0.0, 1.0))
    return 1.0 - entry * (1.0 - floor)


def constant_pose_like(runner, value):
    return np.full(len(runner.policy_order), float(value), dtype=np.float32)


def stand_pose_for_zero_frame(runner, zero_frame, crouch_calibration_value, stand_calibration_value):
    # Hardware zero is established once with the robot standing. The RL default
    # and stand pose therefore remain q=0 for the lifetime of the process.
    return constant_pose_like(runner, stand_calibration_value) + runner.q_stand


def sit_pose_for_zero_frame(runner, zero_frame, crouch_calibration_value, stand_calibration_value):
    # crouch_pose is expressed directly in the fixed stand-zero joint frame.
    return constant_pose_like(runner, stand_calibration_value) + runner.q_crouch


def shifted_safety_filter(
    safety,
    q_target,
    q_previous,
    q_shift,
    apply_rate_limit=True,
    use_policy_limits=False,
):
    q_shift = np.asarray(q_shift, dtype=np.float32)
    if not np.any(np.abs(q_shift) > 1e-8):
        return safety.safety_filter(
            q_target,
            q_previous,
            apply_rate_limit=apply_rate_limit,
            use_policy_limits=use_policy_limits,
        )
    filtered = safety.safety_filter(
        np.asarray(q_target, dtype=np.float32) - q_shift,
        np.asarray(q_previous, dtype=np.float32) - q_shift,
        apply_rate_limit=apply_rate_limit,
        use_policy_limits=use_policy_limits,
    )
    return (filtered + q_shift).astype(np.float32)


def shifted_safety_filter_with_diagnostics(
    safety,
    q_target,
    q_previous,
    q_shift,
    apply_rate_limit=True,
    use_policy_limits=False,
):
    q_target = np.asarray(q_target, dtype=np.float32)
    q_previous = np.asarray(q_previous, dtype=np.float32)
    q_shift = np.asarray(q_shift, dtype=np.float32)
    shifted_target = q_target - q_shift
    shifted_previous = q_previous - q_shift

    clipped = safety.clip_q_target(
        shifted_target,
        use_policy_limits=use_policy_limits,
    )
    joint_limited = np.abs(clipped - shifted_target) > 1.0e-6

    if apply_rate_limit:
        rate_limited_target = safety.rate_limit_q_target(clipped, shifted_previous)
        rate_limited = np.abs(rate_limited_target - clipped) > 1.0e-6
    else:
        rate_limited_target = clipped
        rate_limited = np.zeros_like(clipped, dtype=bool)

    final_clipped = safety.clip_q_target(
        rate_limited_target,
        use_policy_limits=use_policy_limits,
    )
    joint_limited = np.logical_or(
        joint_limited,
        np.abs(final_clipped - rate_limited_target) > 1.0e-6,
    )
    filtered = (final_clipped + q_shift).astype(np.float32)
    return filtered, {
        "target_joint_limited": joint_limited.astype(bool),
        "target_rate_limited": rate_limited.astype(bool),
        "joint_limit_filtered_q_target": (clipped + q_shift).astype(np.float32),
        "rate_limited_q_target": (rate_limited_target + q_shift).astype(np.float32),
    }


def apply_software_zero_calibration(
    estimator,
    motor_layer,
    active_joints,
    feedback_timeout,
    buses,
    mode,
    label,
    target_value=0.0,
):
    # Do not send stop/poll frames here. During live control those frames can
    # drop torque and cause exactly the sit/stand jerk we are avoiding.
    refresh_estimator_feedback(estimator, timeout=feedback_timeout)

    if hasattr(estimator, "apply_software_zero"):
        try:
            updated, missing = estimator.apply_software_zero(
                active_joints=active_joints,
                target_value=target_value,
            )
        except Exception as exc:
            print(f"\n[ZERO CAL] {label}: failed: {exc}")
            return False
    else:
        updated = {joint_name: float(target_value) for joint_name in active_joints}
        missing = []
        if hasattr(estimator, "q_current"):
            estimator.q_current[:] = float(target_value)
        if hasattr(estimator, "qd_current"):
            estimator.qd_current[:] = 0.0

    if missing:
        print(
            f"\n[ZERO CAL] {label}: missing feedback for "
            + ", ".join(missing)
        )
        return False

    q_current, _, _, _, _ = estimator.read()
    print(
        f"\n[ZERO CAL] {label}: software calibration applied as "
        f"q={float(target_value):+.3f} rad for {len(updated)} active joint(s). "
        "No RobStride hardware set-zero frame was sent."
    )
    return q_current.copy()


def run_joint_routing_test(
    runner,
    motor_layer,
    estimator,
    buses,
    mode,
    dt,
    feedback_timeout,
    amplitude_rad=0.04,
    frequency_hz=0.25,
    cycles=1,
    torque_limit_nm=5.0,
):
    if mode != "mit-signal":
        print("ERROR: --joint-routing-test requires --mode mit-signal.")
        return False

    amplitude_rad = float(amplitude_rad)
    frequency_hz = float(frequency_hz)
    cycles = int(cycles)
    torque_limit_nm = float(torque_limit_nm)
    if amplitude_rad <= 0.0 or frequency_hz <= 0.0 or cycles <= 0 or torque_limit_nm <= 0.0:
        print("ERROR: routing-test amplitude/frequency/cycles/torque-limit must be positive.")
        return False

    print("\n################################################################################")
    print("JOINT ROUTING TEST: actor disabled, one tiny sinusoidal joint target at a time")
    print("################################################################################")
    print(
        f"amplitude={amplitude_rad:.3f} rad "
        f"frequency={frequency_hz:.3f} Hz cycles={cycles} "
        f"torque_limit={torque_limit_nm:.2f} Nm"
    )

    motor_layer.set_policy_pd_torque_limit(torque_limit_nm)
    fallback = np.asarray(
        getattr(estimator, "q_current", runner.q_policy_reference),
        dtype=np.float32,
    )
    (
        q_hold,
        feedback_count,
        missing,
        _q_current,
        _qd_current,
        _base_lin_vel,
        _base_ang_vel,
        _gravity,
    ) = acquire_hold_target_from_feedback(
        estimator=estimator,
        motor_layer=motor_layer,
        safety=None,
        q_previous_target=fallback,
        feedback_timeout=feedback_timeout,
        capture_seconds=0.35,
        buses=buses,
        mode=mode,
        allow_poll_snapshot=True,
    )
    if feedback_count <= 0:
        print("ERROR: routing test could not capture any fresh motor feedback.")
        return False
    if missing:
        shown = ", ".join(missing[:6])
        if len(missing) > 6:
            shown += f", +{len(missing) - 6} more"
        print("WARNING: stale/missing joints are held at fallback target:", shown)

    policy_index_by_joint = {name: index for index, name in enumerate(runner.policy_order)}
    active_joints = list(motor_layer.active_joints)
    active_indices = [policy_index_by_joint[name] for name in active_joints]
    steps_per_joint = max(8, int(round(float(cycles) / (frequency_hz * float(dt)))))
    motion_threshold = max(0.006, 0.25 * amplitude_rad)
    unrelated_threshold = max(0.020, 0.50 * amplitude_rad)
    report_rows = []

    for joint_name in active_joints:
        index = policy_index_by_joint[joint_name]
        motor_id = int(motor_layer.motor_ids[joint_name])
        bus_name = motor_layer.joint_can_bus.get(joint_name, "can0")
        direction = float(motor_layer.joint_directions[joint_name])
        offset = float(motor_layer.joint_offsets[joint_name])
        print(
            f"\n[ROUTING] index={index:02d} joint={joint_name} "
            f"bus={bus_name} motor=0x{motor_id:02X} direction={direction:+.0f} "
            f"offset={offset:+.4f}"
        )

        refresh_estimator_feedback(estimator, timeout=feedback_timeout)
        q_start = np.asarray(getattr(estimator, "q_current", q_hold), dtype=np.float32).copy()
        q_target_base = q_start.copy()
        max_delta = np.zeros(len(runner.policy_order), dtype=np.float32)
        positive_peak_delta = 0.0
        next_tick = time.monotonic()

        for step_index in range(steps_per_joint):
            phase = 2.0 * np.pi * frequency_hz * (step_index * float(dt))
            commanded_delta = amplitude_rad * np.sin(phase)
            q_target = q_target_base.copy()
            q_target[index] += commanded_delta

            fresh_feedback, _missing = fresh_feedback_by_joint(
                estimator,
                active_joints,
                max_age_s=max(0.04, 2.5 * float(dt)),
            )
            commands = motor_layer.build_mit_commands(
                q_target,
                phase="policy",
                feedback_by_joint=fresh_feedback,
            )
            if hasattr(estimator, "mark_command_sent"):
                estimator.mark_command_sent(time.monotonic())
            motor_layer.send_signal_commands(buses, commands)
            refresh_estimator_feedback(estimator, timeout=feedback_timeout)

            q_now = np.asarray(getattr(estimator, "q_current", q_start), dtype=np.float32)
            delta = q_now - q_start
            max_delta = np.maximum(max_delta, np.abs(delta))
            if commanded_delta > 0.8 * amplitude_rad:
                positive_peak_delta = float(delta[index])

            moving_elsewhere = [
                runner.policy_order[i]
                for i in active_indices
                if i != index and max_delta[i] > unrelated_threshold
            ]
            if moving_elsewhere:
                print(
                    "[ROUTING] FAIL: unexpected encoder movement while testing "
                    f"{joint_name}: " + ", ".join(moving_elsewhere[:4])
                )
                return False

            feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
            measured_torque_peak = 0.0
            for active_joint in active_joints:
                feedback = feedback_by_joint.get(active_joint, {})
                if feedback.get("joint_torque") is not None:
                    measured_torque_peak = max(
                        measured_torque_peak,
                        abs(float(feedback["joint_torque"])),
                    )
            estimated_torque_peak = max(
                [abs(float(item.get("tau_pd_est") or 0.0)) for item in commands]
                or [0.0]
            )
            if measured_torque_peak > torque_limit_nm or estimated_torque_peak > torque_limit_nm:
                print(
                    "[ROUTING] FAIL: torque exceeded test limit "
                    f"measured={measured_torque_peak:.2f}Nm "
                    f"estimated={estimated_torque_peak:.2f}Nm "
                    f"limit={torque_limit_nm:.2f}Nm"
                )
                return False

            next_tick += float(dt)
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        hold_commands = motor_layer.build_mit_commands(
            q_start,
            phase="hold",
            feedback_by_joint=getattr(estimator, "last_feedback_by_joint", {}) or {},
        )
        motor_layer.send_signal_commands(buses, hold_commands)
        q_hold = q_start.copy()

        measured_index = int(np.argmax(max_delta))
        measured_joint = runner.policy_order[measured_index]
        intended_moved = bool(max_delta[index] >= motion_threshold)
        unrelated_peak = max(
            [float(max_delta[i]) for i in active_indices if i != index] or [0.0]
        )
        unrelated_ok = bool(unrelated_peak <= unrelated_threshold)
        feedback_sign = (
            "+"
            if positive_peak_delta > 1.0e-4
            else "-"
            if positive_peak_delta < -1.0e-4
            else "0"
        )
        passed = intended_moved and unrelated_ok and measured_joint == joint_name
        report_rows.append(
            {
                "policy_index": index,
                "expected_joint": joint_name,
                "motor_id": f"0x{motor_id:02X}",
                "measured_moving_joint": measured_joint,
                "command_sign": "+",
                "feedback_sign": feedback_sign,
                "max_delta_rad": float(max_delta[index]),
                "pass": passed,
            }
        )
        print(
            "[ROUTING] "
            f"{'PASS' if passed else 'FAIL'} "
            f"expected={joint_name} measured={measured_joint} "
            f"max_delta={float(max_delta[index]):.4f}rad "
            f"unrelated_peak={unrelated_peak:.4f}rad "
            f"feedback_sign={feedback_sign}"
        )
        if not passed:
            return False

    print("\nROUTING REPORT")
    print("policy_index,expected_joint,motor_id,measured_moving_joint,command_sign,feedback_sign,max_delta_rad,pass")
    for row in report_rows:
        print(
            f"{row['policy_index']:02d},{row['expected_joint']},{row['motor_id']},"
            f"{row['measured_moving_joint']},{row['command_sign']},"
            f"{row['feedback_sign']},{row['max_delta_rad']:.6f},{int(row['pass'])}"
        )
    return True


def collect_complete_feedback_before_zero(
    estimator,
    motor_layer,
    safety,
    buses,
    mode,
    feedback_timeout,
    gather_seconds=0.5,
):
    if not encoder_feedback_required(mode, estimator):
        refresh_estimator_feedback(estimator, timeout=feedback_timeout)
        return len(motor_layer.active_joints), len(motor_layer.active_joints)

    active_joints = list(motor_layer.active_joints)
    n_active = len(active_joints)
    max_age_s = getattr(safety, "max_feedback_age_s", 0.25)
    deadline = time.monotonic() + max(float(gather_seconds), float(feedback_timeout), 0.02)
    fresh = count_fresh_active_feedback(estimator, active_joints, max_age_s)
    while fresh < n_active and time.monotonic() < deadline:
        request_feedback_snapshot(motor_layer, buses, mode)
        refresh_estimator_feedback(estimator, timeout=feedback_timeout)
        fresh = count_fresh_active_feedback(estimator, active_joints, max_age_s)
        if fresh >= n_active:
            break
        time.sleep(0.002)
    return fresh, n_active


def run_startup_to_stand(
    runner,
    safety,
    motor_layer,
    estimator,
    buses,
    mode,
    standup_seconds,
    log_every,
    show_hex,
    feedback_timeout,
    telemetry=None,
    csv_logger=None,
    can_streamer=None,
):
    dt = runner.control_dt
    steps = max(1, int(standup_seconds / dt))
    from joystick_interface import FixedCommandSource as _FCS
    dummy_command_source = _FCS(0.0, 0.0, 0.0)

    refresh_estimator_feedback(estimator, timeout=feedback_timeout)
    q_start, _, _, _, _ = estimator.read()
    q_previous_target = q_start.copy()
    reason = encoder_safety_stop_reason(
        safety=safety,
        estimator=estimator,
        active_joints=motor_layer.active_joints,
        mode=mode,
    )
    if reason is not None:
        print("\nEMERGENCY STOP:", reason)
        publish_safety_fault(
            telemetry=telemetry,
            csv_logger=csv_logger,
            step=0,
            mode="encoder_fault",
            command=[0.0, 0.0, 0.0],
            command_source=dummy_command_source,
            commands=[],
            estimator=estimator,
            reason=reason,
            action=None,
            phase="startup",
        )
        return q_previous_target, False

    print("\n" + "#" * 80)
    print("STARTUP PHASE: current pose -> STAND / DEFAULT pose")
    print("#" * 80)
    print("standup_seconds:", standup_seconds)
    print("startup steps:", steps)
    print("mode:", mode)

    safety_faulted = False
    for step in range(steps):
        cycle_start = time.monotonic()
        if can_streamer is not None and can_streamer.fault_reason is not None:
            print("\nEMERGENCY STOP:", can_streamer.fault_reason)
            can_streamer.clear()
            safety_faulted = True
            break
        if hasattr(estimator, "imu_stale") and estimator.imu_stale():
            reason = "IMU data missing or stale during startup"
            print("\nEMERGENCY STOP:", reason)
            safety_faulted = True
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="imu_fault",
                command=[0.0, 0.0, 0.0],
                command_source=dummy_command_source,
                commands=[],
                estimator=estimator,
                reason=reason,
                action=None,
                phase="startup",
            )
            break

        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=estimator,
            active_joints=motor_layer.active_joints,
            mode=mode,
        )
        if reason is not None:
            print("\nEMERGENCY STOP:", reason)
            safety_faulted = True
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="encoder_fault",
                command=[0.0, 0.0, 0.0],
                command_source=dummy_command_source,
                commands=[],
                estimator=estimator,
                reason=reason,
                action=None,
                phase="startup",
            )
            break

        alpha = smoothstep((step + 1) / steps)
        q_desired = (1.0 - alpha) * q_start + alpha * runner.q_stand
        q_safe = safety.safety_filter(q_desired, q_previous_target)

        commands = motor_layer.build_mit_commands(
            q_safe,
            phase="startup",
            feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
        )

        if can_streamer is not None:
            can_streamer.submit(commands)
        elif mode == "signal":
            motor_layer.send_harmless_frames(buses, commands)
        elif mode == "mit-signal":
            motor_layer.send_signal_commands(buses, commands)

        active_feedback_timeout = min(
            float(feedback_timeout),
            max(0.002, 0.35 * float(dt)),
        )
        refresh_estimator_feedback(estimator, timeout=active_feedback_timeout)
        require_command_feedback = bool(
            commands and encoder_feedback_required(mode, estimator)
        )
        if require_command_feedback:
            fresh = count_fresh_active_feedback(
                estimator,
                motor_layer.active_joints,
                getattr(safety, "max_feedback_age_s", 0.25),
            )
            if fresh < len(motor_layer.active_joints):
                refresh_active_feedback_before_fault(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    safety=safety,
                    buses=buses,
                    mode=mode,
                    feedback_timeout=feedback_timeout,
                    q_keepalive=q_safe,
                    phase="startup",
                    can_streamer=can_streamer,
                )
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=estimator,
            active_joints=motor_layer.active_joints,
            mode=mode,
            require_feedback=require_command_feedback,
        )
        if reason is not None:
            print("\nEMERGENCY STOP:", reason)
            safety_faulted = True
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="encoder_fault",
                command=[0.0, 0.0, 0.0],
                command_source=dummy_command_source,
                commands=commands,
                estimator=estimator,
                reason=reason,
                action=None,
                phase="startup",
            )
            break

        if step % log_every == 0 or step == steps - 1:
            tau_cmd_mean, tau_cmd_max = command_torque_stats(commands)
            print(
                f"startup_step={step:06d}/{steps - 1:06d} "
                f"mode=stand "
                f"alpha={alpha:.3f} "
                f"vx={0.0: .3f} vy={0.0: .3f} "
                f"vxy={0.0: .3f} yaw={0.0: .3f} "
                f"tau_cmd={tau_cmd_mean: .3f} "
                f"tau_cmd_max={tau_cmd_max: .3f} "
                f"cmds={len(commands):02d}"
            )
            if show_hex:
                print_mit_commands(commands, show_hex=True)
            if csv_logger is not None:
                csv_logger.log(
                    compact_telemetry_record(
                        step=step,
                        mode="stand",
                        command=[0.0, 0.0, 0.0],
                        command_source=dummy_command_source,
                        commands=commands,
                        estimator=estimator,
                        action=None,
                        phase="startup",
                    )
                )

        estimator.dry_update_as_if_robot_followed(q_safe, dt)
        q_previous_target = q_safe.copy()
        if telemetry is not None:
            telemetry.send(step, "stand", [0.0, 0.0, 0.0], dummy_command_source, commands, safety_ok=True)
        cycle_elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, dt - cycle_elapsed))

    if safety_faulted:
        print("\nStartup phase stopped by safety fault.")
    else:
        print("\nStartup phase completed. Robot target is STAND / DEFAULT pose.")
    return q_previous_target, not safety_faulted


def run_policy_loop(
    runner,
    safety,
    motor_layer,
    estimator,
    command_source,
    buses,
    mode,
    q_previous_target,
    steps,
    log_every,
    print_every,
    show_hex,
    start_control_mode,
    feedback_timeout,
    walk_command_threshold,
    walk_command_grace_seconds,
    walk_stop_confirm_seconds,
    joystick_debug,
    joint_debug,
    base_lin_vel_source,
    motion_assist_cfg,
    initial_zero_frame,
    initial_zero_calibrated,
    auto_stand_zero,
    auto_sit_zero,
    stand_zero_error_rad,
    stand_ready_error_rad,
    stand_ready_velocity_rad_s,
    stand_zero_settle_steps,
    pose_sync_error_rad,
    policy_command_gain,
    policy_command_vx_max,
    policy_command_vy_max,
    policy_command_yaw_max,
    policy_action_clip,
    policy_hip_action_clip,
    policy_hip_action_scale,
    policy_action_smoothing,
    policy_action_delta_limit,
    policy_entry_ramp_seconds,
    policy_sim_match,
    exact_policy_after_entry,
    auto_policy_after_stand,
    stand_policy_stabilization,
    hold_capture_seconds,
    hold_command_repeats,
    crouch_calibration_value,
    stand_calibration_value,
    pose_transition_speed_rad_s,
    pose_transition_min_seconds,
    fresh_feedback_max_age_s,
    steady_feedback_budget_s,
    suspension_status_seconds,
    imu_active_max_roll_pitch_deg,
    deadline_tolerance_s,
    deadline_resync_s,
    timing_fault_consecutive,
    policy_shadow_mode=False,
    encoder_calibration_required=True,
    encoder_calibration_passed=False,
    torque_ramp=None,
    measured_torque_soft_limits=None,
    telemetry=None,
    csv_logger=None,
    can_streamer=None,
    pose_test_only=False,
    sit_stand_trace_logger=None,
):
    dt = runner.control_dt
    live_feedback_max_age_s = (
        float(fresh_feedback_max_age_s)
        if float(fresh_feedback_max_age_s) > 0.0
        else min(
            float(getattr(safety, "max_feedback_age_s", 0.25)),
            max(0.02, 2.0 * float(dt)),
        )
    )
    action_dim = len(runner.policy_order)
    measured_torque_soft_limits = dict(
        measured_torque_soft_limits
        or {"hip": 35.0, "thigh": 40.0, "calf": 40.0, "default": 40.0}
    )
    torque_ramp_state_for_log = torque_ramp.telemetry() if torque_ramp is not None else {}
    measured_soft_limit_active_by_joint_for_log = {}
    measured_torque_average_by_joint_for_log = {}
    measured_torque_window_max_by_joint_for_log = {}
    measured_torque_supervisor = MeasuredTorqueSupervisor(
        runner.policy_order,
        measured_torque_soft_limits,
        window=12,
    )
    # In exact mode the raw actor output is also the applied actor-coordinate
    # action. In conditioned hardware mode the applied value is the clipped,
    # smoothed action. Feeding the rejected raw value back into obs[36:48]
    # creates a policy state that never existed in simulation.
    previous_raw_action = np.zeros(action_dim, dtype=np.float32)
    previous_sent_action = np.zeros(action_dim, dtype=np.float32)
    direct_leveling_correction = np.zeros(action_dim, dtype=np.float32)
    policy_entry_elapsed_s = 0.0
    policy_entry_scale = 0.0
    policy_entry_q_start = np.asarray(q_previous_target, dtype=np.float32).copy()
    policy_entry_restart_count = 0
    policy_entry_restart_reason = ""
    last_policy_gain_blend_alpha = 1.0
    stand_recovery_gain_active = False
    stand_recovery_gain_mode = None
    stand_recovery_gain_elapsed_s = 0.0
    stand_recovery_policy_alpha_at_stop = 0.0
    previous_walk_requested = False
    last_walk_command = np.zeros(3, dtype=np.float32)
    last_walk_command_step = -10**9
    walk_stop_candidate_step = -1
    policy_has_started = False
    direct_imu_stabilization_enabled = bool(
        motion_assist_cfg.get("imu_posture", {}).get("enabled", False)
    )

    control_mode = "policy" if policy_shadow_mode else start_control_mode
    zero_frame = str(initial_zero_frame).lower()
    zero_calibrated = zero_frame == "stand" or bool(initial_zero_calibrated)
    has_motion_target = (
        control_mode in ("stand", "sit")
        or bool(zero_calibrated and zero_frame == "crouch")
    )
    stand_zero_pending = bool(
        auto_stand_zero and control_mode == "stand" and zero_frame == "crouch"
    )
    stand_zero_settle_count = 0
    sit_zero_pending = bool(
        auto_sit_zero and control_mode == "sit" and zero_frame == "stand"
    )
    sit_zero_settle_count = 0
    walking_armed = bool(policy_shadow_mode or control_mode == "policy")
    stand_ready_pending = control_mode == "stand"
    stand_ready_settle_count = 0
    calibration_hold_until_step = -1
    active_indices = active_joint_indices(runner.policy_order, motor_layer.active_joints)
    pose_transition_mode = None
    pose_transition_start = np.asarray(q_previous_target, dtype=np.float32).copy()
    pose_transition_target = pose_transition_start.copy()
    pose_transition_velocity_target = np.zeros_like(pose_transition_start)
    pose_support_cfg = dict(motor_layer.cfg.get("pose_support", {}) or {})
    pose_support_enabled = bool(pose_support_cfg.get("enabled", False))
    policy_pose_support_floor = float(pose_support_cfg.get("policy_scale", 0.0))
    if not np.isfinite(policy_pose_support_floor) or not (
        0.0 <= policy_pose_support_floor <= 1.0
    ):
        raise ValueError("pose_support.policy_scale must be finite and within 0.0..1.0")
    pose_support_map = dict(pose_support_cfg.get("stand_joint_tau_ff", {}) or {})
    pose_support_tau_target = np.asarray(
        [float(pose_support_map.get(name, 0.0)) for name in runner.policy_order],
        dtype=np.float32,
    )
    if not pose_support_enabled:
        pose_support_tau_target.fill(0.0)
    pose_support_scale = 1.0 if control_mode == "stand" else 0.0
    pose_transition_support_start = pose_support_scale
    pose_transition_support_target = pose_support_scale
    pose_transition_elapsed_s = 0.0
    pose_transition_duration_s = float(pose_transition_min_seconds)

    def final_pose_target(mode_name):
        if mode_name == "stand":
            return stand_pose_for_zero_frame(
                runner,
                zero_frame,
                crouch_calibration_value,
                stand_calibration_value,
            )
        if mode_name == "sit":
            return sit_pose_for_zero_frame(
                runner,
                zero_frame,
                crouch_calibration_value,
                stand_calibration_value,
            )
        raise ValueError(f"No synchronized pose target for mode {mode_name}")

    def begin_pose_transition(mode_name, start_q, current_step):
        nonlocal pose_transition_mode
        nonlocal pose_transition_start
        nonlocal pose_transition_target
        nonlocal pose_transition_elapsed_s
        nonlocal pose_transition_duration_s
        nonlocal pose_transition_support_start
        nonlocal pose_transition_support_target

        target_q = np.asarray(final_pose_target(mode_name), dtype=np.float32)
        start_q = np.asarray(start_q, dtype=np.float32).copy()
        active_distance = max_active_error(start_q, target_q, active_indices)
        # smoothstep's peak derivative is 1.5. Include it in the duration so
        # no joint exceeds the configured physical pose speed.
        speed = max(float(pose_transition_speed_rad_s), 1.0e-6)
        duration = max(
            float(pose_transition_min_seconds),
            1.5 * float(active_distance) / speed,
        )
        pose_transition_mode = str(mode_name)
        pose_transition_start = start_q
        pose_transition_target = target_q
        pose_transition_elapsed_s = 0.0
        pose_transition_duration_s = duration
        pose_transition_support_start = float(pose_support_scale)
        pose_transition_support_target = 1.0 if mode_name == "stand" else 0.0
        print(
            f"[POSE] synchronized {mode_name} transition: "
            f"distance={active_distance:.3f} rad duration={duration:.2f}s "
            f"speed_limit={speed:.3f} rad/s"
        )
        scheduler.request_resync(f"{mode_name} transition initialized")

    def current_pose_transition_target(mode_name, current_step):
        nonlocal pose_transition_mode
        nonlocal pose_transition_velocity_target
        nonlocal pose_support_scale
        if pose_transition_mode != mode_name:
            begin_pose_transition(mode_name, q_previous_target, current_step)
        target_q, velocity_q, alpha = synchronized_pose_trajectory_state(
            pose_transition_start,
            pose_transition_target,
            pose_transition_elapsed_s,
            pose_transition_duration_s,
        )
        pose_transition_velocity_target = velocity_q
        pose_support_scale = (
            pose_transition_support_start
            + smoothstep(alpha)
            * (pose_transition_support_target - pose_transition_support_start)
        )
        return target_q

    print("\n" + "#" * 80)
    print("RUNTIME CONTROL PHASE")
    print("#" * 80)
    print("mode:", mode)
    print("policy_shadow_mode:", bool(policy_shadow_mode))
    print("pose_test_only:", bool(pose_test_only))
    if policy_shadow_mode:
        print("[SHADOW] Motors remain passive; no MIT movement command is transmitted.")
    print("Joystick buttons:")
    print("  button 4    -> STOP walking and SIT/CROUCH pose")
    print("  button 5    -> STAND pose")
    print("  buttons 0-3 -> EMERGENCY STOP")
    print("  D-pad zero request -> ignored while fixed hardware stand-zero is active")
    print("Terminal keys:")
    print("  c -> SIT/CROUCH, space -> STAND")
    print("  h -> HOLD current position, x -> EMERGENCY STOP")
    if pose_test_only:
        print("  policy and walking commands are disabled for this test")
    else:
        print("Joystick axes:")
        print("  left stick Y  -> forward/back vx")
        print("  left stick X  -> left/right vy")
        print("  right stick X -> turn/yaw")
        print("  w/s -> straight vx, a/d -> lateral vy, combine for xy diagonal")
        print("  q/e -> positive/negative yaw; combine with translation if needed")
        print("  up/down arrows -> increase/decrease speed scale")
    print("start_control_mode:", control_mode)
    print("walk_command_threshold:", walk_command_threshold)
    print("walk_command_grace_seconds:", float(walk_command_grace_seconds))
    print("walk_stop_confirm_seconds:", float(walk_stop_confirm_seconds))
    print("base_lin_vel_source:", base_lin_vel_source, "(state estimate remains zero)")
    print("policy_contract: velocity-command locomotion; obs[0:3]=[0,0,0]")
    print("zero_frame:", zero_frame)
    print("zero_calibrated:", bool(zero_calibrated))
    if has_motion_target and control_mode == "hold" and zero_frame == "crouch":
        print(
            "[ZERO CAL] MIT hold is active at "
            f"q={float(crouch_calibration_value):+.3f} for the crouch/default pose."
        )
    print("auto_stand_zero:", bool(auto_stand_zero))
    print("auto_sit_zero:", bool(auto_sit_zero))
    print("pose_sync_error_rad:", float(pose_sync_error_rad))
    print("stand_ready_error_rad:", float(stand_ready_error_rad))
    print("stand_ready_velocity_rad_s:", float(stand_ready_velocity_rad_s))
    print("policy_command_gain:", float(policy_command_gain))
    print(
        "policy_command_caps:",
        f"vx={float(policy_command_vx_max):.3f}",
        f"vy={float(policy_command_vy_max):.3f}",
        f"yaw={float(policy_command_yaw_max):.3f}",
        "(0 disables each cap)",
    )
    print("policy_action_clip:", float(policy_action_clip), "(0 disables)")
    print(
        "policy_hip_action_clip:",
        float(policy_hip_action_clip),
        "(0 disables)",
    )
    print("policy_hip_action_scale:", float(policy_hip_action_scale))
    print("policy_action_smoothing:", float(policy_action_smoothing), "(0 disables)")
    print("policy_action_delta_limit:", float(policy_action_delta_limit), "(0 disables)")
    print("policy_entry_ramp_seconds:", float(policy_entry_ramp_seconds))
    print(
        "exact_policy_after_entry:",
        bool(exact_policy_after_entry),
        "(sent action equals raw actor action after entry blend)",
    )
    print(
        "previous_action_observation:",
        "raw actor output (training contract)",
    )
    print(
        "policy_sim_match:",
        bool(policy_sim_match),
        "(hard joint, encoder, tilt, and torque safety remain active)",
    )
    if policy_sim_match:
        print(
            "WARNING: policy_sim_match bypasses deployment action clipping, "
            "action slew limiting, and smoothing. Policy target rate limiting "
            "still protects the entry ramp. Per-joint software PD torque "
            "clipping remains active."
        )
    print("stand_policy_stabilization:", bool(stand_policy_stabilization))
    print("walking_armed:", bool(walking_armed))
    print("hold_capture_seconds:", float(hold_capture_seconds))
    print("hold_command_repeats:", int(hold_command_repeats))
    print("fresh_feedback_max_age_s:", float(live_feedback_max_age_s))
    print("steady_feedback_budget_ms:", 1000.0 * float(steady_feedback_budget_s))
    print("suspension_status_seconds:", float(suspension_status_seconds))
    print("pose_transition_speed_rad_s:", float(pose_transition_speed_rad_s))
    print("pose_transition_min_seconds:", float(pose_transition_min_seconds))
    print("policy_pose_support_scale:", float(policy_pose_support_floor))
    print("crouch_calibration_value:", float(crouch_calibration_value))
    print("stand_calibration_value:", float(stand_calibration_value))
    if stand_zero_pending:
        print("[ZERO CAL] initial stand target will auto-zero when settled.")
    if sit_zero_pending:
        print("[ZERO CAL] initial sit/crouch target will auto-zero when settled.")
    print("imu_stabilization:", bool(motion_assist_cfg.get("imu_posture", {}).get("enabled", False)))
    print("gait_assist: unavailable (walking targets are policy-only)")
    if steps is None:
        print("policy_steps: unlimited, running until emergency stop or Ctrl+C")
    else:
        print("policy_steps:", steps)
    scheduler = DeadlineScheduler(
        dt_s=dt,
        deadline_tolerance_s=deadline_tolerance_s,
        deadline_resync_s=deadline_resync_s,
        timing_fault_consecutive=timing_fault_consecutive,
        warning_callback=print,
    )
    print("deadline_tolerance_ms:", 1000.0 * float(scheduler.deadline_tolerance_s))
    print("deadline_resync_ms:", 1000.0 * float(scheduler.deadline_resync_s))
    print("timing_fault_consecutive:", int(scheduler.timing_fault_consecutive))

    step = 0
    previous_cycle_start = None
    policy_steady_cycles = 0
    policy_target_clip_counts = np.zeros(action_dim, dtype=np.int64)
    policy_torque_clip_counts = np.zeros(action_dim, dtype=np.int64)
    policy_joint_summary = init_policy_joint_summary(runner.policy_order)
    root_raw_signals = []
    root_transmitted_signals = []
    root_measured_signals = []
    root_tracking_error_maxima = []
    root_velocity_mit = []
    root_velocity_fd = []
    root_observation_seen = False
    root_encoder_faulted = False
    last_suspension_status_time = -1.0e9
    last_timing_breakdown_print_time = -1.0e9
    policy_summary_stride = max(1, min(5, int(log_every)))
    command_build_s = 0.0

    def build_loop_mit_commands(
        q_target,
        phase,
        feedback_by_joint=None,
        joint_velocity_target=None,
        joint_feedforward_torque_target=None,
        prelimit_q_target=None,
        gain_blend_from_phase=None,
        gain_blend_alpha=1.0,
        previous_command_q=None,
        max_command_delta=None,
    ):
        nonlocal command_build_s
        started = time.monotonic()
        commands_out = motor_layer.build_mit_commands(
            q_target,
            phase=phase,
            feedback_by_joint=feedback_by_joint,
            joint_velocity_target=joint_velocity_target,
            joint_feedforward_torque_target=joint_feedforward_torque_target,
            prelimit_q_target=prelimit_q_target,
            gain_blend_from_phase=gain_blend_from_phase,
            gain_blend_alpha=gain_blend_alpha,
            previous_command_q=previous_command_q,
            max_command_delta=max_command_delta,
        )
        command_build_s += time.monotonic() - started
        return commands_out

    while steps is None or step < steps:
        cycle_start = time.monotonic()
        if can_streamer is not None and can_streamer.fault_reason is not None:
            print("\nEMERGENCY STOP:", can_streamer.fault_reason)
            can_streamer.clear()
            break
        loop_dt_s = (
            None
            if previous_cycle_start is None
            else cycle_start - previous_cycle_start
        )
        previous_cycle_start = cycle_start
        observation_for_log = None
        raw_action = np.zeros(action_dim, dtype=np.float32)
        q_actor_target_for_log = None
        q_entry_blended_target_for_log = None
        q_joint_limit_filtered_target_for_log = None
        q_rate_limited_target_for_log = None
        entry_blend_active_for_log = False
        imu_correction_abs_max = 0.0
        target_joint_limited_mask = np.zeros(action_dim, dtype=bool)
        target_rate_limited_mask = np.zeros(action_dim, dtype=bool)
        measured_soft_limit_active_by_joint_for_log = {}
        measured_torque_average_by_joint_for_log = {}
        measured_torque_window_max_by_joint_for_log = {}
        torque_ramp_state_for_log = torque_ramp.telemetry() if torque_ramp is not None else {}
        policy_inference_s = 0.0
        command_input_s = 0.0
        observation_build_s = 0.0
        policy_target_conversion_s = 0.0
        safety_filter_s = 0.0
        command_build_s = 0.0
        feedback_read_s = 0.0
        pre_feedback_read_s = 0.0
        steady_feedback_read_s = 0.0
        safety_check_s = 0.0
        logging_s = 0.0
        terminal_print_s = 0.0
        expected_feedback_ids = active_feedback_bus_motor_ids(
            estimator,
            motor_layer,
            motor_layer.active_joints,
        )
        if policy_shadow_mode and encoder_feedback_required(mode, estimator):
            request_feedback_snapshot(motor_layer, buses, mode)
        pre_feedback_start = time.monotonic()
        refresh_estimator_feedback(
            estimator,
            timeout=0.0,
            expected_bus_motor_ids=expected_feedback_ids,
        )
        pre_feedback_read_s = time.monotonic() - pre_feedback_start
        feedback_read_s += pre_feedback_read_s
        imu_read_start = time.monotonic()
        (
            q_current,
            qd_current,
            base_lin_vel_b,
            base_ang_vel_b,
            projected_gravity_b,
        ) = read_estimator_state(estimator, refresh_imu=True)
        imu_cache_read_s = time.monotonic() - imu_read_start
        q_coordinate_shift = motor_layer.coordinate_shift_array()

        command_input_start = time.monotonic()
        joystick_emergency_reason = command_source.get_emergency_stop_request()
        if joystick_emergency_reason is not None:
            print("\nEMERGENCY STOP:", joystick_emergency_reason)
            command = command_source.read()
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="joystick_estop",
                command=command,
                command_source=command_source,
                commands=[],
                estimator=estimator,
                reason=joystick_emergency_reason,
                action=np.zeros(action_dim, dtype=np.float32),
                phase="runtime",
            )
            break

        calibration_request = command_source.get_calibration_request()
        if calibration_request == "zero_current_pose":
            if zero_frame == "stand":
                print("[ZERO CAL] ignored because stand/RL zero is already active.")
            elif control_mode not in ("idle", "hold", "sit"):
                print("[ZERO CAL] ignored; zero calibration is only allowed from idle/hold/sit.")
            else:
                if can_streamer is not None:
                    can_streamer.clear()
                # Collect a COMPLETE feedback snapshot before zeroing. The dpad
                # press is a single instant; with a tight --feedback-timeout the
                # last motors on each CAN bus often have not returned a frame
                # yet. Poll/refresh repeatedly until every active joint is fresh
                # (or a short budget elapses) so the software zero is applied to
                # all 12 motors, not just the ones that happened to be ready.
                max_age_s = getattr(safety, "max_feedback_age_s", 0.25)
                n_active = len(motor_layer.active_joints)
                deadline = time.monotonic() + 0.5  # max 500 ms to gather all
                allow_poll_snapshot = control_mode == "idle" and not has_motion_target
                fresh = count_fresh_active_feedback(
                    estimator, motor_layer.active_joints, max_age_s
                )
                while fresh < n_active and time.monotonic() < deadline:
                    if allow_poll_snapshot:
                        request_feedback_snapshot(motor_layer, buses, mode)
                    feedback_read_start = time.monotonic()
                    refresh_estimator_feedback(estimator, timeout=feedback_timeout)
                    feedback_read_s += time.monotonic() - feedback_read_start
                    fresh = count_fresh_active_feedback(
                        estimator, motor_layer.active_joints, max_age_s
                    )
                if fresh < n_active:
                    print(
                        f"[ZERO CAL] aborted: only {fresh}/{n_active} joints "
                        "returned fresh feedback. Check CAN wiring/IDs on the "
                        "missing motors or raise --feedback-timeout."
                    )
                    q_zeroed = False
                else:
                    q_zeroed = apply_software_zero_calibration(
                        estimator=estimator,
                        motor_layer=motor_layer,
                        active_joints=motor_layer.active_joints,
                        feedback_timeout=feedback_timeout,
                        buses=buses,
                        mode=mode,
                        label="crouch/default pose",
                        target_value=crouch_calibration_value,
                    )
                if q_zeroed is not False:
                    scheduler.request_resync("software-zero calibration")
                    q_previous_target = q_zeroed.copy()
                    q_current = q_zeroed.copy()
                    qd_current = np.zeros_like(q_current, dtype=np.float32)
                    q_coordinate_shift = motor_layer.coordinate_shift_array()
                    zero_frame = "crouch"
                    zero_calibrated = True
                    control_mode = "idle"
                    has_motion_target = False
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                    sit_zero_pending = False
                    sit_zero_settle_count = 0
                    calibration_hold_until_step = -1
                    print(
                        "[ZERO CAL] zero_frame -> crouch at "
                        f"q={float(crouch_calibration_value):+.3f} for this pose."
                    )
                    print("[ZERO CAL] No hold commands are sent until H or a pose command.")

        motion_feedback_guard_active = bool(
            not policy_shadow_mode
            and
            (has_motion_target or control_mode not in ("idle", "hold"))
            and encoder_feedback_required(mode, estimator)
        )
        if motion_feedback_guard_active:
            fresh = count_fresh_active_feedback(
                estimator,
                motor_layer.active_joints,
                getattr(safety, "max_feedback_age_s", 0.25),
            )
            if fresh < len(motor_layer.active_joints):
                refresh_active_feedback_before_fault(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    safety=safety,
                    buses=buses,
                    mode=mode,
                    feedback_timeout=feedback_timeout,
                    q_keepalive=q_previous_target,
                    phase="policy" if control_mode == "policy" else "startup",
                    can_streamer=can_streamer,
                )
                (
                    q_current,
                    qd_current,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    projected_gravity_b,
                ) = estimator.read()
            safety_check_start = time.monotonic()
            reason = encoder_safety_stop_reason(
                safety=safety,
                estimator=estimator,
                active_joints=motor_layer.active_joints,
                mode=mode,
                require_feedback=True,
                q_shift=q_coordinate_shift,
            )
            safety_check_s += time.monotonic() - safety_check_start
            if reason is not None:
                root_encoder_faulted = True
                print("\nEMERGENCY STOP:", reason)
                command = command_source.read()
                publish_safety_fault(
                    telemetry=telemetry,
                    csv_logger=csv_logger,
                    step=step,
                    mode="encoder_fault",
                    command=command,
                    command_source=command_source,
                    commands=[],
                    estimator=estimator,
                    reason=reason,
                    action=np.zeros(action_dim, dtype=np.float32),
                    phase="policy",
                )
                break

        safety_check_start = time.monotonic()
        stop, reason = safety.emergency_stop_check(
            projected_gravity_b=projected_gravity_b,
            base_ang_vel_b=base_ang_vel_b,
        )
        safety_check_s += time.monotonic() - safety_check_start
        if stop:
            print("\nEMERGENCY STOP:", reason)
            command = command_source.read()
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="safety_fault",
                command=command,
                command_source=command_source,
                commands=[],
                estimator=estimator,
                reason=reason,
                action=np.zeros(action_dim, dtype=np.float32),
                phase="policy",
            )
            break
        required_imu_fault = validate_required_policy_imu(
            estimator,
            max_roll_pitch_deg=imu_active_max_roll_pitch_deg,
        )
        if required_imu_fault is not None and (
            has_motion_target or control_mode not in ("idle",)
        ):
            reason = required_imu_fault
            print("\nEMERGENCY STOP:", reason)
            command = command_source.read()
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="imu_fault",
                command=command,
                command_source=command_source,
                commands=[],
                estimator=estimator,
                reason=reason,
                action=np.zeros(action_dim, dtype=np.float32),
                phase="policy",
            )
            break

        mode_request = command_source.get_mode_request()
        if pose_test_only and mode_request == "policy":
            print("[POSE TEST] policy request blocked; use C, SPACE, H, or X only.")
            mode_request = None
        if mode_request == control_mode and mode_request in ("stand", "sit", "hold"):
            # Ignore terminal auto-repeat and duplicate controller events. A
            # repeated pose request must not recapture feedback and restart the
            # trajectory, which causes a visible stop and torque ramp.
            mode_request = None
        if mode_request in ("stand", "sit", "hold"):
            if can_streamer is not None and mode_request != "hold":
                # Pose capture can intentionally block longer than one policy
                # cycle. Never keep replaying the previous gait target. Hold
                # snapshots the state already read this cycle and immediately
                # replaces the stream, so clearing here would only create a
                # feedback gap.
                can_streamer.clear()
            required_imu_fault = validate_required_policy_imu(
                estimator,
                max_roll_pitch_deg=imu_active_max_roll_pitch_deg,
            )
            if required_imu_fault is not None:
                print(f"[IMU] active command blocked: {required_imu_fault}")
                mode_request = None
        if mode_request is not None:
            if mode_request == "stand" and zero_frame == "crouch" and not zero_calibrated:
                print("\n[ZERO CAL] first pose command is auto-zeroing current crouch/default pose.")
                fresh, n_active = collect_complete_feedback_before_zero(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    safety=safety,
                    buses=buses,
                    mode=mode,
                    feedback_timeout=feedback_timeout,
                )
                if fresh < n_active:
                    print(
                        f"[ZERO CAL] pose command blocked: only {fresh}/{n_active} "
                        "active joints returned fresh feedback."
                    )
                    q_zeroed = False
                else:
                    q_zeroed = apply_software_zero_calibration(
                        estimator=estimator,
                        motor_layer=motor_layer,
                        active_joints=motor_layer.active_joints,
                        feedback_timeout=feedback_timeout,
                        buses=buses,
                        mode=mode,
                        label="crouch/default pose",
                        target_value=crouch_calibration_value,
                    )
                if q_zeroed is False:
                    print(
                        "[ZERO CAL] pose command blocked because valid feedback was not "
                        "available for all active joints."
                    )
                    mode_request = None
                else:
                    scheduler.request_resync("pose command feedback capture")
                    q_previous_target = q_zeroed.copy()
                    q_current = q_zeroed.copy()
                    qd_current = np.zeros_like(q_current, dtype=np.float32)
                    q_coordinate_shift = motor_layer.coordinate_shift_array()
                    zero_calibrated = True
                    calibration_hold_until_step = -1
            elif mode_request == "sit" and zero_frame == "crouch" and not zero_calibrated:
                print(
                    "\n[POSE] crouch/default requested before software zero; "
                    "commanding q=0 without redefining the current pose."
                )
                print(
                    "[POSE] Press SPACE from the crouch/default pose later if you "
                    "want stand auto-zero for policy walking."
                )
            if mode_request is None:
                pass
            elif mode_request in ("stand", "sit"):
                if (
                    not has_motion_target
                    and encoder_feedback_required(mode, estimator)
                    and count_fresh_active_feedback(
                        estimator,
                        motor_layer.active_joints,
                        getattr(safety, "max_feedback_age_s", 0.25),
                    ) < len(motor_layer.active_joints)
                ):
                    request_feedback_snapshot(motor_layer, buses, mode)
                feedback_read_start = time.monotonic()
                feedback_count = refresh_estimator_feedback(
                    estimator,
                    timeout=feedback_timeout,
                )
                feedback_read_s += time.monotonic() - feedback_read_start
                (
                    q_current,
                    qd_current,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    projected_gravity_b,
                ) = estimator.read()
                q_previous_target = q_current.copy()
                if feedback_count > 0:
                    print(
                        f"\n[FEEDBACK] pose transition starts from "
                        f"{feedback_count} measured motor angle(s)"
                    )
                else:
                    cached_count = count_active_feedback(estimator, motor_layer.active_joints)
                    if cached_count > 0:
                        print(
                            f"\n[FEEDBACK] pose transition uses "
                            f"{cached_count} cached live motor angle(s)"
                        )
                if encoder_feedback_required(mode, estimator):
                    fresh = count_fresh_active_feedback(
                        estimator,
                        motor_layer.active_joints,
                        getattr(safety, "max_feedback_age_s", 0.25),
                    )
                    if fresh < len(motor_layer.active_joints):
                        fresh, n_active = refresh_active_feedback_before_fault(
                            estimator=estimator,
                            motor_layer=motor_layer,
                            safety=safety,
                            buses=buses,
                            mode=mode,
                            feedback_timeout=feedback_timeout,
                            q_keepalive=q_previous_target,
                            phase="startup",
                            can_streamer=can_streamer,
                        )
                        if fresh < n_active:
                            print(
                                f"[FEEDBACK] pose transition still has only "
                                f"{fresh}/{n_active} fresh motor feedback frame(s)."
                            )
                safety_check_start = time.monotonic()
                reason = encoder_safety_stop_reason(
                    safety=safety,
                    estimator=estimator,
                    active_joints=motor_layer.active_joints,
                    mode=mode,
                    q_shift=q_coordinate_shift,
                )
                safety_check_s += time.monotonic() - safety_check_start
                if reason is not None:
                    root_encoder_faulted = True
                    print("\nEMERGENCY STOP:", reason)
                    command = command_source.read()
                    publish_safety_fault(
                        telemetry=telemetry,
                        csv_logger=csv_logger,
                        step=step,
                        mode="encoder_fault",
                        command=command,
                        command_source=command_source,
                        commands=[],
                        estimator=estimator,
                        reason=reason,
                        action=np.zeros(action_dim, dtype=np.float32),
                        phase="policy",
                    )
                    break
            elif mode_request == "hold":
                q_hold, feedback_count, hold_missing = (
                    snapshot_hold_target_from_fresh_feedback(
                        estimator,
                        motor_layer,
                        safety,
                        q_previous_target,
                        q_current,
                    )
                )
                q_previous_target = q_hold.copy()
                previous_raw_action = np.zeros(action_dim, dtype=np.float32)
                previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                print(
                    f"\n[FEEDBACK] hold target captured from "
                    f"{feedback_count}/{len(motor_layer.active_joints)} fresh motor angle(s)"
                )
                if hold_missing:
                    shown = ", ".join(hold_missing[:6])
                    if len(hold_missing) > 6:
                        shown += f", +{len(hold_missing) - 6} more"
                    print(
                        "[FEEDBACK] hold kept previous target for stale/missing joint(s): "
                        + shown
                    )
                scheduler.request_resync("hold feedback capture")

            if mode_request is not None:
                pose_requested_from_policy = bool(
                    previous_walk_requested
                    and mode_request in ("stand", "sit", "hold")
                )
                control_mode = mode_request
                if control_mode in ("hold", "stand", "sit"):
                    has_motion_target = True
                if control_mode == "stand" and zero_frame == "crouch":
                    stand_zero_pending = bool(auto_stand_zero)
                    stand_zero_settle_count = 0
                    sit_zero_pending = False
                    sit_zero_settle_count = 0
                    if stand_zero_pending:
                        print("[ZERO CAL] stand target uses stand_pose_when_sit_zero; stand will auto-zero when settled.")
                    else:
                        print(
                            "[POSE] stand target uses stand_pose_when_sit_zero; "
                            "policy walking remains blocked until stand auto-zero is enabled/applied."
                        )
                elif control_mode == "sit":
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                    sit_zero_pending = bool(auto_sit_zero and zero_frame == "stand")
                    sit_zero_settle_count = 0
                    if sit_zero_pending:
                        print("[ZERO CAL] sit target uses sit_pose_when_stand_zero; crouch will auto-zero when settled.")
                elif control_mode == "hold":
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                    sit_zero_pending = False
                    sit_zero_settle_count = 0
                if control_mode == "stand":
                    walking_armed = False
                    stand_ready_pending = True
                    stand_ready_settle_count = 0
                    print("[POSE] walking remains blocked until the stand target settles.")
                elif control_mode in ("sit", "hold"):
                    walking_armed = False
                    stand_ready_pending = False
                    stand_ready_settle_count = 0
                if pose_requested_from_policy:
                    # Position targets already start from fresh feedback. Keep
                    # impedance continuous as well: switching gains immediately
                    # while a joint is moving can produce a large torque step.
                    stand_recovery_gain_active = True
                    stand_recovery_gain_mode = control_mode
                    stand_recovery_gain_elapsed_s = 0.0
                    stand_recovery_policy_alpha_at_stop = float(
                        last_policy_gain_blend_alpha
                    )
                    print(
                        "[POSE] policy impedance will blend smoothly into "
                        f"{control_mode} impedance."
                    )
                else:
                    stand_recovery_gain_active = False
                    stand_recovery_gain_mode = None
                    stand_recovery_gain_elapsed_s = 0.0
                if control_mode in ("stand", "sit", "hold"):
                    last_walk_command.fill(0.0)
                    last_walk_command_step = -10**9
                    walk_stop_candidate_step = -1
                if control_mode in ("stand", "sit"):
                    # Preserve the last transmitted target across pose changes.
                    # Starting from the sagged measured pose would erase the
                    # supporting PD error and briefly unload a weight-bearing
                    # leg before the new trajectory rebuilt that error.
                    begin_pose_transition(control_mode, q_previous_target, step)
                else:
                    pose_transition_mode = None
                    scheduler.request_resync(f"{control_mode} mode initialized")
                print(f"\n[MODE CHANGE] control_mode -> {control_mode}")

        command = command_source.read()
        if pose_test_only:
            command = np.zeros(3, dtype=np.float32)
        command_input_s += time.monotonic() - command_input_start
        if step < calibration_hold_until_step:
            command = np.zeros(3, dtype=np.float32)
        raw_walk_requested = bool(
            not pose_test_only
            and (
                policy_shadow_mode
                or joystick_walk_requested(command, walk_command_threshold)
            )
        )
        walk_requested = raw_walk_requested
        if raw_walk_requested:
            last_walk_command = np.asarray(command, dtype=np.float32).copy()
            last_walk_command_step = int(step)
            walk_stop_candidate_step = -1
        elif (
            getattr(command_source, "source_name", "") == "keyboard"
            and walking_armed
            and control_mode == "stand"
            and joystick_walk_requested(last_walk_command, walk_command_threshold)
        ):
            elapsed_since_walk_s = (int(step) - int(last_walk_command_step)) * float(dt)
            if elapsed_since_walk_s <= float(walk_command_grace_seconds):
                command = last_walk_command.copy()
                walk_requested = True
                walk_stop_candidate_step = -1
            elif previous_walk_requested and float(walk_stop_confirm_seconds) > 0.0:
                if walk_stop_candidate_step < 0:
                    walk_stop_candidate_step = int(step)
                neutral_elapsed_s = (
                    int(step) - int(walk_stop_candidate_step)
                ) * float(dt)
                if neutral_elapsed_s < float(walk_stop_confirm_seconds):
                    command = last_walk_command.copy()
                    walk_requested = True
                else:
                    walk_requested = False
            else:
                walk_stop_candidate_step = -1
        elif not previous_walk_requested:
            walk_stop_candidate_step = -1
        if not pose_test_only and automatic_policy_takeover_requested(
            enabled=auto_policy_after_stand,
            walking_armed=walking_armed,
            control_mode=control_mode,
        ):
            walk_requested = True
        if walk_requested:
            required_imu_fault = validate_required_policy_imu(
                estimator,
                max_roll_pitch_deg=imu_active_max_roll_pitch_deg,
            )
            if required_imu_fault is not None:
                if step % max(1, print_every) == 0:
                    print(f"[IMU] walking command blocked: {required_imu_fault}")
                walk_requested = False
        policy_command = scaled_policy_command(
            command=command,
            gain=policy_command_gain,
            vx_abs_max=policy_command_vx_max,
            vy_abs_max=policy_command_vy_max,
            yaw_abs_max=policy_command_yaw_max,
        )
        if not walking_armed and walk_requested:
            walk_requested = False
            if step % max(1, print_every) == 0:
                print("[POSE] walking blocked until STAND reaches its target.")
        if should_validate_stand_state_for_policy_entry(
            walking_armed=walking_armed,
            walk_requested=walk_requested,
            control_mode=control_mode,
            previous_walk_requested=previous_walk_requested,
        ):
            q_stand_target = stand_pose_for_zero_frame(
                runner,
                zero_frame,
                crouch_calibration_value,
                stand_calibration_value,
            )
            stand_state_ready, stand_error, stand_velocity = (
                stand_state_ready_for_policy_entry(
                    q_current=q_current,
                    qd_current=qd_current,
                    q_stand_target=q_stand_target,
                    active_indices=active_indices,
                    error_tolerance_rad=stand_ready_error_rad,
                    velocity_tolerance_rad_s=stand_ready_velocity_rad_s,
                )
            )
            if not stand_state_ready:
                walk_requested = False
                policy_command = np.zeros(3, dtype=np.float32)
                last_walk_command.fill(0.0)
                last_walk_command_step = -10**9
                walk_stop_candidate_step = -1
                if step % max(1, print_every) == 0:
                    print(
                        "[POSE] walking blocked: measured stand state is not "
                        f"ready (error={stand_error:.3f} rad, "
                        f"speed={stand_velocity:.3f} rad/s). Press SPACE and "
                        "let stand settle before walking."
                    )
        if (
            walk_requested
            and not has_motion_target
            and encoder_feedback_required(mode, estimator)
            and count_fresh_active_feedback(
                estimator,
                motor_layer.active_joints,
                live_feedback_max_age_s,
            ) < len(motor_layer.active_joints)
        ):
            request_feedback_snapshot(motor_layer, buses, mode)
            feedback_read_start = time.monotonic()
            refresh_estimator_feedback(estimator, timeout=feedback_timeout)
            feedback_read_s += time.monotonic() - feedback_read_start
            fresh = count_fresh_active_feedback(
                estimator,
                motor_layer.active_joints,
                live_feedback_max_age_s,
            )
            if fresh < len(motor_layer.active_joints):
                if step % max(1, print_every) == 0:
                    print(
                        f"[FEEDBACK] walking blocked: only {fresh}/"
                        f"{len(motor_layer.active_joints)} active joints have fresh MIT feedback."
                    )
                walk_requested = False
        if control_mode == "sit":
            active_control_mode = "sit"
        elif walk_requested:
            active_control_mode = "policy"
        elif control_mode == "policy":
            active_control_mode = "hold"
        else:
            active_control_mode = control_mode

        fresh_feedback_for_commands = {}
        live_feedback_missing = []
        live_feedback_required = bool(
            not policy_shadow_mode
            and
            active_control_mode in ("hold", "stand", "sit", "policy")
            and encoder_feedback_required(mode, estimator)
            and (has_motion_target or active_control_mode != "hold")
        )
        if live_feedback_required:
            fresh_feedback_for_commands, live_feedback_missing = fresh_feedback_by_joint(
                estimator,
                motor_layer.active_joints,
                live_feedback_max_age_s,
            )
            if live_feedback_missing:
                keepalive_phase = (
                    "policy"
                    if active_control_mode == "policy" or policy_has_started
                    else "startup"
                )
                refresh_active_feedback_before_fault(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    safety=safety,
                    buses=buses,
                    mode=mode,
                    feedback_timeout=feedback_timeout,
                    q_keepalive=q_previous_target,
                    phase=keepalive_phase,
                    max_wait_s=live_feedback_max_age_s,
                    max_age_s=live_feedback_max_age_s,
                    can_streamer=can_streamer,
                )
                (
                    q_current,
                    qd_current,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    projected_gravity_b,
                ) = estimator.read()
                fresh_feedback_for_commands, live_feedback_missing = fresh_feedback_by_joint(
                    estimator,
                    motor_layer.active_joints,
                    live_feedback_max_age_s,
                )

            if live_feedback_missing:
                if step % max(1, print_every) == 0:
                    shown = ", ".join(live_feedback_missing[:4])
                    if len(live_feedback_missing) > 4:
                        shown += f", +{len(live_feedback_missing) - 4} more"
                    print(
                        "[FEEDBACK] live feedback incomplete; freezing target "
                        f"instead of using stale q/qd: {shown}"
                    )
                if active_control_mode == "policy":
                    walk_requested = False
                active_control_mode = "hold" if has_motion_target else "idle"
                previous_raw_action = np.zeros(action_dim, dtype=np.float32)
                previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                fresh_feedback_for_commands, _ = fresh_feedback_by_joint(
                    estimator,
                    motor_layer.active_joints,
                    live_feedback_max_age_s,
                )

        policy_was_started = bool(policy_has_started)
        if walk_requested:
            has_motion_target = True
            policy_has_started = True

        walk_just_stopped = bool(previous_walk_requested and not walk_requested)
        if walk_requested:
            if not previous_walk_requested:
                policy_entry_restart_count += 1
                policy_entry_restart_reason = (
                    "initial movement command"
                    if not policy_was_started
                    else "movement re-entered after intentional stop"
                )
                print(f"[POLICY ENTRY] started: {policy_entry_restart_reason}")
                policy_entry_elapsed_s = 0.0
                policy_entry_q_start = np.asarray(q_previous_target, dtype=np.float32).copy()
                scheduler.request_resync("policy entry initialized")
            policy_entry_elapsed_s += float(dt)
            if float(policy_entry_ramp_seconds) > 0.0:
                policy_entry_scale = smoothstep(
                    min(1.0, policy_entry_elapsed_s / float(policy_entry_ramp_seconds))
                )
            else:
                policy_entry_scale = 1.0
        else:
            policy_entry_elapsed_s = 0.0
            policy_entry_scale = 0.0
        previous_walk_requested = bool(walk_requested)

        if walk_just_stopped and control_mode == "stand":
            # Continue from the exact last transmitted target so releasing a
            # movement key cannot step either position demand or supporting
            # PD torque. Target and gains then recover smoothly to stand.
            begin_pose_transition("stand", q_previous_target, step)
            walking_armed = False
            stand_ready_pending = True
            stand_ready_settle_count = 0
            stand_recovery_gain_active = True
            stand_recovery_gain_mode = "stand"
            stand_recovery_gain_elapsed_s = 0.0
            stand_recovery_policy_alpha_at_stop = float(
                last_policy_gain_blend_alpha
            )
            print(
                "[POSE] policy stopped; last target preserved and loaded stand "
                "impedance recovery started."
            )

        pose_gain_blend_from_phase = None
        pose_gain_blend_alpha = 1.0
        if stand_recovery_gain_active:
            if active_control_mode != stand_recovery_gain_mode:
                stand_recovery_gain_active = False
                stand_recovery_gain_mode = None
                stand_recovery_gain_elapsed_s = 0.0
            else:
                stand_recovery_gain_elapsed_s += float(dt)
                pose_gain_blend_from_phase = "policy"
                pose_gain_blend_alpha = stand_recovery_gain_blend_scale(
                    stand_recovery_policy_alpha_at_stop,
                    stand_recovery_gain_elapsed_s,
                    policy_entry_ramp_seconds,
                )
                if pose_gain_blend_alpha >= 1.0 - 1.0e-6:
                    stand_recovery_gain_active = False
                    stand_recovery_gain_mode = None
                    stand_recovery_gain_elapsed_s = 0.0
                    pose_gain_blend_from_phase = None
                    pose_gain_blend_alpha = 1.0
                    print("[POSE] loaded pose impedance recovery completed.")

        if active_control_mode == "idle":
            q_safe_target = q_previous_target.copy()
            commands = []
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "hold":
            q_safe_target = q_previous_target.copy()
            commands = (
                build_loop_mit_commands(
                    q_safe_target,
                    phase="hold",
                    feedback_by_joint=fresh_feedback_for_commands,
                    gain_blend_from_phase=pose_gain_blend_from_phase,
                    gain_blend_alpha=pose_gain_blend_alpha,
                )
                if has_motion_target
                else []
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "stand":
            q_policy_target = current_pose_transition_target("stand", step)
            stand_command_phase = runtime_stand_command_phase(
                policy_has_started,
                walking_armed,
            )
            learned_stand_stabilization_active = bool(
                stand_policy_stabilization
                and walking_armed
                and zero_frame == "stand"
                and not stand_zero_pending
            )
            action = np.zeros(action_dim, dtype=np.float32)
            if learned_stand_stabilization_active:
                (
                    imu_correction,
                    observation_for_log,
                    raw_action,
                    action,
                ) = stand_policy_imu_correction(
                    runner=runner,
                    base_ang_vel_b=base_ang_vel_b,
                    projected_gravity_b=projected_gravity_b,
                    q_current=q_current,
                    qd_current=qd_current,
                    cfg=motion_assist_cfg,
                )
                q_policy_target = q_policy_target + imu_correction
                imu_correction_abs_max = float(np.max(np.abs(imu_correction)))
            elif (
                not stand_policy_stabilization
                and direct_imu_stabilization_enabled
                and walking_armed
            ):
                requested_leveling_correction = imu_posture_correction(
                    projected_gravity_b=projected_gravity_b,
                    policy_order=runner.policy_order,
                    cfg=motion_assist_cfg,
                )
                smoothing = float(np.clip(
                    motion_assist_cfg.get("imu_posture", {}).get(
                        "correction_smoothing",
                        0.0,
                    ),
                    0.0,
                    0.98,
                ))
                direct_leveling_correction = (
                    smoothing * direct_leveling_correction
                    + (1.0 - smoothing) * requested_leveling_correction
                ).astype(np.float32)
                imu_correction = direct_leveling_correction.copy()
                imu_correction_abs_max = float(np.max(np.abs(imu_correction)))
                q_policy_target = q_policy_target + imu_correction
                stand_command_phase = "leveling"
            elif not stand_policy_stabilization:
                direct_leveling_correction.fill(0.0)
            q_entry_blended_target_for_log = q_policy_target.copy()
            safety_filter_start = time.monotonic()
            q_safe_target, target_diag = shifted_safety_filter_with_diagnostics(
                safety,
                q_policy_target,
                q_previous_target,
                q_coordinate_shift,
            )
            safety_filter_s += time.monotonic() - safety_filter_start
            target_joint_limited_mask = target_diag["target_joint_limited"]
            target_rate_limited_mask = target_diag["target_rate_limited"]
            q_joint_limit_filtered_target_for_log = target_diag["joint_limit_filtered_q_target"]
            q_rate_limited_target_for_log = target_diag["rate_limited_q_target"]
            commands = build_loop_mit_commands(
                q_safe_target,
                phase=stand_command_phase,
                feedback_by_joint=fresh_feedback_for_commands,
                joint_velocity_target=pose_transition_velocity_target,
                joint_feedforward_torque_target=(
                    pose_support_scale * pose_support_tau_target
                ),
                gain_blend_from_phase=pose_gain_blend_from_phase,
                gain_blend_alpha=pose_gain_blend_alpha,
            )

        elif active_control_mode == "sit":
            q_policy_target = current_pose_transition_target("sit", step)
            q_entry_blended_target_for_log = q_policy_target.copy()
            safety_filter_start = time.monotonic()
            q_safe_target, target_diag = shifted_safety_filter_with_diagnostics(
                safety,
                q_policy_target,
                q_previous_target,
                q_coordinate_shift,
            )
            safety_filter_s += time.monotonic() - safety_filter_start
            target_joint_limited_mask = target_diag["target_joint_limited"]
            target_rate_limited_mask = target_diag["target_rate_limited"]
            q_joint_limit_filtered_target_for_log = target_diag["joint_limit_filtered_q_target"]
            q_rate_limited_target_for_log = target_diag["rate_limited_q_target"]
            commands = build_loop_mit_commands(
                q_safe_target,
                phase="sit",
                feedback_by_joint=fresh_feedback_for_commands,
                joint_velocity_target=pose_transition_velocity_target,
                joint_feedforward_torque_target=(
                    pose_support_scale * pose_support_tau_target
                ),
                gain_blend_from_phase=pose_gain_blend_from_phase,
                gain_blend_alpha=pose_gain_blend_alpha,
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "policy":
            previous_action_observation = policy_previous_action_observation(
                previous_raw_action=previous_raw_action,
                previous_sent_action=previous_sent_action,
                exact_policy_after_entry=exact_policy_after_entry,
            )
            observation_build_start = time.monotonic()
            obs = runner.build_observation(
                base_ang_vel_b=base_ang_vel_b,
                projected_gravity_b=projected_gravity_b,
                command=policy_command,
                q_current=q_current,
                qd_current=qd_current,
                previous_action=previous_action_observation,
            )
            observation_build_s += time.monotonic() - observation_build_start

            policy_inference_start = time.monotonic()
            raw_action = runner.infer_action(obs)
            policy_inference_s = time.monotonic() - policy_inference_start
            observation_for_log = obs.copy()
            root_observation_seen = True
            target_conversion_start = time.monotonic()
            q_actor_target = runner.action_to_q_target(raw_action)
            q_actor_target_for_log = q_actor_target.copy()
            if policy_shadow_mode:
                policy_entry_scale = 1.0
                q_policy_target = q_actor_target.copy()
                action = np.asarray(raw_action, dtype=np.float32).copy()
            elif bool(exact_policy_after_entry):
                if float(policy_entry_scale) < 0.999:
                    q_policy_target = (
                        (1.0 - float(policy_entry_scale)) * policy_entry_q_start
                        + float(policy_entry_scale) * q_actor_target
                    ).astype(np.float32)
                    action = action_equivalent_for_q_target(runner, q_policy_target)
                else:
                    q_policy_target = q_actor_target
                    action = np.asarray(raw_action, dtype=np.float32).copy()
            elif policy_sim_match:
                action = np.asarray(raw_action, dtype=np.float32).copy()
                action = np.asarray(action, dtype=np.float32) * float(policy_entry_scale)
                q_policy_target = runner.action_to_q_target(action)
            else:
                control_action = clip_policy_hip_actions(
                    raw_action,
                    runner.policy_order,
                    hip_clip_abs=policy_hip_action_clip,
                    hip_scale=policy_hip_action_scale,
                )
                action = filtered_policy_action(
                    raw_action=control_action,
                    previous_action=previous_sent_action,
                    clip_abs=policy_action_clip,
                    smoothing=policy_action_smoothing,
                    delta_limit_abs=policy_action_delta_limit,
                )
                action = np.asarray(action, dtype=np.float32) * float(policy_entry_scale)
                q_policy_target = runner.action_to_q_target(action)
            q_entry_blended_target_for_log = q_policy_target.copy()
            entry_blend_active_for_log = float(policy_entry_scale) < 0.999
            imu_correction = imu_posture_correction(
                projected_gravity_b=projected_gravity_b,
                policy_order=runner.policy_order,
                cfg=motion_assist_cfg,
            )
            imu_correction_abs_max = float(np.max(np.abs(imu_correction)))
            if not bool(exact_policy_after_entry):
                if not policy_shadow_mode:
                    q_policy_target = apply_motion_assists(
                        q_target=q_policy_target,
                        projected_gravity_b=projected_gravity_b,
                        runner=runner,
                        cfg=motion_assist_cfg,
                    )
            policy_target_conversion_s += time.monotonic() - target_conversion_start
            policy_entry_rate_limit_active = bool(
                (not bool(exact_policy_after_entry) and not policy_sim_match)
                or float(policy_entry_scale) < 0.999
            )
            # Preserve a configured part of the measured loaded-stand support
            # during policy control. This keeps q_default=0 while preventing
            # the gravity bias from disappearing at the end of entry blending.
            policy_pose_support_tau = (
                policy_pose_support_scale(
                    policy_entry_scale,
                    policy_pose_support_floor,
                )
                * pose_support_tau_target
            )
            # Blend only the position target during policy entry. All phases
            # now use official physical gain units; policy impedance starts on
            # the first actor packet and pose recovery blends gains explicitly.
            last_policy_gain_blend_alpha = 1.0
            if policy_shadow_mode:
                q_safe_target = q_actor_target.copy()
                q_joint_limit_filtered_target_for_log = q_actor_target.copy()
                q_rate_limited_target_for_log = q_actor_target.copy()
            else:
                safety_filter_start = time.monotonic()
                q_safe_target, target_diag = shifted_safety_filter_with_diagnostics(
                    safety,
                    q_policy_target,
                    q_previous_target,
                    q_coordinate_shift,
                    apply_rate_limit=policy_entry_rate_limit_active,
                    # Raw actor targets are logged above. Commands sent to real
                    # motors must remain inside the physical joint envelope.
                    use_policy_limits=False,
                )
                safety_filter_s += time.monotonic() - safety_filter_start
                target_joint_limited_mask = target_diag["target_joint_limited"]
                target_rate_limited_mask = target_diag["target_rate_limited"]
                q_joint_limit_filtered_target_for_log = target_diag["joint_limit_filtered_q_target"]
                q_rate_limited_target_for_log = target_diag["rate_limited_q_target"]
            # Build once with the torque limit currently in force. The ramp
            # guard must evaluate the target that would actually be sent, not
            # the actor or pre-torque-limit safety target.
            commands = (
                []
                if policy_shadow_mode
                else build_loop_mit_commands(
                    q_safe_target,
                    phase="policy",
                    feedback_by_joint=fresh_feedback_for_commands,
                    joint_feedforward_torque_target=policy_pose_support_tau,
                    prelimit_q_target=policy_prelimit_target_for_commands(
                        q_policy_target,
                        q_safe_target,
                        exact_policy_after_entry=exact_policy_after_entry,
                    ),
                    previous_command_q=q_previous_target,
                    max_command_delta=safety.dq_max,
                )
            )
            if (
                torque_ramp is not None
                and not policy_shadow_mode
                and not torque_ramp.is_fixed
                # The entry blend always uses the configured start limit, so
                # running the full per-joint ramp supervisor here cannot alter
                # a command. Once entry is complete, 10 Hz supervision is
                # sufficient for an 8 s authority ramp. Hard encoder, tilt,
                # motor-fault, and measured-torque safety still run at 50 Hz.
                and torque_ramp_supervision_due(policy_entry_scale, step)
            ):
                measured_supervision = measured_torque_supervisor.update(estimator)
                measured_soft_limit_active_by_joint_for_log = measured_supervision[
                    "soft_limit_active_by_joint"
                ]
                measured_torque_average_by_joint_for_log = measured_supervision[
                    "average_by_joint"
                ]
                measured_torque_window_max_by_joint_for_log = measured_supervision[
                    "window_max_by_joint"
                ]
                measured_torque_max_for_ramp = max(
                    float(measured_supervision["max_abs"]),
                    max(measured_torque_window_max_by_joint_for_log.values(), default=0.0),
                )
                pre_fresh_count, pre_feedback_age_max_s = feedback_age_summary(
                    estimator,
                    motor_layer.active_joints,
                    live_feedback_max_age_s,
                )
                pre_fresh_count = 0 if pre_fresh_count is None else int(pre_fresh_count)
                pre_feedback_age_max_s = (
                    float("inf")
                    if pre_feedback_age_max_s is None
                    else float(pre_feedback_age_max_s)
                )
                encoder_margin_rad, encoder_margin_joint = encoder_margin_to_policy_limits(
                    safety,
                    q_current,
                    runner.policy_order,
                )
                q_transmitted_for_ramp = command_targets_in_policy_order(
                    q_safe_target,
                    commands,
                    motor_layer.policy_index_by_joint,
                )
                tracking_error_for_ramp = calculate_tracking_errors(
                    q_actor_target_for_log,
                    q_transmitted_for_ramp,
                    q_current,
                ).tracking_error_max
                feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {}) or {}
                faulted = [
                    joint_name
                    for joint_name in motor_layer.active_joints
                    if int((feedback_by_joint.get(joint_name, {}) or {}).get("fault_bits", 0)) != 0
                ]
                motor_fault_reason = (
                    "motor fault: " + ", ".join(faulted[:4])
                    if faulted
                    else None
                )
                imu_fault_for_ramp = validate_required_policy_imu(
                    estimator,
                    max_roll_pitch_deg=imu_active_max_roll_pitch_deg,
                )
                previous_effective_torque_limits = {
                    joint_name: motor_layer.policy_pd_torque_limit_for_joint(
                        joint_name
                    )
                    for joint_name in motor_layer.active_joints
                }
                effective_torque_limits = torque_ramp.update(
                    steady_policy_elapsed_s=float(policy_steady_cycles) * float(dt),
                    entry_complete=float(policy_entry_scale) >= 0.999,
                    imu_fault=imu_fault_for_ramp,
                    feedback_fresh_count=pre_fresh_count,
                    feedback_count_expected=len(motor_layer.active_joints),
                    feedback_age_max_s=pre_feedback_age_max_s,
                    encoder_margin_rad=encoder_margin_rad,
                    encoder_margin_joint=encoder_margin_joint,
                    tracking_error_max=tracking_error_for_ramp,
                    measured_torque_max=measured_torque_max_for_ramp,
                    measured_soft_limit_active=any(
                        measured_soft_limit_active_by_joint_for_log.values()
                    ),
                    cycle_work_s=scheduler.last_snapshot.cycle_work_s,
                    motor_fault=motor_fault_reason,
                    timing_fault=torque_ramp_timing_fault(
                        scheduler.last_snapshot
                    ),
                    print_fn=print,
                )
                torque_ramp_state_for_log = torque_ramp.telemetry()
                torque_limits_changed = any(
                    abs(
                        float(effective_torque_limits[joint_name])
                        - float(previous_effective_torque_limits[joint_name])
                    )
                    > 1.0e-9
                    for joint_name in motor_layer.active_joints
                )
                if torque_limits_changed:
                    motor_layer.set_policy_pd_torque_limits(
                        effective_torque_limits,
                        start_limits_by_joint=torque_ramp.start_by_joint,
                        final_limits_by_joint=torque_ramp.final_by_joint,
                    )
                    # A changed ramp limit affects this cycle's transmitted
                    # target. A paused/unchanged ramp already has the correct
                    # packet set and must not pay for a duplicate 12-motor build.
                    commands = build_loop_mit_commands(
                        q_safe_target,
                        phase="policy",
                        feedback_by_joint=fresh_feedback_for_commands,
                        joint_feedforward_torque_target=policy_pose_support_tau,
                        prelimit_q_target=q_policy_target,
                        previous_command_q=q_previous_target,
                        max_command_delta=safety.dq_max,
                    )
            if float(policy_entry_scale) >= 0.999:
                policy_steady_cycles += 1
                policy_target_clip_counts += (
                    np.abs(np.asarray(q_safe_target) - np.asarray(q_policy_target))
                    > 1.0e-5
                )
                for command_item in commands:
                    if command_item.get("torque_limited"):
                        joint_name = command_item.get("joint_name")
                        index = motor_layer.policy_index_by_joint.get(joint_name)
                        if index is not None:
                            policy_torque_clip_counts[index] += 1

        else:
            raise RuntimeError(f"Unknown control_mode: {active_control_mode}")

        send_repeats = (
            max(1, int(hold_command_repeats))
            if active_control_mode == "hold"
            else 1
        )
        if sit_stand_trace_logger is not None:
            if (
                active_control_mode in ("stand", "sit")
                and pose_transition_mode == active_control_mode
            ):
                transition_progress = min(
                    1.0,
                    max(
                        0.0,
                        float(pose_transition_elapsed_s)
                        / max(float(pose_transition_duration_s), 1.0e-9),
                    ),
                )
                trace_phase = (
                    f"{active_control_mode}_transition"
                    if transition_progress < 0.999
                    else active_control_mode
                )
                stand_progress = (
                    transition_progress
                    if active_control_mode == "stand"
                    else 1.0 - transition_progress
                )
            else:
                transition_progress = 1.0
                trace_phase = active_control_mode
                stand_progress = 1.0 if active_control_mode == "stand" else 0.0
            sit_stand_trace_logger.update_context(
                estimator=estimator,
                controller_phase=trace_phase,
                transition_progress=transition_progress,
                stand_progress=stand_progress,
                transition_duration_s=pose_transition_duration_s,
                control_frequency_hz=1.0 / float(dt),
                can_frequency_hz=(
                    1.0 / float(can_streamer.command_dt_s)
                    if can_streamer is not None
                    else 1.0 / float(dt)
                ),
                battery_voltage=getattr(
                    sit_stand_trace_logger,
                    "battery_voltage_start",
                    None,
                ),
            )
        command_send_timestamp = time.monotonic() if commands else None
        if (
            command_send_timestamp is not None
            and hasattr(estimator, "mark_command_sent")
        ):
            estimator.mark_command_sent(command_send_timestamp)
        if can_streamer is not None:
            can_streamer.submit(commands, timestamp=command_send_timestamp)
            stream_status = can_streamer.telemetry()
            can_tx_s = 0.001 * float(stream_status["can_command_last_batch_ms"])
        else:
            for _ in range(send_repeats):
                if mode == "signal":
                    motor_layer.send_harmless_frames(buses, commands)
                elif mode == "mit-signal":
                    motor_layer.send_signal_commands(buses, commands)
            can_tx_s = max_can_tx_duration_s(buses)

        steady_read_timeout_s = (
            float(steady_feedback_budget_s)
            if commands and encoder_feedback_required(mode, estimator)
            else 0.0
        )
        feedback_read_start = time.monotonic()
        refresh_estimator_feedback(
            estimator,
            timeout=steady_read_timeout_s,
            expected_bus_motor_ids=expected_feedback_ids,
        )
        steady_feedback_read_s = time.monotonic() - feedback_read_start
        feedback_read_s += steady_feedback_read_s
        post_send_fresh_feedback, post_send_missing_feedback = fresh_feedback_by_joint(
            estimator,
            motor_layer.active_joints,
            live_feedback_max_age_s,
        )
        feedback_recency = feedback_recency_summary(
            estimator,
            motor_layer.active_joints,
            live_feedback_max_age_s,
        )
        feedback_fresh_count, feedback_age_max_s = feedback_age_summary(
            estimator,
            motor_layer.active_joints,
            max_age_s=live_feedback_max_age_s,
        )
        post_send_feedback_incomplete = bool(
            commands
            and encoder_feedback_required(mode, estimator)
            and post_send_missing_feedback
        )
        if post_send_feedback_incomplete and step % max(1, print_every) == 0:
            shown = ", ".join(post_send_missing_feedback[:4])
            if len(post_send_missing_feedback) > 4:
                shown += f", +{len(post_send_missing_feedback) - 4} more"
            print(
                "[FEEDBACK] post-send fresh feedback incomplete; keeping the "
                f"controller alive so the next loop can refresh/freeze: {shown}"
            )
        require_command_feedback = bool(
            commands
            and encoder_feedback_required(mode, estimator)
            and not post_send_feedback_incomplete
        )
        safety_feedback_by_joint = (
            post_send_fresh_feedback if post_send_fresh_feedback else None
        )
        safety_check_start = time.monotonic()
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=estimator,
            active_joints=motor_layer.active_joints,
            mode=mode,
            require_feedback=require_command_feedback,
            q_shift=q_coordinate_shift,
            feedback_by_joint=safety_feedback_by_joint,
            # Feedback is checked against the physical envelope in every mode.
            # The configured margin permits ordinary tracking overshoot without
            # allowing a wide actor diagnostic range to become an encoder range.
            use_policy_limits=False,
        )
        safety_check_s += time.monotonic() - safety_check_start
        if reason is not None:
            root_encoder_faulted = True
            print("\nEMERGENCY STOP:", reason)
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="encoder_fault",
                command=command,
                command_source=command_source,
                commands=commands,
                estimator=estimator,
                reason=reason,
                action=action,
                phase="policy",
            )
            break

        target_advance_scale = 1.0
        if (
            active_control_mode in ("stand", "sit")
            and encoder_feedback_required(mode, estimator)
            and float(pose_sync_error_rad) > 0.0
        ):
            q_feedback = getattr(estimator, "q_current", None)
            if q_feedback is not None:
                # q_safe_target is the requested trajectory point. The MIT
                # torque limiter may send a closer per-joint q_des, so compare
                # feedback against what was actually sent to avoid a permanent
                # false lag at the torque-limit boundary.
                q_sync_target = np.asarray(q_safe_target, dtype=np.float32).copy()
                for command_item in commands:
                    joint_name = command_item.get("joint_name")
                    index = motor_layer.policy_index_by_joint.get(joint_name)
                    if index is not None and "q_des" in command_item:
                        q_sync_target[index] = float(command_item["q_des"])
                sync_error = max_active_error(q_feedback, q_sync_target, active_indices)
                sync_limit = float(pose_sync_error_rad)
                if sync_error > sync_limit:
                    hard_stop_error = 1.5 * sync_limit
                    normalized = np.clip(
                        (hard_stop_error - sync_error)
                        / max(1.0e-6, hard_stop_error - sync_limit),
                        0.0,
                        1.0,
                    )
                    # Do not fully freeze a pose trajectory. A zero advance
                    # creates the visible move-stop-move pulse when one loaded
                    # joint follows more slowly than the others. Hard joint,
                    # rate, torque, encoder, and tilt limits remain enforced.
                    target_advance_scale = max(0.15, smoothstep(normalized))
                    if step % max(1, print_every) == 0:
                        print(
                            f"[SYNC] slowing pose trajectory: max joint lag "
                            f"{sync_error:.3f} rad, advance="
                            f"{100.0 * target_advance_scale:.0f}%"
                        )

        if active_control_mode == "stand" and stand_zero_pending:
            q_feedback = getattr(estimator, "q_current", q_safe_target)
            command_error = max_active_error(q_safe_target, q_policy_target, active_indices)
            feedback_error = max_active_error(q_feedback, q_policy_target, active_indices)
            if (
                command_error <= float(stand_zero_error_rad)
                and feedback_error <= float(stand_zero_error_rad)
            ):
                stand_zero_settle_count += 1
            else:
                stand_zero_settle_count = 0

            if stand_zero_settle_count >= int(stand_zero_settle_steps):
                if can_streamer is not None:
                    can_streamer.clear()
                q_zeroed = apply_software_zero_calibration(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    active_joints=motor_layer.active_joints,
                    feedback_timeout=feedback_timeout,
                    buses=buses,
                    mode=mode,
                    label="stand pose",
                    target_value=stand_calibration_value,
                )
                if q_zeroed is not False:
                    scheduler.request_resync("stand software-zero calibration")
                    zero_frame = "stand"
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                    q_coordinate_shift = motor_layer.coordinate_shift_array()
                    q_safe_target = (
                        constant_pose_like(runner, stand_calibration_value)
                        + runner.q_stand
                    )
                    q_previous_target = q_safe_target.copy()
                    zero_calibrated = True
                    previous_raw_action = np.zeros(action_dim, dtype=np.float32)
                    previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                    commands = build_loop_mit_commands(
                        q_safe_target,
                        phase="startup",
                        feedback_by_joint=fresh_feedback_for_commands,
                    )
                    if can_streamer is not None:
                        can_streamer.submit(commands)
                    elif mode == "signal":
                        motor_layer.send_harmless_frames(buses, commands)
                    elif mode == "mit-signal":
                        motor_layer.send_signal_commands(buses, commands)
                    print("[ZERO CAL] zero_frame -> stand. Policy walking is now enabled in RL zero coordinates.")

        if active_control_mode == "stand" and stand_ready_pending:
            q_stand_target = stand_pose_for_zero_frame(
                runner,
                zero_frame,
                crouch_calibration_value,
                stand_calibration_value,
            )
            q_feedback = getattr(estimator, "q_current", q_safe_target)
            command_error = max_active_error(q_safe_target, q_stand_target, active_indices)
            feedback_error = max_active_error(q_feedback, q_stand_target, active_indices)
            measured_stand_ready, _, _ = stand_state_ready_for_policy_entry(
                q_current=q_feedback,
                qd_current=qd_current,
                q_stand_target=q_stand_target,
                active_indices=active_indices,
                error_tolerance_rad=stand_ready_error_rad,
                velocity_tolerance_rad_s=stand_ready_velocity_rad_s,
            )
            if (
                not stand_recovery_gain_active
                and measured_stand_ready
                and stand_ready_for_walking(
                    command_error=command_error,
                    feedback_error=feedback_error,
                    trajectory_elapsed_s=pose_transition_elapsed_s,
                    trajectory_duration_s=pose_transition_duration_s,
                    error_tolerance_rad=stand_ready_error_rad,
                )
            ):
                stand_ready_settle_count += 1
            else:
                stand_ready_settle_count = 0
            if stand_ready_settle_count >= int(stand_zero_settle_steps):
                walking_armed = not pose_test_only
                stand_ready_pending = False
                stand_ready_settle_count = 0
                previous_raw_action = np.zeros(action_dim, dtype=np.float32)
                previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                if pose_test_only:
                    print(
                        "[POSE TEST] stand settled. Policy and walking remain disabled."
                    )
                elif auto_policy_after_stand:
                    print(
                        "[POSE] stand settled. Automatic zero-command policy "
                        "takeover starts on the next control cycle."
                    )
                elif stand_policy_stabilization:
                    print(
                        "[POSE] stand settled. Differential RL IMU stabilization "
                        "is active; policy walking is armed."
                    )
                elif direct_imu_stabilization_enabled:
                    print(
                        "[POSE] stand settled. Direct IMU body leveling is "
                        "active; policy walking is armed."
                    )
                else:
                    print(
                        "[POSE] stand settled. Policy walking is armed; "
                        "stand target remains fixed until a movement command."
                    )

        if active_control_mode == "sit" and sit_zero_pending:
            q_feedback = getattr(estimator, "q_current", q_safe_target)
            command_error = max_active_error(q_safe_target, q_policy_target, active_indices)
            feedback_error = max_active_error(q_feedback, q_policy_target, active_indices)
            if (
                command_error <= float(stand_zero_error_rad)
                and feedback_error <= float(stand_zero_error_rad)
            ):
                sit_zero_settle_count += 1
            else:
                sit_zero_settle_count = 0

            if sit_zero_settle_count >= int(stand_zero_settle_steps):
                if can_streamer is not None:
                    can_streamer.clear()
                q_zeroed = apply_software_zero_calibration(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    active_joints=motor_layer.active_joints,
                    feedback_timeout=feedback_timeout,
                    buses=buses,
                    mode=mode,
                    label="crouch/sit pose",
                    target_value=crouch_calibration_value,
                )
                if q_zeroed is not False:
                    scheduler.request_resync("sit software-zero calibration")
                    zero_frame = "crouch"
                    sit_zero_pending = False
                    sit_zero_settle_count = 0
                    q_coordinate_shift = motor_layer.coordinate_shift_array()
                    q_safe_target = constant_pose_like(runner, crouch_calibration_value)
                    q_previous_target = q_safe_target.copy()
                    zero_calibrated = True
                    previous_raw_action = np.zeros(action_dim, dtype=np.float32)
                    previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                    commands = build_loop_mit_commands(
                        q_safe_target,
                        phase="startup",
                        feedback_by_joint=fresh_feedback_for_commands,
                    )
                    if can_streamer is not None:
                        can_streamer.submit(commands)
                    elif mode == "signal":
                        motor_layer.send_harmless_frames(buses, commands)
                    elif mode == "mit-signal":
                        motor_layer.send_signal_commands(buses, commands)
                    print(
                        "[ZERO CAL] zero_frame -> crouch. Crouch hold is active at "
                        f"q={float(crouch_calibration_value):+.3f}."
                    )

        q_sent_target = command_targets_in_policy_order(
            q_safe_target,
            commands,
            motor_layer.policy_index_by_joint,
        )
        should_log_csv = step % max(1, log_every) == 0
        should_print = step % max(1, print_every) == 0
        if should_log_csv or should_print:
            logging_start = time.monotonic()
            timing = scheduler.last_snapshot
            telemetry_record = compact_telemetry_record(
                step=step,
                mode=active_control_mode,
                command=command,
                command_source=command_source,
                commands=commands,
                estimator=estimator,
                action=action,
                policy_command=policy_command,
                phase=(
                    "policy"
                    if active_control_mode == "policy"
                    else "pose"
                    if active_control_mode in ("stand", "sit", "hold")
                    else "runtime"
                ),
                observation=observation_for_log,
                raw_action=raw_action,
                sent_action=action,
                q_current=q_current,
                qd_current=qd_current,
                q_actor_target=q_actor_target_for_log,
                q_entry_blended_target=q_entry_blended_target_for_log,
                q_joint_limit_filtered_target=q_joint_limit_filtered_target_for_log,
                q_rate_limited_target=q_rate_limited_target_for_log,
                q_safety_target=q_safe_target,
                q_target=q_sent_target,
                target_joint_limited=target_joint_limited_mask,
                target_rate_limited=target_rate_limited_mask,
                entry_blend_active=entry_blend_active_for_log,
                policy_order=runner.policy_order,
                policy_sha256=runner.policy_sha256,
                policy_entry_scale=policy_entry_scale,
                policy_entry_elapsed_s=policy_entry_elapsed_s,
                policy_entry_restart_count=policy_entry_restart_count,
                policy_entry_restart_reason=policy_entry_restart_reason,
                imu_correction_abs_max=imu_correction_abs_max,
                loop_dt_s=loop_dt_s,
                loop_period_s=loop_dt_s,
                cycle_work_s=timing.cycle_work_s,
                deadline_lateness_s=timing.deadline_lateness_s,
                policy_inference_s=policy_inference_s,
                command_input_s=command_input_s,
                observation_build_s=observation_build_s,
                policy_target_conversion_s=policy_target_conversion_s,
                safety_filter_s=safety_filter_s,
                command_build_s=command_build_s,
                can_tx_s=can_tx_s,
                feedback_read_s=feedback_read_s,
                pre_feedback_read_s=pre_feedback_read_s,
                steady_feedback_read_s=steady_feedback_read_s,
                safety_check_s=safety_check_s,
                logging_s=logging_s,
                terminal_print_s=terminal_print_s,
                imu_cache_read_s=imu_cache_read_s,
                feedback_age_max_s=feedback_age_max_s,
                feedback_fresh_count=feedback_fresh_count,
                feedback_current_cycle_count=feedback_recency["fresh_current_cycle"],
                feedback_previous_cycle_count=feedback_recency["fresh_previous_cycle"],
                feedback_missing_count=feedback_recency["missing"],
                feedback_stale_count=feedback_recency["stale"],
                torque_ramp_state=torque_ramp_state_for_log,
                measured_soft_limit_active_by_joint=measured_soft_limit_active_by_joint_for_log,
                measured_torque_average_by_joint=measured_torque_average_by_joint_for_log,
                measured_torque_window_max_by_joint=measured_torque_window_max_by_joint_for_log,
                missed_deadlines=timing.total_missed_deadlines,
                consecutive_overruns=timing.consecutive_work_overruns,
                max_overrun_s=timing.maximum_lateness_s,
                missed_deadlines_total=timing.total_missed_deadlines,
                missed_deadlines_this_cycle=timing.missed_deadlines_this_cycle,
                consecutive_work_overruns=timing.consecutive_work_overruns,
                scheduler_resync_count=timing.scheduler_resync_count,
                max_cycle_work_s=timing.maximum_work_s,
                max_lateness_s=timing.maximum_lateness_s,
                policy_steady_cycles=policy_steady_cycles,
                policy_target_clip_counts=policy_target_clip_counts,
                policy_torque_clip_counts=policy_torque_clip_counts,
            )
            logging_s = time.monotonic() - logging_start
            telemetry_record["logging_ms"] = 1000.0 * float(logging_s)
            telemetry_record["csv_logging_ms"] = 0.0
            telemetry_record["terminal_print_ms"] = 0.0
            if can_streamer is not None:
                telemetry_record.update(can_streamer.telemetry())
            if should_print:
                terminal_print_start = time.monotonic()
                print(compact_telemetry_line(telemetry_record))
                if show_hex and commands:
                    print_mit_commands(commands, show_hex=True)
                if joystick_debug:
                    print_joystick_debug(command_source)
                if joint_debug:
                    print_joint_debug(commands, estimator)
                terminal_print_s = time.monotonic() - terminal_print_start
                telemetry_record["terminal_print_ms"] = 1000.0 * float(terminal_print_s)
            if csv_logger is not None and should_log_csv:
                csv_logging_s = csv_logger.log(telemetry_record)
                logging_s += csv_logging_s
                telemetry_record["csv_logging_ms"] = 1000.0 * float(csv_logging_s)

        if active_control_mode == "policy":
            if q_actor_target_for_log is not None:
                root_raw_signals.append(np.asarray(raw_action, dtype=np.float32).copy())
                root_transmitted_signals.append(q_sent_target.copy())
                root_measured_signals.append(np.asarray(q_current, dtype=np.float32).copy())
                root_tracking_error_maxima.append(
                    calculate_tracking_errors(
                        q_actor_target_for_log,
                        q_sent_target,
                        q_current,
                    ).tracking_error_max
                )
                feedback = getattr(estimator, "last_feedback_by_joint", {}) or {}
                if all(
                    (feedback.get(name, {}) or {}).get("joint_velocity_mit") is not None
                    and (feedback.get(name, {}) or {}).get(
                        "joint_velocity_finite_difference"
                    ) is not None
                    for name in runner.policy_order
                ):
                    root_velocity_mit.append(np.asarray([
                        float(feedback[name]["joint_velocity_mit"])
                        for name in runner.policy_order
                    ], dtype=np.float32))
                    root_velocity_fd.append(np.asarray([
                        float(feedback[name]["joint_velocity_finite_difference"])
                        for name in runner.policy_order
                    ], dtype=np.float32))
            if step % policy_summary_stride == 0:
                update_policy_joint_summary(
                    policy_joint_summary,
                    runner.policy_order,
                    commands,
                    estimator,
                    q_actor_target_for_log,
                    q_sent_target,
                    target_joint_limited=target_joint_limited_mask,
                    target_rate_limited=target_rate_limited_mask,
                )
            now_status = time.monotonic()
            if (
                float(suspension_status_seconds) > 0.0
                and now_status - last_suspension_status_time
                >= float(suspension_status_seconds)
            ):
                last_suspension_status_time = now_status
                raw_action_max = (
                    0.0
                    if raw_action is None
                    else float(np.max(np.abs(np.asarray(raw_action, dtype=np.float32))))
                )
                actor_motion_max = (
                    0.0
                    if q_actor_target_for_log is None
                    else float(np.max(np.abs(q_actor_target_for_log - q_previous_target)))
                )
                sent_motion_max = float(np.max(np.abs(q_sent_target - q_previous_target)))
                q_fb = np.asarray(getattr(estimator, "q_current", q_sent_target), dtype=np.float32)
                errors = calculate_tracking_errors(
                    q_actor_target_for_log,
                    q_sent_target,
                    q_fb,
                )
                tracking_error_max = errors.tracking_error_max
                policy_authority_loss_max = errors.policy_authority_loss_max
                torque_limited_joints = sum(1 for item in commands if item.get("torque_limited"))
                joint_limited_joints = int(np.count_nonzero(target_joint_limited_mask))
                fresh_count = (
                    0 if feedback_fresh_count in (None, "") else int(feedback_fresh_count)
                )
                print(
                    "[SUSPENSION] "
                    f"mode={active_control_mode} "
                    f"imu={estimator_imu_status(estimator)} "
                    f"command=[{float(policy_command[0]):+.3f},"
                    f"{float(policy_command[1]):+.3f},"
                    f"{float(policy_command[2]):+.3f}] "
                    f"entry={100.0 * float(policy_entry_scale):.0f}% "
                    f"raw_action_max={raw_action_max:.3f} "
                    f"actor_target_motion_max={actor_motion_max:.3f} "
                    f"sent_target_motion_max={sent_motion_max:.3f} "
                    f"tracking_error_max={tracking_error_max:.3f} "
                    f"policy_authority_loss_max={policy_authority_loss_max:.3f} "
                    f"torque_limited_joints={torque_limited_joints} "
                    f"joint_limited_joints={joint_limited_joints} "
                    f"fresh_feedback={fresh_count}/{len(motor_layer.active_joints)} "
                    f"loop_work={1000.0 * scheduler.last_snapshot.cycle_work_s:.1f}ms"
                )

        if not policy_shadow_mode:
            estimator.dry_update_as_if_robot_followed(q_sent_target, dt)

        if active_control_mode == "policy":
            previous_raw_action = raw_action.copy()
            previous_sent_action = action.copy()
        else:
            previous_raw_action = np.zeros(action_dim, dtype=np.float32)
            previous_sent_action = np.zeros(action_dim, dtype=np.float32)

        if (
            active_control_mode in ("stand", "sit")
            and pose_transition_mode == active_control_mode
        ):
            # Lag slows one shared body trajectory clock. Never scale joints
            # independently here: that destroys synchronized pose phase and
            # produces visible stair-step corrections between legs.
            pose_transition_elapsed_s = min(
                pose_transition_duration_s,
                pose_transition_elapsed_s
                + float(dt) * float(target_advance_scale),
            )

        # Use the post-limit q_des values packed into the MIT commands as the
        # sole next-cycle reference. q_safe_target can differ when the policy
        # PD torque limiter pulls a target closer to measured feedback.
        q_previous_target = q_sent_target

        if telemetry is not None and step % 2 == 0:
            telemetry.send(
                step=step,
                mode=active_control_mode,
                command=command,
                command_source=command_source,
                commands=commands,
                action=action,
                safety_ok=True,
        )

        step += 1
        timing = scheduler.finish_cycle(cycle_start)
        timing_warning_now = time.monotonic()
        if (
            timing.work_overrun
            and not timing.timing_fault
            and timing_warning_now - last_timing_breakdown_print_time >= 0.5
        ):
            last_timing_breakdown_print_time = timing_warning_now
            print(
                timing_breakdown_line(
                    cycle_work_s=timing.cycle_work_s,
                    imu_cache_read_s=imu_cache_read_s,
                    command_input_s=command_input_s,
                    observation_build_s=observation_build_s,
                    policy_inference_s=policy_inference_s,
                    policy_target_conversion_s=policy_target_conversion_s,
                    safety_filter_s=safety_filter_s,
                    command_build_s=command_build_s,
                    can_tx_s=can_tx_s,
                    steady_feedback_read_s=steady_feedback_read_s,
                    safety_check_s=safety_check_s,
                    logging_s=logging_s,
                    terminal_print_s=terminal_print_s,
                )
            )
        if timing.timing_fault:
            breakdown = timing_breakdown_line(
                cycle_work_s=timing.cycle_work_s,
                imu_cache_read_s=imu_cache_read_s,
                command_input_s=command_input_s,
                observation_build_s=observation_build_s,
                policy_inference_s=policy_inference_s,
                policy_target_conversion_s=policy_target_conversion_s,
                safety_filter_s=safety_filter_s,
                command_build_s=command_build_s,
                can_tx_s=can_tx_s,
                steady_feedback_read_s=steady_feedback_read_s,
                safety_check_s=safety_check_s,
                logging_s=logging_s,
                terminal_print_s=terminal_print_s,
            )
            print(
                "\nTIMING FAULT: controller exceeded the control-period "
                f"work budget for {timing.consecutive_work_overruns} "
                "consecutive cycles; stopping instead of continuing with "
                "irregular gait timing."
            )
            print(breakdown)
            publish_safety_fault(
                telemetry=telemetry,
                csv_logger=csv_logger,
                step=step,
                mode="timing_fault",
                command=command,
                command_source=command_source,
                commands=commands,
                estimator=estimator,
                reason=(
                    "controller sustained work overrun: "
                    f"cycle_work={1000.0 * timing.cycle_work_s:.1f}ms "
                    f"dt={1000.0 * float(dt):.1f}ms "
                    f"tolerance={1000.0 * scheduler.deadline_tolerance_s:.1f}ms "
                    + breakdown
                ),
                action=action,
                phase="runtime",
            )
            break

    print_policy_joint_summary(
        policy_joint_summary,
        runner.policy_order,
        steady_cycles=policy_steady_cycles,
        steady_torque_counts=policy_torque_clip_counts,
    )
    if can_streamer is not None:
        can_streamer.clear()
    print("\nRuntime control phase completed.")
    can_command_status = "UNKNOWN"
    if can_streamer is not None:
        can_status = can_streamer.telemetry()
        if int(can_status["can_command_send_count"]) > 0:
            can_command_status = (
                "PASS"
                if not can_status["can_command_fault"]
                and int(can_status["can_command_missed_deadlines"]) == 0
                else "FAIL"
            )
    def gait_status(samples, actor_actions=False):
        if len(samples) < 50:
            return "UNKNOWN"
        values = np.asarray(samples, dtype=np.float32)
        if actor_actions:
            values = values * runner.action_scale
        return classify_diagonal_trot(values, 1.0 / float(dt))[0]

    raw_status = gait_status(root_raw_signals, actor_actions=True)
    transmitted_status = "UNKNOWN" if policy_shadow_mode else gait_status(root_transmitted_signals)
    measured_status = "UNKNOWN" if policy_shadow_mode else gait_status(root_measured_signals)
    selected_joint_velocity_source = str(
        getattr(estimator, "joint_velocity_source", "fake")
    )
    if policy_shadow_mode:
        velocity_status = "UNKNOWN"
    elif selected_joint_velocity_source == "finite-difference":
        finite_difference_values = np.asarray(root_velocity_fd, dtype=np.float32)
        velocity_status = (
            "FD SOURCE"
            if len(root_velocity_fd) >= 20
            and np.all(np.isfinite(finite_difference_values))
            else "FAIL"
        )
    elif len(root_velocity_mit) >= 20:
        velocity_passed, _ = validate_joint_velocity_arrays(
            np.asarray(root_velocity_mit),
            np.asarray(root_velocity_fd),
        )
        velocity_status = "PASS" if velocity_passed else "FAIL"
    else:
        velocity_status = "UNKNOWN"
    torque_total = max(1, int(policy_steady_cycles) * len(runner.policy_order))
    torque_clip_ratio = float(np.sum(policy_torque_clip_counts)) / float(torque_total)
    report = {
        "raw_actor_periodic_gait": raw_status,
        "policy_joint_order": "PASS",
        "motor_routing": "PASS",
        "joint_velocity_validation": velocity_status,
        "observation_contract": "PASS" if root_observation_seen else "UNKNOWN",
        "actor_gait_preserved": (
            "UNKNOWN"
            if raw_status == "UNKNOWN" or transmitted_status == "UNKNOWN"
            else "PASS"
            if raw_status == "PASS" and transmitted_status == "PASS"
            else "FAIL"
        ),
        "motor_tracking": (
            "UNKNOWN"
            if policy_shadow_mode or not root_tracking_error_maxima
            else "PASS"
            if float(np.percentile(root_tracking_error_maxima, 95)) <= 0.25
            else "FAIL"
        ),
        "timing_50hz": (
            "PASS"
            if timing_qualification_passed(
                scheduler.total_missed_deadlines,
                step,
                scheduler.consecutive_work_overruns,
            )
            else "FAIL"
        ),
        "can_command_200hz": can_command_status,
        "encoder_calibration": (
            "NOT REQUIRED"
            if not encoder_calibration_required
            else "PASS"
            if encoder_calibration_passed and not root_encoder_faulted
            else "FAIL"
        ),
        "torque_authority": (
            "UNKNOWN"
            if policy_shadow_mode or policy_steady_cycles <= 0
            else "PASS" if torque_clip_ratio < 0.50 else "FAIL"
        ),
        "ground_contact_validity": "NOT TESTED",
        "measured_gait": measured_status,
    }
    return report


def main():
    # The 200 Hz CAN sender shares the CPython process with the 50 Hz policy
    # loop.  CPython's default ~5 ms thread switch interval can starve a CAN
    # lane for an entire command period while observation/target arrays are
    # being built.  A 1 ms interval keeps both SocketCAN lanes schedulable;
    # the transport deadline itself remains the strict 5 ms safety limit.
    sys.setswitchinterval(0.001)

    joystick_defaults = load_joystick_defaults()
    speed_defaults = load_speed_scale_defaults()
    imu_defaults = load_imu_config()
    motion_assist_defaults = load_motion_assist_config()
    policy_deploy_defaults = load_policy_deployment_defaults()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["print", "signal", "mit-signal", "motors"],
        default="print",
        help="print=no serial, signal=harmless empty CAN frames, mit-signal=sends MIT packets, motors=blocked",
    )

    add_can_topology_args(
        parser,
        default_port="slcan0",
        default_can_count=2,
        default_backend="socketcan",
    )
    parser.set_defaults(port_front="slcan0", port_back="slcan1")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--active-joints",
        nargs="*",
        default=None,
        help="only send motor commands to these joint names; default uses config/motor_ids.yaml",
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--allow-policy-hash-mismatch",
        action="store_true",
        help="allow an unrecognized policy SHA256 after explicit artifact verification",
    )
    parser.add_argument(
        "--policy-activation",
        choices=["elu", "relu", "tanh", "identity", "none"],
        default="elu",
        help="activation to use when loading non-TorchScript actor checkpoints",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=0.0,
        help=(
            "runtime controller update rate; 0 uses the policy's trained rate "
            "(50 Hz). Use 25 only for conservative suspended testing"
        ),
    )
    parser.add_argument(
        "--can-command-hz",
        type=float,
        default=200.0,
        help="low-level latest-target MIT retransmission rate; fixed at 200 Hz",
    )
    parser.add_argument(
        "--can-command-stale-timeout",
        type=float,
        default=0.250,
        help=(
            "stop retransmission if the 50 Hz producer does not refresh its "
            "target; BCM safely holds the last complete target through brief "
            "Jetson scheduling stalls"
        ),
    )
    parser.add_argument(
        "--can-command-fault-consecutive",
        type=int,
        default=3,
        help="fault after this many consecutive CAN batches exceed the 5 ms budget",
    )
    parser.add_argument(
        "--can-tx-timeout-ms",
        type=float,
        default=1.0,
        help=(
            "maximum SocketCAN wait for each frame submission; production uses "
            "no retry so a full queue faults instead of delaying newer targets"
        ),
    )
    parser.add_argument(
        "--python-thread-switch-ms",
        type=float,
        default=1.0,
        help="CPython thread handoff interval used by the 200 Hz CAN sender",
    )

    parser.add_argument("--command-source", choices=["fixed", "joystick", "keyboard"], default="keyboard")

    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=float(joystick_defaults["speed_limits"]["max_vx"]))
    parser.add_argument("--max-vy", type=float, default=float(joystick_defaults["speed_limits"]["max_vy"]))
    parser.add_argument("--max-yaw", type=float, default=float(joystick_defaults["speed_limits"]["max_yaw"]))
    parser.add_argument(
        "--keyboard-command-timeout",
        type=float,
        default=0.35,
        help="seconds a terminal movement key remains active unless repeated; hold w/a/s/d to keep moving",
    )
    parser.add_argument(
        "--keyboard-control-mode",
        choices=["repeat", "latched"],
        default="repeat",
        help="repeat uses terminal key-repeat; latched keeps the last movement command active until pose/hold/e-stop",
    )
    parser.add_argument(
        "--keyboard-latched-combo-window",
        type=float,
        default=0.18,
        help="seconds in latched mode for combining w/a/s/d/q/e into diagonal or turn commands",
    )

    parser.add_argument("--axis-vx", type=int, default=int(joystick_defaults["axes"]["vx_axis"]))
    parser.add_argument("--axis-vy", type=int, default=int(joystick_defaults["axes"]["vy_axis"]))
    parser.add_argument("--axis-yaw", type=int, default=int(joystick_defaults["axes"]["yaw_axis"]))
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument(
        "--joystick-wait-seconds",
        type=float,
        default=float(joystick_defaults.get("connection", {}).get("wait_seconds", 5.0)),
    )

    parser.add_argument("--invert-vx", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["vx"]))
    parser.add_argument("--invert-vy", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["vy"]))
    parser.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["yaw"]))

    parser.add_argument("--button-stand", type=int, default=int(joystick_defaults["buttons"]["stand"]))
    parser.add_argument("--button-sit", "--button-sit-stop", type=int, default=int(joystick_defaults["buttons"]["sit"]))
    parser.add_argument("--button-policy", type=int, default=int(joystick_defaults["buttons"]["policy"]))
    parser.add_argument(
        "--button-speed-down",
        type=int,
        default=int(joystick_defaults["buttons"].get("speed_down", 6)),
    )
    parser.add_argument(
        "--button-speed-up",
        type=int,
        default=int(joystick_defaults["buttons"].get("speed_up", 7)),
    )
    parser.add_argument(
        "--button-zero-calibration",
        type=int,
        default=int(joystick_defaults["buttons"].get("zero_calibration", -1)),
    )
    parser.add_argument(
        "--zero-calibration-hat-index",
        type=int,
        default=int(joystick_defaults.get("dpad", {}).get("zero_calibration_hat_index", 0)),
    )
    parser.add_argument(
        "--zero-calibration-hat-direction",
        type=int,
        nargs=2,
        default=[
            int(v)
            for v in joystick_defaults.get("dpad", {}).get(
                "zero_calibration_hat_direction",
                [0, -1],
            )
        ],
    )
    parser.add_argument(
        "--zero-calibration-hat-any",
        action=argparse.BooleanOptionalAction,
        default=bool(joystick_defaults.get("dpad", {}).get("zero_calibration_hat_any", False)),
        help="use any non-centered D-pad/hat direction as the software-zero trigger",
    )
    parser.add_argument(
        "--zero-calibration-axis",
        type=int,
        default=int(joystick_defaults.get("dpad", {}).get("zero_calibration_axis", 1)),
        help="axis used as a software-zero trigger; -1 disables axis trigger",
    )
    parser.add_argument(
        "--zero-calibration-axis-direction",
        type=float,
        default=float(joystick_defaults.get("dpad", {}).get("zero_calibration_axis_direction", 1.0)),
        help="axis sign that triggers software zero: +1 or -1",
    )
    parser.add_argument(
        "--zero-calibration-axis-threshold",
        type=float,
        default=float(joystick_defaults.get("dpad", {}).get("zero_calibration_axis_threshold", 0.8)),
        help="absolute axis threshold for software-zero trigger",
    )
    parser.add_argument(
        "--zero-calibration-cooldown-s",
        type=float,
        default=float(joystick_defaults.get("dpad", {}).get("zero_calibration_cooldown_s", 1.0)),
        help="minimum seconds between repeated D-pad/axis software-zero requests",
    )
    parser.add_argument(
        "--button-emergency-stop",
        type=int,
        nargs="+",
        default=[
            int(button_id)
            for button_id in joystick_defaults["buttons"].get("emergency_stop", [0, 1, 2, 3])
        ],
    )

    parser.add_argument("--deadzone", type=float, default=float(joystick_defaults["filter"]["deadzone"]))
    parser.add_argument("--expo", type=float, default=float(joystick_defaults["filter"]["expo"]))
    parser.add_argument("--smoothing", type=float, default=float(joystick_defaults["filter"]["smoothing"]))
    parser.add_argument(
        "--use-hat-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(joystick_defaults["filter"].get("use_hat_fallback", True)),
    )
    parser.add_argument("--speed-scale-initial", type=float, default=speed_defaults["initial"])
    parser.add_argument("--speed-scale-min", type=float, default=speed_defaults["min"])
    parser.add_argument("--speed-scale-max", type=float, default=speed_defaults["max"])
    parser.add_argument("--speed-scale-step", type=float, default=speed_defaults["step"])

    parser.add_argument(
        "--policy-steps",
        type=int,
        default=0,
        help="policy loop steps; 0 or negative runs until emergency stop",
    )
    parser.add_argument(
        "--policy-shadow-mode",
        action="store_true",
        help=(
            "run the 48D actor at 50 Hz while motors remain passive; no MIT "
            "movement command or torque-limited motor target is transmitted"
        ),
    )
    parser.add_argument(
        "--policy-replay-csv",
        default=None,
        help="offline replay of logged obs_000..047; opens no IMU, CAN, or motor",
    )
    parser.add_argument(
        "--policy-replay-fixed-50-hz",
        action="store_true",
        help="pace replay at fixed 50 Hz instead of source timestamps",
    )
    parser.add_argument(
        "--policy-replay-realtime",
        action="store_true",
        help="pace offline policy replay in real time; default runs without sleeping",
    )
    parser.add_argument(
        "--policy-replay-output",
        default=None,
        help="output CSV for replayed raw actions",
    )
    parser.add_argument("--standup-seconds", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--async-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write CSV rows from a bounded background queue instead of blocking the control loop",
    )
    parser.add_argument("--csv-queue-size", type=int, default=500)
    parser.add_argument("--csv-flush-seconds", type=float, default=1.0)
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="terminal telemetry print interval in control steps; 0 follows --log-every",
    )
    parser.add_argument("--show-hex", action="store_true")
    parser.add_argument("--start-control-mode", choices=["idle", "hold", "stand", "sit", "policy"], default="idle")
    parser.add_argument("--startup-action", choices=["hold", "stand"], default="hold")
    parser.add_argument(
        "--initial-zero-frame",
        choices=["stand"],
        default="stand",
        help="fixed hardware-zero frame; use stand for policy deployment",
    )
    parser.add_argument(
        "--auto-stand-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="legacy software-zero transition; keep disabled with stand hardware zero",
    )
    parser.add_argument(
        "--auto-sit-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="legacy software-zero transition; keep disabled with stand hardware zero",
    )
    parser.add_argument(
        "--auto-zero-on-startup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when starting in crouch frame, software-zero current encoder pose automatically before joystick commands",
    )
    parser.add_argument(
        "--crouch-calibration-value",
        type=float,
        default=0.0,
        help="joint-coordinate value assigned to the current crouch/default pose during software calibration",
    )
    parser.add_argument(
        "--stand-calibration-value",
        type=float,
        default=0.0,
        help="joint-coordinate value assigned to stand during stand auto-calibration; keep 0.0 for RL policy",
    )
    parser.add_argument("--stand-zero-error-rad", type=float, default=0.08)
    parser.add_argument(
        "--stand-ready-error-rad",
        type=float,
        default=0.25,
        help=(
            "maximum loaded stand tracking error used only to arm policy "
            "walking after the complete stand trajectory"
        ),
    )
    parser.add_argument(
        "--stand-ready-velocity-rad-s",
        type=float,
        default=0.15,
        help=(
            "maximum measured active-joint speed for arming or re-entering "
            "policy walking from stand"
        ),
    )
    parser.add_argument("--stand-zero-settle-steps", type=int, default=15)
    parser.add_argument(
        "--pose-sync-error-rad",
        type=float,
        default=0.0,
        help="during sit/stand, slow target progress when live feedback exceeds this max active-joint error; 0 disables",
    )
    parser.add_argument(
        "--pose-transition-speed-rad-s",
        type=float,
        default=0.40,
        help="peak synchronized sit/stand target speed in joint radians/second",
    )
    parser.add_argument(
        "--pose-transition-min-seconds",
        type=float,
        default=1.5,
        help="minimum duration for a synchronized sit/stand transition",
    )
    parser.add_argument(
        "--deadline-tolerance-ms",
        type=float,
        default=1.0,
        help=(
            "current-cycle work-budget tolerance before counting a consecutive "
            "timing overrun"
        ),
    )
    parser.add_argument(
        "--deadline-resync-ms",
        type=float,
        default=50.0,
        help="lateness threshold for resynchronizing the absolute-deadline scheduler",
    )
    parser.add_argument(
        "--timing-fault-consecutive",
        type=int,
        default=25,
        help="stop after this many consecutive current-cycle work overruns",
    )
    parser.add_argument(
        "--walk-command-threshold",
        type=float,
        default=0.02,
        help="minimum absolute vx/vy/yaw command needed to run walking policy",
    )
    parser.add_argument(
        "--walk-command-grace-seconds",
        type=float,
        default=0.25,
        help=(
            "seconds to keep the last nonzero walking command alive after a "
            "terminal key-repeat gap; 0 disables the grace window"
        ),
    )
    parser.add_argument(
        "--walk-stop-confirm-seconds",
        type=float,
        default=0.25,
        help="repeat-mode neutral confirmation time before leaving policy walking",
    )
    parser.add_argument(
        "--joystick-debug",
        action="store_true",
        help="print raw joystick axes/buttons/hats at each telemetry line",
    )
    parser.add_argument(
        "--joint-debug",
        action="store_true",
        help="print active joint target/feedback values at each telemetry line",
    )
    parser.add_argument(
        "--feedback-source",
        choices=["auto", "fake", "mit"],
        default="auto",
        help="auto uses MIT motor feedback in --mode mit-signal, otherwise fake feedback",
    )
    parser.add_argument(
        "--joint-velocity-source",
        choices=["mit", "finite-difference"],
        default="finite-difference",
        help=(
            "joint velocity placed in policy obs[24:36]; finite-difference is "
            "derived from sign-corrected joint-radian feedback and avoids MIT "
            "velocity scale bias"
        ),
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.05,
        help="seconds to wait for MIT feedback after sending commands",
    )
    parser.add_argument(
        "--steady-feedback-budget-ms",
        type=float,
        default=1.5,
        help=(
            "ordinary post-command feedback drain budget in milliseconds; "
            "startup, calibration, hold capture, and recovery still use "
            "--feedback-timeout"
        ),
    )
    parser.add_argument(
        "--fresh-feedback-max-age",
        type=float,
        default=0.0,
        help=(
            "maximum age in seconds for feedback used by policy/pose command "
            "generation; 0 uses min(safety encoder age, 2 control cycles)"
        ),
    )
    parser.add_argument(
        "--encoder-limit-tolerance-rad",
        type=float,
        default=0.0,
        help="extra encoder sanity tolerance for sensor quantization/noise only; does not widen command limits",
    )
    parser.add_argument(
        "--hold-capture-seconds",
        type=float,
        default=0.35,
        help="seconds to gather fresh encoder feedback when h/HOLD is requested",
    )
    parser.add_argument(
        "--hold-command-repeats",
        type=int,
        default=2,
        help="number of repeated MIT command batches to send each hold cycle",
    )
    parser.add_argument(
        "--enable-retries",
        type=int,
        default=3,
        help="number of motor enable command rounds before control starts",
    )
    parser.add_argument(
        "--enable-retry-delay",
        type=float,
        default=0.05,
        help="seconds between repeated motor enable command rounds",
    )
    parser.add_argument(
        "--imu-source",
        choices=["auto", "fake", "none", "xsens", "serial-json", "serial-csv"],
        default="auto",
        help="auto uses config/imu.yaml; serial sources feed gyro/gravity into policy observation",
    )
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument(
        "--async-imu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read Xsens in a background thread so the 50 Hz control loop only copies the latest cached sample",
    )
    parser.add_argument(
        "--imu-stale-timeout",
        type=float,
        default=None,
        help="seconds before a required live IMU sample is considered stale; default uses config/imu.yaml or 0.20 for async Xsens",
    )
    parser.add_argument(
        "--imu-startup-samples",
        type=int,
        default=5,
        help="consecutive valid live IMU samples required before active control when the IMU source is required",
    )
    parser.add_argument(
        "--imu-startup-timeout",
        type=float,
        default=5.0,
        help="seconds to wait for required live IMU startup validation",
    )
    parser.add_argument(
        "--imu-max-startup-roll-pitch-deg",
        type=float,
        default=60.0,
        help="maximum absolute roll/pitch accepted during startup IMU validation",
    )
    parser.add_argument(
        "--imu-max-active-roll-pitch-deg",
        type=float,
        default=75.0,
        help="maximum absolute roll/pitch accepted for required IMU active-motion validation",
    )
    parser.add_argument(
        "--base-lin-vel-source",
        choices=["zero"],
        default="zero",
        help=(
            "force policy obs[0:3] to [0,0,0], matching locomotion training"
        ),
    )
    parser.add_argument(
        "--imu-stabilization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable small projected-gravity posture corrections",
    )
    parser.add_argument(
        "--stand-policy-stabilization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after stand settles, apply only the difference between live-IMU "
            "and upright-IMU RL actions; cancels nominal policy motion"
        ),
    )
    parser.add_argument(
        "--gait-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "deprecated compatibility flag; --no-gait-assist is accepted, "
            "but enabling hard-coded gait substitution is rejected"
        ),
    )
    parser.add_argument(
        "--policy-command-gain",
        type=float,
        default=float(policy_deploy_defaults.get("command_gain", 1.0)),
        help="multiply joystick vx/vy/yaw only for the policy observation; motor speed scale stays unchanged",
    )
    parser.add_argument(
        "--policy-command-vx-max",
        type=float,
        default=float(policy_deploy_defaults.get("command_vx_abs_max", 0.0)),
        help="absolute cap for policy-observed vx after gain; 0 disables this extra cap",
    )
    parser.add_argument(
        "--policy-command-vy-max",
        type=float,
        default=float(policy_deploy_defaults.get("command_vy_abs_max", 0.0)),
        help="absolute cap for policy-observed vy after gain; 0 disables this extra cap",
    )
    parser.add_argument(
        "--policy-command-yaw-max",
        type=float,
        default=float(policy_deploy_defaults.get("command_yaw_abs_max", 0.0)),
        help="absolute cap for policy-observed yaw after gain; 0 disables this extra cap",
    )
    parser.add_argument(
        "--policy-action-clip",
        type=float,
        default=float(policy_deploy_defaults.get("action_clip_abs", 0.0)),
        help="absolute clip on raw policy actions before action_scale; 0 matches IsaacSim no-clip behavior",
    )
    parser.add_argument(
        "--policy-hip-action-clip",
        type=float,
        default=float(policy_deploy_defaults.get("hip_action_clip_abs", 0.0)),
        help=(
            "hip-only actor-output clip before the global action filter; "
            "0 disables it and leaves thigh/calf authority unchanged"
        ),
    )
    parser.add_argument(
        "--policy-hip-action-scale",
        type=float,
        default=float(policy_deploy_defaults.get("hip_action_scale", 1.0)),
        help=(
            "motor-side scale applied to clipped hip actor outputs only; "
            "does not alter thigh/calf actions"
        ),
    )
    parser.add_argument(
        "--policy-action-smoothing",
        type=float,
        default=float(policy_deploy_defaults.get("action_smoothing", 0.0)),
        help="blend current policy action with previous sent action; 0 disables, larger is smoother/slower",
    )
    parser.add_argument(
        "--policy-action-delta-limit",
        type=float,
        default=float(policy_deploy_defaults.get("action_delta_limit_abs", 0.0)),
        help=(
            "maximum per-cycle change of the sent policy action; 0 disables. "
            "Conditioned mode reports this applied actor-coordinate action in "
            "the next previous_action observation"
        ),
    )
    parser.add_argument(
        "--policy-entry-ramp-seconds",
        type=float,
        default=float(policy_deploy_defaults.get("policy_entry_ramp_seconds", 1.5)),
        help="seconds used to blend smoothly from stand into policy walking targets",
    )
    parser.add_argument(
        "--policy-pd-torque-limit",
        type=float,
        default=0.0,
        help=(
            "override all YAML policy PD torque limits in Nm; 0 uses "
            "config/control_limits.yaml, including per-joint limits"
        ),
    )
    parser.add_argument(
        "--policy-kp-override",
        type=float,
        default=None,
        help="uniform physical RS04 Kp for policy mode; pose gains are unchanged",
    )
    parser.add_argument(
        "--policy-kd-override",
        type=float,
        default=None,
        help="uniform physical RS04 Kd for policy mode; pose gains are unchanged",
    )
    parser.add_argument("--policy-pd-torque-limit-start", type=float, default=14.0)
    parser.add_argument("--policy-pd-torque-limit-final", type=float, default=14.0)
    parser.add_argument("--policy-torque-ramp-delay-seconds", type=float, default=2.0)
    parser.add_argument("--policy-torque-ramp-seconds", type=float, default=8.0)
    parser.add_argument(
        "--policy-torque-ramp-require-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--policy-torque-ramp-max-tracking-error-rad", type=float, default=0.25)
    parser.add_argument("--policy-torque-ramp-max-measured-torque", type=float, default=30.0)
    parser.add_argument("--policy-torque-ramp-min-encoder-margin-rad", type=float, default=0.08)
    parser.add_argument("--policy-torque-ramp-max-feedback-age", type=float, default=0.04)
    parser.add_argument("--policy-torque-ramp-max-cycle-work-ms", type=float, default=20.0)
    parser.add_argument("--policy-pd-torque-profile", default=None)
    parser.add_argument("--policy-pd-torque-scale", type=float, default=1.0)
    parser.add_argument("--policy-absolute-torque-ceiling", type=float, default=40.0)
    parser.add_argument(
        "--torque-profile-stage",
        choices=sorted(TORQUE_PROFILE_STAGES),
        default="stage14",
    )
    parser.add_argument(
        "--calf-calibration-recommendation",
        default=str(ROOT / "config" / "calf_endpoint_recommendation.yaml"),
        help="passive endpoint recommendation required for stages above stage14",
    )
    parser.add_argument(
        "--acknowledge-40nm-suspension-test",
        action="store_true",
        help="explicitly acknowledge a suspended stage40 test",
    )
    parser.add_argument(
        "--acknowledge-40nm-loaded-ground-test",
        action="store_true",
        help="explicitly acknowledge a loaded-ground stage40 test",
    )
    parser.add_argument(
        "--acknowledge-100nm-loaded-ground-test",
        action="store_true",
        help="explicitly acknowledge the 100 Nm loaded-ground authority ceiling",
    )
    parser.add_argument("--measured-torque-soft-hip", type=float, default=35.0)
    parser.add_argument("--measured-torque-soft-thigh", type=float, default=40.0)
    parser.add_argument("--measured-torque-soft-calf", type=float, default=40.0)
    parser.add_argument(
        "--calf-range-check",
        action="store_true",
        help="read-only calf range report; no policy actor or walking commands",
    )
    parser.add_argument(
        "--pose-pd-torque-limit",
        type=float,
        default=0.0,
        help=(
            "override sit/stand/hold estimated PD torque limits in Nm; "
            "0 preserves config/control_limits.yaml startup/hold limits"
        ),
    )
    parser.add_argument(
        "--policy-sim-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use raw policy actions and bypass policy-target slew limiting; "
            "hard joint, encoder, tilt, and torque safety remain active"
        ),
    )
    parser.add_argument(
        "--exact-policy-after-entry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "blend safely into walking in joint-target space, then send the "
            "raw actor target exactly; disable only for filtered suspended debugging"
        ),
    )
    parser.add_argument(
        "--auto-policy-after-stand",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "start policy inference with a zero velocity command immediately "
            "after measured stand readiness passes"
        ),
    )
    parser.add_argument(
        "--pose-test-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow only sit, stand, hold, and emergency-stop motor modes",
    )
    parser.add_argument(
        "--pose-gains-config",
        default=None,
        help="dedicated sit/stand gain YAML used only by --pose-test-only",
    )
    parser.add_argument(
        "--sit-stand-trace-200hz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the dedicated per-CAN-cycle sit/stand trace",
    )
    parser.add_argument("--robot-mass-kg", type=float, default=50.0)
    parser.add_argument(
        "--battery-voltage-start",
        type=float,
        default=None,
        help="manually measured starting battery voltage; blank when omitted",
    )
    parser.add_argument(
        "--motor-firmware-version",
        default="unknown",
        help="motor firmware/version text stored in trace metadata",
    )
    parser.add_argument(
        "--suspension-status-seconds",
        type=float,
        default=0.5,
        help="seconds between [SUSPENSION] diagnostic lines; 0 disables",
    )
    parser.add_argument(
        "--joint-routing-test",
        action="store_true",
        help="safe suspended one-joint-at-a-time routing test; actor/policy is not run",
    )
    parser.add_argument("--routing-test-amplitude-rad", type=float, default=0.04)
    parser.add_argument("--routing-test-frequency-hz", type=float, default=0.25)
    parser.add_argument("--routing-test-cycles", type=int, default=1)
    parser.add_argument("--routing-test-torque-limit", type=float, default=5.0)
    parser.add_argument(
        "--fake-start",
        choices=["stand", "crouch", "random_small"],
        default="stand",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the real-time telemetry GUI window alongside the controller",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=TELEMETRY_PORT_DEFAULT,
        help="UDP port used to stream telemetry to the GUI (default 57543)",
    )
    parser.add_argument(
        "--log-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write compact terminal telemetry rows to a per-run CSV file",
    )
    parser.add_argument(
        "--log-dir",
        default=str(ROOT / "logs"),
        help="directory for timestamped CSV run logs",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="exact CSV log file path; overrides --log-dir",
    )
    parser.add_argument(
        "--log-prefix",
        default="grallator_run",
        help="filename prefix for timestamped CSV logs",
    )
    parser.add_argument(
        "--csv-flush-every",
        type=int,
        default=25,
        help="flush the CSV file after this many logged rows; increase to reduce disk stalls",
    )
    parser.add_argument(
        "--auto-push-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after motors are stopped and the CSV is closed, commit and push "
            "only that run log"
        ),
    )
    parser.add_argument(
        "--log-git-remote",
        default="origin",
        help="Git remote used by --auto-push-log (default: origin)",
    )

    args = parser.parse_args()
    if args.pose_test_only:
        if args.start_control_mode == "policy" or args.policy_shadow_mode:
            parser.error("--pose-test-only cannot start in policy mode")
        if not args.pose_gains_config:
            parser.error("--pose-test-only requires --pose-gains-config")
        args.auto_policy_after_stand = False
    elif args.pose_gains_config:
        parser.error("--pose-gains-config is valid only with --pose-test-only")
    if args.policy_shadow_mode:
        args.startup_action = "hold"
        args.start_control_mode = "policy"
    args.log_every = max(1, args.log_every)
    args.print_every = args.log_every if args.print_every <= 0 else max(1, args.print_every)
    args.csv_flush_every = max(1, int(args.csv_flush_every))
    args.policy_steps = None if args.policy_steps <= 0 else args.policy_steps
    args.walk_command_threshold = max(0.0, args.walk_command_threshold)
    args.walk_command_grace_seconds = max(0.0, float(args.walk_command_grace_seconds))
    args.walk_stop_confirm_seconds = max(0.0, float(args.walk_stop_confirm_seconds))
    args.keyboard_latched_combo_window = max(0.02, float(args.keyboard_latched_combo_window))
    args.policy_command_gain = max(0.0, float(args.policy_command_gain))
    args.policy_command_vx_max = max(0.0, float(args.policy_command_vx_max))
    args.policy_command_vy_max = max(0.0, float(args.policy_command_vy_max))
    args.policy_command_yaw_max = max(0.0, float(args.policy_command_yaw_max))
    args.policy_action_clip = max(0.0, float(args.policy_action_clip))
    args.policy_hip_action_clip = max(0.0, float(args.policy_hip_action_clip))
    if (
        not np.isfinite(args.policy_hip_action_scale)
        or args.policy_hip_action_scale < 0.0
    ):
        parser.error("--policy-hip-action-scale must be finite and >= 0")
    args.policy_action_smoothing = float(np.clip(args.policy_action_smoothing, 0.0, 0.98))
    args.policy_action_delta_limit = max(0.0, float(args.policy_action_delta_limit))
    if not np.isfinite(args.policy_entry_ramp_seconds) or args.policy_entry_ramp_seconds < 0.0:
        parser.error("--policy-entry-ramp-seconds must be finite and >= 0")
    if (
        args.mode == "mit-signal"
        and bool(args.policy_sim_match)
        and float(args.policy_entry_ramp_seconds) <= 0.0
    ):
        print(
            "WARNING: --policy-sim-match with --mode mit-signal needs a walking "
            "entry ramp on hardware; forcing --policy-entry-ramp-seconds 1.5"
        )
        args.policy_entry_ramp_seconds = 1.5
    if not np.isfinite(args.policy_pd_torque_limit) or args.policy_pd_torque_limit < 0.0:
        parser.error("--policy-pd-torque-limit must be finite and >= 0")
    if int(args.csv_queue_size) <= 0:
        parser.error("--csv-queue-size must be > 0")
    if not np.isfinite(args.csv_flush_seconds) or args.csv_flush_seconds <= 0.0:
        parser.error("--csv-flush-seconds must be finite and > 0")
    torque_numeric_args = (
        ("--policy-pd-torque-limit-start", args.policy_pd_torque_limit_start),
        ("--policy-pd-torque-limit-final", args.policy_pd_torque_limit_final),
        ("--policy-torque-ramp-delay-seconds", args.policy_torque_ramp_delay_seconds),
        ("--policy-torque-ramp-seconds", args.policy_torque_ramp_seconds),
        ("--policy-torque-ramp-max-tracking-error-rad", args.policy_torque_ramp_max_tracking_error_rad),
        ("--policy-torque-ramp-max-measured-torque", args.policy_torque_ramp_max_measured_torque),
        ("--policy-torque-ramp-min-encoder-margin-rad", args.policy_torque_ramp_min_encoder_margin_rad),
        ("--policy-torque-ramp-max-feedback-age", args.policy_torque_ramp_max_feedback_age),
        ("--policy-torque-ramp-max-cycle-work-ms", args.policy_torque_ramp_max_cycle_work_ms),
        ("--policy-pd-torque-scale", args.policy_pd_torque_scale),
        ("--policy-absolute-torque-ceiling", args.policy_absolute_torque_ceiling),
        ("--measured-torque-soft-hip", args.measured_torque_soft_hip),
        ("--measured-torque-soft-thigh", args.measured_torque_soft_thigh),
        ("--measured-torque-soft-calf", args.measured_torque_soft_calf),
    )
    for name, value in torque_numeric_args:
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"{name} must be finite and >= 0")
    if args.policy_torque_ramp_seconds <= 0.0:
        parser.error("--policy-torque-ramp-seconds must be > 0")
    if args.policy_absolute_torque_ceiling <= 0.0:
        parser.error("--policy-absolute-torque-ceiling must be > 0")
    if args.torque_profile_stage == "stage40" and not (
        args.acknowledge_40nm_suspension_test
        or args.acknowledge_40nm_loaded_ground_test
    ):
        parser.error(
            "stage40 requires --acknowledge-40nm-suspension-test or "
            "--acknowledge-40nm-loaded-ground-test"
        )
    if (
        args.torque_profile_stage == "stage100"
        and not args.acknowledge_100nm_loaded_ground_test
    ):
        parser.error(
            "stage100 requires --acknowledge-100nm-loaded-ground-test"
        )
    if not np.isfinite(args.pose_pd_torque_limit) or args.pose_pd_torque_limit < 0.0:
        parser.error("--pose-pd-torque-limit must be finite and >= 0")
    if (
        not np.isfinite(args.deadline_tolerance_ms)
        or args.deadline_tolerance_ms < 0.0
    ):
        parser.error("--deadline-tolerance-ms must be finite and >= 0")
    if (
        not np.isfinite(args.deadline_resync_ms)
        or args.deadline_resync_ms <= 0.0
    ):
        parser.error("--deadline-resync-ms must be finite and > 0")
    if int(args.timing_fault_consecutive) <= 0:
        parser.error("--timing-fault-consecutive must be > 0")
    if int(args.imu_startup_samples) < 1:
        parser.error("--imu-startup-samples must be >= 1")
    if args.imu_stale_timeout is not None and (
        not np.isfinite(args.imu_stale_timeout) or args.imu_stale_timeout <= 0.0
    ):
        parser.error("--imu-stale-timeout must be finite and > 0")
    if not np.isfinite(args.imu_startup_timeout) or args.imu_startup_timeout <= 0.0:
        parser.error("--imu-startup-timeout must be finite and > 0")
    if (
        not np.isfinite(args.imu_max_startup_roll_pitch_deg)
        or args.imu_max_startup_roll_pitch_deg <= 0.0
        or args.imu_max_startup_roll_pitch_deg > 89.0
    ):
        parser.error("--imu-max-startup-roll-pitch-deg must be within 0..89")
    if (
        not np.isfinite(args.imu_max_active_roll_pitch_deg)
        or args.imu_max_active_roll_pitch_deg <= 0.0
        or args.imu_max_active_roll_pitch_deg > 89.0
    ):
        parser.error("--imu-max-active-roll-pitch-deg must be within 0..89")
    if not np.isfinite(args.suspension_status_seconds) or args.suspension_status_seconds < 0.0:
        parser.error("--suspension-status-seconds must be finite and >= 0")
    if args.joint_routing_test:
        if (
            not np.isfinite(args.routing_test_amplitude_rad)
            or args.routing_test_amplitude_rad <= 0.0
            or args.routing_test_amplitude_rad > 0.12
        ):
            parser.error("--routing-test-amplitude-rad must be finite and within 0.0..0.12")
        if (
            not np.isfinite(args.routing_test_frequency_hz)
            or args.routing_test_frequency_hz <= 0.0
            or args.routing_test_frequency_hz > 1.0
        ):
            parser.error("--routing-test-frequency-hz must be finite and within 0.0..1.0")
        if int(args.routing_test_cycles) <= 0:
            parser.error("--routing-test-cycles must be > 0")
        if (
            not np.isfinite(args.routing_test_torque_limit)
            or args.routing_test_torque_limit <= 0.0
            or args.routing_test_torque_limit > 12.0
        ):
            parser.error("--routing-test-torque-limit must be finite and within 0.0..12.0")
    if not np.isfinite(args.fresh_feedback_max_age) or args.fresh_feedback_max_age < 0.0:
        parser.error("--fresh-feedback-max-age must be finite and >= 0")
    if (
        not np.isfinite(args.stand_ready_error_rad)
        or args.stand_ready_error_rad <= 0.0
        or args.stand_ready_error_rad > 0.50
    ):
        parser.error("--stand-ready-error-rad must be finite and within 0.0..0.50")
    if (
        not np.isfinite(args.stand_ready_velocity_rad_s)
        or args.stand_ready_velocity_rad_s <= 0.0
        or args.stand_ready_velocity_rad_s > 2.0
    ):
        parser.error(
            "--stand-ready-velocity-rad-s must be finite and within 0.0..2.0"
        )
    if (
        not np.isfinite(args.encoder_limit_tolerance_rad)
        or args.encoder_limit_tolerance_rad < 0.0
        or args.encoder_limit_tolerance_rad > 0.10
    ):
        parser.error("--encoder-limit-tolerance-rad must be finite and within 0.0..0.10")
    if (
        not np.isfinite(args.steady_feedback_budget_ms)
        or args.steady_feedback_budget_ms < 0.0
        or args.steady_feedback_budget_ms > 5.0
    ):
        parser.error("--steady-feedback-budget-ms must be finite and within 0.0..5.0")
    if not np.isfinite(args.control_hz) or args.control_hz < 0.0:
        parser.error("--control-hz must be finite and >= 0")
    if args.control_hz > 0.0 and not np.isclose(args.control_hz, 50.0):
        parser.error("policy/control rate is fixed at 50 Hz; use --control-hz 50 or 0")
    if not np.isfinite(args.can_command_hz) or not np.isclose(args.can_command_hz, 200.0):
        parser.error("--can-command-hz is fixed at 200 Hz for a 0.005 s CAN dt")
    if (
        not np.isfinite(args.can_command_stale_timeout)
        or args.can_command_stale_timeout < 0.040
    ):
        parser.error("--can-command-stale-timeout must be finite and >= 0.040 s")
    if int(args.can_command_fault_consecutive) <= 0:
        parser.error("--can-command-fault-consecutive must be > 0")
    if (
        not np.isfinite(args.can_tx_timeout_ms)
        or args.can_tx_timeout_ms < 0.0
        or args.can_tx_timeout_ms > 2.5
    ):
        parser.error("--can-tx-timeout-ms must be finite and within 0.0..2.5")
    if (
        not np.isfinite(args.python_thread_switch_ms)
        or args.python_thread_switch_ms < 0.25
        or args.python_thread_switch_ms > 2.5
    ):
        parser.error("--python-thread-switch-ms must be finite and within 0.25..2.5")
    if (
        not np.isfinite(args.pose_transition_speed_rad_s)
        or args.pose_transition_speed_rad_s <= 0.0
    ):
        parser.error("--pose-transition-speed-rad-s must be finite and > 0")
    if (
        not np.isfinite(args.pose_transition_min_seconds)
        or args.pose_transition_min_seconds <= 0.0
    ):
        parser.error("--pose-transition-min-seconds must be finite and > 0")
    if args.auto_zero_on_startup or args.auto_stand_zero or args.auto_sit_zero:
        parser.error(
            "automatic software-zero transitions are disabled; set RobStride "
            "hardware zero once in stand, then use stand_pose=0 and crouch_pose YAML targets"
        )
    try:
        port_by_bus = resolve_port_by_bus(args)
    except ValueError as exc:
        print("ERROR:", exc)
        return 1

    # The low-level CAN sender is a dedicated Python thread. CPython's usual
    # 5 ms handoff interval equals the entire 200 Hz CAN period, allowing a
    # busy 50 Hz policy/pose cycle to starve it for a complete deadline.
    sys.setswitchinterval(0.001 * float(args.python_thread_switch_ms))

    if args.mode == "motors":
        print("ERROR: --mode motors is intentionally blocked for now.")
        print("Use --mode mit-signal for the real RobStride MIT CAN path.")
        return 1

    active_imu_source = imu_source_name(args.imu_source, imu_defaults)
    active_imu_port = args.imu_port if args.imu_port is not None else imu_defaults.get("port")

    runner = PolicyRunner(
        policy_path=args.policy_path,
        policy_activation=args.policy_activation,
        allow_policy_hash_mismatch=args.allow_policy_hash_mismatch,
    )
    if args.policy_replay_csv:
        if not args.policy_shadow_mode:
            parser.error("--policy-replay-csv requires --policy-shadow-mode")
        replay = replay_policy_csv(
            runner,
            args.policy_replay_csv,
            output_path=args.policy_replay_output,
            fixed_50_hz=bool(args.policy_replay_fixed_50_hz),
            realtime=bool(args.policy_replay_realtime),
        )
        print("POLICY SHADOW REPLAY")
        print("Policy SHA256:", runner.policy_sha256)
        print("Observation rows:", replay["row_count"])
        print("Output:", replay["output_path"])
        print(
            "Maximum absolute error versus logged raw action:",
            f"{replay['maximum_absolute_error']:.9g}",
        )
        return 0
    trained_control_dt = float(runner.control_dt)
    trained_control_hz = 1.0 / trained_control_dt
    if args.control_hz > 0.0:
        runner.control_dt = 1.0 / float(args.control_hz)
    runtime_control_hz = 1.0 / float(runner.control_dt)
    if not np.isclose(float(runner.control_dt), 0.02):
        parser.error("the deployed policy requires control_dt=0.02 s (50 Hz)")
    if args.policy_shadow_mode and not np.isclose(runtime_control_hz, 50.0):
        parser.error("--policy-shadow-mode requires exactly --control-hz 50 (or 0)")
    motion_assist_cfg = motion_assist_defaults
    if args.imu_stabilization is not None:
        motion_assist_cfg.setdefault("imu_posture", {})["enabled"] = bool(args.imu_stabilization)
    if args.gait_assist:
        parser.error(
            "--gait-assist was removed: walking motor targets must come from "
            "the loaded policy"
        )

    safety = SafetyMonitor(
        runner.policy_order,
        control_dt=runner.control_dt,
    )
    safety.set_encoder_limit_tolerance(args.encoder_limit_tolerance_rad)
    motor_ids = load_motor_ids()
    joint_can_bus = resolve_joint_can_bus(runner.policy_order, args.can_count)
    active_joints = args.active_joints if args.active_joints is not None else load_active_joints()
    motor_layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=active_joints,
        joint_can_bus=joint_can_bus,
    )
    pose_gain_profile = None
    if args.pose_gains_config:
        try:
            pose_gain_profile = motor_layer.apply_sit_stand_gain_profile(
                args.pose_gains_config
            )
        except (KeyError, OSError, ValueError) as exc:
            parser.error(f"invalid sit/stand gain profile: {exc}")
    try:
        motor_layer.set_policy_gains(
            kp=args.policy_kp_override,
            kd=args.policy_kd_override,
        )
    except ValueError as exc:
        parser.error(str(exc))
    four_bar_cfg = load_yaml(ROOT / "config" / "four_bar_transmission.yaml")
    four_bar_enabled = bool(
        (four_bar_cfg or {}).get("four_bar_transmission", {}).get("enabled", False)
    )
    joint_mapping = AuthoritativeJointMapping(
        motor_ids=motor_ids,
        motor_directions=motor_layer.joint_directions,
        encoder_offsets=motor_layer.joint_offsets,
        joint_can_bus=joint_can_bus,
        estimator_order=runner.policy_order,
        policy_order=runner.policy_order,
    )
    supplied_options = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            supplied_options.add(token.split("=", 1)[0])
    legacy_torque_supplied = "--policy-pd-torque-limit" in supplied_options
    profile_controls_supplied = bool(
        {
            "--policy-pd-torque-limit-start",
            "--policy-pd-torque-limit-final",
            "--policy-pd-torque-profile",
            "--torque-profile-stage",
        }
        & supplied_options
    )
    if legacy_torque_supplied and args.policy_pd_torque_limit > 0.0 and not profile_controls_supplied:
        torque_start_by_joint = constant_joint_map(runner.policy_order, args.policy_pd_torque_limit)
        torque_final_by_joint = constant_joint_map(runner.policy_order, args.policy_pd_torque_limit)
    else:
        stage_start, stage_final = TORQUE_PROFILE_STAGES[str(args.torque_profile_stage)]
        torque_start_value = (
            float(args.policy_pd_torque_limit_start)
            if "--policy-pd-torque-limit-start" in supplied_options
            else stage_start
        )
        torque_final_value = (
            float(args.policy_pd_torque_limit_final)
            if "--policy-pd-torque-limit-final" in supplied_options
            else stage_final
        )
        torque_start_by_joint = constant_joint_map(runner.policy_order, torque_start_value)
        torque_final_by_joint = constant_joint_map(runner.policy_order, torque_final_value)
        if args.policy_pd_torque_profile:
            torque_start_by_joint, torque_final_by_joint = load_policy_torque_profile(
                args.policy_pd_torque_profile,
                runner.policy_order,
                torque_start_by_joint,
                torque_final_by_joint,
            )
        if float(args.policy_pd_torque_scale) != 1.0:
            torque_start_by_joint = {
                joint_name: float(value) * float(args.policy_pd_torque_scale)
                for joint_name, value in torque_start_by_joint.items()
            }
            torque_final_by_joint = {
                joint_name: float(value) * float(args.policy_pd_torque_scale)
                for joint_name, value in torque_final_by_joint.items()
            }
    calibration_required = requires_calf_endpoint_gate(
        four_bar_enabled,
        torque_final_by_joint,
    )
    calibration_ok = True
    calibration_reason = "not required for disabled nonlinear transmission"
    if calibration_required:
        calibration_ok, calibration_reason = calf_calibration_gate(
            args.calf_calibration_recommendation,
            joint_name="FL_calf_joint",
        )
        if not calibration_ok:
            print("ERROR: torque stages above stage14 are locked.")
            print("FL calf calibration gate:", calibration_reason)
            print(
                "Run scripts/calibrate_calf_endpoints.py passively and review "
                "the recommendation before selecting stage18 or higher."
            )
            return 1
    try:
        validate_torque_profile(
            torque_start_by_joint,
            torque_final_by_joint,
            runner.policy_order,
            args.policy_absolute_torque_ceiling,
        )
    except (KeyError, ValueError) as exc:
        print("ERROR:", exc)
        return 1
    motor_layer.set_policy_pd_torque_limits(
        torque_start_by_joint,
        start_limits_by_joint=torque_start_by_joint,
        final_limits_by_joint=torque_final_by_joint,
    )
    if args.pose_pd_torque_limit > 0.0:
        motor_layer.set_pose_pd_torque_limit(args.pose_pd_torque_limit)
    active_port_by_bus = ports_for_active_joints(
        port_by_bus,
        joint_can_bus,
        motor_layer.active_joints,
    )
    if not active_port_by_bus:
        active_port_by_bus = port_by_bus
    if args.mode == "mit-signal" and len(motor_layer.active_joints) > 6:
        physical_can_ports = {
            os.path.realpath(str(port)) for port in active_port_by_bus.values()
        }
        if len(physical_can_ports) < 2:
            print(
                "ERROR: 12-motor control at 200 Hz requires two distinct CAN "
                "interfaces. Use --can-count 2 --can-ports slcan0 slcan1."
            )
            return 1
    try:
        validate_unique_motor_ids_per_physical_bus(
            motor_ids=motor_ids,
            joint_can_bus=joint_can_bus,
            active_joints=motor_layer.active_joints,
            port_by_bus=active_port_by_bus,
        )
    except ValueError as exc:
        print("ERROR:", exc)
        return 1
    if four_bar_enabled:
        print(
            "ERROR: four-bar transmission is enabled, but this deployment "
            "contract requires the active walking path to remain 1:1."
        )
        return 1
    try:
        for bus_name, port in active_port_by_bus.items():
            selected_backend = backend_for_port(port, args.can_backend)
            if selected_backend == "socketcan" and str(port).startswith("/dev/tty"):
                print(
                    "ERROR:",
                    f"{bus_name} uses {port!r} with SocketCAN backend. "
                    "SocketCAN needs a Linux CAN interface such as slcan0, "
                    "not the raw serial device.",
                )
                return 1
    except ValueError as exc:
        print("ERROR:", exc)
        return 1
    if (
        args.mode in ("signal", "mit-signal")
        and active_imu_source in ("xsens", "xsens_binary", "mtdata2", "serial_json", "serial_csv")
        and active_imu_port in set(active_port_by_bus.values())
    ):
        print("ERROR: IMU port", active_imu_port, "conflicts with an active CAN bus port.")
        print("Active CAN ports in use:", sorted(set(active_port_by_bus.values())))
        print("Use a separate device, e.g. --imu-port /dev/ttyUSB2")
        return 1

    mit_cfg = load_yaml(ROOT / "config" / "mit_motor_control.yaml")
    standup_seconds = (
        float(args.standup_seconds)
        if args.standup_seconds is not None
        else float(mit_cfg["startup"]["standup_seconds"])
    )

    try:
        command_source = CommandSource(
            source=args.command_source,
            vx=args.vx,
            vy=args.vy,
            yaw=args.yaw,
            max_vx=args.max_vx,
            max_vy=args.max_vy,
            max_yaw=args.max_yaw,
            axis_vx=args.axis_vx,
            axis_vy=args.axis_vy,
            axis_yaw=args.axis_yaw,
            invert_vx=args.invert_vx,
            invert_vy=args.invert_vy,
            invert_yaw=args.invert_yaw,
            joystick_index=args.joystick_index,
            joystick_wait_seconds=args.joystick_wait_seconds,
            button_stand=args.button_stand,
            button_sit=args.button_sit,
            button_policy=args.button_policy,
            button_speed_down=args.button_speed_down,
            button_speed_up=args.button_speed_up,
            button_zero_calibration=args.button_zero_calibration,
            zero_calibration_hat_index=args.zero_calibration_hat_index,
            zero_calibration_hat_direction=args.zero_calibration_hat_direction,
            zero_calibration_hat_any=args.zero_calibration_hat_any,
            zero_calibration_axis=args.zero_calibration_axis,
            zero_calibration_axis_direction=args.zero_calibration_axis_direction,
            zero_calibration_axis_threshold=args.zero_calibration_axis_threshold,
            zero_calibration_cooldown_s=args.zero_calibration_cooldown_s,
            emergency_stop_buttons=args.button_emergency_stop,
            speed_scale_initial=args.speed_scale_initial,
            speed_scale_min=args.speed_scale_min,
            speed_scale_max=args.speed_scale_max,
            speed_scale_step=args.speed_scale_step,
            keyboard_command_timeout=args.keyboard_command_timeout,
            keyboard_control_mode=args.keyboard_control_mode,
            keyboard_latched_combo_window=args.keyboard_latched_combo_window,
            deadzone=args.deadzone,
            expo=args.expo,
            smoothing=args.smoothing,
            use_hat_fallback=args.use_hat_fallback,
        )
    except (ImportError, RuntimeError) as exc:
        print("ERROR:", exc)
        return 1

    q_fake_start = fake_start_pose_array(runner, args.fake_start)
    feedback_source = args.feedback_source
    if feedback_source == "auto":
        feedback_source = "mit" if args.mode == "mit-signal" else "fake"

    print("==== GRALLATOR JETSON MIT CONTROLLER ====")
    print("Mode:", args.mode)
    print("Command source:", args.command_source)
    print("Initial command:", command_source.read())
    print("Initial speed scale:", f"{command_speed_scale(command_source):.3f}")
    print("Policy command gain:", f"{args.policy_command_gain:.2f}")
    print(
        "Policy command caps:",
        f"vx={args.policy_command_vx_max:.3f}",
        f"vy={args.policy_command_vy_max:.3f}",
        f"yaw={args.policy_command_yaw_max:.3f}",
    )
    print("Policy action clip:", f"{args.policy_action_clip:.3f}")
    print("Policy hip action clip:", f"{args.policy_hip_action_clip:.3f}")
    print("Policy hip action scale:", f"{args.policy_hip_action_scale:.3f}")
    print("Policy action smoothing:", f"{args.policy_action_smoothing:.2f}")
    print("Policy action delta limit:", f"{args.policy_action_delta_limit:.3f}")
    print("Policy entry ramp:", f"{args.policy_entry_ramp_seconds:.2f} s")
    print("Automatic policy takeover after stand:", bool(args.auto_policy_after_stand))
    print(
        "Policy MIT gain override:",
        f"Kp={args.policy_kp_override if args.policy_kp_override is not None else 'config'}",
        f"Kd={args.policy_kd_override if args.policy_kd_override is not None else 'config'}",
    )
    print("TORQUE QUALIFICATION STAGE:", args.torque_profile_stage)
    print("Previous lower torque stage must have passed.")
    if args.torque_profile_stage == "stage40":
        if args.acknowledge_40nm_loaded_ground_test:
            print("40 Nm loaded-ground stage explicitly acknowledged.")
        else:
            print("40 Nm suspended stage explicitly acknowledged.")
    elif args.torque_profile_stage == "stage100":
        print("100 Nm loaded-ground stage explicitly acknowledged.")
    print(
        "Exact policy after entry:",
        "enabled" if args.exact_policy_after_entry else "disabled",
    )
    torque_start_values = list(torque_start_by_joint.values())
    torque_final_values = list(torque_final_by_joint.values())
    print(
        "Policy PD torque ramp:",
        f"start min={min(torque_start_values):.2f} max={max(torque_start_values):.2f} Nm",
        f"final min={min(torque_final_values):.2f} max={max(torque_final_values):.2f} Nm",
        f"delay={float(args.policy_torque_ramp_delay_seconds):.2f}s",
        f"ramp={float(args.policy_torque_ramp_seconds):.2f}s",
        f"require_clean={bool(args.policy_torque_ramp_require_clean)}",
    )
    if args.exact_policy_after_entry:
        virtual_stop_status = "suppressed after exact-policy entry"
    else:
        virtual_stop_status = (
            "enabled" if motor_layer.virtual_joint_stop_enabled else "disabled"
        )
    print(
        "Virtual joint-stop preload:",
        virtual_stop_status,
        f"max={motor_layer.virtual_joint_stop_max_preload_nm:.2f} Nm",
        "(conditioned policy path only; fresh feedback required)",
    )
    pose_torque_limits = motor_layer.pose_pd_torque_limits()
    if pose_gain_profile is not None:
        print("Sit/stand gain-test profile:", pose_gain_profile["path"])
        for phase_name in ("sit", "stand"):
            phase_gains = pose_gain_profile["gains"][phase_name]
            print(
                f"  {phase_name} gains:",
                " ".join(
                    f"{group}=Kp{phase_gains[group]['kp']:.1f}/"
                    f"Kd{phase_gains[group]['kd']:.1f}"
                    for group in ("hip", "thigh", "calf")
                ),
            )
    if args.pose_pd_torque_limit > 0.0:
        print("Pose PD torque limit:", f"{args.pose_pd_torque_limit:.2f} Nm override")
    else:
        print(
            "Pose PD torque limit:",
            "config",
            f"startup={pose_torque_limits['startup']:.2f} Nm",
            f"sit={pose_torque_limits['sit']:.2f} Nm",
            f"stand={pose_torque_limits['stand']:.2f} Nm",
            f"hold={pose_torque_limits['hold']:.2f} Nm",
        )
    print("Sit estimated PD torque limit:", f"{pose_torque_limits['sit']:.2f} Nm")
    print("Stand estimated PD torque limit:", f"{pose_torque_limits['stand']:.2f} Nm")
    print("Hold estimated PD torque limit:", f"{pose_torque_limits['hold']:.2f} Nm")
    print(
        "Measured torque emergency limit:",
        f"{float(safety.max_abs_feedback_torque):.2f} Nm",
        f"samples={int(safety.max_abs_feedback_torque_fault_samples)}",
    )
    measured_torque_soft_limits = {
        "hip": float(args.measured_torque_soft_hip),
        "thigh": float(args.measured_torque_soft_thigh),
        "calf": float(args.measured_torque_soft_calf),
        "default": float(args.measured_torque_soft_calf),
    }
    print(
        "Measured torque soft limits:",
        f"hip={measured_torque_soft_limits['hip']:.1f} Nm",
        f"thigh={measured_torque_soft_limits['thigh']:.1f} Nm",
        f"calf={measured_torque_soft_limits['calf']:.1f} Nm",
    )
    torque_ramp = PolicyTorqueRamp(
        policy_order=runner.policy_order,
        start_by_joint=torque_start_by_joint,
        final_by_joint=torque_final_by_joint,
        delay_s=float(args.policy_torque_ramp_delay_seconds),
        ramp_s=float(args.policy_torque_ramp_seconds),
        require_clean=bool(args.policy_torque_ramp_require_clean),
        max_tracking_error_rad=float(args.policy_torque_ramp_max_tracking_error_rad),
        max_measured_torque=float(args.policy_torque_ramp_max_measured_torque),
        min_encoder_margin_rad=float(args.policy_torque_ramp_min_encoder_margin_rad),
        max_feedback_age_s=float(args.policy_torque_ramp_max_feedback_age),
        max_cycle_work_s=0.001 * float(args.policy_torque_ramp_max_cycle_work_ms),
    )
    print("Encoder limit tolerance:", f"{float(args.encoder_limit_tolerance_rad):.3f} rad")
    print("Walk command grace:", f"{args.walk_command_grace_seconds:.2f} s")
    print("Start control mode:", args.start_control_mode)
    print("Startup action:", args.startup_action)
    print("Policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Policy hash verified:", runner.policy_hash_matches)
    print("Policy format:", runner.policy_format)
    print("Policy obs/actions:", runner.observation_dim, runner.action_dim)
    print("Policy frame origin:", runner.policy_frame_origin.tolist())
    print("Policy reference pose:", runner.q_policy_reference.tolist())
    print("Policy Torch CPU threads:", runner.torch_thread_count)
    print(
        "Control rate:",
        f"runtime={runtime_control_hz:.2f} Hz",
        f"dt={runner.control_dt:.4f}s",
        f"trained={trained_control_hz:.2f} Hz",
    )
    print(
        "CAN command rate:",
        f"{float(args.can_command_hz):.2f} Hz",
        f"dt={1.0 / float(args.can_command_hz):.4f}s",
        "(latest 50 Hz target; no target queue)",
    )
    if not np.isclose(runtime_control_hz, trained_control_hz):
        print(
            "WARNING: runtime control rate differs from policy training; "
            "use only for suspended motion testing."
        )
    for line in topology_lines(args.can_count, port_by_bus):
        print(line)
    print("CAN backend:", args.can_backend)
    print("CAN bitrate:", args.can_bitrate)
    print("Four-bar transmission:", "enabled" if four_bar_enabled else "disabled")
    if args.can_backend == "serial-at" or any(
        str(port).startswith("/dev/tty") for port in port_by_bus.values()
    ):
        print("Serial adapter baud:", args.baud)
    print("Feedback source:", feedback_source)
    print("IMU source:", active_imu_source)
    imu_policy_filter_cfg = imu_defaults.get("policy_filter", {})
    print(
        "IMU policy filter:",
        "enabled" if bool(imu_policy_filter_cfg.get("enabled", False)) else "disabled",
        f"gyro_alpha={float(imu_policy_filter_cfg.get('gyro_lowpass_alpha', 1.0)):.2f}",
        f"gyro_clip={imu_policy_filter_cfg.get('gyro_clip_abs', 0.0)}",
    )
    print(
        "Active joints:",
        ", ".join(motor_layer.active_joints)
        if len(motor_layer.active_joints) < len(runner.policy_order)
        else "all",
    )
    print("Fallback fake start pose:", args.fake_start)
    print()

    print("Joint order and motor IDs:")
    for route in joint_mapping.routes:
        print(
            f"{route.policy_index:02d}: {route.policy_joint_name:16s} "
            f"-> motor_id=0x{route.motor_id:02X}"
        )

    buses = None
    imu_sensor = None
    telemetry = None
    gui_proc = None
    csv_logger = None
    can_streamer = None
    sit_stand_trace_logger = None
    startup_zero_calibrated = False
    root_cause_results = {
        "policy_joint_order": "PASS",
        "motor_routing": "PASS",
        "encoder_calibration": (
            "PASS" if calibration_required and calibration_ok else
            "FAIL" if calibration_required else
            "NOT REQUIRED"
        ),
        "ground_contact_validity": "NOT TESTED",
    }

    try:
        csv_logger = CsvRunLogger(
            enabled=args.log_csv,
            log_dir=args.log_dir,
            log_file=args.log_file,
            policy_order=runner.policy_order,
            flush_every=args.csv_flush_every,
            async_enabled=bool(args.async_csv),
            queue_size=int(args.csv_queue_size),
            flush_seconds=float(args.csv_flush_seconds),
            log_prefix=args.log_prefix,
            run_metadata={
                "command_line": " ".join(sys.argv),
                "runtime_control_hz": f"{runtime_control_hz:.6f}",
                "policy_action_scale": f"{runner.action_scale:.6f}",
                "policy_action_formula": (
                    "q_policy_reference + 0.25 * raw_action"
                ),
                "policy_frame_origin": json.dumps(
                    [float(value) for value in runner.policy_frame_origin],
                    separators=(",", ":"),
                ),
                "policy_hip_action_clip": f"{args.policy_hip_action_clip:.6f}",
                "policy_hip_action_scale": f"{args.policy_hip_action_scale:.6f}",
                "exact_policy_after_entry": str(bool(args.exact_policy_after_entry)),
                "previous_action_source": "raw_actor",
                "can_topology": "; ".join(topology_lines(args.can_count, port_by_bus)),
                "can_backend": str(args.can_backend),
                "can_command_hz": f"{float(args.can_command_hz):.6f}",
                "can_command_dt_s": f"{1.0 / float(args.can_command_hz):.6f}",
                "torque_profile_stage": str(args.torque_profile_stage),
                "policy_torque_start_max_nm": f"{max(torque_start_by_joint.values()):.6f}",
                "policy_torque_final_max_nm": f"{max(torque_final_by_joint.values()):.6f}",
                "policy_torque_ramp_delay_s": f"{float(args.policy_torque_ramp_delay_seconds):.6f}",
                "policy_torque_ramp_seconds": f"{float(args.policy_torque_ramp_seconds):.6f}",
                "pose_test_only": str(bool(args.pose_test_only)),
                "pose_gains_config": str(args.pose_gains_config or ""),
            },
        )
        if csv_logger.enabled:
            print("\nCSV log:", csv_logger.path)
            print(
                "CSV automatic Git push:",
                "enabled" if args.auto_push_log else "disabled",
            )
            print(
                "CSV writer:",
                "async" if args.async_csv else "sync",
                f"queue={int(args.csv_queue_size)}",
                f"flush={float(args.csv_flush_seconds):.2f}s",
            )
            print(
                "CSV policy I/O: input=obs_000..047, "
                "raw_output=action_00..11, sent_output=sent_action_00..11, "
                "motor_target=q_target_00..11/qd_target_00..11, "
                "feedback=q_00..11/qd_00..11, "
                "named_joint_columns=<joint>_q_fb/<joint>_q_target/"
                "<joint>_fb_position_raw/<joint>_fb_torque"
            )

        imu_sensor = create_imu_sensor(
            source=args.imu_source,
            port=args.imu_port,
            baud=args.imu_baud,
            background=bool(
                args.async_imu
                and active_imu_source in ("xsens", "xsens_binary", "mtdata2")
            ),
            startup_samples=int(args.imu_startup_samples),
            stale_timeout=args.imu_stale_timeout,
        )
        if imu_sensor is None:
            print("\nIMU source disabled. Policy IMU fields will stay at fallback values.")
        else:
            print(
                "\nIMU opened:",
                getattr(imu_sensor, "source_name", "imu"),
                "port:",
                getattr(imu_sensor, "port", "none"),
                "async:",
                bool(
                    args.async_imu
                    and active_imu_source in ("xsens", "xsens_binary", "mtdata2")
                ),
            )

        if args.mode in ["signal", "mit-signal"]:
            if args.mode == "signal":
                print("\n--mode signal: sends harmless empty CAN frames only.")
                print("This tests USB-CAN transmission even if motors are absent.")
            elif args.mode == "mit-signal":
                print("\nWARNING: --mode mit-signal sends MIT control packets.")
                print("MIT motor feedback is read when the motors reply.")
                print("Use only with the robot secured or suspended for first tests.")

            print("\nOpening CAN interfaces...")
            for bus_name, port in active_port_by_bus.items():
                if backend_for_port(port, args.can_backend) == "socketcan":
                    socketcan_preflight(port, require_exists=True)
            try:
                buses = open_can_buses(
                    active_port_by_bus,
                    baud=args.baud,
                    backend=args.can_backend,
                    bitrate=args.can_bitrate,
                    timeout=0.001 * float(args.can_tx_timeout_ms),
                    socketcan_tx_retry_count=0,
                )
            except Exception:
                raise
            for bus_name, port in active_port_by_bus.items():
                shared = [
                    other_name
                    for other_name, other_bus in buses.items()
                    if other_name != bus_name and other_bus is buses[bus_name]
                ]
                if shared:
                    print(f"USB-CAN {bus_name} shares {port}.")
                else:
                    print(f"USB-CAN {bus_name} ({port}) opened.")

            if feedback_source == "mit":
                feedback_types = motor_layer._feedback_comm_types(motor_layer.proto)
                configured = set()
                for bus in buses.values():
                    if id(bus) in configured:
                        continue
                    configured.add(id(bus))
                    if hasattr(bus, "configure_feedback_filters"):
                        bus.configure_feedback_filters(feedback_types)

        if feedback_source == "mit":
            if args.mode != "mit-signal" or buses is None:
                print("ERROR: --feedback-source mit requires --mode mit-signal.")
                return 1

            estimator = MitFeedbackStateEstimator(
                q_initial=q_fake_start,
                policy_order=runner.policy_order,
                motor_ids=motor_ids,
                motor_layer=motor_layer,
                bus=buses,
                imu_sensor=imu_sensor,
                imu_filter_cfg=imu_policy_filter_cfg,
                pose_references={
                    "default": runner.q_policy_reference,
                    "stand": runner.q_stand,
                    "crouch": runner.q_crouch,
                },
                pose_snap_tolerance=0.35,
                joint_velocity_source=args.joint_velocity_source,
            )
            print("Polling initial motor encoder feedback...")
            motor_layer.send_raw_commands(buses, motor_layer.build_feedback_poll_commands())
            try:
                initial_feedback = estimator.refresh_from_bus(timeout=0.10)
            except Exception as exc:
                if args.calf_range_check:
                    print("Initial decoded feedback failed during calf range check:", exc)
                    initial_feedback = 0
                else:
                    raise
            if initial_feedback <= 0:
                print(
                    "\nMIT feedback: no initial frames yet. "
                    "The estimator will update from motor replies after MIT commands."
                )
        else:
            estimator = FakeStateEstimator(
                q_initial=q_fake_start,
                imu_sensor=imu_sensor,
                imu_filter_cfg=imu_policy_filter_cfg,
            )

        print_joint_coordinate_contract(joint_mapping, runner, estimator)

        if args.calf_range_check:
            ok = run_calf_range_check(
                runner=runner,
                safety=safety,
                motor_layer=motor_layer,
                estimator=estimator,
                buses=buses,
                mode=args.mode,
                feedback_timeout=args.feedback_timeout,
            )
            return 0 if ok else 1

        if hasattr(estimator, "imu_required") and estimator.imu_required():
            imu_ok = wait_for_live_policy_imu(
                estimator,
                source_name=getattr(imu_sensor, "source_name", active_imu_source),
                samples=int(args.imu_startup_samples),
                timeout_s=float(args.imu_startup_timeout),
                max_roll_pitch_deg=float(args.imu_max_startup_roll_pitch_deg),
            )
            if not imu_ok:
                return 1

        if args.auto_zero_on_startup and str(args.initial_zero_frame).lower() == "crouch":
            print("\n[ZERO CAL] startup auto-zero enabled for current crouch/default pose.")
            fresh, n_active = collect_complete_feedback_before_zero(
                estimator=estimator,
                motor_layer=motor_layer,
                safety=safety,
                buses=buses,
                mode=args.mode,
                feedback_timeout=args.feedback_timeout,
            )
            if fresh < n_active:
                print(
                    f"[ZERO CAL] startup auto-zero blocked: only {fresh}/{n_active} "
                    "active joints returned fresh feedback."
                )
                q_auto_zero = False
            else:
                q_auto_zero = apply_software_zero_calibration(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    active_joints=motor_layer.active_joints,
                    feedback_timeout=args.feedback_timeout,
                    buses=buses,
                    mode=args.mode,
                    label="startup crouch/default pose",
                    target_value=args.crouch_calibration_value,
                )
            if q_auto_zero is not False:
                startup_zero_calibrated = True
                print(
                    "[ZERO CAL] startup crouch/default pose is now "
                    f"q={float(args.crouch_calibration_value):+.3f}. D-pad zero is optional."
                )
            else:
                print(
                    "[ZERO CAL] startup auto-zero did not complete. The first stand/sit "
                    "request will try again from live MIT feedback."
                )

        # ── optional GUI subprocess ────────────────────────────────────────────
        if args.gui or args.telemetry_port != TELEMETRY_PORT_DEFAULT:
            telemetry = TelemetrySender(
                port=args.telemetry_port,
                policy_order=runner.policy_order,
                estimator=estimator,
                joint_can_bus=joint_can_bus,
            )
        if args.gui:
            gui_script = Path(__file__).parent / "telemetry_gui.py"
            if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                print("\nWARNING: DISPLAY/WAYLAND_DISPLAY is not set; Tk GUI may not open from this terminal.")
            gui_proc = subprocess.Popen(
                [sys.executable, str(gui_script), "--port", str(args.telemetry_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time.sleep(0.6)   # let tkinter window open before control starts
            if gui_proc.poll() is None:
                print(f"\nTelemetry GUI launched (PID {gui_proc.pid}).")
            else:
                gui_output, _ = gui_proc.communicate(timeout=0.2)
                print("\nERROR: Telemetry GUI exited before opening.")
                print("GUI return code:", gui_proc.returncode)
                if gui_output.strip():
                    print("GUI output:")
                    print(gui_output.strip())
                print("You can run src/telemetry_gui.py directly to see the full error.")
                gui_proc = None

        if args.mode == "mit-signal" and not args.policy_shadow_mode:
            if feedback_source == "mit":
                reason = encoder_safety_stop_reason(
                    safety=safety,
                    estimator=estimator,
                    active_joints=motor_layer.active_joints,
                    mode=args.mode,
                    require_feedback=False,
                    q_shift=motor_layer.coordinate_shift_array(),
                )
                if reason is not None:
                    print("\nWARNING:", reason)
                    print(
                        "Startup is continuing so auto-zero or joystick D-pad "
                        "software-zero can calibrate before commanding sit, stand, or walk."
                    )
            print("Sending motor enable frames...")
            enable_commands = motor_layer.build_enable_commands()
            for retry_index in range(max(1, int(args.enable_retries))):
                motor_layer.send_raw_commands(buses, enable_commands)
                if retry_index + 1 < max(1, int(args.enable_retries)):
                    time.sleep(max(0.0, float(args.enable_retry_delay)))
            print("Motor enable frames sent.")

        if args.joint_routing_test:
            ok = run_joint_routing_test(
                runner=runner,
                motor_layer=motor_layer,
                estimator=estimator,
                buses=buses,
                mode=args.mode,
                dt=runner.control_dt,
                feedback_timeout=args.feedback_timeout,
                amplitude_rad=args.routing_test_amplitude_rad,
                frequency_hz=args.routing_test_frequency_hz,
                cycles=args.routing_test_cycles,
                torque_limit_nm=args.routing_test_torque_limit,
            )
            return 0 if ok else 1

        if args.mode in ("signal", "mit-signal") and not args.policy_shadow_mode:
            if args.pose_test_only and args.sit_stand_trace_200hz:
                trace_metadata = {
                    "robot_mass_kg": float(args.robot_mass_kg),
                    "battery_voltage_start": args.battery_voltage_start,
                    "kp_kd_configuration": (
                        None if pose_gain_profile is None else pose_gain_profile["gains"]
                    ),
                    "torque_limit_nm": pose_torque_limits,
                    "sit_pose_rad_policy_order": [
                        float(value) for value in runner.q_crouch
                    ],
                    "stand_pose_rad_policy_order": [
                        float(value) for value in runner.q_stand
                    ],
                    "pose_transition_speed_rad_s": float(args.pose_transition_speed_rad_s),
                    "pose_transition_min_seconds": float(args.pose_transition_min_seconds),
                    "control_frequency_hz": runtime_control_hz,
                    "can_frequency_hz": float(args.can_command_hz),
                    "motor_firmware_version": str(args.motor_firmware_version),
                    "command_encoding": str(motor_layer.command_encoding),
                    "protocol": "RobStride extended-CAN MIT operation control",
                    "base_height_available": False,
                    "foot_contacts_available": False,
                }
                sit_stand_trace_logger = SitStandTraceLogger(
                    log_dir=args.log_dir,
                    motor_layer=motor_layer,
                    metadata=trace_metadata,
                )
                sit_stand_trace_logger.battery_voltage_start = (
                    None
                    if args.battery_voltage_start is None
                    else float(args.battery_voltage_start)
                )
                print("Sit/stand 200 Hz trace:", sit_stand_trace_logger.path)
                print("Sit/stand trace metadata:", sit_stand_trace_logger.metadata_path)
                print(
                    "WARNING: RS04 MIT feedback has no motor-current or bus-voltage "
                    "fields; those trace columns remain blank."
                )

            def send_latest_can_snapshot(command_snapshot):
                if args.mode == "signal":
                    motor_layer.send_harmless_frames(buses, command_snapshot)
                else:
                    motor_layer.update_periodic_commands(
                        buses,
                        command_snapshot,
                        period_s=1.0 / float(args.can_command_hz),
                    )

            def clear_latest_can_snapshot():
                motor_layer.stop_periodic_commands(buses)

            def receive_latest_can_feedback():
                return MotorCommandLayer.read_all_frames(
                    buses,
                    timeout=0.0,
                    proto=motor_layer.proto,
                    max_frames=24,
                )

            can_streamer = CanCommandStreamer(
                send_callback=send_latest_can_snapshot,
                receive_callback=receive_latest_can_feedback,
                clear_callback=(
                    clear_latest_can_snapshot if args.mode == "mit-signal" else None
                ),
                send_only_on_change=(args.mode == "mit-signal"),
                # The 0c17450 hardware logs show this 100 Hz receive cadence
                # keeps policy work near 9 ms and feedback near 9 ms. Receiving
                # every 5 ms contends for the CAN lock, doubles policy work,
                # and paradoxically makes the actor consume older feedback.
                receive_every_n_cycles=CAN_FEEDBACK_RECEIVE_EVERY_N_CYCLES,
                initial_stale_timeout_s=max(
                    0.250,
                    float(args.can_command_stale_timeout),
                ),
                command_dt_s=1.0 / float(args.can_command_hz),
                stale_timeout_s=float(args.can_command_stale_timeout),
                fault_consecutive_overruns=int(args.can_command_fault_consecutive),
                transport_label=f"{int(args.can_count)}-ADAPTER",
                cycle_callback=(
                    None
                    if sit_stand_trace_logger is None
                    else sit_stand_trace_logger.record_can_cycle
                ),
            )
            can_streamer.start()
            if hasattr(estimator, "update_from_frames"):
                estimator.can_feedback_streamer = can_streamer
            print(
                "Two-rate command streamer started: policy=50 Hz, "
                f"CAN={float(args.can_command_hz):.0f} Hz"
            )

        if args.startup_action == "stand":
            q_previous_target, startup_ok = run_startup_to_stand(
                runner=runner,
                safety=safety,
                motor_layer=motor_layer,
                estimator=estimator,
                buses=buses,
                mode=args.mode,
                standup_seconds=standup_seconds,
                log_every=args.log_every,
                show_hex=args.show_hex,
                feedback_timeout=args.feedback_timeout,
                telemetry=telemetry,
                csv_logger=csv_logger,
                can_streamer=can_streamer,
            )
            if not startup_ok:
                print("Controller aborted because startup-to-stand did not complete safely.")
                return 1
        else:
            q_previous_target = initialize_hold_target(
                estimator=estimator,
                feedback_timeout=args.feedback_timeout,
            )

        root_cause_results = run_policy_loop(
            runner=runner,
            safety=safety,
            motor_layer=motor_layer,
            estimator=estimator,
            command_source=command_source,
            buses=buses,
            mode=args.mode,
            q_previous_target=q_previous_target,
            steps=args.policy_steps,
            log_every=args.log_every,
            print_every=args.print_every,
            show_hex=args.show_hex,
            start_control_mode=args.start_control_mode,
            feedback_timeout=args.feedback_timeout,
            walk_command_threshold=args.walk_command_threshold,
            walk_command_grace_seconds=args.walk_command_grace_seconds,
            walk_stop_confirm_seconds=args.walk_stop_confirm_seconds,
            joystick_debug=args.joystick_debug,
            joint_debug=args.joint_debug,
            base_lin_vel_source=args.base_lin_vel_source,
            motion_assist_cfg=motion_assist_cfg,
            initial_zero_frame=args.initial_zero_frame,
            initial_zero_calibrated=startup_zero_calibrated,
            auto_stand_zero=args.auto_stand_zero,
            auto_sit_zero=args.auto_sit_zero,
            stand_zero_error_rad=args.stand_zero_error_rad,
            stand_ready_error_rad=args.stand_ready_error_rad,
            stand_ready_velocity_rad_s=args.stand_ready_velocity_rad_s,
            stand_zero_settle_steps=max(1, args.stand_zero_settle_steps),
            pose_sync_error_rad=max(0.0, args.pose_sync_error_rad),
            policy_command_gain=args.policy_command_gain,
            policy_command_vx_max=args.policy_command_vx_max,
            policy_command_vy_max=args.policy_command_vy_max,
            policy_command_yaw_max=args.policy_command_yaw_max,
            policy_action_clip=args.policy_action_clip,
            policy_hip_action_clip=args.policy_hip_action_clip,
            policy_hip_action_scale=args.policy_hip_action_scale,
            policy_action_smoothing=args.policy_action_smoothing,
            policy_action_delta_limit=args.policy_action_delta_limit,
            policy_entry_ramp_seconds=args.policy_entry_ramp_seconds,
            policy_sim_match=bool(args.policy_sim_match),
            exact_policy_after_entry=bool(args.exact_policy_after_entry),
            auto_policy_after_stand=bool(args.auto_policy_after_stand),
            stand_policy_stabilization=bool(args.stand_policy_stabilization),
            hold_capture_seconds=max(0.02, args.hold_capture_seconds),
            hold_command_repeats=max(1, args.hold_command_repeats),
            crouch_calibration_value=float(args.crouch_calibration_value),
            stand_calibration_value=float(args.stand_calibration_value),
            pose_transition_speed_rad_s=float(args.pose_transition_speed_rad_s),
            pose_transition_min_seconds=float(args.pose_transition_min_seconds),
            fresh_feedback_max_age_s=float(args.fresh_feedback_max_age),
            steady_feedback_budget_s=0.001 * float(args.steady_feedback_budget_ms),
            suspension_status_seconds=float(args.suspension_status_seconds),
            imu_active_max_roll_pitch_deg=float(args.imu_max_active_roll_pitch_deg),
            deadline_tolerance_s=0.001 * float(args.deadline_tolerance_ms),
            deadline_resync_s=0.001 * float(args.deadline_resync_ms),
            timing_fault_consecutive=int(args.timing_fault_consecutive),
            policy_shadow_mode=bool(args.policy_shadow_mode),
            encoder_calibration_required=bool(calibration_required),
            encoder_calibration_passed=bool(calibration_ok),
            torque_ramp=torque_ramp,
            measured_torque_soft_limits=measured_torque_soft_limits,
            telemetry=telemetry,
            csv_logger=csv_logger,
            can_streamer=can_streamer,
            pose_test_only=bool(args.pose_test_only),
            sit_stand_trace_logger=sit_stand_trace_logger,
        )

    except KeyboardInterrupt:
        print("\nKeyboard interrupt: stopping controller.")

    finally:
        command_source.close()
        if imu_sensor is not None:
            imu_sensor.close()
        if can_streamer is not None:
            can_streamer.stop()
        if sit_stand_trace_logger is not None:
            sit_stand_trace_logger.close()
            print(
                "Sit/stand 200 Hz trace saved:",
                sit_stand_trace_logger.path,
                f"({sit_stand_trace_logger.rows_written} rows)",
            )
        if buses is not None:
            if args.mode == "mit-signal":
                try:
                    print("\nSending motor stop frames...")
                    motor_layer.send_raw_commands(buses, motor_layer.build_stop_commands())
                    print("Motor stop frames sent.")
                except Exception as exc:
                    print("\nWARNING: failed to send motor stop frames:", exc)
            motor_layer.close()
            close_can_buses(buses)
            print("\nCAN interfaces closed.")
        if telemetry is not None:
            telemetry.close()
        if csv_logger is not None:
            csv_path = csv_logger.path
            csv_logger.close()
            if csv_path is not None:
                print("CSV log saved:", csv_path)
                if args.auto_push_log:
                    pushed, message = publish_run_log_to_git(
                        csv_path,
                        remote=args.log_git_remote,
                    )
                    prefix = (
                        "CSV log Git push:"
                        if pushed
                        else "WARNING: CSV log Git push skipped:"
                    )
                    print(prefix, message)
        if gui_proc is not None:
            gui_proc.terminate()
            try:
                gui_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                gui_proc.kill()
            print("Telemetry GUI closed.")
        print()
        for line in root_cause_report_lines(root_cause_results):
            print(line)

    print("\nController finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
