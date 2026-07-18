#!/usr/bin/env python3
import argparse
import csv
import json
import os
import socket
import subprocess
import sys
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
from imu_interface import create_imu_sensor, load_imu_config
from motor_command_layer import MotorCommandLayer, print_mit_commands
from can_topology import (
    add_can_topology_args,
    close_can_buses,
    open_can_buses,
    ports_for_active_joints,
    resolve_joint_can_bus,
    resolve_port_by_bus,
    topology_lines,
)


ROOT = Path(__file__).resolve().parents[1]

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


def load_motion_assist_config():
    return load_yaml(ROOT / "config" / "motion_assist.yaml")


def load_policy_deployment_defaults():
    cfg = load_yaml(ROOT / "config" / "control_limits.yaml")
    return cfg.get("policy_deployment", {})


def smoothstep(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def synchronized_pose_trajectory(start, target, elapsed_s, duration_s):
    """Interpolate every joint with one smooth phase so they finish together."""
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    duration_s = max(float(duration_s), 1.0e-6)
    alpha = float(np.clip(float(elapsed_s) / duration_s, 0.0, 1.0))
    blend = smoothstep(alpha)
    return (start + blend * (target - start)).astype(np.float32), alpha


def fake_start_pose_array(runner, name):
    if name == "stand":
        return runner.q_stand.copy()
    if name == "crouch":
        return runner.q_crouch.copy()
    if name == "random_small":
        rng = np.random.default_rng(7)
        return rng.uniform(-0.25, 0.25, size=len(runner.policy_order)).astype(np.float32)
    raise ValueError(f"Unknown fake start pose: {name}")


def refresh_estimator_feedback(estimator, timeout=0.0):
    if hasattr(estimator, "refresh_from_bus"):
        return estimator.refresh_from_bus(timeout=timeout)
    return 0


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


def projected_gravity_to_roll_pitch(projected_gravity_b):
    g = np.asarray(projected_gravity_b, dtype=np.float32)
    down_z = max(1e-6, -float(g[2]))
    # projected_gravity_b = R_world_from_body.T @ [0, 0, -1]. Therefore a
    # positive body roll produces negative gravity-Y, and positive pitch
    # produces positive gravity-X. These signs match IsaacLab and the Xsens
    # world-from-body quaternion conversion in imu_interface.py.
    roll = float(np.arctan2(-float(g[1]), down_z))
    pitch = float(np.arctan2(float(g[0]), down_z))
    return roll, pitch


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


def joint_pose_sign(reference_pose_value):
    sign = float(np.sign(reference_pose_value))
    return sign if abs(sign) > 0.0 else 1.0


def apply_gait_assist(q_target, command, elapsed_time, runner, cfg):
    gait_cfg = cfg.get("gait_assist", {})
    if not bool(gait_cfg.get("enabled", False)):
        return q_target

    command = np.asarray(command, dtype=np.float32)
    command_mag = float(max(np.linalg.norm(command[:2]), abs(command[2])))
    min_command = float(gait_cfg.get("min_command", 0.03))
    if command_mag < min_command:
        return q_target

    full_scale_command = max(min_command + 1e-6, float(gait_cfg.get("full_scale_command", 0.15)))
    scale = float(np.clip((command_mag - min_command) / (full_scale_command - min_command), 0.0, 1.0))
    frequency_hz = float(gait_cfg.get("frequency_hz", 1.0))
    thigh_amp = float(gait_cfg.get("thigh_amplitude", 0.0)) * scale
    calf_amp = float(gait_cfg.get("calf_amplitude", 0.0)) * scale
    hip_amp = float(gait_cfg.get("hip_amplitude", 0.0)) * scale

    q_target = np.asarray(q_target, dtype=np.float32).copy()
    index_by_joint = {joint_name: i for i, joint_name in enumerate(runner.policy_order)}
    phase_by_leg = {
        "FL": 0.0,
        "BR": 0.0,
        "FR": np.pi,
        "BL": np.pi,
    }
    reverse_phase = np.pi if command[0] < -min_command else 0.0

    for leg, leg_phase in phase_by_leg.items():
        phase = 2.0 * np.pi * frequency_hz * float(elapsed_time) + leg_phase + reverse_phase
        swing = float(np.sin(phase))
        lift = max(0.0, swing)

        hip_name = f"{leg}_hip_joint"
        thigh_name = f"{leg}_thigh_joint"
        calf_name = f"{leg}_calf_joint"

        if hip_name in index_by_joint:
            lateral = float(command[1]) + 0.5 * float(command[2])
            q_target[index_by_joint[hip_name]] += hip_amp * np.sign(lateral) * swing if abs(lateral) > min_command else 0.0
        if thigh_name in index_by_joint:
            i = index_by_joint[thigh_name]
            q_target[i] += joint_pose_sign(runner.q_crouch[i]) * thigh_amp * swing
        if calf_name in index_by_joint:
            i = index_by_joint[calf_name]
            q_target[i] += joint_pose_sign(runner.q_crouch[i]) * calf_amp * lift

    return q_target


def apply_motion_assists(q_target, command, elapsed_time, projected_gravity_b, runner, cfg, use_gait):
    q = apply_imu_posture_stabilization(
        q_target=q_target,
        projected_gravity_b=projected_gravity_b,
        policy_order=runner.policy_order,
        cfg=cfg,
    )
    if use_gait:
        q = apply_gait_assist(
            q_target=q,
            command=command,
            elapsed_time=elapsed_time,
            runner=runner,
            cfg=cfg,
        )
    return q


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
            value = cmd["tau_ff"]
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
    q_target=None,
    policy_order=None,
    policy_sha256=None,
    imu_correction_abs_max=0.0,
    loop_dt_s=None,
    can_tx_s=None,
    feedback_age_max_s=None,
    feedback_fresh_count=None,
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
        "can_tx_ms": None if can_tx_s is None else 1000.0 * float(can_tx_s),
        "feedback_age_max_ms": (
            None if feedback_age_max_s is None else 1000.0 * float(feedback_age_max_s)
        ),
        "feedback_fresh": "" if feedback_fresh_count is None else int(feedback_fresh_count),
        "tau_fb": None,
        "tau_fb_max": None,
        "fault_reason": "",
        "policy_joint_order": "" if policy_order is None else ",".join(policy_order),
        "policy_sha256": "" if policy_sha256 is None else str(policy_sha256),
    }
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

    if policy_order is not None:
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
        obs_values = (
            None if observation is None else np.asarray(observation, dtype=np.float32).reshape(-1)
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
            if q_target_values is not None and q_target_values.shape[0] > index:
                record[f"{prefix}_q_target"] = float(q_target_values[index])
                if q_values is not None and q_values.shape[0] > index:
                    record[f"{prefix}_q_error"] = float(q_target_values[index] - q_values[index])
            if qd_target_values is not None and qd_target_values.shape[0] > index:
                record[f"{prefix}_qd_target"] = float(qd_target_values[index])
            if raw_action_values is not None and raw_action_values.shape[0] > index:
                record[f"{prefix}_action_raw"] = float(raw_action_values[index])
            if sent_action_values is not None and sent_action_values.shape[0] > index:
                record[f"{prefix}_action_sent"] = float(sent_action_values[index])
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
                    "q_des",
                    "q_before_torque_limit",
                    "torque_limited",
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
    return record


def joint_telemetry_fieldnames(policy_order):
    fields = []
    scalar_fields = [
        "index",
        "q_fb",
        "qd_fb",
        "q_target",
        "qd_target",
        "q_error",
        "action_raw",
        "action_sent",
        "obs_joint_pos",
        "obs_joint_vel",
        "obs_prev_action",
    ]
    command_fields = [
        "motor_id",
        "bus_name",
        "phase",
        "command_encoding",
        "q_requested",
        "q_des",
        "q_before_torque_limit",
        "torque_limited",
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
        f"speed={float(record['speed']):.2f} "
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
            f" policy_cmd=["
            f"{float(record['policy_vx']): .3f},"
            f"{float(record['policy_vy']): .3f},"
            f"{float(record['policy_yaw']): .3f}]"
        )

    if record["tau_fb"] is None:
        line += " tau_fb=na tau_fb_max=na"
    else:
        line += (
            f" tau_fb={float(record['tau_fb']): .3f} "
            f"tau_fb_max={float(record['tau_fb_max']): .3f}"
        )
    return line


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
        "imu_correction_abs_max",
        "act_max",
        "tau_cmd",
        "tau_cmd_max",
        "cmds",
        "bus_counts",
        "loop_dt_ms",
        "loop_hz",
        "can_tx_ms",
        "feedback_age_max_ms",
        "feedback_fresh",
        "tau_fb",
        "tau_fb_max",
        "fault_reason",
        "policy_joint_order",
        "policy_sha256",
        "compact_line",
    ]
    BASE_FIELDNAMES += [f"obs_{index:03d}" for index in range(48)]
    BASE_FIELDNAMES += [f"action_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"sent_action_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"qd_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"q_target_{index:02d}" for index in range(12)]
    BASE_FIELDNAMES += [f"qd_target_{index:02d}" for index in range(12)]

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

    def __init__(self, enabled=True, log_dir=None, log_file=None, policy_order=None):
        self.enabled = bool(enabled)
        self.path = None
        self.run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        self.start_time = time.monotonic()
        self._file = None
        self._writer = None
        self.fieldnames = list(self.BASE_FIELDNAMES)
        self.fieldnames.extend(joint_telemetry_fieldnames(policy_order or []))

        if not self.enabled:
            return

        if log_file:
            self.path = Path(log_file).expanduser()
            self.run_id = self.path.stem
        else:
            directory = Path(log_dir).expanduser() if log_dir else ROOT / "logs"
            self.path = self._unique_log_path(directory, f"grallator_run_{self.run_id}")
            self.run_id = self.path.stem.removeprefix("grallator_run_")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._file.flush()

    def log(self, record):
        if not self.enabled or self._writer is None:
            return

        row = {field: "" for field in self.fieldnames}
        row.update(record)
        row["run_id"] = self.run_id
        row["wall_time"] = datetime.now().isoformat(timespec="milliseconds")
        row["elapsed_s"] = f"{time.monotonic() - self.start_time:.6f}"
        row["compact_line"] = compact_telemetry_line(record)
        if row["tau_fb"] is None:
            row["tau_fb"] = ""
        if row["tau_fb_max"] is None:
            row["tau_fb_max"] = ""
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
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
    max_age_s = (
        getattr(safety, "max_feedback_age_s", 0.25)
        if max_age_s is None
        else float(max_age_s)
    )
    deadline = time.monotonic() + max(float(max_wait_s), float(feedback_timeout), 0.02)

    refresh_estimator_feedback(estimator, timeout=0.0)
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
                motor_layer.send_signal_commands(buses, keepalive_commands)
            elif allow_poll_snapshot:
                request_feedback_snapshot(motor_layer, buses, mode)

        remaining = max(0.0, deadline - time.monotonic())
        refresh_estimator_feedback(
            estimator,
            timeout=min(float(feedback_timeout), remaining),
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

        if mode == "signal":
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
    show_hex,
    start_control_mode,
    feedback_timeout,
    walk_command_threshold,
    walk_command_grace_seconds,
    joystick_debug,
    joint_debug,
    base_lin_vel_source,
    motion_assist_cfg,
    initial_zero_frame,
    initial_zero_calibrated,
    auto_stand_zero,
    auto_sit_zero,
    stand_zero_error_rad,
    stand_zero_settle_steps,
    pose_sync_error_rad,
    policy_command_gain,
    policy_command_vx_max,
    policy_command_vy_max,
    policy_command_yaw_max,
    policy_action_clip,
    policy_action_smoothing,
    policy_action_delta_limit,
    policy_entry_ramp_seconds,
    policy_sim_match,
    stand_policy_stabilization,
    hold_capture_seconds,
    hold_command_repeats,
    crouch_calibration_value,
    stand_calibration_value,
    pose_transition_speed_rad_s,
    pose_transition_min_seconds,
    fresh_feedback_max_age_s,
    telemetry=None,
    csv_logger=None,
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
    # Preserve the trained observation contract: slots 36:48 contain the
    # previous raw actor output. Hardware clipping, smoothing, and target slew
    # limiting are applied downstream without changing this policy input.
    previous_action = np.zeros(action_dim, dtype=np.float32)
    previous_sent_action = np.zeros(action_dim, dtype=np.float32)
    direct_leveling_correction = np.zeros(action_dim, dtype=np.float32)
    policy_entry_elapsed_s = 0.0
    policy_entry_scale = 0.0
    previous_walk_requested = False
    last_walk_command = np.zeros(3, dtype=np.float32)
    last_walk_command_step = -10**9
    policy_has_started = False
    direct_imu_stabilization_enabled = bool(
        motion_assist_cfg.get("imu_posture", {}).get("enabled", False)
    )

    control_mode = start_control_mode  # options: idle, hold, policy, stand, sit
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
    walking_armed = control_mode == "policy"
    stand_ready_pending = control_mode == "stand"
    stand_ready_settle_count = 0
    calibration_hold_until_step = -1
    active_indices = active_joint_indices(runner.policy_order, motor_layer.active_joints)
    pose_transition_mode = None
    pose_transition_start = np.asarray(q_previous_target, dtype=np.float32).copy()
    pose_transition_target = pose_transition_start.copy()
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
        print(
            f"[POSE] synchronized {mode_name} transition: "
            f"distance={active_distance:.3f} rad duration={duration:.2f}s "
            f"speed_limit={speed:.3f} rad/s"
        )

    def current_pose_transition_target(mode_name, current_step):
        nonlocal pose_transition_mode
        if pose_transition_mode != mode_name:
            begin_pose_transition(mode_name, q_previous_target, current_step)
        target_q, _ = synchronized_pose_trajectory(
            pose_transition_start,
            pose_transition_target,
            pose_transition_elapsed_s,
            pose_transition_duration_s,
        )
        return target_q

    print("\n" + "#" * 80)
    print("RUNTIME CONTROL PHASE")
    print("#" * 80)
    print("mode:", mode)
    print("Joystick buttons:")
    print("  button 4    -> STOP walking and SIT/CROUCH pose")
    print("  button 5    -> STAND pose")
    print("  buttons 0-3 -> EMERGENCY STOP")
    print("  D-pad zero request -> ignored while fixed hardware stand-zero is active")
    print("Joystick axes:")
    print("  left stick Y  -> forward/back vx")
    print("  left stick X  -> left/right vy")
    print("  right stick X -> turn/yaw")
    print("Terminal keys:")
    print("  w/s -> straight vx, a/d -> lateral vy, combine for xy diagonal")
    print("  q/e -> positive/negative yaw; combine with translation if needed")
    print("  c -> SIT/CROUCH, space -> STAND")
    print("  up/down arrows -> increase/decrease speed scale")
    print("  h -> HOLD current position, x -> EMERGENCY STOP")
    print("start_control_mode:", control_mode)
    print("walk_command_threshold:", walk_command_threshold)
    print("walk_command_grace_seconds:", float(walk_command_grace_seconds))
    print("base_lin_vel_source:", base_lin_vel_source)
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
    print("policy_command_gain:", float(policy_command_gain))
    print(
        "policy_command_caps:",
        f"vx={float(policy_command_vx_max):.3f}",
        f"vy={float(policy_command_vy_max):.3f}",
        f"yaw={float(policy_command_yaw_max):.3f}",
        "(0 disables each cap)",
    )
    print("policy_action_clip:", float(policy_action_clip), "(0 disables)")
    print("policy_action_smoothing:", float(policy_action_smoothing), "(0 disables)")
    print("policy_action_delta_limit:", float(policy_action_delta_limit), "(0 disables)")
    print("policy_entry_ramp_seconds:", float(policy_entry_ramp_seconds))
    print(
        "policy_sim_match:",
        bool(policy_sim_match),
        "(hard joint, encoder, tilt, and torque safety remain active)",
    )
    if policy_sim_match:
        print(
            "WARNING: policy_sim_match bypasses deployment action clipping, "
            "action slew limiting, smoothing, and policy target rate limiting. "
            "Use it only for suspended/dry sim comparison, not first ground walking."
        )
    print("stand_policy_stabilization:", bool(stand_policy_stabilization))
    print("walking_armed:", bool(walking_armed))
    print("hold_capture_seconds:", float(hold_capture_seconds))
    print("hold_command_repeats:", int(hold_command_repeats))
    print("fresh_feedback_max_age_s:", float(live_feedback_max_age_s))
    print("pose_transition_speed_rad_s:", float(pose_transition_speed_rad_s))
    print("pose_transition_min_seconds:", float(pose_transition_min_seconds))
    print("crouch_calibration_value:", float(crouch_calibration_value))
    print("stand_calibration_value:", float(stand_calibration_value))
    if stand_zero_pending:
        print("[ZERO CAL] initial stand target will auto-zero when settled.")
    if sit_zero_pending:
        print("[ZERO CAL] initial sit/crouch target will auto-zero when settled.")
    print("imu_stabilization:", bool(motion_assist_cfg.get("imu_posture", {}).get("enabled", False)))
    print("gait_assist:", bool(motion_assist_cfg.get("gait_assist", {}).get("enabled", False)))
    if steps is None:
        print("policy_steps: unlimited, running until emergency stop or Ctrl+C")
    else:
        print("policy_steps:", steps)

    step = 0
    previous_cycle_start = None
    while steps is None or step < steps:
        cycle_start = time.monotonic()
        loop_dt_s = (
            None
            if previous_cycle_start is None
            else cycle_start - previous_cycle_start
        )
        previous_cycle_start = cycle_start
        observation_for_log = None
        raw_action = np.zeros(action_dim, dtype=np.float32)
        imu_correction_abs_max = 0.0
        (
            q_current,
            qd_current,
            base_lin_vel_b,
            base_ang_vel_b,
            projected_gravity_b,
        ) = estimator.read()
        q_coordinate_shift = motor_layer.coordinate_shift_array()

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
                    refresh_estimator_feedback(estimator, timeout=feedback_timeout)
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
                )
                (
                    q_current,
                    qd_current,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    projected_gravity_b,
                ) = estimator.read()
            reason = encoder_safety_stop_reason(
                safety=safety,
                estimator=estimator,
                active_joints=motor_layer.active_joints,
                mode=mode,
                require_feedback=True,
                q_shift=q_coordinate_shift,
            )
            if reason is not None:
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

        stop, reason = safety.emergency_stop_check(
            projected_gravity_b=projected_gravity_b,
            base_ang_vel_b=base_ang_vel_b,
        )
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
        if hasattr(estimator, "imu_stale") and estimator.imu_stale():
            reason = "IMU data missing or stale"
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
        if mode_request == control_mode and mode_request in ("stand", "sit", "hold"):
            # Ignore terminal auto-repeat and duplicate controller events. A
            # repeated pose request must not recapture feedback and restart the
            # trajectory, which causes a visible stop and torque ramp.
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
                feedback_count = refresh_estimator_feedback(
                    estimator,
                    timeout=feedback_timeout,
                )
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
                        )
                        if fresh < n_active:
                            print(
                                f"[FEEDBACK] pose transition still has only "
                                f"{fresh}/{n_active} fresh motor feedback frame(s)."
                            )
                reason = encoder_safety_stop_reason(
                    safety=safety,
                    estimator=estimator,
                    active_joints=motor_layer.active_joints,
                    mode=mode,
                    q_shift=q_coordinate_shift,
                )
                if reason is not None:
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
                (
                    q_hold,
                    feedback_count,
                    hold_missing,
                    q_current,
                    qd_current,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    projected_gravity_b,
                ) = acquire_hold_target_from_feedback(
                    estimator,
                    motor_layer,
                    safety,
                    q_previous_target,
                    feedback_timeout=feedback_timeout,
                    capture_seconds=hold_capture_seconds,
                    buses=buses,
                    mode=mode,
                    allow_poll_snapshot=not has_motion_target,
                )
                q_previous_target = q_hold.copy()
                previous_action = np.zeros(action_dim, dtype=np.float32)
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

            if mode_request is not None:
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
                if control_mode in ("stand", "sit", "hold"):
                    last_walk_command.fill(0.0)
                    last_walk_command_step = -10**9
                if control_mode in ("stand", "sit"):
                    begin_pose_transition(control_mode, q_current, step)
                else:
                    pose_transition_mode = None
                print(f"\n[MODE CHANGE] control_mode -> {control_mode}")

        command = command_source.read()
        if step < calibration_hold_until_step:
            command = np.zeros(3, dtype=np.float32)
        raw_walk_requested = joystick_walk_requested(command, walk_command_threshold)
        walk_requested = raw_walk_requested
        if raw_walk_requested:
            last_walk_command = np.asarray(command, dtype=np.float32).copy()
            last_walk_command_step = int(step)
        elif (
            walking_armed
            and control_mode == "stand"
            and float(walk_command_grace_seconds) > 0.0
            and joystick_walk_requested(last_walk_command, walk_command_threshold)
        ):
            elapsed_since_walk_s = (int(step) - int(last_walk_command_step)) * float(dt)
            if elapsed_since_walk_s <= float(walk_command_grace_seconds):
                command = last_walk_command.copy()
                walk_requested = True
        policy_command = scaled_policy_command(
            command=command,
            gain=policy_command_gain,
            vx_abs_max=policy_command_vx_max,
            vy_abs_max=policy_command_vy_max,
            yaw_abs_max=policy_command_yaw_max,
        )
        if not walking_armed and walk_requested:
            walk_requested = False
            if step % max(1, log_every) == 0:
                print("[POSE] walking blocked until STAND reaches its target.")
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
            refresh_estimator_feedback(estimator, timeout=feedback_timeout)
            fresh = count_fresh_active_feedback(
                estimator,
                motor_layer.active_joints,
                live_feedback_max_age_s,
            )
            if fresh < len(motor_layer.active_joints):
                if step % max(1, log_every) == 0:
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
                if step % max(1, log_every) == 0:
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
                previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                fresh_feedback_for_commands, _ = fresh_feedback_by_joint(
                    estimator,
                    motor_layer.active_joints,
                    live_feedback_max_age_s,
                )

        if walk_requested:
            has_motion_target = True
            policy_has_started = True

        walk_just_stopped = bool(previous_walk_requested and not walk_requested)
        if walk_requested:
            if not previous_walk_requested:
                policy_entry_elapsed_s = 0.0
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
            # A terminal movement key can briefly time out between key-repeat
            # events. Restart the stand trajectory from the exact previously
            # sent motor targets instead of snapping from a gait target to
            # stand q=0 under the stronger startup impedance.
            begin_pose_transition("stand", q_previous_target, step)

        if active_control_mode == "idle":
            q_safe_target = q_previous_target.copy()
            commands = []
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "hold":
            q_safe_target = q_previous_target.copy()
            commands = (
                motor_layer.build_mit_commands(
                    q_safe_target,
                    phase="hold",
                    feedback_by_joint=fresh_feedback_for_commands,
                )
                if has_motion_target
                else []
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "stand":
            q_policy_target = current_pose_transition_target("stand", step)
            # Initial crouch-to-stand lifting retains the proven startup
            # impedance. Once gait has started, every return to stand uses the
            # bounded policy impedance to avoid a gain-switch torque impulse.
            stand_command_phase = "policy" if policy_has_started else "startup"
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
            q_safe_target = shifted_safety_filter(
                safety,
                q_policy_target,
                q_previous_target,
                q_coordinate_shift,
            )
            commands = motor_layer.build_mit_commands(
                q_safe_target,
                phase=stand_command_phase,
                feedback_by_joint=fresh_feedback_for_commands,
            )

        elif active_control_mode == "sit":
            q_policy_target = current_pose_transition_target("sit", step)
            q_safe_target = shifted_safety_filter(
                safety,
                q_policy_target,
                q_previous_target,
                q_coordinate_shift,
            )
            commands = motor_layer.build_mit_commands(
                q_safe_target,
                phase="startup",
                feedback_by_joint=fresh_feedback_for_commands,
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "policy":
            obs = runner.build_observation(
                base_ang_vel_b=base_ang_vel_b,
                projected_gravity_b=projected_gravity_b,
                command=policy_command,
                q_current=q_current,
                qd_current=qd_current,
                previous_action=previous_action,
            )

            raw_action = runner.infer_action(obs)
            observation_for_log = obs.copy()
            if policy_sim_match:
                action = np.asarray(raw_action, dtype=np.float32).copy()
            else:
                action = filtered_policy_action(
                    raw_action=raw_action,
                    previous_action=previous_sent_action,
                    clip_abs=policy_action_clip,
                    smoothing=policy_action_smoothing,
                    delta_limit_abs=policy_action_delta_limit,
                )
            action = np.asarray(action, dtype=np.float32) * float(policy_entry_scale)
            q_policy_target = runner.action_to_q_target(action)
            imu_correction = imu_posture_correction(
                projected_gravity_b=projected_gravity_b,
                policy_order=runner.policy_order,
                cfg=motion_assist_cfg,
            )
            imu_correction_abs_max = float(np.max(np.abs(imu_correction)))
            q_policy_target = apply_motion_assists(
                q_target=q_policy_target,
                command=policy_command,
                elapsed_time=step * dt,
                projected_gravity_b=projected_gravity_b,
                runner=runner,
                cfg=motion_assist_cfg,
                use_gait=True,
            )
            q_safe_target = shifted_safety_filter(
                safety,
                q_policy_target,
                q_previous_target,
                q_coordinate_shift,
                apply_rate_limit=not policy_sim_match,
                use_policy_limits=True,
            )
            commands = motor_layer.build_mit_commands(
                q_safe_target,
                phase="policy",
                feedback_by_joint=fresh_feedback_for_commands,
            )

        else:
            raise RuntimeError(f"Unknown control_mode: {active_control_mode}")

        send_repeats = (
            max(1, int(hold_command_repeats))
            if active_control_mode == "hold"
            else 1
        )
        for _ in range(send_repeats):
            if mode == "signal":
                motor_layer.send_harmless_frames(buses, commands)
            elif mode == "mit-signal":
                motor_layer.send_signal_commands(buses, commands)
        can_tx_s = max_can_tx_duration_s(buses)

        active_feedback_timeout = min(
            float(feedback_timeout),
            max(0.002, 0.35 * float(dt)),
        )
        refresh_estimator_feedback(estimator, timeout=active_feedback_timeout)
        post_send_fresh_feedback, post_send_missing_feedback = fresh_feedback_by_joint(
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
        if post_send_feedback_incomplete and step % max(1, log_every) == 0:
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
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=estimator,
            active_joints=motor_layer.active_joints,
            mode=mode,
            require_feedback=require_command_feedback,
            q_shift=q_coordinate_shift,
            feedback_by_joint=safety_feedback_by_joint,
        )
        if reason is not None:
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
                    if step % max(1, log_every) == 0:
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
                    previous_action = np.zeros(action_dim, dtype=np.float32)
                    previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                    commands = motor_layer.build_mit_commands(
                        q_safe_target,
                        phase="startup",
                        feedback_by_joint=fresh_feedback_for_commands,
                    )
                    if mode == "signal":
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
            if (
                command_error <= float(stand_zero_error_rad)
                and feedback_error <= float(stand_zero_error_rad)
            ):
                stand_ready_settle_count += 1
            else:
                stand_ready_settle_count = 0
            if stand_ready_settle_count >= int(stand_zero_settle_steps):
                walking_armed = True
                stand_ready_pending = False
                stand_ready_settle_count = 0
                previous_action = np.zeros(action_dim, dtype=np.float32)
                previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                if stand_policy_stabilization:
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
                    zero_frame = "crouch"
                    sit_zero_pending = False
                    sit_zero_settle_count = 0
                    q_coordinate_shift = motor_layer.coordinate_shift_array()
                    q_safe_target = constant_pose_like(runner, crouch_calibration_value)
                    q_previous_target = q_safe_target.copy()
                    zero_calibrated = True
                    previous_action = np.zeros(action_dim, dtype=np.float32)
                    previous_sent_action = np.zeros(action_dim, dtype=np.float32)
                    commands = motor_layer.build_mit_commands(
                        q_safe_target,
                        phase="startup",
                        feedback_by_joint=fresh_feedback_for_commands,
                    )
                    if mode == "signal":
                        motor_layer.send_harmless_frames(buses, commands)
                    elif mode == "mit-signal":
                        motor_layer.send_signal_commands(buses, commands)
                    print(
                        "[ZERO CAL] zero_frame -> crouch. Crouch hold is active at "
                        f"q={float(crouch_calibration_value):+.3f}."
                    )

        if step % log_every == 0:
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
                q_target=q_safe_target,
                policy_order=runner.policy_order,
                policy_sha256=runner.policy_sha256,
                imu_correction_abs_max=imu_correction_abs_max,
                loop_dt_s=loop_dt_s,
                can_tx_s=can_tx_s,
                feedback_age_max_s=feedback_age_max_s,
                feedback_fresh_count=feedback_fresh_count,
            )
            print(compact_telemetry_line(telemetry_record))
            if csv_logger is not None:
                csv_logger.log(telemetry_record)
            if show_hex and commands:
                print_mit_commands(commands, show_hex=True)
            if joystick_debug:
                print_joystick_debug(command_source)
            if joint_debug:
                print_joint_debug(commands, estimator)

        q_sent_target = np.asarray(q_safe_target, dtype=np.float32).copy()
        for command_item in commands:
            joint_name = command_item.get("joint_name")
            index = motor_layer.policy_index_by_joint.get(joint_name)
            if index is not None and "q_des" in command_item:
                q_sent_target[index] = float(command_item["q_des"])

        estimator.dry_update_as_if_robot_followed(q_sent_target, dt)

        if active_control_mode == "policy":
            previous_action = raw_action.copy()
            previous_sent_action = action.copy()
        else:
            previous_action = np.zeros(action_dim, dtype=np.float32)
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
        cycle_elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, dt - cycle_elapsed))

    print("\nRuntime control phase completed.")


def main():
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

    add_can_topology_args(parser, default_port="/dev/ttyUSB0", default_can_count=2)
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

    parser.add_argument("--command-source", choices=["fixed", "joystick", "keyboard"], default="joystick")

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
    parser.add_argument("--standup-seconds", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=25)
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
        "--walk-command-threshold",
        type=float,
        default=0.02,
        help="minimum absolute vx/vy/yaw command needed to run walking policy",
    )
    parser.add_argument(
        "--walk-command-grace-seconds",
        type=float,
        default=0.0,
        help=(
            "seconds to keep the last nonzero walking command alive after a "
            "terminal key-repeat gap; 0 disables the grace window"
        ),
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
        "--feedback-timeout",
        type=float,
        default=0.05,
        help="seconds to wait for MIT feedback after sending commands",
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
        "--base-lin-vel-source",
        choices=["zero"],
        default="zero",
        help="fixed policy contract: observation indices 0:3 are always [0,0,0]",
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
        help="enable/disable walking-style sinusoidal leg overlay for suspended tests",
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
            "This does not change the raw policy output stored in previous_action obs slots"
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
        default=float(policy_deploy_defaults.get("estimated_pd_torque_limit", 25.0)),
        help="estimated per-joint policy PD torque ceiling in Nm; must be positive",
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
        "--auto-push-log",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    args.log_every = max(1, args.log_every)
    args.policy_steps = None if args.policy_steps <= 0 else args.policy_steps
    args.walk_command_threshold = max(0.0, args.walk_command_threshold)
    args.walk_command_grace_seconds = max(0.0, float(args.walk_command_grace_seconds))
    args.policy_command_gain = max(0.0, float(args.policy_command_gain))
    args.policy_command_vx_max = max(0.0, float(args.policy_command_vx_max))
    args.policy_command_vy_max = max(0.0, float(args.policy_command_vy_max))
    args.policy_command_yaw_max = max(0.0, float(args.policy_command_yaw_max))
    args.policy_action_clip = max(0.0, float(args.policy_action_clip))
    args.policy_action_smoothing = float(np.clip(args.policy_action_smoothing, 0.0, 0.98))
    args.policy_action_delta_limit = max(0.0, float(args.policy_action_delta_limit))
    if not np.isfinite(args.policy_entry_ramp_seconds) or args.policy_entry_ramp_seconds < 0.0:
        parser.error("--policy-entry-ramp-seconds must be finite and >= 0")
    if not np.isfinite(args.policy_pd_torque_limit) or args.policy_pd_torque_limit <= 0.0:
        parser.error("--policy-pd-torque-limit must be finite and > 0")
    if not np.isfinite(args.fresh_feedback_max_age) or args.fresh_feedback_max_age < 0.0:
        parser.error("--fresh-feedback-max-age must be finite and >= 0")
    if not np.isfinite(args.control_hz) or args.control_hz < 0.0:
        parser.error("--control-hz must be finite and >= 0")
    if 0.0 < args.control_hz < 10.0 or args.control_hz > 100.0:
        parser.error("--control-hz must be 0 or within 10..100 Hz")
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
    trained_control_dt = float(runner.control_dt)
    trained_control_hz = 1.0 / trained_control_dt
    if args.control_hz > 0.0:
        runner.control_dt = 1.0 / float(args.control_hz)
    runtime_control_hz = 1.0 / float(runner.control_dt)
    motion_assist_cfg = motion_assist_defaults
    if args.imu_stabilization is not None:
        motion_assist_cfg.setdefault("imu_posture", {})["enabled"] = bool(args.imu_stabilization)
    if args.gait_assist is not None:
        motion_assist_cfg.setdefault("gait_assist", {})["enabled"] = bool(args.gait_assist)

    safety = SafetyMonitor(
        runner.policy_order,
        control_dt=runner.control_dt,
    )
    motor_ids = load_motor_ids()
    joint_can_bus = resolve_joint_can_bus(runner.policy_order, args.can_count)
    active_joints = args.active_joints if args.active_joints is not None else load_active_joints()
    motor_layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=active_joints,
        joint_can_bus=joint_can_bus,
    )
    motor_layer.set_policy_pd_torque_limit(args.policy_pd_torque_limit)
    active_port_by_bus = ports_for_active_joints(
        port_by_bus,
        joint_can_bus,
        motor_layer.active_joints,
    )
    if not active_port_by_bus:
        active_port_by_bus = port_by_bus
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
    print("Initial speed scale:", f"{command_speed_scale(command_source):.2f}")
    print("Policy command gain:", f"{args.policy_command_gain:.2f}")
    print(
        "Policy command caps:",
        f"vx={args.policy_command_vx_max:.3f}",
        f"vy={args.policy_command_vy_max:.3f}",
        f"yaw={args.policy_command_yaw_max:.3f}",
    )
    print("Policy action clip:", f"{args.policy_action_clip:.3f}")
    print("Policy action smoothing:", f"{args.policy_action_smoothing:.2f}")
    print("Policy action delta limit:", f"{args.policy_action_delta_limit:.3f}")
    print("Policy entry ramp:", f"{args.policy_entry_ramp_seconds:.2f} s")
    print("Policy PD torque limit:", f"{args.policy_pd_torque_limit:.2f} Nm")
    print("Walk command grace:", f"{args.walk_command_grace_seconds:.2f} s")
    print("Start control mode:", args.start_control_mode)
    print("Startup action:", args.startup_action)
    print("Policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Policy hash verified:", runner.policy_hash_matches)
    print("Policy format:", runner.policy_format)
    print("Policy obs/actions:", runner.observation_dim, runner.action_dim)
    print(
        "Control rate:",
        f"runtime={runtime_control_hz:.2f} Hz",
        f"dt={runner.control_dt:.4f}s",
        f"trained={trained_control_hz:.2f} Hz",
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
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name:16s} -> motor_id=0x{int(motor_ids[name]):02X}")

    buses = None
    imu_sensor = None
    telemetry = None
    gui_proc = None
    csv_logger = None
    startup_zero_calibrated = False

    try:
        csv_logger = CsvRunLogger(
            enabled=args.log_csv,
            log_dir=args.log_dir,
            log_file=args.log_file,
            policy_order=runner.policy_order,
        )
        if csv_logger.enabled:
            print("\nCSV log:", csv_logger.path)
            print(
                "CSV automatic Git push:",
                "enabled" if args.auto_push_log else "disabled",
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
        )
        if imu_sensor is None:
            print("\nIMU source disabled. Policy IMU fields will stay at fallback values.")
        else:
            print(
                "\nIMU opened:",
                getattr(imu_sensor, "source_name", "imu"),
                "port:",
                getattr(imu_sensor, "port", "none"),
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
            try:
                buses = open_can_buses(
                    active_port_by_bus,
                    baud=args.baud,
                    backend=args.can_backend,
                    bitrate=args.can_bitrate,
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
                    "default": runner.q_default,
                    "stand": runner.q_stand,
                    "crouch": runner.q_crouch,
                },
                pose_snap_tolerance=0.35,
            )
            print("Polling initial motor encoder feedback...")
            motor_layer.send_raw_commands(buses, motor_layer.build_feedback_poll_commands())
            initial_feedback = estimator.refresh_from_bus(timeout=0.10)
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

        if args.mode == "mit-signal":
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
            )
            if not startup_ok:
                print("Controller aborted because startup-to-stand did not complete safely.")
                return 1
        else:
            q_previous_target = initialize_hold_target(
                estimator=estimator,
                feedback_timeout=args.feedback_timeout,
            )

        run_policy_loop(
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
            show_hex=args.show_hex,
            start_control_mode=args.start_control_mode,
            feedback_timeout=args.feedback_timeout,
            walk_command_threshold=args.walk_command_threshold,
            walk_command_grace_seconds=args.walk_command_grace_seconds,
            joystick_debug=args.joystick_debug,
            joint_debug=args.joint_debug,
            base_lin_vel_source=args.base_lin_vel_source,
            motion_assist_cfg=motion_assist_cfg,
            initial_zero_frame=args.initial_zero_frame,
            initial_zero_calibrated=startup_zero_calibrated,
            auto_stand_zero=args.auto_stand_zero,
            auto_sit_zero=args.auto_sit_zero,
            stand_zero_error_rad=args.stand_zero_error_rad,
            stand_zero_settle_steps=max(1, args.stand_zero_settle_steps),
            pose_sync_error_rad=max(0.0, args.pose_sync_error_rad),
            policy_command_gain=args.policy_command_gain,
            policy_command_vx_max=args.policy_command_vx_max,
            policy_command_vy_max=args.policy_command_vy_max,
            policy_command_yaw_max=args.policy_command_yaw_max,
            policy_action_clip=args.policy_action_clip,
            policy_action_smoothing=args.policy_action_smoothing,
            policy_action_delta_limit=args.policy_action_delta_limit,
            policy_entry_ramp_seconds=args.policy_entry_ramp_seconds,
            policy_sim_match=bool(args.policy_sim_match),
            stand_policy_stabilization=bool(args.stand_policy_stabilization),
            hold_capture_seconds=max(0.02, args.hold_capture_seconds),
            hold_command_repeats=max(1, args.hold_command_repeats),
            crouch_calibration_value=float(args.crouch_calibration_value),
            stand_calibration_value=float(args.stand_calibration_value),
            pose_transition_speed_rad_s=float(args.pose_transition_speed_rad_s),
            pose_transition_min_seconds=float(args.pose_transition_min_seconds),
            fresh_feedback_max_age_s=float(args.fresh_feedback_max_age),
            telemetry=telemetry,
            csv_logger=csv_logger,
        )

    except KeyboardInterrupt:
        print("\nKeyboard interrupt: stopping controller.")

    finally:
        command_source.close()
        if imu_sensor is not None:
            imu_sensor.close()
        if buses is not None:
            if args.mode == "mit-signal":
                try:
                    print("\nSending motor stop frames...")
                    motor_layer.send_raw_commands(buses, motor_layer.build_stop_commands())
                    print("Motor stop frames sent.")
                except Exception as exc:
                    print("\nWARNING: failed to send motor stop frames:", exc)
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

    print("\nController finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
