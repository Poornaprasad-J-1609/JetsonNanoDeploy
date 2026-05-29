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
from robstride_can_interface import ATUsbCan
from motor_command_layer import MotorCommandLayer, print_mit_commands


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


def smoothstep(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


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


def select_policy_base_lin_vel(source, estimator_base_lin_vel_b, command):
    source = str(source).replace("-", "_").lower()
    if source == "zero":
        return np.zeros(3, dtype=np.float32)
    if source == "command":
        command = np.asarray(command, dtype=np.float32)
        return np.array([command[0], command[1], 0.0], dtype=np.float32)
    if source == "imu":
        return np.asarray(estimator_base_lin_vel_b, dtype=np.float32)
    raise ValueError(f"Unknown base linear velocity source: {source}")


def projected_gravity_to_roll_pitch(projected_gravity_b):
    g = np.asarray(projected_gravity_b, dtype=np.float32)
    down_z = max(1e-6, -float(g[2]))
    roll = float(np.arctan2(float(g[1]), down_z))
    pitch = float(np.arctan2(-float(g[0]), down_z))
    return roll, pitch


def apply_imu_posture_stabilization(q_target, projected_gravity_b, policy_order, cfg):
    imu_cfg = cfg.get("imu_posture", {})
    if not bool(imu_cfg.get("enabled", False)):
        return q_target

    q_target = np.asarray(q_target, dtype=np.float32).copy()
    roll, pitch = projected_gravity_to_roll_pitch(projected_gravity_b)
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
    for index, joint_name in enumerate(policy_order):
        joint_gain = gains.get(joint_name, {})
        q_target[index] += (
            float(joint_gain.get("roll", 0.0)) * roll_corr
            + float(joint_gain.get("pitch", 0.0)) * pitch_corr
        )

    return q_target


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

    tau = np.asarray([float(cmd["tau_ff"]) for cmd in commands], dtype=np.float32)
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


def compact_telemetry_record(step, mode, command, command_source, commands, estimator, action=None, phase="policy"):
    command = np.asarray(command, dtype=np.float32)
    tau_cmd_mean, tau_cmd_max = command_torque_stats(commands)
    bus_counts = command_bus_counts(commands)
    tau_fb = feedback_torque_stats(estimator)
    action_abs_max = 0.0 if action is None else float(np.max(np.abs(action)))

    record = {
        "phase": str(phase),
        "step": int(step),
        "mode": str(mode),
        "vx": float(command[0]),
        "vy": float(command[1]),
        "vxy": float(np.linalg.norm(command[:2])),
        "yaw": float(command[2]),
        "speed": float(command_speed_scale(command_source)),
        "imu": estimator_imu_status(estimator),
        "act_max": action_abs_max,
        "tau_cmd": tau_cmd_mean,
        "tau_cmd_max": tau_cmd_max,
        "cmds": int(len(commands)),
        "bus_counts": format_bus_counts(bus_counts),
        "tau_fb": None,
        "tau_fb_max": None,
        "fault_reason": "",
    }
    if tau_fb is not None:
        record["tau_fb"] = float(tau_fb[0])
        record["tau_fb_max"] = float(tau_fb[1])
    return record


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
        f"act_max={float(record['act_max']): .3f} "
        f"tau_cmd={float(record['tau_cmd']): .3f} "
        f"tau_cmd_max={float(record['tau_cmd_max']): .3f} "
        f"cmds={int(record['cmds']):02d} "
        f"bus={record.get('bus_counts', 'none')}"
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
    FIELDNAMES = [
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
        "speed",
        "imu",
        "act_max",
        "tau_cmd",
        "tau_cmd_max",
        "cmds",
        "bus_counts",
        "tau_fb",
        "tau_fb_max",
        "fault_reason",
        "compact_line",
    ]

    def __init__(self, enabled=True, log_dir=None, log_file=None):
        self.enabled = bool(enabled)
        self.path = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.start_time = time.monotonic()
        self._file = None
        self._writer = None

        if not self.enabled:
            return

        if log_file:
            self.path = Path(log_file).expanduser()
        else:
            directory = Path(log_dir).expanduser() if log_dir else ROOT / "logs"
            self.path = directory / f"grallator_run_{self.run_id}_{os.getpid()}.csv"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log(self, record):
        if not self.enabled or self._writer is None:
            return

        row = {field: "" for field in self.FIELDNAMES}
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


def encoder_feedback_required(mode, estimator):
    return mode == "mit-signal" and hasattr(estimator, "last_feedback_by_joint")


def encoder_safety_stop_reason(safety, estimator, active_joints, mode, require_feedback=None):
    q_current = getattr(estimator, "q_current", None)
    if q_current is None:
        return "ABNORMAL ENCODER ANGLE: estimator has no joint position vector"

    if require_feedback is None:
        require_feedback = encoder_feedback_required(mode, estimator)

    stop, reason = safety.encoder_sanity_check(
        q_current=q_current,
        active_joints=active_joints,
        feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
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
    print("STARTUP PHASE: HOLD current pose")
    print("#" * 80)
    print("No sit, stand, or walking command is sent until joystick input.")
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


def apply_software_zero_calibration(
    estimator,
    motor_layer,
    active_joints,
    feedback_timeout,
    buses,
    mode,
    label,
):
    # Do not send stop/poll frames here. During live control those frames can
    # drop torque and cause exactly the sit/stand jerk we are avoiding.
    refresh_estimator_feedback(estimator, timeout=feedback_timeout)

    if hasattr(estimator, "apply_software_zero"):
        try:
            updated, missing = estimator.apply_software_zero(active_joints=active_joints)
        except Exception as exc:
            print(f"\n[ZERO CAL] {label}: failed: {exc}")
            return False
    else:
        updated = {joint_name: 0.0 for joint_name in active_joints}
        missing = []
        if hasattr(estimator, "q_current"):
            estimator.q_current[:] = 0.0
        if hasattr(estimator, "qd_current"):
            estimator.qd_current[:] = 0.0

    if missing:
        print(
            f"\n[ZERO CAL] {label}: missing feedback for "
            + ", ".join(missing[:6])
            + (" ..." if len(missing) > 6 else "")
        )
        return False

    q_current, _, _, _, _ = estimator.read()
    print(
        f"\n[ZERO CAL] {label}: software zero applied to "
        f"{len(updated)} active joint(s). No RobStride hardware set-zero frame was sent."
    )
    return q_current.copy()


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
        return q_previous_target

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
    return q_previous_target


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
    joystick_debug,
    joint_debug,
    base_lin_vel_source,
    motion_assist_cfg,
    initial_zero_frame,
    initial_zero_calibrated,
    auto_stand_zero,
    stand_zero_error_rad,
    stand_zero_settle_steps,
    pose_sync_error_rad,
    telemetry=None,
    csv_logger=None,
):
    dt = runner.control_dt
    action_dim = len(runner.policy_order)
    previous_action = np.zeros(action_dim, dtype=np.float32)

    control_mode = start_control_mode  # options: hold, policy, stand, sit
    has_motion_target = control_mode in ("stand", "sit")
    zero_frame = str(initial_zero_frame).lower()
    zero_calibrated = zero_frame == "stand" or bool(initial_zero_calibrated)
    stand_zero_pending = bool(
        auto_stand_zero and control_mode == "stand" and zero_frame == "crouch"
    )
    stand_zero_settle_count = 0
    calibration_hold_until_step = -1
    active_indices = active_joint_indices(runner.policy_order, motor_layer.active_joints)

    print("\n" + "#" * 80)
    print("POLICY / POSE PHASE")
    print("#" * 80)
    print("mode:", mode)
    print("Joystick buttons:")
    print("  button 4    -> STOP walking and SIT/CROUCH pose")
    print("  button 5    -> STAND pose")
    print("  button 6    -> reduce speed scale")
    print("  button 7    -> increase speed scale")
    print("  buttons 0-3 -> EMERGENCY STOP")
    print("  D-pad down / configured zero axis -> software-zero current crouch/default pose")
    print("Joystick axes:")
    print("  left stick Y  -> forward/back vx")
    print("  left stick X  -> left/right vy")
    print("  right stick X -> turn/yaw")
    print("start_control_mode:", control_mode)
    print("walk_command_threshold:", walk_command_threshold)
    print("base_lin_vel_source:", base_lin_vel_source)
    print("zero_frame:", zero_frame)
    print("zero_calibrated:", bool(zero_calibrated))
    print("auto_stand_zero:", bool(auto_stand_zero))
    print("pose_sync_error_rad:", float(pose_sync_error_rad))
    if stand_zero_pending:
        print("[ZERO CAL] initial stand target will auto-zero when settled.")
    print("imu_stabilization:", bool(motion_assist_cfg.get("imu_posture", {}).get("enabled", False)))
    print("gait_assist:", bool(motion_assist_cfg.get("gait_assist", {}).get("enabled", False)))
    if steps is None:
        print("policy_steps: unlimited, running until emergency stop or Ctrl+C")
    else:
        print("policy_steps:", steps)

    step = 0
    while steps is None or step < steps:
        cycle_start = time.monotonic()
        (
            q_current,
            qd_current,
            base_lin_vel_b,
            base_ang_vel_b,
            projected_gravity_b,
        ) = estimator.read()

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
                phase="policy",
            )
            break

        calibration_request = command_source.get_calibration_request()
        if calibration_request == "zero_current_pose":
            if zero_frame == "stand":
                print("[ZERO CAL] ignored because stand/RL zero is already active.")
            elif control_mode not in ("hold", "sit"):
                print("[ZERO CAL] ignored; zero calibration is only allowed from hold/sit.")
            else:
                q_zeroed = apply_software_zero_calibration(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    active_joints=motor_layer.active_joints,
                    feedback_timeout=feedback_timeout,
                    buses=buses,
                    mode=mode,
                    label="crouch/default pose",
                )
                if q_zeroed is not False:
                    q_previous_target = q_zeroed.copy()
                    q_current = q_zeroed.copy()
                    qd_current = np.zeros_like(q_current, dtype=np.float32)
                    zero_frame = "crouch"
                    zero_calibrated = True
                    control_mode = "hold"
                    has_motion_target = True
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                    calibration_hold_until_step = step + 10
                    print("[ZERO CAL] zero_frame -> crouch. Press STAND to move to stand and auto-zero for policy.")

        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=estimator,
            active_joints=motor_layer.active_joints,
            mode=mode,
            require_feedback=bool(
                (has_motion_target or control_mode != "hold")
                and encoder_feedback_required(mode, estimator)
            ),
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
        if mode_request is not None:
            if mode_request in ("stand", "sit") and zero_frame == "crouch" and not zero_calibrated:
                print("\n[ZERO CAL] first pose command is auto-zeroing current crouch/default pose.")
                q_zeroed = apply_software_zero_calibration(
                    estimator=estimator,
                    motor_layer=motor_layer,
                    active_joints=motor_layer.active_joints,
                    feedback_timeout=feedback_timeout,
                    buses=buses,
                    mode=mode,
                    label="crouch/default pose",
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
                    zero_calibrated = True
                    calibration_hold_until_step = step + 5
            if mode_request is None:
                pass
            elif mode_request in ("stand", "sit"):
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
                reason = encoder_safety_stop_reason(
                    safety=safety,
                    estimator=estimator,
                    active_joints=motor_layer.active_joints,
                    mode=mode,
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

            if mode_request is not None:
                control_mode = mode_request
                if control_mode in ("stand", "sit"):
                    has_motion_target = True
                if control_mode == "stand" and zero_frame == "crouch":
                    stand_zero_pending = bool(auto_stand_zero)
                    stand_zero_settle_count = 0
                    print("[ZERO CAL] stand target uses stand_pose_when_sit_zero; stand will auto-zero when settled.")
                elif control_mode == "sit":
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                print(f"\n[MODE CHANGE] control_mode -> {control_mode}")

        command = command_source.read()
        if step < calibration_hold_until_step:
            command = np.zeros(3, dtype=np.float32)
        policy_base_lin_vel_b = select_policy_base_lin_vel(
            base_lin_vel_source,
            base_lin_vel_b,
            command,
        )
        walk_requested = joystick_walk_requested(command, walk_command_threshold)
        if zero_frame != "stand" and walk_requested:
            walk_requested = False
            if step % max(1, log_every) == 0:
                print("[ZERO CAL] walking blocked until STAND auto-zero completes.")
        if control_mode == "sit":
            active_control_mode = "sit"
        elif walk_requested:
            active_control_mode = "policy"
        elif control_mode == "policy":
            active_control_mode = "hold"
        else:
            active_control_mode = control_mode

        if walk_requested:
            has_motion_target = True

        if active_control_mode == "hold":
            q_safe_target = q_previous_target.copy()
            commands = (
                motor_layer.build_mit_commands(
                    q_safe_target,
                    phase="startup",
                    feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
                )
                if has_motion_target
                else []
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "stand":
            if zero_frame == "crouch":
                q_policy_target = runner.q_stand_when_sit_zero.copy()
            else:
                q_policy_target = runner.q_stand.copy()
            q_policy_target = apply_motion_assists(
                q_target=q_policy_target,
                command=command,
                elapsed_time=step * dt,
                projected_gravity_b=projected_gravity_b,
                runner=runner,
                cfg=motion_assist_cfg,
                use_gait=False,
            )
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(
                q_safe_target,
                phase="startup",
                feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "sit":
            if zero_frame == "crouch":
                q_policy_target = np.zeros(len(runner.policy_order), dtype=np.float32)
            else:
                q_policy_target = runner.q_sit_when_stand_zero.copy()
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(
                q_safe_target,
                phase="startup",
                feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
            )
            action = np.zeros(action_dim, dtype=np.float32)

        elif active_control_mode == "policy":
            obs = runner.build_observation(
                base_lin_vel_b=policy_base_lin_vel_b,
                base_ang_vel_b=base_ang_vel_b,
                projected_gravity_b=projected_gravity_b,
                command=command,
                q_current=q_current,
                qd_current=qd_current,
                previous_action=previous_action,
            )

            action = runner.infer_action(obs)
            q_policy_target = runner.action_to_q_target(action)
            q_policy_target = apply_motion_assists(
                q_target=q_policy_target,
                command=command,
                elapsed_time=step * dt,
                projected_gravity_b=projected_gravity_b,
                runner=runner,
                cfg=motion_assist_cfg,
                use_gait=True,
            )
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(
                q_safe_target,
                phase="policy",
                feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
            )

        else:
            raise RuntimeError(f"Unknown control_mode: {active_control_mode}")

        if mode == "signal":
            motor_layer.send_harmless_frames(buses, commands)
        elif mode == "mit-signal":
            motor_layer.send_signal_commands(buses, commands)

        active_feedback_timeout = min(
            float(feedback_timeout),
            max(0.002, 0.35 * float(dt)),
        )
        refresh_estimator_feedback(estimator, timeout=active_feedback_timeout)
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

        advance_target = True
        if (
            active_control_mode in ("stand", "sit")
            and encoder_feedback_required(mode, estimator)
            and float(pose_sync_error_rad) > 0.0
        ):
            q_feedback = getattr(estimator, "q_current", None)
            if q_feedback is not None:
                sync_error = max_active_error(q_feedback, q_safe_target, active_indices)
                if sync_error > float(pose_sync_error_rad):
                    advance_target = False
                    if step % max(1, log_every) == 0:
                        print(
                            f"[SYNC] holding pose target: max joint lag "
                            f"{sync_error:.3f} rad > {float(pose_sync_error_rad):.3f} rad"
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
                )
                if q_zeroed is not False:
                    zero_frame = "stand"
                    stand_zero_pending = False
                    stand_zero_settle_count = 0
                    q_safe_target = runner.q_stand.copy()
                    q_previous_target = q_safe_target.copy()
                    zero_calibrated = True
                    previous_action = np.zeros(action_dim, dtype=np.float32)
                    commands = motor_layer.build_mit_commands(
                        q_safe_target,
                        phase="startup",
                        feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
                    )
                    if mode == "signal":
                        motor_layer.send_harmless_frames(buses, commands)
                    elif mode == "mit-signal":
                        motor_layer.send_signal_commands(buses, commands)
                    print("[ZERO CAL] zero_frame -> stand. Policy walking is now enabled in RL zero coordinates.")

        if step % log_every == 0:
            telemetry_record = compact_telemetry_record(
                step=step,
                mode=active_control_mode,
                command=command,
                command_source=command_source,
                commands=commands,
                estimator=estimator,
                action=action,
                phase="policy",
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

        estimator.dry_update_as_if_robot_followed(q_safe_target, dt)

        if active_control_mode == "policy":
            previous_action = action.copy()
        else:
            previous_action = np.zeros(action_dim, dtype=np.float32)

        if advance_target:
            q_previous_target = q_safe_target.copy()

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

    print("\nPolicy / pose phase completed.")


def main():
    joystick_defaults = load_joystick_defaults()
    speed_defaults = load_speed_scale_defaults()
    imu_defaults = load_imu_config()
    motion_assist_defaults = load_motion_assist_config()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["print", "signal", "mit-signal", "motors"],
        default="print",
        help="print=no serial, signal=harmless empty CAN frames, mit-signal=sends MIT packets, motors=blocked",
    )

    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="fallback USB-CAN port used when --port-front / --port-back are not given")
    parser.add_argument("--port-front", default=None,
                        help="USB-CAN port for front legs (FL/FR, 6 motors); overrides --port")
    parser.add_argument("--port-back", default=None,
                        help="USB-CAN port for back legs (BL/BR, 6 motors); overrides --port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--active-joints",
        nargs="*",
        default=None,
        help="only send motor commands to these joint names; default uses config/motor_ids.yaml",
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--policy-activation",
        choices=["elu", "relu", "tanh", "identity", "none"],
        default="elu",
        help="activation to use when loading non-TorchScript actor checkpoints",
    )

    parser.add_argument("--command-source", choices=["fixed", "joystick"], default="joystick")

    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=float(joystick_defaults["speed_limits"]["max_vx"]))
    parser.add_argument("--max-vy", type=float, default=float(joystick_defaults["speed_limits"]["max_vy"]))
    parser.add_argument("--max-yaw", type=float, default=float(joystick_defaults["speed_limits"]["max_yaw"]))

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
    parser.add_argument("--start-control-mode", choices=["hold", "stand", "sit", "policy"], default="hold")
    parser.add_argument("--startup-action", choices=["hold", "stand"], default="hold")
    parser.add_argument(
        "--initial-zero-frame",
        choices=["stand", "crouch"],
        default="crouch",
        help="coordinate frame before joystick zero calibration; policy walking requires stand",
    )
    parser.add_argument(
        "--auto-stand-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after standing from crouch-zero frame, make current stand pose software zero",
    )
    parser.add_argument(
        "--auto-zero-on-startup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when starting in crouch frame, software-zero current encoder pose automatically before joystick commands",
    )
    parser.add_argument("--stand-zero-error-rad", type=float, default=0.08)
    parser.add_argument("--stand-zero-settle-steps", type=int, default=15)
    parser.add_argument(
        "--pose-sync-error-rad",
        type=float,
        default=0.12,
        help="during sit/stand, hold the next target step until live feedback is within this max active-joint error; 0 disables",
    )
    parser.add_argument(
        "--walk-command-threshold",
        type=float,
        default=0.02,
        help="minimum absolute vx/vy/yaw command needed to run walking policy",
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
        "--imu-source",
        choices=["auto", "fake", "none", "xsens", "serial-json", "serial-csv"],
        default="auto",
        help="auto uses config/imu.yaml; serial sources feed gyro/gravity into policy observation",
    )
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument(
        "--base-lin-vel-source",
        choices=["command", "imu", "zero"],
        default=str(imu_defaults.get("base_linear_velocity_source", "zero")),
        help="policy base linear velocity input: command=[vx,vy,0], imu=sensor/estimator value, zero=[0,0,0]",
    )
    parser.add_argument(
        "--imu-stabilization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable small projected-gravity posture corrections",
    )
    parser.add_argument(
        "--gait-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable walking-style sinusoidal leg overlay for suspended tests",
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

    args = parser.parse_args()
    args.log_every = max(1, args.log_every)
    args.policy_steps = None if args.policy_steps <= 0 else args.policy_steps
    args.walk_command_threshold = max(0.0, args.walk_command_threshold)

    port_front = args.port_front if args.port_front is not None else args.port
    port_back  = args.port_back  if args.port_back  is not None else args.port

    if args.mode == "motors":
        print("ERROR: --mode motors is intentionally blocked for now.")
        print("Use --mode mit-signal for the real RobStride MIT CAN path.")
        return 1

    active_imu_source = imu_source_name(args.imu_source, imu_defaults)
    active_imu_port = args.imu_port if args.imu_port is not None else imu_defaults.get("port")
    if (
        args.mode in ("signal", "mit-signal")
        and active_imu_source in ("xsens", "xsens_binary", "mtdata2", "serial_json", "serial_csv")
        and active_imu_port in {port_front, port_back}
    ):
        print("ERROR: IMU port", active_imu_port, "conflicts with a CAN bus port.")
        print("CAN ports in use:", sorted({port_front, port_back}))
        print("Use a separate device, e.g. --imu-port /dev/ttyUSB2")
        return 1

    runner = PolicyRunner(
        policy_path=args.policy_path,
        policy_activation=args.policy_activation,
    )
    motion_assist_cfg = motion_assist_defaults
    if args.imu_stabilization is not None:
        motion_assist_cfg.setdefault("imu_posture", {})["enabled"] = bool(args.imu_stabilization)
    if args.gait_assist is not None:
        motion_assist_cfg.setdefault("gait_assist", {})["enabled"] = bool(args.gait_assist)

    safety = SafetyMonitor(runner.policy_order)
    motor_ids = load_motor_ids()
    joint_can_bus = load_joint_can_bus()
    active_joints = args.active_joints if args.active_joints is not None else load_active_joints()
    motor_layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=active_joints,
        joint_can_bus=joint_can_bus,
    )

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
            zero_calibration_axis=args.zero_calibration_axis,
            zero_calibration_axis_direction=args.zero_calibration_axis_direction,
            zero_calibration_axis_threshold=args.zero_calibration_axis_threshold,
            emergency_stop_buttons=args.button_emergency_stop,
            speed_scale_initial=args.speed_scale_initial,
            speed_scale_min=args.speed_scale_min,
            speed_scale_max=args.speed_scale_max,
            speed_scale_step=args.speed_scale_step,
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
    print("Start control mode:", args.start_control_mode)
    print("Startup action:", args.startup_action)
    print("Policy:", runner.policy_path)
    print("Policy format:", runner.policy_format)
    print("Policy obs/actions:", runner.observation_dim, runner.action_dim)
    print("Control dt:", runner.control_dt)
    print("Port front (FL/FR):", port_front)
    print("Port back  (BL/BR):", port_back)
    print("Baud:", args.baud)
    print("Feedback source:", feedback_source)
    print("IMU source:", active_imu_source)
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
        )
        if csv_logger.enabled:
            print("\nCSV log:", csv_logger.path)

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

            print("\nOpening USB-CAN serial ports...")
            bus_front = ATUsbCan(port=port_front, baud=args.baud).open()
            print(f"USB-CAN front ({port_front}) opened.")
            try:
                if port_back != port_front:
                    bus_back = ATUsbCan(port=port_back, baud=args.baud).open()
                    print(f"USB-CAN back ({port_back}) opened.")
                else:
                    bus_back = bus_front
                    print(f"USB-CAN back shares front port ({port_front}).")
            except Exception:
                bus_front.close()
                raise
            buses = {"front": bus_front, "back": bus_back}

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
            estimator = FakeStateEstimator(q_initial=q_fake_start, imu_sensor=imu_sensor)

        if args.auto_zero_on_startup and str(args.initial_zero_frame).lower() == "crouch":
            print("\n[ZERO CAL] startup auto-zero enabled for current crouch/default pose.")
            q_auto_zero = apply_software_zero_calibration(
                estimator=estimator,
                motor_layer=motor_layer,
                active_joints=motor_layer.active_joints,
                feedback_timeout=args.feedback_timeout,
                buses=buses,
                mode=args.mode,
                label="startup crouch/default pose",
            )
            if q_auto_zero is not False:
                startup_zero_calibrated = True
                print("[ZERO CAL] startup crouch/default pose is now q=0. D-pad zero is optional.")
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
                )
                if reason is not None:
                    print("\nWARNING:", reason)
                    print(
                        "Startup is continuing so auto-zero or joystick D-pad "
                        "software-zero can calibrate before commanding sit, stand, or walk."
                    )
            print("Sending motor enable frames...")
            motor_layer.send_raw_commands(buses, motor_layer.build_enable_commands())
            print("Motor enable frames sent.")

        if args.startup_action == "stand":
            q_previous_target = run_startup_to_stand(
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
            joystick_debug=args.joystick_debug,
            joint_debug=args.joint_debug,
            base_lin_vel_source=args.base_lin_vel_source,
            motion_assist_cfg=motion_assist_cfg,
            initial_zero_frame=args.initial_zero_frame,
            initial_zero_calibrated=startup_zero_calibrated,
            auto_stand_zero=args.auto_stand_zero,
            stand_zero_error_rad=args.stand_zero_error_rad,
            stand_zero_settle_steps=max(1, args.stand_zero_settle_steps),
            pose_sync_error_rad=max(0.0, args.pose_sync_error_rad),
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
            closed_ids = set()
            for b in buses.values():
                if id(b) not in closed_ids:
                    b.close()
                    closed_ids.add(id(b))
            print("\nUSB-CAN closed.")
        if telemetry is not None:
            telemetry.close()
        if csv_logger is not None:
            csv_path = csv_logger.path
            csv_logger.close()
            if csv_path is not None:
                print("CSV log saved:", csv_path)
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
