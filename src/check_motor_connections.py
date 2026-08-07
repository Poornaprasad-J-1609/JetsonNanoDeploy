#!/usr/bin/env python3
import argparse
import json
import math
import os
import select
import socket
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from motor_command_layer import (
    MotorCommandLayer as DirectMotorCommandLayer,
    decode_mit_feedback_frame,
    motor_position_to_joint_angle,
)
from can_topology import (
    add_can_topology_args,
    close_can_buses,
    open_can_buses,
    ports_for_active_joints,
    resolve_joint_can_bus,
    resolve_port_by_bus,
    topology_lines,
    validate_unique_motor_ids_per_physical_bus,
)


ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_PORT_DEFAULT = 57543
POSE_SNAP_TOLERANCE_RAD = 0.0
DISPLAY_LEG_ORDER = ("FR", "FL", "BR", "BL")


class KeyboardReader:
    def __init__(self, enabled=True):
        self.enabled = bool(enabled) and sys.stdin.isatty()
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self):
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp_path.replace(path)


def load_config():
    joint_cfg = load_yaml(ROOT / "config" / "joint_map.yaml")
    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return joint_cfg, motor_cfg


def load_pose_references(policy_order):
    pose_cfg = load_yaml(ROOT / "config" / "default_pose.yaml")
    pose_names = ("default_pose", "stand_pose", "crouch_pose")
    refs_by_joint = {joint_name: [] for joint_name in policy_order}
    poses = {}
    for pose_name in pose_names:
        pose = pose_cfg.get(pose_name, {})
        values = {}
        for joint_name in policy_order:
            if joint_name in pose:
                value = float(pose[joint_name])
                refs_by_joint[joint_name].append(value)
                values[joint_name] = value
        poses[pose_name] = values
    return refs_by_joint, poses


def resolve_joints(policy_order, motor_cfg, active_joints_arg):
    if active_joints_arg is not None:
        joints = list(active_joints_arg)
    else:
        joints = list(motor_cfg.get("active_joints", []) or policy_order)

    unknown = [name for name in joints if name not in policy_order]
    if unknown:
        raise KeyError(f"Unknown joint name(s): {unknown}")

    missing_ids = [name for name in joints if name not in motor_cfg["motor_ids"]]
    if missing_ids:
        raise KeyError(f"Missing motor ID(s) in motor_ids.yaml: {missing_ids}")

    return joints


def connection_display_order(joints, motor_ids):
    """Order diagnostics by physical leg, then by ascending CAN motor ID."""
    leg_rank = {leg: rank for rank, leg in enumerate(DISPLAY_LEG_ORDER)}

    def sort_key(joint_name):
        leg = str(joint_name).split("_", 1)[0]
        return (
            leg_rank.get(leg, len(leg_rank)),
            int(motor_ids[joint_name]),
            str(joint_name),
        )

    return sorted(joints, key=sort_key)


def feedback_status(feedback, now, stale_seconds):
    if feedback is None:
        return "NOT_CONNECTED"
    age = now - feedback["timestamp"]
    if age > stale_seconds:
        return "STALE"
    if int(feedback.get("fault_bits", 0)) != 0:
        return "FAULT"
    return "CONNECTED"


def feedback_joint_position(joint_name, layer, feedback, reference=None, pose_references=None):
    if feedback is None:
        return None
    if "joint_position" in feedback:
        return float(feedback["joint_position"])
    offset = float(layer.joint_offsets.get(joint_name, 0.0))
    direction = float(layer.joint_directions.get(joint_name, 1.0))
    return motor_position_to_joint_angle(
        feedback["position"],
        offset=offset,
        direction=direction,
    )


def resolve_feedback_joint_positions(
    joints,
    motor_ids,
    joint_can_bus,
    layer,
    pose_references,
    poses,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
    joint_positions,
    stale_seconds,
):
    now = time.monotonic()
    for joint_name in joints:
        feedback = feedback_for_joint(
            joint_name,
            motor_ids,
            joint_can_bus,
            feedback_by_bus_motor_id,
            feedback_by_motor_id,
        )
        if feedback is None:
            continue
        if now - feedback["timestamp"] > stale_seconds:
            continue
        if int(feedback.get("fault_bits", 0)) != 0:
            continue

        velocity_raw = float(feedback["velocity"])
        torque_raw = float(feedback["torque"])
        raw_position = float(feedback["position"])
        offset = float(layer.joint_offsets.get(joint_name, 0.0))
        direction = float(layer.joint_directions.get(joint_name, 1.0))

        q_joint = motor_position_to_joint_angle(
            raw_position,
            offset=offset,
            direction=direction,
        )
        qd_joint = direction * velocity_raw
        tau_joint = direction * torque_raw
        motor_position = q_joint
        motor_velocity = qd_joint
        motor_torque = tau_joint
        transmission_jacobian = 1.0
        transmission_efficiency = 1.0
        transmission_enabled = False

        joint_positions[joint_name] = q_joint
        feedback["position_raw"] = raw_position
        feedback["velocity_raw"] = velocity_raw
        feedback["torque_raw"] = torque_raw
        feedback["motor_position"] = motor_position
        feedback["motor_velocity"] = motor_velocity
        feedback["motor_torque"] = motor_torque
        feedback["joint_position"] = q_joint
        feedback["joint_velocity"] = qd_joint
        feedback["joint_torque"] = tau_joint
        feedback["velocity"] = qd_joint
        feedback["torque"] = tau_joint
        feedback["joint_direction"] = direction
        feedback["transmission_jacobian"] = transmission_jacobian
        feedback["transmission_efficiency"] = transmission_efficiency
        feedback["transmission_enabled"] = transmission_enabled
    

def estimate_pose_from_angles(joints, angles, poses):
    if not angles:
        return "unknown", None, None

    best_name = "unknown"
    best_rms = None
    best_max = None
    for pose_name, pose in poses.items():
        errors = []
        for joint_name in joints:
            if joint_name in angles and joint_name in pose:
                errors.append(float(angles[joint_name]) - float(pose[joint_name]))
        if not errors:
            continue
        rms = float((sum(err * err for err in errors) / len(errors)) ** 0.5)
        max_err = float(max(abs(err) for err in errors))
        if best_rms is None or rms < best_rms:
            best_name = pose_name.replace("_pose", "")
            best_rms = rms
            best_max = max_err

    if best_rms is None:
        return "unknown", None, None
    if best_rms > 0.35 or best_max > 0.80:
        return "unknown", best_rms, best_max
    return best_name, best_rms, best_max


def update_feedback_from_frames(
    frames,
    layer,
    active_bus_motor_keys,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
):
    updated = 0
    for frame in frames:
        feedback = decode_mit_feedback_frame(
            frame.can_id,
            frame.data,
            layer.proto,
        )
        if feedback is None:
            continue

        motor_id = int(feedback["motor_id"])
        bus_name = getattr(frame, "bus_name", None)
        if bus_name is None:
            continue
        if (bus_name, motor_id) not in active_bus_motor_keys:
            continue
        feedback["timestamp"] = frame.timestamp
        feedback["bus_name"] = bus_name
        feedback_by_bus_motor_id[(bus_name, motor_id)] = feedback
        feedback_by_motor_id.pop(motor_id, None)
        updated += 1
    return updated


def print_table(
    joints,
    motor_ids,
    joint_can_bus,
    layer,
    pose_references,
    poses,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
    stale_seconds,
    set_zero_key,
    quit_key,
    keyboard_enabled,
    set_zero_enabled,
    status_message,
    crouch_key,
    crouch_enabled,
    clear_screen=True,
):
    now = time.monotonic()
    if clear_screen:
        print("\033[2J\033[H", end="")
    print("=" * 190)
    print("GRALLATOR ROBSTRIDE MOTOR CONNECTION CHECK")
    print("=" * 190)
    print("Poll loop sends RobStride comm-type 4 stop/poll frames. It does not enable MIT torque control.")
    if getattr(getattr(layer, "transmissions", None), "enabled", False):
        print("Four-bar transmission: ENABLED for encoder decoding; Joint rad is virtual Isaac/URDF knee angle.")
    else:
        print("Four-bar transmission: disabled; Joint rad is direct sign/offset-corrected motor angle.")
    if keyboard_enabled:
        actions = []
        if set_zero_enabled:
            actions.append(
                f"'{set_zero_key}' = set current motor position as zero"
            )
        if crouch_enabled:
            actions.append(
                f"'{crouch_key}' = save current sign-corrected joint angles as YAML crouch_pose"
            )
        actions.append(f"'{quit_key}' = quit")
        print("Keys: " + "; ".join(actions))
    else:
        print("Keyboard input is disabled because stdin is not an interactive terminal.")
    if status_message:
        print(status_message)
    print("-" * 190)
    print(
        f"{'Joint':20s} | {'Bus':>5s} | {'Motor':>7s} | {'State':>13s} | {'Age ms':>8s} | "
        f"{'Joint rad':>10s} | {'Raw rad':>10s} | {'Motor th':>10s} | {'J=dq/dth':>9s} | "
        f"{'Vel rad/s':>10s} | {'Torque':>9s} | {'Temp C':>7s} | {'4bar':>5s} | Fault"
    )
    print("-" * 190)

    connected = 0
    angles = {}
    for joint_name in joints:
        motor_id = int(motor_ids[joint_name])
        bus_name = joint_can_bus.get(joint_name, "front")
        feedback = feedback_for_joint(
            joint_name,
            motor_ids,
            joint_can_bus,
            feedback_by_bus_motor_id,
            feedback_by_motor_id,
        )
        state = feedback_status(feedback, now, stale_seconds)
        if state == "CONNECTED":
            connected += 1

        if feedback is None:
            print(
                f"{joint_name:20s} | {bus_name:>5s} | 0x{motor_id:02X}    | {state:>13s} | "
                f"{'-':>8s} | {'-':>10s} | {'-':>10s} | {'-':>10s} | {'-':>9s} | "
                f"{'-':>10s} | {'-':>9s} | {'-':>7s} | {'-':>5s} | -"
            )
            continue

        age_ms = 1000.0 * (now - feedback["timestamp"])
        q_joint = feedback_joint_position(
            joint_name,
            layer,
            feedback,
            pose_references=pose_references,
        )
        motor_position = float(feedback.get("motor_position", q_joint))
        transmission_jacobian = float(feedback.get("transmission_jacobian", 1.0))
        transmission_enabled = bool(feedback.get("transmission_enabled", False))
        angles[joint_name] = q_joint
        print(
            f"{joint_name:20s} | {bus_name:>5s} | 0x{motor_id:02X}    | {state:>13s} | "
            f"{age_ms:8.1f} | "
            f"{q_joint:+10.4f} | "
            f"{feedback['position']:+10.4f} | "
            f"{motor_position:+10.4f} | "
            f"{transmission_jacobian:+9.4f} | "
            f"{feedback['velocity']:+10.4f} | "
            f"{feedback['torque']:+9.4f} | "
            f"{feedback['temperature_c']:7.1f} | "
            f"{'yes' if transmission_enabled else 'no':>5s} | "
            f"0x{int(feedback['fault_bits']):02X}"
        )

    print("-" * 190)
    print(f"Connected: {connected}/{len(joints)}")
    pose_name, pose_rms, pose_max = estimate_pose_from_angles(joints, angles, poses)
    if pose_rms is None:
        print("Pose estimate: unknown")
    else:
        print(
            f"Pose estimate: {pose_name} "
            f"(rms_error={pose_rms:.4f} rad, max_error={pose_max:.4f} rad)"
        )
    print("=" * 190)


def feedback_for_joint(
    joint_name,
    motor_ids,
    joint_can_bus,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
):
    """Fetch feedback strictly by CAN bus and motor ID."""
    motor_id = int(motor_ids[joint_name])
    bus_name = joint_can_bus.get(joint_name, "front")
    return feedback_by_bus_motor_id.get((bus_name, motor_id))


def active_joint_angles_from_feedback(
    joints,
    motor_ids,
    joint_can_bus,
    layer,
    pose_references,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
    stale_seconds,
):
    now = time.monotonic()
    angles = {}
    missing = []
    stale = []
    faults = []

    for joint_name in joints:
        feedback = feedback_for_joint(
            joint_name,
            motor_ids,
            joint_can_bus,
            feedback_by_bus_motor_id,
            feedback_by_motor_id,
        )
        if feedback is None:
            missing.append(joint_name)
            continue
        if now - feedback["timestamp"] > stale_seconds:
            stale.append(joint_name)
            continue
        if int(feedback.get("fault_bits", 0)) != 0:
            faults.append(joint_name)
            continue

        angles[joint_name] = feedback_joint_position(
            joint_name,
            layer,
            feedback,
            pose_references=pose_references,
        )

    if missing or stale or faults:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:4]))
        if stale:
            details.append("stale=" + ", ".join(stale[:4]))
        if faults:
            details.append("fault=" + ", ".join(faults[:4]))
        return None, "Cannot capture pose; " + "; ".join(details)

    return angles, ""


def validate_angles_against_joint_limits(angles):
    limits = load_yaml(ROOT / "config" / "joint_limits.yaml")["joint_limits"]
    bad = []
    for joint_name, q in angles.items():
        if joint_name not in limits:
            bad.append(f"{joint_name}: missing limit")
            continue
        q_min = float(limits[joint_name]["min"])
        q_max = float(limits[joint_name]["max"])
        if q < q_min or q > q_max:
            bad.append(f"{joint_name}={q:+.4f} outside [{q_min:+.4f}, {q_max:+.4f}]")
    return bad


def update_default_pose_yaml(pose_updates_by_name):
    path = ROOT / "config" / "default_pose.yaml"
    cfg = load_yaml(path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(path.read_text())

    for pose_name, values in pose_updates_by_name.items():
        cfg.setdefault(pose_name, {})
        for joint_name, value in values.items():
            cfg[pose_name][joint_name] = round(float(value), 6)

    save_yaml(path, cfg)
    return path, backup_path


def capture_crouch_pose(
    joints,
    motor_ids,
    joint_can_bus,
    layer,
    pose_references,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
    stale_seconds,
):
    angles, reason = active_joint_angles_from_feedback(
        joints=joints,
        motor_ids=motor_ids,
        joint_can_bus=joint_can_bus,
        layer=layer,
        pose_references=pose_references,
        feedback_by_bus_motor_id=feedback_by_bus_motor_id,
        feedback_by_motor_id=feedback_by_motor_id,
        stale_seconds=stale_seconds,
    )
    if angles is None:
        return reason

    violations = validate_angles_against_joint_limits(angles)
    if violations:
        return "CROUCH capture refused; joint limit violation: " + "; ".join(violations[:4])

    path, backup_path = update_default_pose_yaml({"crouch_pose": angles})
    ordered_joints = connection_display_order(list(angles), motor_ids)
    saved_values = "\n".join(
        f"  {joint_name:20s} joint_rad={angles[joint_name]:+.6f}"
        for joint_name in ordered_joints
    )
    return (
        f"CROUCH pose saved from sign-corrected joint radians for "
        f"{len(angles)} joint(s):\n{saved_values}\n"
        f"YAML: {path}; backup={backup_path}"
    )


def set_stand_default_yaml_value(joints, value):
    values = {joint_name: float(value) for joint_name in joints}
    path, backup_path = update_default_pose_yaml({
        "default_pose": values,
        "stand_pose": values,
    })
    return (
        f"default_pose and stand_pose set to {float(value):+.3f} rad "
        f"for {len(values)} joint(s) "
        f"in {path}; backup={backup_path}"
    )


def build_gui_packet(
    step,
    joints,
    motor_ids,
    joint_can_bus,
    layer,
    pose_references,
    poses,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
    stale_seconds,
):
    now = time.monotonic()
    joint_rows = []
    has_fault = False

    for joint_name in joints:
        motor_id = int(motor_ids[joint_name])
        bus_name = joint_can_bus.get(joint_name, "front")
        feedback = feedback_for_joint(
            joint_name=joint_name,
            motor_ids=motor_ids,
            joint_can_bus=joint_can_bus,
            feedback_by_bus_motor_id=feedback_by_bus_motor_id,
            feedback_by_motor_id=feedback_by_motor_id,
        )

        row = {
            "n": joint_name,
            "id": motor_id,
            "bus": bus_name,
            "qd": None,
            "qf": None,
            "vf": None,
            "tc": None,
            "tf": None,
            "temp": None,
            "fault": 0,
            "mode": 0,
        }

        if feedback is not None and now - feedback["timestamp"] <= stale_seconds:
            row.update({
                "qf": feedback_joint_position(
                    joint_name,
                    layer,
                    feedback,
                    pose_references=pose_references,
                ),
                "vf": float(feedback["velocity"]),
                "tf": float(feedback["torque"]),
                "temp": float(feedback["temperature_c"]),
                "fault": int(feedback.get("fault_bits", 0)),
                "mode": int(feedback.get("mode_status", 0)),
            })
            has_fault = has_fault or row["fault"] != 0

        joint_rows.append(row)

    angles = {row["n"]: row["qf"] for row in joint_rows if row["qf"] is not None}
    pose_name, pose_rms, pose_max = estimate_pose_from_angles(joints, angles, poses)

    return {
        "step": int(step),
        "mode": "encoder",
        "cmd": [0.0, 0.0, 0.0],
        "speed": 0.0,
        "imu": "none",
        "safe": not has_fault,
        "fault_reason": "motor feedback fault bit set" if has_fault else "",
        "pose_estimate": pose_name,
        "pose_error_rms": pose_rms,
        "pose_error_max": pose_max,
        "act_max": 0.0,
        "base_vel": [0.0, 0.0, 0.0],
        "ang_vel": [0.0, 0.0, 0.0],
        "gravity": [0.0, 0.0, -1.0],
        "joints": joint_rows,
        "ts": time.monotonic(),
    }


def send_gui_packet(sock, port, packet):
    if sock is None:
        return
    try:
        data = json.dumps(packet, separators=(",", ":")).encode()
        sock.sendto(data, ("127.0.0.1", int(port)))
    except Exception:
        pass


def launch_gui(port):
    gui_script = ROOT / "src" / "telemetry_gui.py"
    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("WARNING: DISPLAY/WAYLAND_DISPLAY is not set; Tk GUI may not open from this terminal.")

    proc = subprocess.Popen(
        [sys.executable, str(gui_script), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(0.6)
    if proc.poll() is None:
        print(f"Telemetry GUI launched (PID {proc.pid}).")
        return proc

    output, _ = proc.communicate(timeout=0.2)
    print("ERROR: Telemetry GUI exited before opening.")
    print("GUI return code:", proc.returncode)
    if output.strip():
        print("GUI output:")
        print(output.strip())
    return None


def print_compact_status(
    step,
    joints,
    motor_ids,
    joint_can_bus,
    layer,
    pose_references,
    poses,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
    stale_seconds,
):
    now = time.monotonic()
    connected = 0
    stale = 0
    faults = 0
    angles = {}
    for joint_name in joints:
        feedback = feedback_for_joint(
            joint_name=joint_name,
            motor_ids=motor_ids,
            joint_can_bus=joint_can_bus,
            feedback_by_bus_motor_id=feedback_by_bus_motor_id,
            feedback_by_motor_id=feedback_by_motor_id,
        )
        state = feedback_status(feedback, now, stale_seconds)
        if state == "CONNECTED":
            connected += 1
            angles[joint_name] = feedback_joint_position(
                joint_name,
                layer,
                feedback,
                pose_references=pose_references,
            )
        elif state == "STALE":
            stale += 1
        elif state == "FAULT":
            faults += 1

    pose_name, pose_rms, pose_max = estimate_pose_from_angles(joints, angles, poses)
    pose_text = "pose=unknown"
    if pose_rms is not None:
        pose_text = f"pose={pose_name} pose_rms={pose_rms:.4f} pose_max={pose_max:.4f}"

    print(
        f"encoder_gui step={step:06d} connected={connected:02d}/{len(joints):02d} "
        f"stale={stale:02d} faults={faults:02d} {pose_text}",
        flush=True,
    )


def main():
    joint_cfg, motor_cfg = load_config()
    policy_order = list(joint_cfg["policy_to_real_order"])
    pose_references, poses = load_pose_references(policy_order)

    parser = argparse.ArgumentParser()
    add_can_topology_args(parser, default_port="/dev/ttyUSB0", default_can_count=2)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--active-joints", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="ignore config active_joints and check all configured joints")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--stale-seconds", type=float, default=0.5)
    parser.add_argument("--print-every", type=float, default=0.25)
    parser.add_argument("--set-zero-key", default="s",
                        help="keyboard key that sends RobStride set-zero to active joints")
    parser.add_argument("--crouch-key", default="c",
                        help="keyboard key that saves current sign-corrected joint feedback as config/default_pose.yaml crouch_pose")
    parser.add_argument("--quit-key", default="q", help="keyboard key that quits the checker")
    parser.add_argument("--set-zero-cooldown", type=float, default=1.0,
                        help="minimum seconds between repeated set-zero keypresses")
    parser.add_argument("--disable-set-zero-key", action="store_true",
                        help="disable keyboard set-zero command")
    parser.add_argument("--disable-crouch-key", action="store_true",
                        help="disable keyboard crouch-pose capture")
    parser.add_argument("--set-zero-yaml", action=argparse.BooleanOptionalAction, default=True,
                        help="when pressing set-zero, also write default_pose and stand_pose")
    parser.add_argument("--set-zero-value-rad", type=float, default=0.0,
                        help="legacy compatibility option; hardware set-zero requires 0.0")
    parser.add_argument("--gui", action="store_true",
                        help="launch telemetry GUI and stream encoder values only")
    parser.add_argument("--gui-only", action="store_true",
                        help="launch GUI and print compact status instead of the full terminal table")
    parser.add_argument("--telemetry-port", type=int, default=TELEMETRY_PORT_DEFAULT,
                        help="UDP port used by the GUI")
    parser.add_argument("--no-clear", action="store_true",
                        help="do not clear/redraw the terminal table")
    args = parser.parse_args()
    if not math.isclose(float(args.set_zero_value_rad), 0.0, abs_tol=1e-12):
        parser.error("--set-zero-value-rad must be 0.0 for persistent motor hardware zero")
    if args.gui_only:
        args.gui = True

    try:
        port_by_bus = resolve_port_by_bus(args)
    except ValueError as exc:
        print("ERROR:", exc)
        return 1

    if args.all:
        joints = policy_order
    else:
        joints = resolve_joints(policy_order, motor_cfg, args.active_joints)

    motor_ids = motor_cfg["motor_ids"]
    display_joints = connection_display_order(joints, motor_ids)
    joint_can_bus = resolve_joint_can_bus(policy_order, args.can_count)
    layer = DirectMotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=joints,
        joint_can_bus=joint_can_bus,
    )
    active_port_by_bus = ports_for_active_joints(
        port_by_bus,
        joint_can_bus,
        layer.active_joints,
    )
    if not active_port_by_bus:
        active_port_by_bus = port_by_bus
    try:
        validate_unique_motor_ids_per_physical_bus(
            motor_ids=motor_ids,
            joint_can_bus=joint_can_bus,
            active_joints=layer.active_joints,
            port_by_bus=active_port_by_bus,
        )
    except ValueError as exc:
        print("ERROR:", exc)
        return 1
    poll_commands = layer.build_feedback_poll_commands()
    set_zero_commands = layer.build_set_zero_commands()
    save_parameter_commands = layer.build_save_parameter_commands()
    set_zero_key = (args.set_zero_key or "s")[0].lower()
    crouch_key = (args.crouch_key or "c")[0].lower()
    quit_key = (args.quit_key or "q")[0].lower()

    print("Opening RobStride CAN interfaces...")
    for line in topology_lines(args.can_count, port_by_bus):
        print(line)
    print(
        f"backend={args.can_backend} bitrate={args.can_bitrate} "
        f"serial_baud={args.baud}"
    )
    print("Checking joints (FR, FL, BR, BL; ascending motor ID within each leg):")
    for joint_name in display_joints:
        bus_name = joint_can_bus.get(joint_name, "front")
        print(f"  {joint_name:20s} -> 0x{int(motor_ids[joint_name]):02X}  [{bus_name}]")

    buses = open_can_buses(
        active_port_by_bus,
        baud=args.baud,
        timeout=0.002,
        backend=args.can_backend,
        bitrate=args.can_bitrate,
    )

    gui_proc = None
    telemetry_sock = None
    if args.gui:
        telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        telemetry_sock.setblocking(False)
        gui_proc = launch_gui(args.telemetry_port)

    feedback_by_bus_motor_id = {}
    feedback_by_motor_id = {}
    joint_positions = {joint_name: 0.0 for joint_name in layer.active_joints}
    active_bus_motor_keys = {
        (joint_can_bus.get(name, "front"), int(motor_ids[name]))
        for name in layer.active_joints
    }
    dt = 1.0 / max(1e-6, args.rate)
    deadline = None if args.seconds <= 0.0 else time.monotonic() + args.seconds
    next_print = 0.0
    last_set_zero_time = -1e9
    status_message = ""
    keyboard_enabled = sys.stdin.isatty()
    set_zero_enabled = not args.disable_set_zero_key
    crouch_enabled = not args.disable_crouch_key
    step = 0

    try:
        with KeyboardReader(enabled=keyboard_enabled) as keyboard:
            while deadline is None or time.monotonic() < deadline:
                now = time.monotonic()
                key = keyboard.read_key()
                if key is not None:
                    key = key.lower()
                    if key == quit_key:
                        print("\nStopped by keyboard quit key.")
                        break
                    if key == set_zero_key and set_zero_enabled:
                        if now - last_set_zero_time >= max(0.0, args.set_zero_cooldown):
                            layer.send_raw_commands(buses, layer.build_stop_commands())
                            time.sleep(0.02)
                            layer.send_raw_commands(buses, set_zero_commands)
                            time.sleep(0.02)
                            layer.send_raw_commands(buses, save_parameter_commands)
                            last_set_zero_time = now
                            joint_positions = {
                                joint_name: 0.0
                                for joint_name in layer.active_joints
                            }
                            feedback_by_bus_motor_id.clear()
                            feedback_by_motor_id.clear()
                            status_message = (
                                f"PERSISTENT MOTOR ZERO and SAVE sent to "
                                f"{len(set_zero_commands)} active motor(s) "
                                f"at {time.strftime('%H:%M:%S')}; "
                                "joint q=+0.000 rad; no software offset applied"
                            )
                            if args.set_zero_yaml:
                                try:
                                    yaml_status = set_stand_default_yaml_value(
                                        layer.active_joints,
                                        0.0,
                                    )
                                    status_message += "\n" + yaml_status
                                except Exception as exc:
                                    status_message += f"\nYAML zero update failed: {exc}"
                            print(status_message, flush=True)
                            time.sleep(0.05)
                        else:
                            status_message = "SET ZERO ignored: cooldown active."
                            print(status_message, flush=True)
                    elif key == set_zero_key and not set_zero_enabled:
                        status_message = "SET ZERO ignored: disabled by --disable-set-zero-key."
                        print(status_message, flush=True)
                    elif key == crouch_key and crouch_enabled:
                        status_message = capture_crouch_pose(
                            joints=layer.active_joints,
                            motor_ids=motor_ids,
                            joint_can_bus=joint_can_bus,
                            layer=layer,
                            pose_references=pose_references,
                            feedback_by_bus_motor_id=feedback_by_bus_motor_id,
                            feedback_by_motor_id=feedback_by_motor_id,
                            stale_seconds=args.stale_seconds,
                        )
                        print(status_message, flush=True)
                    elif key == crouch_key and not crouch_enabled:
                        status_message = "CROUCH capture ignored: disabled by --disable-crouch-key."
                        print(status_message, flush=True)

                layer.send_raw_commands(buses, poll_commands)

                frames = layer_cls.read_all_frames(buses, timeout=min(0.05, dt))
                now = time.monotonic()
                update_feedback_from_frames(
                    frames=frames,
                    layer=layer,
                    active_bus_motor_keys=active_bus_motor_keys,
                    feedback_by_bus_motor_id=feedback_by_bus_motor_id,
                    feedback_by_motor_id=feedback_by_motor_id,
                )
                resolve_feedback_joint_positions(
                    joints=layer.active_joints,
                    motor_ids=motor_ids,
                    joint_can_bus=joint_can_bus,
                    layer=layer,
                    pose_references=pose_references,
                    poses=poses,
                    feedback_by_bus_motor_id=feedback_by_bus_motor_id,
                    feedback_by_motor_id=feedback_by_motor_id,
                    joint_positions=joint_positions,
                    stale_seconds=args.stale_seconds,
                )

                if now >= next_print:
                    if args.gui_only:
                        print_compact_status(
                            step=step,
                            joints=layer.active_joints,
                            motor_ids=motor_ids,
                            joint_can_bus=joint_can_bus,
                            layer=layer,
                            pose_references=pose_references,
                            poses=poses,
                            feedback_by_bus_motor_id=feedback_by_bus_motor_id,
                            feedback_by_motor_id=feedback_by_motor_id,
                            stale_seconds=args.stale_seconds,
                        )
                    else:
                        print_table(
                            joints=display_joints,
                            motor_ids=motor_ids,
                            joint_can_bus=joint_can_bus,
                            layer=layer,
                            pose_references=pose_references,
                            poses=poses,
                            feedback_by_bus_motor_id=feedback_by_bus_motor_id,
                            feedback_by_motor_id=feedback_by_motor_id,
                            stale_seconds=args.stale_seconds,
                            set_zero_key=set_zero_key,
                            quit_key=quit_key,
                            keyboard_enabled=keyboard_enabled,
                            set_zero_enabled=set_zero_enabled,
                            status_message=status_message,
                            crouch_key=crouch_key,
                            crouch_enabled=crouch_enabled,
                            clear_screen=not args.no_clear,
                        )
                    next_print = now + args.print_every

                send_gui_packet(
                    telemetry_sock,
                    args.telemetry_port,
                    build_gui_packet(
                        step=step,
                        joints=layer.active_joints,
                        motor_ids=motor_ids,
                        joint_can_bus=joint_can_bus,
                        layer=layer,
                        pose_references=pose_references,
                        poses=poses,
                        feedback_by_bus_motor_id=feedback_by_bus_motor_id,
                        feedback_by_motor_id=feedback_by_motor_id,
                        stale_seconds=args.stale_seconds,
                    ),
                )
                step += 1
                time.sleep(max(0.0, dt - 0.001 * len(poll_commands)))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        close_can_buses(buses)
        if telemetry_sock is not None:
            telemetry_sock.close()
        if gui_proc is not None:
            gui_proc.terminate()
            try:
                gui_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                gui_proc.kill()
            print("Telemetry GUI closed.")
        print("Serial closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
