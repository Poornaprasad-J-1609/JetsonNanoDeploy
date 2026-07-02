#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]
TWO_PI = 2.0 * np.pi


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def float_to_uint(x, x_min, x_max, bits):
    x = float(np.clip(x, x_min, x_max))
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)


def uint_to_float(x, x_min, x_max, bits):
    x = int(np.clip(x, 0, (1 << bits) - 1))
    span = x_max - x_min
    return float(x_min + span * x / ((1 << bits) - 1))


def wrap_to_pi(angle):
    """Map an angle or angle error into [-pi, pi)."""
    return float((float(angle) + np.pi) % TWO_PI - np.pi)


def motor_position_to_joint_angle(
    position,
    offset=0.0,
    direction=1.0,
    reference=None,
    references=None,
    pose_snap_tolerance=0.0,
):
    """Convert raw motor feedback to deployed joint coordinates.

    RobStride can report the same physical zero as 0, +2*pi, or -2*pi after
    power cycling near an encoder branch boundary. Apply the motor mounting
    direction first, then normalize the phase in true joint space so inverted
    left/right motors unwrap with the correct sign.
    """
    oriented_angle = float(direction) * (float(position) - float(offset))
    return wrap_to_pi(oriented_angle)


def motor_command_position_near_feedback(
    q_des,
    offset=0.0,
    direction=1.0,
    feedback_position=None,
    feedback_joint_position=None,
    p_min=None,
    p_max=None,
):
    """Convert desired joint angle to the raw MIT motor branch nearest feedback."""
    direction = float(direction)
    p_base = float(offset) + direction * float(q_des)
    if feedback_position is None or not np.isfinite(feedback_position):
        return p_base

    if p_min is None:
        p_min = -np.inf
    if p_max is None:
        p_max = np.inf
    p_min = float(p_min)
    p_max = float(p_max)

    feedback_position = float(feedback_position)
    if feedback_joint_position is not None and np.isfinite(feedback_joint_position):
        q_error = wrap_to_pi(float(q_des) - float(feedback_joint_position))
        p_branch = feedback_position + direction * q_error
        if p_min <= p_branch <= p_max:
            return float(p_branch)

    # Pick the equivalent command position separated by whole 2*pi turns that
    # is closest to the live encoder branch. This prevents a target near zero
    # from commanding a full-turn jump when feedback booted at +2*pi/-2*pi.
    k_nearest = int(round((feedback_position - p_base) / TWO_PI))
    candidates = [p_base + TWO_PI * k for k in (k_nearest - 1, k_nearest, k_nearest + 1)]
    valid = [candidate for candidate in candidates if p_min <= candidate <= p_max]
    if valid:
        return float(min(valid, key=lambda candidate: abs(candidate - feedback_position)))

    p_near = feedback_position + wrap_to_pi(p_base - feedback_position)
    if p_min <= p_near <= p_max:
        return float(p_near)

    return p_base


def signed_offset_to_uint(x, x_min, x_max):
    limit = max(abs(float(x_min)), abs(float(x_max)))
    x = float(np.clip(x, -limit, limit))
    raw = int((x / limit + 1.0) * 0x7FFF)
    return int(np.clip(raw, 0, 0xFFFF))


def uint_to_signed_offset(x, x_min, x_max):
    limit = max(abs(float(x_min)), abs(float(x_max)))
    x = int(np.clip(x, 0, 0xFFFF))
    value = (x / 0x7FFF - 1.0) * limit
    return float(np.clip(value, -limit, limit))


def unsigned_to_uint(x, x_min, x_max):
    if float(x_min) != 0.0:
        raise ValueError("RobStride/CyberGear unsigned MIT fields must have min=0.0")
    x = float(np.clip(x, x_min, x_max))
    return int(np.clip(int(x / float(x_max) * 0xFFFF), 0, 0xFFFF))


def pack_mit_command(p_des, v_des, kp, kd, proto):
    """
    Pack RobStride/CyberGear-style operation-control command into 8 bytes.

    Feed-forward torque is NOT in these 8 bytes; it lives in the extended CAN
    ID extra-data field and is handled separately by mit_can_id().
    """
    p_int = float_to_uint(p_des, proto["p_min"], proto["p_max"], 16)
    v_int = float_to_uint(v_des, proto["v_min"], proto["v_max"], 16)
    kp_int = float_to_uint(kp, proto["kp_min"], proto["kp_max"], 16)
    kd_int = float_to_uint(kd, proto["kd_min"], proto["kd_max"], 16)

    return (
        p_int.to_bytes(2, "big") +
        v_int.to_bytes(2, "big") +
        kp_int.to_bytes(2, "big") +
        kd_int.to_bytes(2, "big")
    )


def mit_can_id(motor_id, proto, tau_ff=0.0):
    comm_type = int(proto["comm_type_mit_control"])
    tau_int = float_to_uint(tau_ff, proto["tau_min"], proto["tau_max"], 16)
    motor_id = int(motor_id)

    # RobStride/CyberGear-style extended ID layout:
    # bits 28..24: comm_type, bits 23..8: torque extra-data, bits 7..0: motor id.
    return (comm_type << 24) | (tau_int << 8) | motor_id


def motor_management_can_id(motor_id, proto, comm_type_key):
    comm_type = int(proto[comm_type_key])
    master_id = int(proto["master_id"])
    motor_id = int(motor_id)
    return (comm_type << 24) | (master_id << 8) | motor_id


def decode_mit_feedback_frame(can_id, data, proto):
    data = bytes(data)
    if len(data) != 8:
        return None

    comm_type = (int(can_id) >> 24) & 0x1F
    feedback_types = {
        int(proto.get("comm_type_feedback", 2)),
        int(proto.get("comm_type_active_feedback", 24)),
    }
    if comm_type not in feedback_types:
        return None

    extra = (int(can_id) >> 8) & 0xFFFF
    motor_id = extra & 0xFF
    fault_bits = (extra >> 8) & 0x3F
    mode_status = (extra >> 14) & 0x03

    p_int = int.from_bytes(data[0:2], "big")
    v_int = int.from_bytes(data[2:4], "big")
    tau_int = int.from_bytes(data[4:6], "big")
    temp_int = int.from_bytes(data[6:8], "big")

    return {
        "comm_type": comm_type,
        "motor_id": motor_id,
        "fault_bits": fault_bits,
        "mode_status": mode_status,
        "position": uint_to_float(p_int, proto["p_min"], proto["p_max"], 16),
        "velocity": uint_to_float(v_int, proto["v_min"], proto["v_max"], 16),
        "torque": uint_to_float(tau_int, proto["tau_min"], proto["tau_max"], 16),
        "temperature_c": 0.1 * temp_int,
    }


def joint_group(joint_name):
    if "hip" in joint_name:
        return "hip"
    if "thigh" in joint_name:
        return "thigh"
    if "calf" in joint_name:
        return "calf"
    raise ValueError(f"Cannot infer joint group from {joint_name}")


class MotorCommandLayer:
    def __init__(self, policy_order, motor_ids, active_joints=None, joint_can_bus=None):
        self.policy_order = policy_order
        self.motor_ids = motor_ids
        self.policy_index_by_joint = {
            joint_name: index for index, joint_name in enumerate(self.policy_order)
        }

        self.control_limit_path = ROOT / "config" / "control_limits.yaml"
        self.joint_limit_path = ROOT / "config" / "joint_limits.yaml"
        self.control_limit_mtime_ns = None
        self.joint_limit_mtime_ns = None
        self.mit_parameter_limits_enabled = True
        self.mit_parameter_limits = {}
        self.hard_joint_limits = {}

        self.cfg = load_yaml(ROOT / "config" / "mit_motor_control.yaml")
        self.offset_cfg = load_yaml(ROOT / "config" / "joint_offsets.yaml")
        self.proto = self.cfg["mit_protocol"]
        self.gains = self.cfg["gains"]
        self.feedforward = self.cfg["feedforward"]
        communication_cfg = self.cfg.get("communication", {})
        self.frame_gap_s = float(communication_cfg.get("frame_gap_s", 0.0))
        self.batch_writes = bool(communication_cfg.get("batch_writes", False))
        self.group_flush = bool(communication_cfg.get("group_flush", True))
        self.joint_offsets = self.offset_cfg["joint_offsets"]
        self.joint_directions = self._load_joint_directions()
        self.active_joints = self.resolve_active_joints(active_joints)
        self.joint_can_bus = dict(joint_can_bus) if joint_can_bus else {}
        self.joint_coordinate_shifts = {joint_name: 0.0 for joint_name in self.policy_order}
        self.reload_joint_limits(force=True)
        self.reload_control_limits(force=True)

    def _load_joint_directions(self):
        configured = self.offset_cfg.get("joint_directions", {}) or {}
        directions = {}
        for joint_name in self.policy_order:
            value = float(configured.get(joint_name, 1.0))
            if not np.isfinite(value) or abs(value) < 1e-9:
                raise ValueError(f"{joint_name}: joint direction must be +1 or -1")
            directions[joint_name] = 1.0 if value > 0.0 else -1.0
        return directions

    def resolve_active_joints(self, active_joints):
        if not active_joints:
            active_joints = self.policy_order

        resolved = []
        for joint_name in active_joints:
            if joint_name not in self.policy_index_by_joint:
                raise KeyError(f"Unknown active joint: {joint_name}")
            if joint_name not in self.motor_ids:
                raise KeyError(f"Missing motor ID for active joint: {joint_name}")
            if joint_name not in self.joint_offsets:
                raise KeyError(f"Missing joint offset for active joint: {joint_name}")
            if joint_name not in self.joint_directions:
                raise KeyError(f"Missing joint direction for active joint: {joint_name}")
            motor_id = int(self.motor_ids[joint_name])
            if motor_id < 0 or motor_id > 0xFF:
                raise ValueError(
                    f"{joint_name}: motor ID 0x{motor_id:X} is outside 8-bit range"
                )
            resolved.append(joint_name)

        return resolved

    def reload_control_limits(self, force=False):
        mtime_ns = self.control_limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.control_limit_mtime_ns:
            return False

        cfg = load_yaml(self.control_limit_path)
        mit_cfg = cfg.get("mit_parameters", {})
        self.mit_parameter_limits_enabled = bool(mit_cfg.get("enabled", True))
        self.mit_parameter_limits = {
            "p_min": float(mit_cfg.get("p_min", self.proto["p_min"])),
            "p_max": float(mit_cfg.get("p_max", self.proto["p_max"])),
            "v_min": float(mit_cfg.get("v_min", self.proto["v_min"])),
            "v_max": float(mit_cfg.get("v_max", self.proto["v_max"])),
            "kp_min": float(mit_cfg.get("kp_min", self.proto["kp_min"])),
            "kp_max": float(mit_cfg.get("kp_max", self.proto["kp_max"])),
            "kd_min": float(mit_cfg.get("kd_min", self.proto["kd_min"])),
            "kd_max": float(mit_cfg.get("kd_max", self.proto["kd_max"])),
            "tau_ff_min": float(mit_cfg.get("tau_ff_min", self.proto["tau_min"])),
            "tau_ff_max": float(mit_cfg.get("tau_ff_max", self.proto["tau_max"])),
        }
        for prefix in ("p", "v", "kp", "kd", "tau_ff"):
            lo = self.mit_parameter_limits[f"{prefix}_min"]
            hi = self.mit_parameter_limits[f"{prefix}_max"]
            if lo > hi:
                raise ValueError(f"Invalid MIT parameter limits for {prefix}: min > max")

        self.control_limit_mtime_ns = mtime_ns
        return True

    def reload_joint_limits(self, force=False):
        mtime_ns = self.joint_limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.joint_limit_mtime_ns:
            return False

        cfg = load_yaml(self.joint_limit_path)
        limits = cfg["joint_limits"]
        hard_limits = {}

        for joint_name in self.policy_order:
            if joint_name not in limits:
                raise KeyError(f"Missing joint limit for {joint_name} in {self.joint_limit_path}")

            joint_limit = limits[joint_name]
            q_min = float(joint_limit["min"])
            q_max = float(joint_limit["max"])
            if not np.all(np.isfinite([q_min, q_max])):
                raise ValueError(f"{joint_name}: joint limits must be finite")
            if q_min > q_max:
                raise ValueError(f"{joint_name}: min {q_min} is greater than max {q_max}")
            offset = float(self.joint_offsets[joint_name])
            direction = float(self.joint_directions[joint_name])
            p_a = offset + direction * q_min
            p_b = offset + direction * q_max
            p_min = min(p_a, p_b)
            p_max = max(p_a, p_b)
            if p_min < float(self.proto["p_min"]) or p_max > float(self.proto["p_max"]):
                raise ValueError(
                    f"{joint_name}: joint limits plus offset [{p_min}, {p_max}] exceed "
                    f"MIT position range [{self.proto['p_min']}, {self.proto['p_max']}]"
                )
            hard_limits[joint_name] = (q_min, q_max)

        self.hard_joint_limits = hard_limits
        self.joint_limit_mtime_ns = mtime_ns
        return True

    def apply_hard_joint_limit(self, joint_name, q_des):
        self.reload_joint_limits()
        if not np.isfinite(q_des):
            raise ValueError(f"{joint_name}: requested joint target is NaN or Inf")
        q_min, q_max = self.hard_joint_limits[joint_name]
        shift = float(self.joint_coordinate_shifts.get(joint_name, 0.0))
        q_min += shift
        q_max += shift
        return float(np.clip(q_des, q_min, q_max))

    def apply_hard_joint_limit_to_motor_position(self, joint_name, p_des, offset, direction=1.0):
        q_des = motor_position_to_joint_angle(
            p_des,
            offset=offset,
            direction=direction,
        )
        q_des = self.apply_hard_joint_limit(joint_name, q_des)
        return float(offset) + float(direction) * q_des, q_des

    def coordinate_shift_array(self):
        return np.array(
            [self.joint_coordinate_shifts.get(joint_name, 0.0) for joint_name in self.policy_order],
            dtype=np.float32,
        )

    def set_software_zero_from_feedback(self, feedback_by_joint, active_joints=None, target_value=0.0):
        """
        Make the current measured motor position read as target_value in software.

        This updates only this process' joint_offsets. It does not send a
        RobStride set-zero frame, so MIT control can keep holding the same raw
        motor position without a torque drop or one-turn jump.
        """
        if feedback_by_joint is None:
            feedback_by_joint = {}
        if active_joints is None:
            active_joints = self.active_joints

        # First pass: validate that every active joint has fresh feedback
        # BEFORE mutating any offset. A partial calibration leaves some motors
        # referenced to a stale raw position, so they fail to hold while the
        # rest do. Make the whole operation all-or-nothing.
        target_value = float(target_value)
        new_offsets = {}
        new_shifts = {}
        missing = []
        for joint_name in active_joints:
            feedback = feedback_by_joint.get(joint_name)
            if feedback is None or "position" not in feedback:
                missing.append(joint_name)
                continue
            direction = float(self.joint_directions[joint_name])
            new_offsets[joint_name] = float(feedback["position"]) - direction * target_value
            new_shifts[joint_name] = target_value

        if missing:
            # Do not touch self.joint_offsets at all.
            return {}, missing

        updated = {}
        old_offsets = dict(self.joint_offsets)
        old_shifts = dict(self.joint_coordinate_shifts)
        for joint_name, position in new_offsets.items():
            self.joint_offsets[joint_name] = position
            self.joint_coordinate_shifts[joint_name] = new_shifts[joint_name]
            updated[joint_name] = position
        try:
            self.reload_joint_limits(force=True)
        except Exception:
            self.joint_offsets = old_offsets
            self.joint_coordinate_shifts = old_shifts
            self.reload_joint_limits(force=True)
            raise
        return updated, missing

    def apply_mit_parameter_limits(self, p_des, v_des, kp, kd, tau_ff):
        self.reload_control_limits()
        values = np.asarray([p_des, v_des, kp, kd, tau_ff], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("MIT command parameters contain NaN or Inf")
        if not self.mit_parameter_limits_enabled:
            return p_des, v_des, kp, kd, tau_ff

        lim = self.mit_parameter_limits
        return (
            float(np.clip(p_des, lim["p_min"], lim["p_max"])),
            float(np.clip(v_des, lim["v_min"], lim["v_max"])),
            float(np.clip(kp, lim["kp_min"], lim["kp_max"])),
            float(np.clip(kd, lim["kd_min"], lim["kd_max"])),
            float(np.clip(tau_ff, lim["tau_ff_min"], lim["tau_ff_max"])),
        )

    def build_mit_commands(self, q_target, phase="policy", feedback_by_joint=None):
        q_target = np.asarray(q_target, dtype=np.float32)
        if q_target.shape != (len(self.policy_order),):
            raise ValueError(
                f"q_target has shape {list(q_target.shape)}, "
                f"expected [{len(self.policy_order)}]"
            )
        if not np.all(np.isfinite(q_target)):
            raise ValueError("q_target contains NaN or Inf")

        commands = []
        feedback_by_joint = feedback_by_joint or {}

        if phase not in self.gains:
            raise ValueError(f"Unknown phase {phase}. Expected one of {list(self.gains.keys())}")

        for joint_name in self.active_joints:
            i = self.policy_index_by_joint[joint_name]
            motor_id = int(self.motor_ids[joint_name])
            group = joint_group(joint_name)
            offset = float(self.joint_offsets[joint_name])
            direction = float(self.joint_directions[joint_name])

            kp = float(self.gains[phase][group]["kp"])
            kd = float(self.gains[phase][group]["kd"])
            v_des = float(self.feedforward["v_des"])
            tau_ff = float(self.feedforward["tau_ff"])
            q_requested = float(q_target[i])
            q_des = self.apply_hard_joint_limit(joint_name, q_requested)
            p_base = offset + direction * q_des
            feedback = feedback_by_joint.get(joint_name, {})
            feedback_position = feedback.get("position") if isinstance(feedback, dict) else None
            feedback_joint_position = feedback.get("joint_position") if isinstance(feedback, dict) else None
            p_des = motor_command_position_near_feedback(
                q_des=q_des,
                offset=offset,
                direction=direction,
                feedback_position=feedback_position,
                feedback_joint_position=feedback_joint_position,
                p_min=self.proto["p_min"],
                p_max=self.proto["p_max"],
            )
            p_des, v_des, kp, kd, tau_ff = self.apply_mit_parameter_limits(
                p_des=p_des,
                v_des=v_des,
                kp=kp,
                kd=kd,
                tau_ff=tau_ff,
            )
            q_des_sent = q_des

            can_id = mit_can_id(motor_id, self.proto, tau_ff=tau_ff)
            data = pack_mit_command(
                p_des=p_des,
                v_des=v_des,
                kp=kp,
                kd=kd,
                proto=self.proto,
            )

            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "bus_name": self.joint_can_bus.get(joint_name, "front"),
                "phase": phase,
                "q_des": q_des_sent,
                "q_requested": q_requested,
                "offset": offset,
                "direction": direction,
                "p_des": p_des,
                "p_base": p_base,
                "p_wrap_adjustment": p_des - p_base,
                "v_des": v_des,
                "kp": kp,
                "kd": kd,
                "tau_ff": tau_ff,
                "can_id": can_id,
                "data": data,
            })

        return commands

    @staticmethod
    def _resolve_bus(buses, bus_name):
        """Return the ATUsbCan for bus_name; falls back to the single bus when buses is not a dict."""
        if not isinstance(buses, dict):
            return buses
        if bus_name not in buses:
            raise KeyError(f"Unknown CAN bus '{bus_name}'. Available buses: {sorted(buses)}")
        return buses[bus_name]

    @staticmethod
    def read_all_frames(buses, timeout=0.0):
        """Read available frames from all buses and return them as a single combined list."""
        if not isinstance(buses, dict):
            return buses.read_available_frames(timeout=timeout)

        frames = []
        unique_buses = []
        seen_bus_ids = set()
        for bus_name, bus in buses.items():
            if id(bus) in seen_bus_ids:
                continue
            seen_bus_ids.add(id(bus))
            unique_buses.append((bus_name, bus))

        per_bus_timeout = float(timeout)
        if unique_buses and per_bus_timeout > 0.0:
            per_bus_timeout /= float(len(unique_buses))

        for bus_name, bus in unique_buses:
            bus_frames = bus.read_available_frames(timeout=per_bus_timeout)
            for frame in bus_frames:
                frame.bus_name = bus_name
            frames.extend(bus_frames)
        return frames

    def send_harmless_frames(self, buses, commands):
        """Send a harmless signal frame per motor, routed to the correct bus."""
        for cmd in commands:
            bus = self._resolve_bus(buses, cmd.get("bus_name", "front"))
            bus.send_signal_frame(cmd["motor_id"])
            if self.frame_gap_s > 0.0:
                time.sleep(self.frame_gap_s)

    def send_signal_commands(self, buses, commands):
        """
        Sends MIT packets through USB-CAN, routing each command to the correct bus.

        Motors do not have to be connected for the serial adapter to transmit.
        If motors ARE connected and powered, these packets can move them.
        """
        return self._send_raw_like_commands(buses, commands)

    def build_enable_commands(self):
        commands = []
        for joint_name in self.active_joints:
            motor_id = int(self.motor_ids[joint_name])
            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "bus_name": self.joint_can_bus.get(joint_name, "front"),
                "can_id": motor_management_can_id(
                    motor_id,
                    self.proto,
                    "comm_type_enable",
                ),
                "data": bytes(8),
            })
        return commands

    def build_feedback_poll_commands(self):
        commands = []
        for joint_name in self.active_joints:
            motor_id = int(self.motor_ids[joint_name])
            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "bus_name": self.joint_can_bus.get(joint_name, "front"),
                "can_id": motor_management_can_id(
                    motor_id,
                    self.proto,
                    "comm_type_stop",
                ),
                "data": bytes(8),
            })
        return commands

    def build_set_zero_commands(self):
        data = bytearray(8)
        data[0] = 1

        commands = []
        for joint_name in self.active_joints:
            motor_id = int(self.motor_ids[joint_name])
            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "bus_name": self.joint_can_bus.get(joint_name, "front"),
                "can_id": motor_management_can_id(
                    motor_id,
                    self.proto,
                    "comm_type_set_zero",
                ),
                "data": bytes(data),
            })
        return commands

    def build_stop_commands(self, clear_fault=False):
        data = bytearray(8)
        if clear_fault:
            data[0] = 1

        commands = []
        for joint_name in self.active_joints:
            motor_id = int(self.motor_ids[joint_name])
            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "bus_name": self.joint_can_bus.get(joint_name, "front"),
                "can_id": motor_management_can_id(
                    motor_id,
                    self.proto,
                    "comm_type_stop",
                ),
                "data": bytes(data),
            })
        return commands

    def send_raw_commands(self, buses, commands):
        return self._send_raw_like_commands(buses, commands)

    def _send_raw_like_commands(self, buses, commands):
        commands = list(commands)
        if not commands:
            return []

        def send_items(bus, items):
            frames = [
                (cmd["can_id"], cmd["data"])
                for _, cmd in items
            ]
            if self.group_flush and hasattr(bus, "send_raw_sequence"):
                packets = bus.send_raw_sequence(
                    frames,
                    frame_gap_s=self.frame_gap_s,
                )
                return [
                    (index, packet)
                    for (index, _), packet in zip(items, packets)
                ]
            if self.batch_writes and hasattr(bus, "send_raw_batch"):
                packets = bus.send_raw_batch(frames)
                return [
                    (index, packet)
                    for (index, _), packet in zip(items, packets)
                ]

            results = []
            for index, cmd in items:
                packet = bus.send_raw(cmd["can_id"], cmd["data"])
                results.append((index, packet))
                if self.frame_gap_s > 0.0:
                    time.sleep(self.frame_gap_s)
            return results

        if not isinstance(buses, dict):
            indexed = list(enumerate(commands))
            return [packet for _, packet in send_items(buses, indexed)]

        grouped = {}
        group_order = []
        for index, cmd in enumerate(commands):
            bus = self._resolve_bus(buses, cmd.get("bus_name", "front"))
            key = id(bus)
            if key not in grouped:
                grouped[key] = {
                    "bus": bus,
                    "items": [],
                }
                group_order.append(key)
            grouped[key]["items"].append((index, cmd))

        if len(group_order) == 1:
            bus = grouped[group_order[0]]["bus"]
            sent = [None] * len(commands)
            for index, packet in send_items(bus, grouped[group_order[0]]["items"]):
                sent[index] = packet
            return sent

        sent = [None] * len(commands)

        def send_group(group):
            return send_items(group["bus"], group["items"])

        with ThreadPoolExecutor(max_workers=len(group_order)) as executor:
            futures = [executor.submit(send_group, grouped[key]) for key in group_order]
            for future in futures:
                for index, pkt in future.result():
                    sent[index] = pkt

        return sent


def print_mit_commands(commands, show_hex=False):
    for cmd in commands:
        line = (
            f"motor_id=0x{cmd['motor_id']:02X} "
            f"{cmd['joint_name']:16s} "
            f"phase={cmd['phase']:7s} "
            f"p={cmd['p_des']: .4f} "
            f"kp={cmd['kp']: .2f} "
            f"kd={cmd['kd']: .2f} "
            f"tau={cmd['tau_ff']: .2f}"
        )
        if abs(cmd["offset"]) > 1e-6:
            line += f" offset={cmd['offset']: .4f}"
        if abs(float(cmd.get("direction", 1.0)) - 1.0) > 1e-6:
            line += f" direction={cmd['direction']: .0f}"
        if show_hex:
            line += f" can_id=0x{cmd['can_id']:08X} data={cmd['data'].hex()}"
        print(line)


if __name__ == "__main__":
    from policy_runner import PolicyRunner

    runner = PolicyRunner()
    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_cfg["motor_ids"],
        active_joints=motor_cfg.get("active_joints"),
    )

    cmds = layer.build_mit_commands(runner.q_stand, phase="startup")
    print("MIT command example for STAND pose:")
    print_mit_commands(cmds, show_hex=True)
