#!/usr/bin/env python3
import math
from pathlib import Path
import time
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clip_scalar(value, lower, upper):
    """Clamp one numeric value without constructing a NumPy scalar array."""
    return min(max(float(value), float(lower)), float(upper))


def float_to_uint(x, x_min, x_max, bits):
    x = clip_scalar(x, x_min, x_max)
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)


def uint_to_float(x, x_min, x_max, bits):
    x = int(clip_scalar(x, 0, (1 << bits) - 1))
    span = x_max - x_min
    return float(x_min + span * x / ((1 << bits) - 1))


def motor_position_to_joint_angle(
    position,
    offset=0.0,
    direction=1.0,
    reference=None,
    references=None,
    pose_snap_tolerance=0.0,
):
    """Convert raw motor feedback directly into policy joint coordinates.

    The motor's persistent zero_state firmware owns shortest-path selection.
    Deployment deliberately performs no modulo, phase unwrap, or pose snap.
    """
    return float(direction) * (float(position) - float(offset))


def motor_command_position_near_feedback(
    q_des,
    offset=0.0,
    direction=1.0,
    feedback_position=None,
    feedback_joint_position=None,
    p_min=None,
    p_max=None,
):
    """Convert desired joint angle directly to the calibrated motor frame."""
    return float(offset) + float(direction) * float(q_des)


def signed_offset_to_uint(x, x_min, x_max):
    limit = max(abs(float(x_min)), abs(float(x_max)))
    x = clip_scalar(x, -limit, limit)
    raw = int((x / limit + 1.0) * 0x7FFF)
    return int(clip_scalar(raw, 0, 0xFFFF))


def uint_to_signed_offset(x, x_min, x_max):
    limit = max(abs(float(x_min)), abs(float(x_max)))
    x = int(clip_scalar(x, 0, 0xFFFF))
    value = (x / 0x7FFF - 1.0) * limit
    return clip_scalar(value, -limit, limit)


def unsigned_to_uint(x, x_min, x_max):
    if float(x_min) != 0.0:
        raise ValueError("RobStride/CyberGear unsigned MIT fields must have min=0.0")
    x = clip_scalar(x, x_min, x_max)
    return int(clip_scalar(int(x / float(x_max) * 0xFFFF), 0, 0xFFFF))


def pack_mit_command(p_des, v_des, kp, kd, proto):
    """
    Pack RobStride/CyberGear-style operation-control command into 8 bytes.

    Feed-forward torque is NOT in these 8 bytes; it lives in the extended CAN
    ID extra-data field and is handled separately by mit_can_id().
    """
    if bool(proto.get("use_float_to_uint", False)):
        # Exact command quantization used by the proven pre-SocketCAN MIT path
        # at commit 9b03a77. Feedback decoding remains on the official RS04
        # ranges and is intentionally independent of this compatibility path.
        p_int = float_to_uint(p_des, proto["p_min"], proto["p_max"], 16)
        v_int = float_to_uint(v_des, proto["v_min"], proto["v_max"], 16)
        kp_int = float_to_uint(kp, proto["kp_min"], proto["kp_max"], 16)
        kd_int = float_to_uint(kd, proto["kd_min"], proto["kd_max"], 16)
    else:
        p_int = signed_offset_to_uint(p_des, proto["p_min"], proto["p_max"])
        v_int = signed_offset_to_uint(v_des, proto["v_min"], proto["v_max"])
        kp_int = unsigned_to_uint(kp, proto["kp_min"], proto["kp_max"])
        kd_int = unsigned_to_uint(kd, proto["kd_min"], proto["kd_max"])

    return (
        p_int.to_bytes(2, "big") +
        v_int.to_bytes(2, "big") +
        kp_int.to_bytes(2, "big") +
        kd_int.to_bytes(2, "big")
    )


def mit_can_id(motor_id, proto, tau_ff=0.0):
    comm_type = int(proto["comm_type_mit_control"])
    if bool(proto.get("use_float_to_uint", False)):
        tau_int = float_to_uint(
            tau_ff,
            proto["tau_min"],
            proto["tau_max"],
            16,
        )
    else:
        tau_int = signed_offset_to_uint(
            tau_ff,
            proto["tau_min"],
            proto["tau_max"],
        )
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
        "position": uint_to_signed_offset(p_int, proto["p_min"], proto["p_max"]),
        "velocity": uint_to_signed_offset(v_int, proto["v_min"], proto["v_max"]),
        "torque": uint_to_signed_offset(tau_int, proto["tau_min"], proto["tau_max"]),
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
        self.control_limit_reload_interval_s = 1.0
        self.joint_limit_reload_interval_s = 1.0
        self._last_control_limit_reload_check_s = 0.0
        self._last_joint_limit_reload_check_s = 0.0
        self.mit_parameter_limits_enabled = True
        self.mit_parameter_limits = {}
        self.policy_pd_torque_limit = 0.0
        self.policy_pd_torque_limits = {}
        self.policy_pd_torque_limit_start = {}
        self.policy_pd_torque_limit_final = {}
        self.policy_pd_torque_limit_override = None
        self.policy_pd_torque_limits_override = None
        self.pose_pd_torque_limit_override = None
        self.startup_pd_torque_limit = 0.0
        self.sit_pd_torque_limit = 0.0
        self.stand_pd_torque_limit = 0.0
        self.hold_pd_torque_limit = 0.0
        self.leveling_pd_torque_limit = 0.0
        self.hard_joint_limits = {}

        self.cfg = load_yaml(ROOT / "config" / "mit_motor_control.yaml")
        self.offset_cfg = load_yaml(ROOT / "config" / "joint_offsets.yaml")
        self.direction_cfg = load_yaml(ROOT / "config" / "motor_directions.yaml")
        self.proto = dict(self.cfg["mit_protocol"])
        motor_cfg = self.cfg.get("motor", {})
        if bool(motor_cfg.get("use_official_mit_ranges", True)):
            self._load_official_mit_ranges(str(motor_cfg.get("model", "rs-04")))
        self.command_encoding = str(
            motor_cfg.get("command_encoding", "official")
        ).strip().lower()
        self.command_proto = self._command_proto_for_encoding(self.command_encoding)
        phase_encodings = motor_cfg.get("phase_command_encoding", {}) or {}
        self.phase_command_encoding = {
            str(phase): str(encoding).strip().lower()
            for phase, encoding in phase_encodings.items()
        }
        self.phase_command_proto = {
            phase: self._command_proto_for_encoding(encoding)
            for phase, encoding in self.phase_command_encoding.items()
        }
        self.gains = self.cfg["gains"]
        self.feedforward = self.cfg["feedforward"]
        virtual_stop_cfg = self.cfg.get("virtual_joint_stop", {}) or {}
        self.virtual_joint_stop_enabled = bool(virtual_stop_cfg.get("enabled", False))
        self.virtual_joint_stop_max_preload_nm = float(
            virtual_stop_cfg.get("max_preload_nm", 0.0)
        )
        if (
            not math.isfinite(self.virtual_joint_stop_max_preload_nm)
            or self.virtual_joint_stop_max_preload_nm < 0.0
        ):
            raise ValueError("virtual_joint_stop.max_preload_nm must be finite and >= 0")
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

    def close(self):
        return None

    def set_policy_gains(self, kp=None, kd=None):
        """Override walking MIT gains uniformly without changing pose gains."""
        if kp is None and kd is None:
            return
        values = {"kp": kp, "kd": kd}
        for field, value in values.items():
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"policy {field} override must be finite and >= 0")
            protocol_max = float(self.proto[f"{field}_max"])
            if value > protocol_max:
                raise ValueError(
                    f"policy {field} override {value} exceeds RS04 maximum "
                    f"{protocol_max}"
                )
            for group in ("hip", "thigh", "calf"):
                self.gains["policy"].setdefault(group, {})[field] = value
            for joint_name in self.policy_order:
                self.gains["policy"].setdefault("joints", {}).setdefault(
                    joint_name,
                    {},
                )[field] = value

    def apply_sit_stand_gain_profile(self, path):
        """Load isolated sit/stand gains without changing policy gains."""
        profile_path = Path(path).expanduser().resolve()
        cfg = load_yaml(profile_path) or {}
        profile = cfg.get("sit_stand_gain_test", {})
        gains = profile.get("gains", {})
        if not isinstance(gains, dict):
            raise ValueError("sit_stand_gain_test.gains must be a mapping")

        resolved = {}
        for phase in ("sit", "stand"):
            phase_cfg = gains.get(phase)
            if not isinstance(phase_cfg, dict):
                raise ValueError(
                    f"sit_stand_gain_test.gains.{phase} must be a mapping"
                )
            resolved_phase = {}
            for group in ("hip", "thigh", "calf"):
                group_cfg = phase_cfg.get(group)
                if not isinstance(group_cfg, dict):
                    raise ValueError(
                        f"sit_stand_gain_test.gains.{phase}.{group} "
                        "must be a mapping"
                    )
                resolved_group = {}
                for field in ("kp", "kd"):
                    value = float(group_cfg.get(field, float("nan")))
                    protocol_limit = float(self.proto[f"{field}_max"])
                    runtime_limit = (
                        float(self.mit_parameter_limits[f"{field}_max"])
                        if self.mit_parameter_limits_enabled
                        else protocol_limit
                    )
                    limit = min(protocol_limit, runtime_limit)
                    if not math.isfinite(value) or value < 0.0 or value > limit:
                        raise ValueError(
                            f"{phase}.{group}.{field} must be finite within "
                            f"[0, {limit}]"
                        )
                    resolved_group[field] = value
                resolved_phase[group] = resolved_group

            joint_overrides = phase_cfg.get("joints", {}) or {}
            if not isinstance(joint_overrides, dict):
                raise ValueError(
                    f"sit_stand_gain_test.gains.{phase}.joints must be a mapping"
                )
            unknown = sorted(set(joint_overrides) - set(self.policy_order))
            if unknown:
                raise KeyError(
                    f"Unknown {phase} gain override joint(s): " + ", ".join(unknown)
                )
            resolved_phase["joints"] = {}
            for joint_name, joint_cfg in joint_overrides.items():
                if not isinstance(joint_cfg, dict):
                    raise ValueError(f"{phase}.{joint_name} gain override must be a mapping")
                group = joint_group(joint_name)
                resolved_joint = dict(resolved_phase[group])
                for field in ("kp", "kd"):
                    if field not in joint_cfg:
                        continue
                    value = float(joint_cfg[field])
                    protocol_limit = float(self.proto[f"{field}_max"])
                    runtime_limit = (
                        float(self.mit_parameter_limits[f"{field}_max"])
                        if self.mit_parameter_limits_enabled
                        else protocol_limit
                    )
                    limit = min(protocol_limit, runtime_limit)
                    if not math.isfinite(value) or value < 0.0 or value > limit:
                        raise ValueError(
                            f"{phase}.{joint_name}.{field} must be finite within "
                            f"[0, {limit}]"
                        )
                    resolved_joint[field] = value
                resolved_phase["joints"][joint_name] = resolved_joint
            resolved[phase] = resolved_phase

        torque_limit = float(profile.get("torque_limit_nm", float("nan")))
        protocol_limit = max(abs(float(self.proto["tau_min"])), abs(float(self.proto["tau_max"])))
        if (
            not math.isfinite(torque_limit)
            or torque_limit < 0.0
            or torque_limit > protocol_limit
        ):
            raise ValueError(
                "sit_stand_gain_test.torque_limit_nm must be finite within "
                f"[0, {protocol_limit}]"
            )

        self.gains["sit"] = resolved["sit"]
        self.gains["stand"] = resolved["stand"]
        self.set_pose_pd_torque_limit(torque_limit)
        self.sit_stand_gain_profile_path = str(profile_path)
        return {
            "path": str(profile_path),
            "torque_limit_nm": torque_limit,
            "gains": resolved,
        }

    def _resolved_gain_phase(self, phase):
        phase = str(phase)
        if phase in self.gains:
            return phase
        if phase in ("sit", "stand"):
            return "startup"
        return phase

    def _load_official_mit_ranges(self, model):
        try:
            from robstride_dynamics.table import (
                MODEL_MIT_KD_TABLE,
                MODEL_MIT_KP_TABLE,
                MODEL_MIT_POSITION_TABLE,
                MODEL_MIT_TORQUE_TABLE,
                MODEL_MIT_VELOCITY_TABLE,
            )
        except ImportError as exc:
            raise ImportError(
                "Install robstride-dynamics to load official MIT parameters"
            ) from exc

        tables = (
            MODEL_MIT_POSITION_TABLE,
            MODEL_MIT_VELOCITY_TABLE,
            MODEL_MIT_KP_TABLE,
            MODEL_MIT_KD_TABLE,
            MODEL_MIT_TORQUE_TABLE,
        )
        if any(model not in table for table in tables):
            raise KeyError(f"Official RobStride MIT table has no model '{model}'")

        position = float(MODEL_MIT_POSITION_TABLE[model])
        velocity = float(MODEL_MIT_VELOCITY_TABLE[model])
        torque = float(MODEL_MIT_TORQUE_TABLE[model])
        self.proto.update({
            "p_min": -position,
            "p_max": position,
            "v_min": -velocity,
            "v_max": velocity,
            "kp_min": 0.0,
            "kp_max": float(MODEL_MIT_KP_TABLE[model]),
            "kd_min": 0.0,
            "kd_max": float(MODEL_MIT_KD_TABLE[model]),
            "tau_min": -torque,
            "tau_max": torque,
        })

    def _command_proto_for_encoding(self, encoding):
        encoding = str(encoding).strip().lower()
        command_proto = dict(self.proto)
        if encoding == "official":
            command_proto["use_float_to_uint"] = False
            return command_proto
        if encoding != "legacy_9b03a77":
            raise ValueError(
                "command encoding must be official or legacy_9b03a77"
            )

        legacy_proto = self.cfg.get("legacy_command_protocol", {})
        required = (
            "p_min", "p_max", "v_min", "v_max", "kp_min", "kp_max",
            "kd_min", "kd_max", "tau_min", "tau_max",
        )
        missing = [key for key in required if key not in legacy_proto]
        if missing:
            raise KeyError(
                "legacy_command_protocol is missing: " + ", ".join(missing)
            )
        command_proto.update({key: float(legacy_proto[key]) for key in required})
        command_proto["use_float_to_uint"] = bool(
            legacy_proto.get("use_float_to_uint", True)
        )
        return command_proto

    def command_proto_for_phase(self, phase):
        return self.phase_command_proto.get(str(phase), self.command_proto)

    def _load_joint_directions(self):
        configured = self.direction_cfg.get("motor_directions", {}) or {}
        directions = {}
        missing = []
        for joint_name in self.policy_order:
            if joint_name not in configured:
                missing.append(joint_name)
                continue
            value = float(configured[joint_name])
            if not np.isfinite(value) or value not in (-1.0, 1.0):
                raise ValueError(
                    f"{joint_name}: motor direction must be exactly +1 or -1"
                )
            directions[joint_name] = value
        if missing:
            raise KeyError(
                "Missing motor direction(s) in config/motor_directions.yaml: "
                + ", ".join(missing)
            )
        return directions

    def _effective_unsigned_wire_value(self, value, field, command_proto=None):
        """Return the value RS04 decodes from the configured command bits."""
        command_proto = self.command_proto if command_proto is None else command_proto
        command_min = float(command_proto[f"{field}_min"])
        command_max = float(command_proto[f"{field}_max"])
        if bool(command_proto.get("use_float_to_uint", False)):
            raw = float_to_uint(value, command_min, command_max, 16)
        else:
            raw = unsigned_to_uint(value, command_min, command_max)
        return uint_to_float(
            raw,
            float(self.proto[f"{field}_min"]),
            float(self.proto[f"{field}_max"]),
            16,
        )

    def _effective_signed_wire_value(self, value, field, command_proto=None):
        """Return a signed value after command encoding and official decoding."""
        command_proto = self.command_proto if command_proto is None else command_proto
        command_min = float(command_proto[f"{field}_min"])
        command_max = float(command_proto[f"{field}_max"])
        if bool(command_proto.get("use_float_to_uint", False)):
            raw = float_to_uint(value, command_min, command_max, 16)
        else:
            raw = signed_offset_to_uint(value, command_min, command_max)
        return uint_to_signed_offset(
            raw,
            float(self.proto[f"{field}_min"]),
            float(self.proto[f"{field}_max"]),
        )

    def _configured_unsigned_value_for_effective(
        self,
        effective_value,
        field,
        command_proto=None,
    ):
        """Map a desired motor-side gain into the selected packet encoding."""
        command_proto = self.command_proto if command_proto is None else command_proto
        effective_min = float(self.proto[f"{field}_min"])
        effective_max = float(self.proto[f"{field}_max"])
        command_min = float(command_proto[f"{field}_min"])
        command_max = float(command_proto[f"{field}_max"])
        normalized = (
            (float(effective_value) - effective_min)
            / max(effective_max - effective_min, 1.0e-12)
        )
        return clip_scalar(
            command_min + normalized * (command_max - command_min),
            command_min,
            command_max,
        )

    def _joint_gains(self, phase, joint_name, group):
        phase_cfg = self.gains[phase]
        joint_cfg = phase_cfg.get("joints", {}).get(joint_name)
        gain_cfg = phase_cfg[group] if joint_cfg is None else joint_cfg
        return float(gain_cfg["kp"]), float(gain_cfg["kd"])

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
        policy_cfg = cfg.get("policy_deployment", {})
        configured_policy_torque_limit = float(
            policy_cfg.get("estimated_pd_torque_limit", 0.0)
        )
        configured_joint_limits = policy_cfg.get("estimated_pd_torque_limits", {}) or {}
        if configured_joint_limits and not isinstance(configured_joint_limits, dict):
            raise ValueError("policy_deployment.estimated_pd_torque_limits must be a mapping")
        if (
            not np.isfinite(configured_policy_torque_limit)
            or configured_policy_torque_limit < 0.0
        ):
            raise ValueError("policy_deployment.estimated_pd_torque_limit must be finite and >= 0")
        configured_joint_torque_limits = {}
        for joint_name in self.policy_order:
            joint_limit = float(
                configured_joint_limits.get(joint_name, configured_policy_torque_limit)
            )
            if not np.isfinite(joint_limit) or joint_limit < 0.0:
                raise ValueError(
                    "policy_deployment.estimated_pd_torque_limits."
                    f"{joint_name} must be finite and >= 0"
                )
            configured_joint_torque_limits[joint_name] = joint_limit

        if self.policy_pd_torque_limits_override is not None:
            configured_joint_torque_limits = dict(self.policy_pd_torque_limits_override)
            configured_policy_torque_limit = max(configured_joint_torque_limits.values(), default=0.0)
            self.policy_pd_torque_limit = configured_policy_torque_limit
            self.policy_pd_torque_limits = configured_joint_torque_limits
        elif self.policy_pd_torque_limit_override is None:
            self.policy_pd_torque_limit = configured_policy_torque_limit
            self.policy_pd_torque_limits = configured_joint_torque_limits
        else:
            override = float(self.policy_pd_torque_limit_override)
            self.policy_pd_torque_limit = override
            self.policy_pd_torque_limits = {
                joint_name: override for joint_name in self.policy_order
            }
        self.policy_pd_torque_limit_start = dict(self.policy_pd_torque_limits)
        self.policy_pd_torque_limit_final = dict(self.policy_pd_torque_limits)
        mit_cfg = cfg.get("mit_parameters", {})
        configured_startup_torque_limit = float(
            mit_cfg.get("startup_pd_torque_limit", 0.0)
        )
        if (
            not np.isfinite(configured_startup_torque_limit)
            or configured_startup_torque_limit < 0.0
        ):
            raise ValueError("mit_parameters.startup_pd_torque_limit must be finite and >= 0")
        configured_hold_torque_limit = float(
            mit_cfg.get("hold_pd_torque_limit", 0.0)
        )
        if (
            not np.isfinite(configured_hold_torque_limit)
            or configured_hold_torque_limit < 0.0
        ):
            raise ValueError("mit_parameters.hold_pd_torque_limit must be finite and >= 0")
        configured_sit_torque_limit = float(
            mit_cfg.get("sit_pd_torque_limit", configured_startup_torque_limit)
        )
        configured_stand_torque_limit = float(
            mit_cfg.get("stand_pd_torque_limit", configured_startup_torque_limit)
        )
        if not np.isfinite(configured_sit_torque_limit) or configured_sit_torque_limit < 0.0:
            raise ValueError("mit_parameters.sit_pd_torque_limit must be finite and >= 0")
        if not np.isfinite(configured_stand_torque_limit) or configured_stand_torque_limit < 0.0:
            raise ValueError("mit_parameters.stand_pd_torque_limit must be finite and >= 0")
        if self.pose_pd_torque_limit_override is None:
            self.startup_pd_torque_limit = configured_startup_torque_limit
            self.sit_pd_torque_limit = configured_sit_torque_limit
            self.stand_pd_torque_limit = configured_stand_torque_limit
            self.hold_pd_torque_limit = configured_hold_torque_limit
        else:
            override = float(self.pose_pd_torque_limit_override)
            self.startup_pd_torque_limit = override
            self.sit_pd_torque_limit = override
            self.stand_pd_torque_limit = override
            self.hold_pd_torque_limit = override
        self.leveling_pd_torque_limit = float(
            mit_cfg.get("leveling_pd_torque_limit", 0.0)
        )
        if not np.isfinite(self.leveling_pd_torque_limit) or self.leveling_pd_torque_limit < 0.0:
            raise ValueError("mit_parameters.leveling_pd_torque_limit must be finite and >= 0")
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
        if (
            self.virtual_joint_stop_enabled
            and self.virtual_joint_stop_max_preload_nm > 0.0
            and self.mit_parameter_limits_enabled
        ):
            preload = self.virtual_joint_stop_max_preload_nm
            if (
                self.mit_parameter_limits["tau_ff_min"] > -preload
                or self.mit_parameter_limits["tau_ff_max"] < preload
            ):
                raise ValueError(
                    "MIT tau_ff limits would clip virtual joint-stop preload: "
                    f"need at least [-{preload:.3f}, +{preload:.3f}] Nm, got "
                    f"[{self.mit_parameter_limits['tau_ff_min']:.3f}, "
                    f"{self.mit_parameter_limits['tau_ff_max']:.3f}] Nm"
                )

        self.control_limit_mtime_ns = mtime_ns
        return True

    def maybe_reload_control_limits(self):
        now = time.monotonic()
        if now - self._last_control_limit_reload_check_s < self.control_limit_reload_interval_s:
            return False
        self._last_control_limit_reload_check_s = now
        return self.reload_control_limits(force=False)

    def set_policy_pd_torque_limit(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("policy PD torque limit override must be finite and >= 0")
        self.policy_pd_torque_limit_override = value
        self.policy_pd_torque_limits_override = None
        self.policy_pd_torque_limit = value
        self.policy_pd_torque_limits = {
            joint_name: value for joint_name in self.policy_order
        }
        self.policy_pd_torque_limit_start = dict(self.policy_pd_torque_limits)
        self.policy_pd_torque_limit_final = dict(self.policy_pd_torque_limits)

    def set_policy_pd_torque_limits(self, limits_by_joint, start_limits_by_joint=None, final_limits_by_joint=None):
        def resolve_map(values, label):
            if not isinstance(values, dict):
                raise ValueError(f"{label} must be a mapping")
            unknown = sorted(set(values) - set(self.policy_order))
            missing = sorted(set(self.policy_order) - set(values))
            if unknown:
                raise KeyError(f"Unknown {label} joint(s): " + ", ".join(unknown))
            if missing:
                raise KeyError(f"Missing {label} joint(s): " + ", ".join(missing))
            resolved_map = {}
            for joint_name in self.policy_order:
                value = float(values[joint_name])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{joint_name}: {label} must be finite and >= 0")
                resolved_map[joint_name] = value
            return resolved_map

        resolved = resolve_map(limits_by_joint, "policy PD torque limit")
        resolved_start = (
            dict(resolved)
            if start_limits_by_joint is None
            else resolve_map(start_limits_by_joint, "policy PD torque start limit")
        )
        resolved_final = (
            dict(resolved)
            if final_limits_by_joint is None
            else resolve_map(final_limits_by_joint, "policy PD torque final limit")
        )
        self.policy_pd_torque_limit_override = None
        self.policy_pd_torque_limits_override = dict(resolved)
        self.policy_pd_torque_limit = max(resolved.values(), default=0.0)
        self.policy_pd_torque_limits = dict(resolved)
        self.policy_pd_torque_limit_start = resolved_start
        self.policy_pd_torque_limit_final = resolved_final

    def set_pose_pd_torque_limit(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("pose PD torque limit override must be finite and >= 0")
        self.pose_pd_torque_limit_override = value
        self.startup_pd_torque_limit = value
        self.sit_pd_torque_limit = value
        self.stand_pd_torque_limit = value
        self.hold_pd_torque_limit = value

    def policy_pd_torque_limit_for_joint(self, joint_name):
        return float(
            self.policy_pd_torque_limits.get(joint_name, self.policy_pd_torque_limit)
        )

    def pose_pd_torque_limits(self):
        return {
            "startup": float(self.startup_pd_torque_limit),
            "sit": float(self.sit_pd_torque_limit),
            "stand": float(self.stand_pd_torque_limit),
            "hold": float(self.hold_pd_torque_limit),
        }

    def reload_joint_limits(self, force=False):
        mtime_ns = self.joint_limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.joint_limit_mtime_ns:
            return False

        cfg = load_yaml(self.joint_limit_path)
        limits = cfg["joint_limits"]
        hard_limits = {}
        policy_target_limits = {}

        for joint_name in self.policy_order:
            if joint_name not in limits:
                raise KeyError(f"Missing joint limit for {joint_name} in {self.joint_limit_path}")

            joint_limit = limits[joint_name]
            q_min = float(joint_limit["min"])
            q_max = float(joint_limit["max"])
            policy_q_min = float(joint_limit.get("policy_min", q_min))
            policy_q_max = float(joint_limit.get("policy_max", q_max))
            if not np.all(np.isfinite([q_min, q_max, policy_q_min, policy_q_max])):
                raise ValueError(f"{joint_name}: joint limits must be finite")
            if q_min > q_max:
                raise ValueError(f"{joint_name}: min {q_min} is greater than max {q_max}")
            if policy_q_min > policy_q_max:
                raise ValueError(
                    f"{joint_name}: policy_min {policy_q_min} is greater than "
                    f"policy_max {policy_q_max}"
                )
            offset = float(self.joint_offsets[joint_name])
            direction = float(self.joint_directions[joint_name])
            p_a = offset + direction * min(q_min, policy_q_min)
            p_b = offset + direction * max(q_max, policy_q_max)
            p_min = min(p_a, p_b)
            p_max = max(p_a, p_b)
            if p_min < float(self.proto["p_min"]) or p_max > float(self.proto["p_max"]):
                raise ValueError(
                    f"{joint_name}: joint limits plus offset [{p_min}, {p_max}] exceed "
                    f"MIT position range [{self.proto['p_min']}, {self.proto['p_max']}]"
                )
            hard_limits[joint_name] = (q_min, q_max)
            policy_target_limits[joint_name] = (policy_q_min, policy_q_max)

        self.hard_joint_limits = hard_limits
        self.policy_target_limits = policy_target_limits
        self.joint_limit_mtime_ns = mtime_ns
        return True

    def maybe_reload_joint_limits(self):
        now = time.monotonic()
        if now - self._last_joint_limit_reload_check_s < self.joint_limit_reload_interval_s:
            return False
        self._last_joint_limit_reload_check_s = now
        return self.reload_joint_limits(force=False)

    def apply_hard_joint_limit(self, joint_name, q_des, phase=None):
        self.maybe_reload_joint_limits()
        if not math.isfinite(float(q_des)):
            raise ValueError(f"{joint_name}: requested joint target is NaN or Inf")
        # The actor may request targets beyond the physical articulation range,
        # just as a simulator actuator target may sit beyond a joint stop. Real
        # hardware has no simulator constraint solver to absorb that request,
        # so the final packet boundary must always enforce the physical limits.
        # Wider policy limits remain useful for raw-actor diagnostics only.
        q_min, q_max = self.hard_joint_limits[joint_name]
        shift = float(self.joint_coordinate_shifts.get(joint_name, 0.0))
        q_min += shift
        q_max += shift
        return clip_scalar(q_des, q_min, q_max)

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
            if feedback is None or (
                "position_raw" not in feedback and "position" not in feedback
            ):
                missing.append(joint_name)
                continue
            direction = float(self.joint_directions[joint_name])
            motor_position = feedback.get("position_raw", feedback.get("position"))
            new_offsets[joint_name] = float(motor_position) - direction * target_value
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
        self.maybe_reload_control_limits()
        values = (p_des, v_des, kp, kd, tau_ff)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("MIT command parameters contain NaN or Inf")
        if not self.mit_parameter_limits_enabled:
            return p_des, v_des, kp, kd, tau_ff

        lim = self.mit_parameter_limits
        return (
            clip_scalar(p_des, lim["p_min"], lim["p_max"]),
            clip_scalar(v_des, lim["v_min"], lim["v_max"]),
            clip_scalar(kp, lim["kp_min"], lim["kp_max"]),
            clip_scalar(kd, lim["kd_min"], lim["kd_max"]),
            clip_scalar(tau_ff, lim["tau_ff_min"], lim["tau_ff_max"]),
        )

    def build_mit_commands(
        self,
        q_target,
        phase="policy",
        feedback_by_joint=None,
        joint_velocity_target=None,
        joint_feedforward_torque_target=None,
        prelimit_q_target=None,
        gain_blend_from_phase=None,
        gain_blend_alpha=1.0,
        previous_command_q=None,
        max_command_delta=None,
    ):
        self.maybe_reload_control_limits()
        self.maybe_reload_joint_limits()
        q_target = np.asarray(q_target, dtype=np.float32)
        if q_target.shape != (len(self.policy_order),):
            raise ValueError(
                f"q_target has shape {list(q_target.shape)}, "
                f"expected [{len(self.policy_order)}]"
            )
        if not np.all(np.isfinite(q_target)):
            raise ValueError("q_target contains NaN or Inf")
        if prelimit_q_target is not None:
            prelimit_q_target = np.asarray(prelimit_q_target, dtype=np.float32)
            if prelimit_q_target.shape != q_target.shape:
                raise ValueError(
                    "prelimit_q_target has shape "
                    f"{list(prelimit_q_target.shape)}, expected {list(q_target.shape)}"
                )
            if not np.all(np.isfinite(prelimit_q_target)):
                raise ValueError("prelimit_q_target contains NaN or Inf")
        if joint_velocity_target is not None:
            joint_velocity_target = np.asarray(
                joint_velocity_target,
                dtype=np.float32,
            )
            if joint_velocity_target.shape != q_target.shape:
                raise ValueError(
                    "joint_velocity_target has shape "
                    f"{list(joint_velocity_target.shape)}, expected "
                    f"{list(q_target.shape)}"
                )
            if not np.all(np.isfinite(joint_velocity_target)):
                raise ValueError("joint_velocity_target contains NaN or Inf")
        if joint_feedforward_torque_target is not None:
            joint_feedforward_torque_target = np.asarray(
                joint_feedforward_torque_target,
                dtype=np.float32,
            )
            if joint_feedforward_torque_target.shape != q_target.shape:
                raise ValueError(
                    "joint_feedforward_torque_target has shape "
                    f"{list(joint_feedforward_torque_target.shape)}, expected "
                    f"{list(q_target.shape)}"
                )
            if not np.all(np.isfinite(joint_feedforward_torque_target)):
                raise ValueError(
                    "joint_feedforward_torque_target contains NaN or Inf"
                )
        if previous_command_q is not None:
            previous_command_q = np.asarray(previous_command_q, dtype=np.float32)
            if previous_command_q.shape != q_target.shape:
                raise ValueError(
                    "previous_command_q has shape "
                    f"{list(previous_command_q.shape)}, expected {list(q_target.shape)}"
                )
            if not np.all(np.isfinite(previous_command_q)):
                raise ValueError("previous_command_q contains NaN or Inf")
        if max_command_delta is not None:
            max_command_delta = np.asarray(max_command_delta, dtype=np.float32)
            if max_command_delta.ndim == 0:
                max_command_delta = np.full_like(q_target, float(max_command_delta))
            if max_command_delta.shape != q_target.shape:
                raise ValueError(
                    "max_command_delta has shape "
                    f"{list(max_command_delta.shape)}, expected {list(q_target.shape)}"
                )
            if not np.all(np.isfinite(max_command_delta)) or np.any(
                max_command_delta < 0.0
            ):
                raise ValueError(
                    "max_command_delta must contain finite non-negative values"
                )

        commands = []
        feedback_by_joint = feedback_by_joint or {}

        gain_phase = self._resolved_gain_phase(phase)
        pure_position_pd_phase = str(phase) in ("sit", "stand")
        if gain_phase not in self.gains:
            raise ValueError(f"Unknown phase {phase}. Expected one of {list(self.gains.keys())}")
        command_proto = self.command_proto_for_phase(gain_phase)
        command_encoding = self.phase_command_encoding.get(
            gain_phase,
            self.command_encoding,
        )
        gain_blend_source_phase = None
        gain_blend_alpha = float(gain_blend_alpha)
        if not math.isfinite(gain_blend_alpha):
            raise ValueError("gain_blend_alpha must be finite")
        gain_blend_alpha = clip_scalar(gain_blend_alpha, 0.0, 1.0)
        if gain_blend_from_phase is not None and gain_blend_alpha < 1.0:
            gain_blend_source_phase = self._resolved_gain_phase(
                gain_blend_from_phase
            )
            if gain_blend_source_phase not in self.gains:
                raise ValueError(
                    "Unknown gain_blend_from_phase "
                    f"{gain_blend_from_phase}. Expected one of "
                    f"{list(self.gains.keys())}"
                )
            if phase == "policy" and gain_blend_source_phase != "policy":
                raise ValueError(
                    "policy packets cannot blend gains from a pose phase; "
                    "blend the position target and use official policy gains"
                )

        base_phase_torque_limit = 0.0
        if phase == "startup":
            base_phase_torque_limit = self.startup_pd_torque_limit
        elif phase == "sit":
            base_phase_torque_limit = self.sit_pd_torque_limit
        elif phase == "stand":
            base_phase_torque_limit = self.stand_pd_torque_limit
        elif phase == "hold":
            base_phase_torque_limit = self.hold_pd_torque_limit
        elif phase == "policy":
            base_phase_torque_limit = self.policy_pd_torque_limit
        elif phase == "leveling":
            base_phase_torque_limit = self.leveling_pd_torque_limit

        for joint_name in self.active_joints:
            i = self.policy_index_by_joint[joint_name]
            motor_id = int(self.motor_ids[joint_name])
            group = joint_group(joint_name)
            offset = float(self.joint_offsets[joint_name])
            direction = float(self.joint_directions[joint_name])

            kp, kd = self._joint_gains(gain_phase, joint_name, group)
            if gain_blend_source_phase is not None:
                source_kp, source_kd = self._joint_gains(
                    gain_blend_source_phase,
                    joint_name,
                    group,
                )
                source_proto = self.command_proto_for_phase(
                    gain_blend_source_phase
                )
                source_kp_effective = self._effective_unsigned_wire_value(
                    source_kp,
                    "kp",
                    source_proto,
                )
                source_kd_effective = self._effective_unsigned_wire_value(
                    source_kd,
                    "kd",
                    source_proto,
                )
                target_kp_effective = self._effective_unsigned_wire_value(
                    kp,
                    "kp",
                    command_proto,
                )
                target_kd_effective = self._effective_unsigned_wire_value(
                    kd,
                    "kd",
                    command_proto,
                )
                # Policy packets use official RS04 units, so their configured
                # values are the effective motor-side gains. Blend in that
                # physical space to avoid an impedance step when the legacy
                # loaded-stand packet path hands control to the policy.
                desired_kp_effective = (
                    (1.0 - gain_blend_alpha) * source_kp_effective
                    + gain_blend_alpha * target_kp_effective
                )
                desired_kd_effective = (
                    (1.0 - gain_blend_alpha) * source_kd_effective
                    + gain_blend_alpha * target_kd_effective
                )
                kp = self._configured_unsigned_value_for_effective(
                    desired_kp_effective,
                    "kp",
                    command_proto,
                )
                kd = self._configured_unsigned_value_for_effective(
                    desired_kd_effective,
                    "kd",
                    command_proto,
                )
            kp_effective = self._effective_unsigned_wire_value(
                kp,
                "kp",
                command_proto,
            )
            kd_effective = self._effective_unsigned_wire_value(
                kd,
                "kd",
                command_proto,
            )
            joint_v_des = 0.0 if pure_position_pd_phase else (
                float(self.feedforward["v_des"])
                if joint_velocity_target is None
                else float(joint_velocity_target[i])
            )
            joint_v_des_requested = joint_v_des
            q_requested = float(q_target[i])
            q_des = self.apply_hard_joint_limit(joint_name, q_requested, phase=phase)
            q_prelimit_requested = (
                q_requested
                if prelimit_q_target is None
                else float(prelimit_q_target[i])
            )
            q_prelimit_hard_limited = self.apply_hard_joint_limit(
                joint_name,
                q_prelimit_requested,
                phase=phase,
            )
            feedback = feedback_by_joint.get(joint_name, {})
            feedback_position = feedback.get("position_raw") if isinstance(feedback, dict) else None
            feedback_joint_position = feedback.get("joint_position") if isinstance(feedback, dict) else None
            feedback_joint_velocity = feedback.get("joint_velocity") if isinstance(feedback, dict) else None
            phase_torque_limit = (
                self.policy_pd_torque_limit_for_joint(joint_name)
                if phase == "policy"
                else base_phase_torque_limit
            )
            joint_limit_preload_error = 0.0
            joint_limit_preload_tau_ff_requested = 0.0
            joint_limit_preload_tau_ff = 0.0
            preload_feedback_valid = (
                feedback_joint_position is not None
                and feedback_joint_velocity is not None
                and math.isfinite(float(feedback_joint_position))
                and math.isfinite(float(feedback_joint_velocity))
            )
            if (
                phase == "policy"
                and self.virtual_joint_stop_enabled
                and self.virtual_joint_stop_max_preload_nm > 0.0
                and phase_torque_limit > 0.0
                and preload_feedback_valid
            ):
                # Reproduce only target error discarded by the physical joint
                # boundary. q_des can also differ because of target slew or
                # estimated-torque limiting; turning those differences into
                # feedforward torque bypasses the very limits intended to
                # smooth and bound the command.
                joint_limit_preload_error = (
                    q_prelimit_requested - q_prelimit_hard_limited
                )
                if abs(joint_limit_preload_error) > 1.0e-7:
                    joint_limit_preload_tau_ff_requested = (
                        kp_effective * joint_limit_preload_error
                    )
                    preload_limit = min(
                        self.virtual_joint_stop_max_preload_nm,
                        float(phase_torque_limit),
                    )
                    joint_limit_preload_tau_ff = clip_scalar(
                        joint_limit_preload_tau_ff_requested,
                        -preload_limit,
                        preload_limit,
                    )
            joint_tau_ff = 0.0 if pure_position_pd_phase else (
                float(self.feedforward["tau_ff"])
                + (
                    0.0
                    if joint_feedforward_torque_target is None
                    else float(joint_feedforward_torque_target[i])
                )
                + joint_limit_preload_tau_ff
            )
            joint_tau_ff_effective = self._effective_signed_wire_value(
                joint_tau_ff,
                "tau",
                command_proto,
            )
            q_before_command_rate_limit = q_des
            command_rate_limited = False
            if previous_command_q is not None and max_command_delta is not None:
                q_des = clip_scalar(
                    q_des,
                    float(previous_command_q[i] - max_command_delta[i]),
                    float(previous_command_q[i] + max_command_delta[i]),
                )
                q_des = self.apply_hard_joint_limit(
                    joint_name,
                    q_des,
                    phase=phase,
                )
                command_rate_limited = (
                    abs(q_des - q_before_command_rate_limit) > 1.0e-7
                )

            q_before_torque_limit = q_des
            torque_limited = False
            impedance_scale = 1.0
            kp_scale = 1.0
            kd_scale = 1.0
            tau_pd_est = None
            if (
                phase_torque_limit > 0.0
                and feedback_joint_position is not None
                and feedback_joint_velocity is not None
                and math.isfinite(float(feedback_joint_position))
                and math.isfinite(float(feedback_joint_velocity))
                and kp_effective > 0.0
            ):
                q_feedback = float(feedback_joint_position)
                qd_feedback = float(feedback_joint_velocity)
                velocity_and_ff_torque = (
                    kd_effective * (joint_v_des - qd_feedback)
                    + joint_tau_ff_effective
                )
                tau_pd_est = kp_effective * (q_des - q_feedback) + velocity_and_ff_torque

                if abs(tau_pd_est) > phase_torque_limit:
                    # Keep the policy target fixed, but do not remove damping
                    # when the position error consumes the torque budget. The
                    # July 31 loaded run showed a reversing hip at -3 rad/s
                    # with both gains scaled almost to zero; the joint then
                    # crossed its physical limit. Reserve torque for damping
                    # first and use only the remaining authority for stiffness.
                    position_error = q_des - q_feedback
                    velocity_error = joint_v_des - qd_feedback
                    damping_torque = kd_effective * velocity_error
                    damping_plus_ff = damping_torque + joint_tau_ff_effective
                    if abs(damping_plus_ff) > phase_torque_limit:
                        target_damping_torque = (
                            clip_scalar(
                                damping_plus_ff,
                                -phase_torque_limit,
                                phase_torque_limit,
                            )
                            - joint_tau_ff_effective
                        )
                        kd_scale = (
                            clip_scalar(
                                target_damping_torque / damping_torque,
                                0.0,
                                1.0,
                            )
                            if abs(damping_torque) > 1.0e-9
                            else 0.0
                        )
                        kd *= kd_scale
                        kd_effective = self._effective_unsigned_wire_value(
                            kd,
                            "kd",
                            command_proto,
                        )

                    damping_plus_ff = (
                        kd_effective * velocity_error + joint_tau_ff_effective
                    )
                    position_torque = kp_effective * position_error
                    combined_torque = damping_plus_ff + position_torque
                    if abs(combined_torque) > phase_torque_limit:
                        target_position_torque = (
                            clip_scalar(
                                combined_torque,
                                -phase_torque_limit,
                                phase_torque_limit,
                            )
                            - damping_plus_ff
                        )
                        kp_scale = (
                            clip_scalar(
                                target_position_torque / position_torque,
                                0.0,
                                1.0,
                            )
                            if abs(position_torque) > 1.0e-9
                            else 0.0
                        )
                        kp *= kp_scale

                    impedance_scale = min(kp_scale, kd_scale)
                    kp_effective = self._effective_unsigned_wire_value(
                        kp,
                        "kp",
                        command_proto,
                    )
                    velocity_and_ff_torque = (
                        kd_effective * (joint_v_des - qd_feedback)
                        + joint_tau_ff_effective
                    )
                    tau_pd_est = (
                        kp_effective * (q_des - q_feedback)
                        + velocity_and_ff_torque
                    )
                    torque_limited = True

            if (
                tau_pd_est is None
                and feedback_joint_position is not None
                and feedback_joint_velocity is not None
                and math.isfinite(float(feedback_joint_position))
                and math.isfinite(float(feedback_joint_velocity))
            ):
                q_feedback = float(feedback_joint_position)
                qd_feedback = float(feedback_joint_velocity)
                tau_pd_est = (
                    kp_effective * (q_des - q_feedback)
                    + kd_effective * (joint_v_des - qd_feedback)
                    + joint_tau_ff_effective
                )

            p_base = offset + direction * q_des
            p_des = motor_command_position_near_feedback(
                q_des=q_des,
                offset=offset,
                direction=direction,
                feedback_position=feedback_position,
                feedback_joint_position=feedback_joint_position,
                p_min=command_proto["p_min"],
                p_max=command_proto["p_max"],
            )
            motor_v_des = direction * joint_v_des
            motor_tau_ff = direction * joint_tau_ff
            p_des, motor_v_des, kp, kd, motor_tau_ff = self.apply_mit_parameter_limits(
                p_des=p_des,
                v_des=motor_v_des,
                kp=kp,
                kd=kd,
                tau_ff=motor_tau_ff,
            )
            q_des_sent = q_des

            can_id = mit_can_id(
                motor_id,
                command_proto,
                tau_ff=motor_tau_ff,
            )
            data = pack_mit_command(
                p_des=p_des,
                v_des=motor_v_des,
                kp=kp,
                kd=kd,
                proto=command_proto,
            )

            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "bus_name": self.joint_can_bus.get(joint_name, "front"),
                "phase": phase,
                "command_encoding": command_encoding,
                "gain_blend_from_phase": gain_blend_source_phase,
                "gain_blend_alpha": gain_blend_alpha,
                "q_des": q_des_sent,
                "q_requested": q_requested,
                "q_prelimit_requested": q_prelimit_requested,
                "q_prelimit_hard_limited": q_prelimit_hard_limited,
                "q_before_torque_limit": q_before_torque_limit,
                "q_before_command_rate_limit": q_before_command_rate_limit,
                "command_rate_limited": command_rate_limited,
                "torque_limited": torque_limited,
                "impedance_scale": impedance_scale,
                "kp_scale": kp_scale,
                "kd_scale": kd_scale,
                "torque_limit_effective": phase_torque_limit,
                "torque_limit_start": float(
                    self.policy_pd_torque_limit_start.get(joint_name, phase_torque_limit)
                ),
                "torque_limit_final": float(
                    self.policy_pd_torque_limit_final.get(joint_name, phase_torque_limit)
                ),
                "tau_pd_est": tau_pd_est,
                "offset": offset,
                "direction": direction,
                "p_des": p_des,
                "p_base": p_base,
                "p_limit_adjustment": p_des - p_base,
                "joint_v_des": joint_v_des,
                "joint_v_des_requested": joint_v_des_requested,
                "v_des": motor_v_des,
                "kp": kp,
                "kd": kd,
                "kp_effective": kp_effective,
                "kd_effective": kd_effective,
                "joint_tau_ff": joint_tau_ff,
                "joint_tau_ff_effective": joint_tau_ff_effective,
                "joint_limit_preload_error": joint_limit_preload_error,
                "joint_limit_preload_tau_ff_requested": joint_limit_preload_tau_ff_requested,
                "joint_limit_preload_tau_ff": joint_limit_preload_tau_ff,
                "tau_ff": motor_tau_ff,
                "can_id": can_id,
                "data": data,
            })

        return commands

    @staticmethod
    def _resolve_bus(buses, bus_name):
        """Return the CAN transport for bus_name, or the supplied single transport."""
        if not isinstance(buses, dict):
            return buses
        if bus_name not in buses:
            raise KeyError(f"Unknown CAN bus '{bus_name}'. Available buses: {sorted(buses)}")
        return buses[bus_name]

    @staticmethod
    def _feedback_comm_types(proto):
        if proto is None:
            return None
        return {
            int(proto.get("comm_type_feedback", 2)),
            int(proto.get("comm_type_active_feedback", 24)),
        }

    @staticmethod
    def read_all_frames(
        buses,
        timeout=0.0,
        expected_bus_motor_ids=None,
        proto=None,
        max_frames=256,
    ):
        """Read available frames from all buses and return them as a single combined list."""
        expected_by_bus = {}
        if expected_bus_motor_ids:
            for bus_name, motor_id in expected_bus_motor_ids:
                expected_by_bus.setdefault(str(bus_name), set()).add(int(motor_id) & 0xFF)
        feedback_comm_types = MotorCommandLayer._feedback_comm_types(proto)

        if not isinstance(buses, dict):
            expected_motor_ids = {
                motor_id
                for motor_ids in expected_by_bus.values()
                for motor_id in motor_ids
            }
            try:
                return buses.read_available_frames(
                    timeout=timeout,
                    max_frames=max_frames,
                    expected_motor_ids=expected_motor_ids or None,
                    feedback_comm_types=feedback_comm_types,
                )
            except TypeError:
                return buses.read_available_frames(timeout=timeout, max_frames=max_frames)

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
            try:
                bus_frames = bus.read_available_frames(
                    timeout=per_bus_timeout,
                    max_frames=max_frames,
                    expected_motor_ids=expected_by_bus.get(str(bus_name)) or None,
                    feedback_comm_types=feedback_comm_types,
                )
            except TypeError:
                bus_frames = bus.read_available_frames(
                    timeout=per_bus_timeout,
                    max_frames=max_frames,
                )
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

    def build_save_parameter_commands(self):
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
                    "comm_type_save_parameters",
                ),
                "data": bytes(8),
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

    def update_periodic_commands(self, buses, commands, period_s):
        """Update latest targets while SocketCAN repeats them in the kernel.

        This method runs inside the dedicated CAN owner thread. Keep both
        adapter updates in that one thread: dispatching each lane through a
        Python executor adds two GIL contenders at 200 Hz and can starve the
        50 Hz policy producer. SocketCAN BCM continues retransmitting the last
        complete snapshot while these lightweight task updates are applied.
        """
        commands = list(commands)
        grouped = {}
        for cmd in commands:
            bus = self._resolve_bus(buses, cmd.get("bus_name", "front"))
            grouped.setdefault(id(bus), {"bus": bus, "frames": []})["frames"].append(
                (cmd["can_id"], cmd["data"])
            )

        count = 0
        for group in grouped.values():
            bus = group["bus"]
            if not hasattr(bus, "update_periodic_sequence"):
                raise RuntimeError("CAN transport does not support kernel periodic commands")
            count += int(
                bus.update_periodic_sequence(group["frames"], period_s)
            )
        return count

    @staticmethod
    def stop_periodic_commands(buses):
        seen = set()
        values = buses.values() if isinstance(buses, dict) else (buses,)
        for bus in values:
            if id(bus) in seen:
                continue
            seen.add(id(bus))
            if hasattr(bus, "stop_periodic_sequence"):
                bus.stop_periodic_sequence()

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
                    frame_gap_s=(
                        self.frame_gap_s
                        if getattr(bus, "requires_frame_gap", True)
                        else 0.0
                    ),
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

        # The caller is already the dedicated 200 Hz CAN sender. Dispatching
        # each lane through another Python worker pool makes that sender wait
        # for two additional GIL-scheduled threads. Under the active 50 Hz
        # policy loop this added 6-13 ms stalls even though direct SocketCAN
        # submission for both six-motor lanes takes less than the 5 ms budget.
        # Submit each complete lane directly and deterministically instead.
        for key in group_order:
            group = grouped[key]
            for index, pkt in send_items(group["bus"], group["items"]):
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
        if (
            abs(float(cmd.get("kp_effective", cmd["kp"])) - float(cmd["kp"])) > 1e-3
            or abs(float(cmd.get("kd_effective", cmd["kd"])) - float(cmd["kd"])) > 1e-3
        ):
            line += (
                f" wire_kp={float(cmd['kp_effective']):.2f}"
                f" wire_kd={float(cmd['kd_effective']):.2f}"
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
