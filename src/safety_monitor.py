#!/usr/bin/env python3
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


class SafetyMonitor:
    def __init__(self, policy_order):
        self.root = ROOT
        self.policy_order = policy_order
        self.limit_path = self.root / "config" / "joint_limits.yaml"
        self.control_limit_path = self.root / "config" / "control_limits.yaml"
        self.safety_limit_path = self.root / "config" / "safety_limits.yaml"
        self.limit_mtime_ns = None
        self.control_limit_mtime_ns = None
        self.safety_limit_mtime_ns = None

        self.q_min = None
        self.q_max = None
        self.dq_max = None
        self.joint_position_enabled = True
        self.joint_rate_enabled = True
        self.encoder_sanity_enabled = True
        self.require_feedback_for_motion = True
        self.max_abs_encoder_position_rad = 3.5
        self.max_feedback_age_s = 0.25
        self.encoder_joint_limit_margin_rad = 0.75
        self.encoder_report_max_joints = 4
        self.projected_gravity_gz_min = -0.75
        self.max_body_ang_vel_norm = 8.0
        self.reload_joint_limits(force=True)
        self.reload_control_limits(force=True)
        self.reload_safety_limits(force=True)

    def _dict_to_array(self, d):
        return np.array([d[name] for name in self.policy_order], dtype=np.float32)

    def reload_joint_limits(self, force=False):
        mtime_ns = self.limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.limit_mtime_ns:
            return False

        cfg = load_yaml(self.limit_path)
        limits = cfg["joint_limits"]

        q_min = []
        q_max = []
        dq_max = []

        for joint_name in self.policy_order:
            if joint_name not in limits:
                raise KeyError(f"Missing joint limit for {joint_name} in {self.limit_path}")

            joint_limit = limits[joint_name]
            q_lo = float(joint_limit["min"])
            q_hi = float(joint_limit["max"])
            dq_step = float(joint_limit["dq_max_per_step"])

            if q_lo > q_hi:
                raise ValueError(f"{joint_name}: min {q_lo} is greater than max {q_hi}")
            if dq_step < 0.0:
                raise ValueError(f"{joint_name}: dq_max_per_step must be >= 0")

            q_min.append(q_lo)
            q_max.append(q_hi)
            dq_max.append(dq_step)

        self.q_min = np.asarray(q_min, dtype=np.float32)
        self.q_max = np.asarray(q_max, dtype=np.float32)
        self.dq_max = np.asarray(dq_max, dtype=np.float32)
        self.limit_mtime_ns = mtime_ns
        return True

    def reload_control_limits(self, force=False):
        mtime_ns = self.control_limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.control_limit_mtime_ns:
            return False

        cfg = load_yaml(self.control_limit_path)
        self.joint_position_enabled = bool(
            cfg.get("joint_position", {}).get("enabled", True)
        )
        self.joint_rate_enabled = bool(
            cfg.get("joint_rate", {}).get("enabled", True)
        )
        self.control_limit_mtime_ns = mtime_ns
        return True

    def reload_safety_limits(self, force=False):
        mtime_ns = self.safety_limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.safety_limit_mtime_ns:
            return False

        cfg = load_yaml(self.safety_limit_path)
        emergency = cfg["emergency"]
        encoder = cfg.get("encoder", {})

        self.projected_gravity_gz_min = float(emergency["projected_gravity_gz_min"])
        self.max_body_ang_vel_norm = float(emergency["max_body_ang_vel_norm"])

        self.encoder_sanity_enabled = bool(encoder.get("enabled", True))
        self.require_feedback_for_motion = bool(
            encoder.get("require_feedback_for_motion", True)
        )
        self.max_abs_encoder_position_rad = float(
            encoder.get("max_abs_position_rad", 3.5)
        )
        self.max_feedback_age_s = float(
            encoder.get("max_feedback_age_s", 0.25)
        )
        self.encoder_joint_limit_margin_rad = float(
            encoder.get("joint_limit_margin_rad", 0.75)
        )
        self.encoder_report_max_joints = int(encoder.get("report_max_joints", 4))

        if self.max_abs_encoder_position_rad <= 0.0:
            raise ValueError("encoder.max_abs_position_rad must be > 0")
        if self.max_feedback_age_s <= 0.0:
            raise ValueError("encoder.max_feedback_age_s must be > 0")
        if self.encoder_joint_limit_margin_rad < 0.0:
            raise ValueError("encoder.joint_limit_margin_rad must be >= 0")
        if self.encoder_report_max_joints < 1:
            raise ValueError("encoder.report_max_joints must be >= 1")

        self.safety_limit_mtime_ns = mtime_ns
        return True

    def clip_q_target(self, q_target):
        q_target = np.asarray(q_target, dtype=np.float32)
        if not self.joint_position_enabled:
            return q_target

        return np.clip(q_target, self.q_min, self.q_max)

    def rate_limit_q_target(self, q_desired, q_previous):
        q_desired = np.asarray(q_desired, dtype=np.float32)
        q_previous = np.asarray(q_previous, dtype=np.float32)
        if not self.joint_rate_enabled:
            return q_desired

        dq = q_desired - q_previous
        dq = np.clip(dq, -self.dq_max, self.dq_max)
        return q_previous + dq

    def safety_filter(self, q_policy_target, q_previous_target, apply_rate_limit=True):
        self.reload_control_limits()
        self.reload_joint_limits()

        q_previous_target = np.asarray(q_previous_target, dtype=np.float32)

        q = self.clip_q_target(q_policy_target)
        if apply_rate_limit:
            q = self.rate_limit_q_target(q, q_previous_target)
        q = self.clip_q_target(q)
        return q.astype(np.float32)

    def emergency_stop_check(self, projected_gravity_b, base_ang_vel_b):
        self.reload_safety_limits()

        projected_gravity_b = np.asarray(projected_gravity_b, dtype=np.float32)
        base_ang_vel_b = np.asarray(base_ang_vel_b, dtype=np.float32)

        if projected_gravity_b[2] > self.projected_gravity_gz_min:
            return True, f"bad tilt: projected_gravity={projected_gravity_b}"

        if np.linalg.norm(base_ang_vel_b) > self.max_body_ang_vel_norm:
            return True, f"high body angular velocity: {base_ang_vel_b}"

        return False, ""

    def encoder_sanity_check(
        self,
        q_current,
        active_joints=None,
        feedback_by_joint=None,
        require_feedback=False,
    ):
        """
        Stop motion when measured encoder angles are clearly impossible/unsafe.

        q_current must be in deployed joint coordinates, i.e. motor encoder
        position after subtracting the configured joint offset.
        """
        self.reload_safety_limits()
        self.reload_joint_limits()

        if not self.encoder_sanity_enabled:
            return False, ""

        q_current = np.asarray(q_current, dtype=np.float32)
        if q_current.shape[0] != len(self.policy_order):
            return (
                True,
                "ABNORMAL ENCODER ANGLE: feedback vector has "
                f"{q_current.shape[0]} joints, expected {len(self.policy_order)}",
            )

        active_joints = list(active_joints or self.policy_order)
        active_indices = []
        for joint_name in active_joints:
            if joint_name not in self.policy_order:
                return True, f"ABNORMAL ENCODER ANGLE: unknown active joint {joint_name}"
            active_indices.append((joint_name, self.policy_order.index(joint_name)))

        feedback_names = set(feedback_by_joint or {})
        require_feedback = bool(require_feedback and self.require_feedback_for_motion)
        if require_feedback:
            missing = [name for name, _ in active_indices if name not in feedback_names]
            if missing:
                shown = ", ".join(missing[:self.encoder_report_max_joints])
                if len(missing) > self.encoder_report_max_joints:
                    shown += f", +{len(missing) - self.encoder_report_max_joints} more"
                return (
                    True,
                    "ABNORMAL ENCODER ANGLE: missing MIT encoder feedback before motion "
                    f"for active joint(s): {shown}",
                )

            now = time.monotonic()
            stale = []
            for name, _ in active_indices:
                feedback = (feedback_by_joint or {}).get(name, {})
                timestamp = feedback.get("timestamp") if isinstance(feedback, dict) else None
                try:
                    age = now - float(timestamp)
                except (TypeError, ValueError):
                    stale.append(f"{name}=no timestamp")
                    continue
                if not np.isfinite(age) or age > self.max_feedback_age_s:
                    stale.append(f"{name} age={age:.3f}s")
            if stale:
                shown = ", ".join(stale[:self.encoder_report_max_joints])
                if len(stale) > self.encoder_report_max_joints:
                    shown += f", +{len(stale) - self.encoder_report_max_joints} more"
                return (
                    True,
                    "ABNORMAL ENCODER ANGLE: stale MIT encoder feedback before motion "
                    f"for active joint(s): {shown}",
                )

        violations = []
        margin = self.encoder_joint_limit_margin_rad
        for joint_name, index in active_indices:
            if feedback_by_joint is not None and joint_name not in feedback_names:
                continue

            q = float(q_current[index])
            q_min = float(self.q_min[index]) - margin
            q_max = float(self.q_max[index]) + margin

            if not np.isfinite(q):
                violations.append(f"{joint_name}=non-finite")
                continue

            reasons = []
            if abs(q) > self.max_abs_encoder_position_rad:
                reasons.append(f"|q|>{self.max_abs_encoder_position_rad:.3f}")
            if q < q_min or q > q_max:
                reasons.append(f"outside [{q_min:+.3f}, {q_max:+.3f}]")

            if reasons:
                violations.append(
                    f"{joint_name}={q:+.3f} rad ({np.degrees(q):+.1f} deg; "
                    + ", ".join(reasons)
                    + ")"
                )

        if not violations:
            return False, ""

        shown = "; ".join(violations[:self.encoder_report_max_joints])
        if len(violations) > self.encoder_report_max_joints:
            shown += f"; +{len(violations) - self.encoder_report_max_joints} more"
        return (
            True,
            "ABNORMAL ENCODER ANGLE: "
            + shown
            + ". Motor command blocked; set zero/check encoder before sit, stand, or walk.",
        )


if __name__ == "__main__":
    from policy_runner import PolicyRunner

    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order)

    print("Safety limits:")
    for i, name in enumerate(runner.policy_order):
        print(
            f"{i:02d} {name:16s} "
            f"min={safety.q_min[i]: .3f} "
            f"max={safety.q_max[i]: .3f} "
            f"dq_step={safety.dq_max[i]: .3f}"
        )
