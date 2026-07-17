#!/usr/bin/env python3
"""MotorCommandLayer wrapper for nonlinear calf four-bar transmissions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from four_bar_transmission import FourBarTransmissionSet
from motor_command_layer import MotorCommandLayer, mit_can_id, pack_mit_command

ROOT = Path(__file__).resolve().parents[1]


class FourBarMotorCommandLayer(MotorCommandLayer):
    """Keep policy/safety logic in virtual joint space and convert at CAN I/O."""

    def __init__(
        self,
        policy_order,
        motor_ids,
        active_joints=None,
        joint_can_bus=None,
        transmission_config_path: Optional[str] = None,
    ):
        config_path = (
            Path(transmission_config_path)
            if transmission_config_path is not None
            else ROOT / "config" / "four_bar_transmission.yaml"
        )
        self.transmissions = FourBarTransmissionSet.from_yaml(
            config_path, policy_order=policy_order
        )
        super().__init__(
            policy_order,
            motor_ids,
            active_joints=active_joints,
            joint_can_bus=joint_can_bus,
        )
        self._validate_transmission_ranges()

    def _validate_transmission_ranges(self):
        """Verify every configured virtual limit has a measured motor mapping."""
        if not self.transmissions.enabled:
            return
        for joint_name in self.policy_order:
            if not self.transmissions.is_enabled(joint_name):
                continue
            hard_min, hard_max = self.hard_joint_limits[joint_name]
            policy_min, policy_max = self.policy_target_limits.get(
                joint_name, (hard_min, hard_max)
            )
            limits = [float(hard_min), float(hard_max)]
            if not self.transmissions.clamp_policy_to_hard_limits:
                limits.extend([float(policy_min), float(policy_max)])

            offset = float(self.joint_offsets[joint_name])
            direction = float(self.joint_directions[joint_name])
            raw_positions = []
            for q_value in limits:
                theta = self.transmissions.motor_from_virtual(joint_name, q_value)
                raw_positions.append(offset + direction * theta)

            p_min = min(raw_positions)
            p_max = max(raw_positions)
            if p_min < float(self.proto["p_min"]) or p_max > float(self.proto["p_max"]):
                raise ValueError(
                    f"{joint_name}: four-bar transformed raw motor range "
                    f"[{p_min:.6f}, {p_max:.6f}] exceeds MIT feedback range "
                    f"[{self.proto['p_min']}, {self.proto['p_max']}]"
                )

    def apply_hard_joint_limit(self, joint_name, q_des, phase=None):
        # The current repository has small policy-only calf overtravel beyond
        # the mechanical hard limit. With a real nonlinear linkage there is no
        # simulator stop to absorb that request, so the safe default is to use
        # the hard range for transmitted calves.
        mapped_phase = phase
        if (
            phase == "policy"
            and self.transmissions.is_enabled(joint_name)
            and self.transmissions.clamp_policy_to_hard_limits
        ):
            mapped_phase = None
        return super().apply_hard_joint_limit(
            joint_name, q_des, phase=mapped_phase
        )

    def apply_hard_joint_limit_to_motor_position(
        self, joint_name, p_des, offset, direction=1.0
    ):
        if not self.transmissions.is_enabled(joint_name):
            return super().apply_hard_joint_limit_to_motor_position(
                joint_name, p_des, offset, direction=direction
            )
        theta = float(direction) * (float(p_des) - float(offset))
        q_des = self.transmissions.virtual_from_motor(joint_name, theta)
        q_des = self.apply_hard_joint_limit(joint_name, q_des)
        theta = self.transmissions.motor_from_virtual(joint_name, q_des)
        return float(offset) + float(direction) * theta, q_des

    def decode_joint_feedback(
        self, joint_name, position_raw, velocity_raw, torque_raw
    ):
        return self.transmissions.decode_feedback(
            joint_name=joint_name,
            position_raw=position_raw,
            velocity_raw=velocity_raw,
            torque_raw=torque_raw,
            offset=float(self.joint_offsets[joint_name]),
            direction=float(self.joint_directions[joint_name]),
        )

    def set_software_zero_from_feedback(
        self, feedback_by_joint, active_joints=None, target_value=0.0
    ):
        """Recompute raw offsets while preserving the nonlinear calibration."""
        feedback_by_joint = feedback_by_joint or {}
        active_joints = self.active_joints if active_joints is None else active_joints
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
            raw_position = float(
                feedback.get("position_raw", feedback.get("position"))
            )
            direction = float(self.joint_directions[joint_name])
            target_motor_coordinate = self.transmissions.motor_from_virtual(
                joint_name, target_value
            )
            new_offsets[joint_name] = (
                raw_position - direction * target_motor_coordinate
            )
            new_shifts[joint_name] = target_value

        if missing:
            return {}, missing

        old_offsets = dict(self.joint_offsets)
        old_shifts = dict(self.joint_coordinate_shifts)
        updated = {}
        for joint_name, offset in new_offsets.items():
            self.joint_offsets[joint_name] = offset
            self.joint_coordinate_shifts[joint_name] = new_shifts[joint_name]
            updated[joint_name] = offset

        try:
            self.reload_joint_limits(force=True)
            self._validate_transmission_ranges()
        except Exception:
            self.joint_offsets = old_offsets
            self.joint_coordinate_shifts = old_shifts
            self.reload_joint_limits(force=True)
            raise
        return updated, missing

    def build_mit_commands(
        self,
        q_target,
        phase="policy",
        feedback_by_joint=None,
        joint_velocity_target=None,
    ):
        # Parent code performs virtual joint clipping, phase-specific gains,
        # virtual PD torque limiting, and command bookkeeping first.
        commands = super().build_mit_commands(
            q_target=q_target,
            phase=phase,
            feedback_by_joint=feedback_by_joint,
            joint_velocity_target=joint_velocity_target,
        )
        feedback_by_joint = feedback_by_joint or {}
        for command in commands:
            joint_name = command["joint_name"]
            command["transmission_enabled"] = self.transmissions.is_enabled(
                joint_name
            )
            if not command["transmission_enabled"]:
                command.setdefault("transmission_jacobian", 1.0)
                command.setdefault("transmission_efficiency", 1.0)
                continue
            self._rewrite_four_bar_command(
                command, feedback_by_joint.get(joint_name, {})
            )
        return commands

    def _rewrite_four_bar_command(self, command, feedback):
        joint_name = command["joint_name"]
        phase = command["phase"]
        command_proto = self.command_proto_for_phase(phase)
        offset = float(command["offset"])
        direction = float(command["direction"])

        q_virtual_des = float(command["q_des"])
        qd_virtual_des = float(command["joint_v_des"])
        tau_virtual_ff = float(command["joint_tau_ff"])
        kp_virtual = float(command["kp"])
        kd_virtual = float(command["kd"])

        feedback_ok = isinstance(feedback, dict) and all(
            key in feedback
            for key in (
                "joint_position",
                "motor_position",
                "motor_velocity",
            )
        )
        # Policy/leveling commands use the live motor-side Jacobian to estimate
        # and limit PD torque around the measured linkage state. Startup,
        # sit/stand and hold pose moves can still be mapped safely from the
        # calibrated virtual target when feedback is momentarily stale; this
        # avoids a bootstrapping deadlock where we cannot send the next MIT
        # command needed to refresh active feedback.
        feedback_required = (
            self.transmissions.require_feedback_for_commands
            and phase in ("policy", "leveling")
        )
        if feedback_required and not feedback_ok:
            raise RuntimeError(
                f"{joint_name}: nonlinear four-bar command requires fresh "
                "motor feedback"
            )

        q_reference = (
            float(feedback["joint_position"])
            if feedback_ok
            else q_virtual_des
        )
        theta_reference = (
            float(feedback["motor_position"])
            if feedback_ok
            else self.transmissions.motor_from_virtual(
                joint_name, q_reference
            )
        )
        theta_dot_feedback = (
            float(feedback["motor_velocity"]) if feedback_ok else 0.0
        )
        jacobian_reference = self.transmissions.jacobian_from_motor(
            joint_name, theta_reference
        )

        theta_des = self.transmissions.motor_from_virtual(
            joint_name, q_virtual_des
        )
        theta_dot_des = qd_virtual_des / jacobian_reference
        kp_motor, kd_motor = self.transmissions.motor_gains_from_virtual(
            joint_name,
            theta_reference,
            kp_virtual,
            kd_virtual,
        )
        tau_motor_ff = self.transmissions.motor_torque_from_virtual(
            joint_name,
            theta_reference,
            tau_virtual_ff,
        )

        kp_motor_effective = self._effective_unsigned_wire_value(
            kp_motor, "kp", command_proto
        )
        kd_motor_effective = self._effective_unsigned_wire_value(
            kd_motor, "kd", command_proto
        )
        tau_motor_ff_effective = self._effective_signed_wire_value(
            tau_motor_ff, "tau", command_proto
        )

        protocol_torque_limit = max(
            abs(float(self.proto["tau_min"])),
            abs(float(self.proto["tau_max"])),
        )
        motor_torque_limit = min(
            self.transmissions.motor_torque_limit(joint_name),
            protocol_torque_limit,
        )
        motor_torque_limited = False
        tau_motor_pd_est = None

        if feedback_ok and motor_torque_limit > 0.0:
            velocity_and_ff_torque = (
                kd_motor_effective
                * (theta_dot_des - theta_dot_feedback)
                + tau_motor_ff_effective
            )
            position_torque_requested = (
                kp_motor_effective * (theta_des - theta_reference)
            )
            position_torque = float(
                np.clip(
                    position_torque_requested,
                    -motor_torque_limit - velocity_and_ff_torque,
                    motor_torque_limit - velocity_and_ff_torque,
                )
            )
            if kp_motor_effective > 0.0:
                theta_des = theta_reference + position_torque / kp_motor_effective
            tau_motor_pd_est = (
                kp_motor_effective * (theta_des - theta_reference)
                + velocity_and_ff_torque
            )
            if abs(tau_motor_pd_est) > motor_torque_limit and kd_motor_effective > 0.0:
                target_torque = float(
                    np.clip(
                        tau_motor_pd_est,
                        -motor_torque_limit,
                        motor_torque_limit,
                    )
                )
                theta_dot_des = theta_dot_feedback + (
                    target_torque
                    - kp_motor_effective * (theta_des - theta_reference)
                    - tau_motor_ff_effective
                ) / kd_motor_effective
                tau_motor_pd_est = (
                    kp_motor_effective * (theta_des - theta_reference)
                    + kd_motor_effective
                    * (theta_dot_des - theta_dot_feedback)
                    + tau_motor_ff_effective
                )
            motor_torque_limited = bool(
                abs(position_torque - position_torque_requested) > 1.0e-7
                or abs(tau_motor_pd_est) > motor_torque_limit + 1.0e-6
            )

        p_base = offset + direction * self.transmissions.motor_from_virtual(
            joint_name, q_virtual_des
        )
        p_des = offset + direction * theta_des
        v_des = direction * theta_dot_des
        tau_ff = direction * tau_motor_ff
        p_des, v_des, kp_motor, kd_motor, tau_ff = self.apply_mit_parameter_limits(
            p_des=p_des,
            v_des=v_des,
            kp=kp_motor,
            kd=kd_motor,
            tau_ff=tau_ff,
        )

        theta_sent = direction * (p_des - offset)
        theta_dot_sent = direction * v_des
        tau_motor_ff_sent = direction * tau_ff
        q_sent = self.transmissions.virtual_from_motor(joint_name, theta_sent)
        jacobian_sent = self.transmissions.jacobian_from_motor(
            joint_name, theta_sent
        )
        qd_virtual_sent = jacobian_sent * theta_dot_sent

        kp_effective_sent = self._effective_unsigned_wire_value(
            kp_motor, "kp", command_proto
        )
        kd_effective_sent = self._effective_unsigned_wire_value(
            kd_motor, "kd", command_proto
        )
        tau_motor_ff_effective_sent = self._effective_signed_wire_value(
            tau_motor_ff_sent, "tau", command_proto
        )
        tau_virtual_ff_effective = self.transmissions.virtual_torque_from_motor(
            joint_name,
            theta_reference,
            tau_motor_ff_effective_sent,
        )

        if feedback_ok:
            tau_motor_pd_est = (
                kp_effective_sent * (theta_sent - theta_reference)
                + kd_effective_sent * (theta_dot_sent - theta_dot_feedback)
                + tau_motor_ff_effective_sent
            )
            tau_virtual_pd_est = self.transmissions.virtual_torque_from_motor(
                joint_name,
                theta_reference,
                tau_motor_pd_est,
            )
        else:
            tau_virtual_pd_est = None

        can_id = mit_can_id(
            int(command["motor_id"]), command_proto, tau_ff=tau_ff
        )
        data = pack_mit_command(
            p_des=p_des,
            v_des=v_des,
            kp=kp_motor,
            kd=kd_motor,
            proto=command_proto,
        )

        command.update(
            {
                "q_before_motor_torque_limit": q_virtual_des,
                "q_des": q_sent,
                "p_des": p_des,
                "p_base": p_base,
                "p_limit_adjustment": p_des - p_base,
                "joint_v_des": qd_virtual_sent,
                "v_des": v_des,
                "kp_virtual": kp_virtual,
                "kd_virtual": kd_virtual,
                "kp": kp_motor,
                "kd": kd_motor,
                "kp_effective": kp_effective_sent,
                "kd_effective": kd_effective_sent,
                "joint_tau_ff": tau_virtual_ff,
                "joint_tau_ff_effective": tau_virtual_ff_effective,
                "tau_ff": tau_ff,
                "tau_pd_est": tau_virtual_pd_est,
                "tau_motor_pd_est": tau_motor_pd_est,
                "motor_torque_limit_nm": motor_torque_limit,
                "motor_torque_limited": motor_torque_limited,
                "torque_limited": bool(
                    command.get("torque_limited", False)
                    or motor_torque_limited
                ),
                "motor_position_des": theta_sent,
                "motor_velocity_des": theta_dot_sent,
                "transmission_jacobian": jacobian_sent,
                "transmission_efficiency": self.transmissions.efficiency(
                    joint_name
                ),
                "can_id": can_id,
                "data": data,
            }
        )
