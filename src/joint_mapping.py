#!/usr/bin/env python3
"""Authoritative policy, estimator, hardware, and CAN joint mapping."""

from dataclasses import dataclass


POLICY_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "BL_hip_joint",
    "BR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "BL_thigh_joint",
    "BR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "BL_calf_joint",
    "BR_calf_joint",
)


@dataclass(frozen=True)
class JointRoute:
    policy_index: int
    policy_joint_name: str
    estimator_index: int
    hardware_index: int
    motor_id: int
    motor_direction: float
    encoder_offset: float
    can_interface: str


class AuthoritativeJointMapping:
    """Name-based mapping that never depends on YAML/dict insertion order."""

    def __init__(
        self,
        motor_ids,
        motor_directions,
        encoder_offsets,
        joint_can_bus,
        estimator_order=None,
        policy_order=None,
    ):
        self.policy_order = tuple(policy_order or POLICY_JOINT_ORDER)
        if self.policy_order != POLICY_JOINT_ORDER:
            raise ValueError(
                "Policy joint order differs from the verified actor contract: "
                f"{list(self.policy_order)}"
            )
        self.estimator_order = tuple(estimator_order or self.policy_order)
        self._validate_complete_order(self.estimator_order, "estimator")

        required = set(self.policy_order)
        for label, values in (
            ("motor IDs", motor_ids),
            ("motor directions", motor_directions),
            ("encoder offsets", encoder_offsets),
            ("CAN routing", joint_can_bus),
        ):
            missing = sorted(required - set(values))
            if missing:
                raise KeyError(f"Missing {label} for: " + ", ".join(missing))

        self.policy_index_by_joint = {
            name: index for index, name in enumerate(self.policy_order)
        }
        self.estimator_index_by_joint = {
            name: index for index, name in enumerate(self.estimator_order)
        }
        self.hardware_order = tuple(
            sorted(self.policy_order, key=lambda name: (int(motor_ids[name]), name))
        )
        self.hardware_index_by_joint = {
            name: index for index, name in enumerate(self.hardware_order)
        }
        self.routes = tuple(
            JointRoute(
                policy_index=self.policy_index_by_joint[name],
                policy_joint_name=name,
                estimator_index=self.estimator_index_by_joint[name],
                hardware_index=self.hardware_index_by_joint[name],
                motor_id=int(motor_ids[name]),
                motor_direction=float(motor_directions[name]),
                encoder_offset=float(encoder_offsets[name]),
                can_interface=str(joint_can_bus[name]),
            )
            for name in self.policy_order
        )
        self._validate_routes()

    def _validate_complete_order(self, order, label):
        if len(order) != len(POLICY_JOINT_ORDER) or set(order) != set(POLICY_JOINT_ORDER):
            raise ValueError(
                f"{label} order must contain every policy joint exactly once"
            )

    def _validate_routes(self):
        if len({route.motor_id for route in self.routes}) != len(self.routes):
            raise ValueError("Motor IDs must be unique for the active one-CAN mapping")
        invalid = [
            route.policy_joint_name
            for route in self.routes
            if route.motor_direction not in (-1.0, 1.0)
        ]
        if invalid:
            raise ValueError("Motor directions must be +1 or -1 for: " + ", ".join(invalid))

    def policy_to_estimator(self, values):
        return self._reorder(values, self.policy_order, self.estimator_order)

    def estimator_to_policy(self, values):
        return self._reorder(values, self.estimator_order, self.policy_order)

    def policy_to_hardware(self, values):
        return self._reorder(values, self.policy_order, self.hardware_order)

    def hardware_to_policy(self, values):
        return self._reorder(values, self.hardware_order, self.policy_order)

    @staticmethod
    def _reorder(values, source_order, target_order):
        import numpy as np

        array = np.asarray(values)
        if array.shape[-1] != len(source_order):
            raise ValueError(
                f"Joint vector has trailing dimension {array.shape[-1]}; "
                f"expected {len(source_order)}"
            )
        source_index = {name: index for index, name in enumerate(source_order)}
        return np.take(array, [source_index[name] for name in target_order], axis=-1)

    def startup_table_lines(self):
        header = (
            "policy_index policy_joint_name estimator_index hardware_index "
            "motor_id motor_direction encoder_offset CAN_interface"
        )
        lines = [header]
        for route in self.routes:
            lines.append(
                f"{route.policy_index:>12d} "
                f"{route.policy_joint_name:<18s} "
                f"{route.estimator_index:>15d} "
                f"{route.hardware_index:>14d} "
                f"0x{route.motor_id:02X} "
                f"{route.motor_direction:+.0f} "
                f"{route.encoder_offset:+.6f} "
                f"{route.can_interface}"
            )
        return lines


def hardware_policy_position(raw_motor_position, motor_direction, encoder_offset):
    return float(motor_direction) * (
        float(raw_motor_position) - float(encoder_offset)
    )


def observation_joint_position(hardware_policy_q, training_q_default):
    return float(hardware_policy_q) - float(training_q_default)
