#!/usr/bin/env python3
import time
import numpy as np

from motor_command_layer import (
    MotorCommandLayer,
    decode_mit_feedback_frame,
    motor_position_to_joint_angle,
)   


class FakeStateEstimator:
    """
    Dry-run estimator.

    Replace this later with:
      - q_current from RobStride encoder feedback
      - qd_current from RobStride velocity feedback
      - base_ang_vel_b from IMU gyro
      - projected_gravity_b from IMU orientation
      - base_lin_vel_b from velocity estimator
    """

    def __init__(
        self,
        q_initial,
        imu_sensor=None,
        dry_follow_alpha=0.10,
        dry_max_joint_velocity=10.0,
    ):
        # Policy-facing joint state is always expressed in deployed joint
        # radians. Raw MIT motor values are kept only in feedback dictionaries.
        self.joint_state_units = "joint_radians"
        self.q_current = np.asarray(q_initial, dtype=np.float32).copy()
        self.qd_current = np.zeros_like(self.q_current, dtype=np.float32)
        self.dry_follow_alpha = float(np.clip(dry_follow_alpha, 0.0, 1.0))
        self.dry_max_joint_velocity = max(0.0, float(dry_max_joint_velocity))

        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.imu_sensor = imu_sensor
        self.last_imu_timestamp = None
        self.last_imu_update_time = None
        self.last_imu_reading = None

    def refresh_imu(self):
        if self.imu_sensor is None:
            return False

        reading = self.imu_sensor.read()
        if reading is None:
            return False

        base_ang_vel_b = np.asarray(reading.base_ang_vel_b, dtype=np.float32)
        projected_gravity_b = np.asarray(
            reading.projected_gravity_b,
            dtype=np.float32,
        )
        if base_ang_vel_b.shape != (3,) or not np.all(np.isfinite(base_ang_vel_b)):
            raise ValueError(f"Invalid IMU angular velocity: {base_ang_vel_b}")
        if projected_gravity_b.shape != (3,) or not np.all(np.isfinite(projected_gravity_b)):
            raise ValueError(f"Invalid IMU projected gravity: {projected_gravity_b}")
        gravity_norm = float(np.linalg.norm(projected_gravity_b))
        if gravity_norm < 1e-6:
            raise ValueError("Invalid IMU projected gravity: zero-length vector")

        # The deployed policy is run with base linear velocity fixed at zero.
        # Keep this observation term identical even when an IMU helper can
        # estimate velocity from acceleration.
        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = base_ang_vel_b.copy()
        self.projected_gravity_b = (
            projected_gravity_b / gravity_norm
        ).astype(np.float32)
        self.last_imu_timestamp = float(reading.timestamp)
        # Use the sensor packet timestamp, not the time at which cached data was
        # read. Otherwise an unplugged Xsens can appear live forever.
        self.last_imu_update_time = float(reading.timestamp)
        self.last_imu_reading = reading
        return True

    def imu_required(self):
        return bool(getattr(self.imu_sensor, "required", False))

    def imu_age(self):
        if self.last_imu_update_time is None:
            return None
        return time.monotonic() - self.last_imu_update_time

    def imu_stale(self):
        if not self.imu_required():
            return False
        age = self.imu_age()
        stale_timeout = float(getattr(self.imu_sensor, "stale_timeout", 0.25))
        return age is None or age > stale_timeout

    def imu_status(self):
        if self.imu_sensor is None:
            return "none"
        if not self.imu_required():
            return getattr(self.imu_sensor, "source_name", "imu")
        age = self.imu_age()
        if age is None:
            return f"{getattr(self.imu_sensor, 'source_name', 'imu')}:missing"
        if self.imu_stale():
            return f"{getattr(self.imu_sensor, 'source_name', 'imu')}:stale"
        return f"{getattr(self.imu_sensor, 'source_name', 'imu')}:live"

    def read(self):
        self.refresh_imu()
        return (
            self.q_current.copy(),
            self.qd_current.copy(),
            self.base_lin_vel_b.copy(),
            self.base_ang_vel_b.copy(),
            self.projected_gravity_b.copy(),
        )

    def read_policy_joint_state(self):
        """Return only converted joint-space position/velocity for policy input."""
        if self.joint_state_units != "joint_radians":
            raise RuntimeError(
                "Policy joint state must be converted joint radians, not raw motor radians"
            )
        if not np.all(np.isfinite(self.q_current)) or not np.all(np.isfinite(self.qd_current)):
            raise ValueError("Policy joint state contains NaN or infinite values")
        return self.q_current.copy(), self.qd_current.copy()

    def dry_update_as_if_robot_followed(self, q_target, dt):
        q_target = np.asarray(q_target, dtype=np.float32)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dry-run update dt must be > 0")

        # A position-controlled motor cannot teleport to each new target in one
        # 20 ms cycle. Doing that made fake feedback report enormous qd values,
        # which destabilized the recurrent observation even though real encoder
        # feedback was slow. Use a bounded first-order response for dry runs.
        dq = self.dry_follow_alpha * (q_target - self.q_current)
        if self.dry_max_joint_velocity > 0.0:
            max_step = self.dry_max_joint_velocity * dt
            dq = np.clip(dq, -max_step, max_step)
        self.qd_current = (dq / dt).astype(np.float32)
        self.q_current = (self.q_current + dq).astype(np.float32)


class MitFeedbackStateEstimator(FakeStateEstimator):
    """
    State estimator backed by RobStride/CyberGear MIT feedback frames.

    Motor feedback position is converted back into policy joint coordinates by
    subtracting the configured joint offset used when commands are sent.
    """

    def __init__(
        self,
        q_initial,
        policy_order,
        motor_ids,
        motor_layer,
        bus,
        imu_sensor=None,
        pose_references=None,
        pose_snap_tolerance=0.0,
    ):
        super().__init__(q_initial=q_initial, imu_sensor=imu_sensor)

        self.policy_order = list(policy_order)
        self.motor_layer = motor_layer
        self.bus = bus

        self.joint_index_by_name = {
            name: index for index, name in enumerate(self.policy_order)
        }
        self.pose_references_by_joint = self._build_pose_references(pose_references)
        self.pose_reference_arrays = self._build_pose_reference_arrays(pose_references)
        self.pose_snap_tolerance = max(0.0, float(pose_snap_tolerance))
        self.joint_name_by_bus_motor_id = {
            (self.motor_layer.joint_can_bus.get(joint_name, "front"), int(motor_id)): joint_name
            for joint_name, motor_id in motor_ids.items()
        }

        self.last_feedback_by_joint = {}
        self.last_feedback_count = 0
        self.position_branch_offsets = {}

    def _pose_reference_iterable(self, pose_references):
        if pose_references is None:
            return []
        if isinstance(pose_references, dict):
            return list(pose_references.values())
        return list(pose_references)

    def _build_pose_references(self, pose_references):
        refs_by_joint = {joint_name: [] for joint_name in self.policy_order}
        for pose in self._pose_reference_iterable(pose_references):
            pose = np.asarray(pose, dtype=np.float32)
            if pose.shape[0] != len(self.policy_order):
                continue
            for index, joint_name in enumerate(self.policy_order):
                value = float(pose[index])
                if np.isfinite(value):
                    refs_by_joint[joint_name].append(value)
        return refs_by_joint

    def _build_pose_reference_arrays(self, pose_references):
        arrays = []
        for pose in self._pose_reference_iterable(pose_references):
            pose = np.asarray(pose, dtype=np.float32)
            if pose.shape[0] == len(self.policy_order):
                arrays.append(pose.copy())
        return arrays

    def _poses_differ_on_feedback(self, pose_a, pose_b, feedback_items):
        for _, index, _ in feedback_items:
            if abs(float(pose_a[index]) - float(pose_b[index])) > 1e-5:
                return True
        return False

    def infer_initial_pose_reference(self, feedback_items):
        return None

    def update_from_frames(self, frames):
        feedback_items = []

        for frame in frames:
            feedback = decode_mit_feedback_frame(
                frame.can_id,
                frame.data,
                self.motor_layer.proto,
            )
            if feedback is None:
                continue

            motor_id = int(feedback["motor_id"])
            bus_name = getattr(frame, "bus_name", None)
            if bus_name is None:
                continue
            joint_name = self.joint_name_by_bus_motor_id.get((bus_name, motor_id))
            if joint_name is None:
                continue

            index = self.joint_index_by_name[joint_name]
            feedback_items.append((
                joint_name,
                index,
                feedback,
                float(getattr(frame, "timestamp", time.monotonic())),
                bus_name,
            ))

        count = 0
        for joint_name, index, feedback, timestamp, bus_name in feedback_items:
            offset = float(self.motor_layer.joint_offsets[joint_name])
            direction = float(self.motor_layer.joint_directions[joint_name])
            raw_position = float(feedback["position"])
            # This is the sole raw-position -> policy joint-position boundary.
            # Apply offset, mounting direction, and 2*pi normalization before
            # q_current can be observed by the policy.
            q_joint = motor_position_to_joint_angle(
                feedback["position"],
                offset=offset,
                direction=direction,
            )
            position_unwrapped = offset + direction * q_joint
            self.position_branch_offsets[joint_name] = raw_position - position_unwrapped

            self.q_current[index] = q_joint
            velocity_raw = float(feedback["velocity"])
            torque_raw = float(feedback["torque"])
            self.qd_current[index] = direction * velocity_raw
            feedback = dict(feedback)
            feedback["timestamp"] = timestamp
            feedback["bus_name"] = bus_name
            feedback["position_raw"] = float(feedback["position"])
            feedback["velocity_raw"] = velocity_raw
            feedback["torque_raw"] = torque_raw
            feedback["joint_position"] = q_joint
            feedback["joint_velocity"] = direction * velocity_raw
            feedback["joint_torque"] = direction * torque_raw
            feedback["velocity"] = direction * velocity_raw
            feedback["torque"] = direction * torque_raw
            feedback["position_unwrapped"] = position_unwrapped
            feedback["position_branch_offset"] = float(self.position_branch_offsets[joint_name])
            feedback["joint_direction"] = direction
            self.last_feedback_by_joint[joint_name] = feedback
            count += 1

        self.last_feedback_count = count
        return count

    def refresh_from_bus(self, timeout=0.0):
        frames = MotorCommandLayer.read_all_frames(self.bus, timeout=timeout)
        return self.update_from_frames(frames)

    def read(self):
        self.refresh_from_bus(timeout=0.0)
        return super().read()

    def dry_update_as_if_robot_followed(self, q_target, dt):
        # Live feedback owns q_current/qd_current. Keep the method for the
        # controller loop interface, but do not overwrite measured state.
        return None

    def has_all_joint_feedback(self):
        return len(self.last_feedback_by_joint) >= len(self.policy_order)

    def apply_software_zero(self, active_joints=None, target_value=0.0):
        target_value = float(target_value)
        updated, missing = self.motor_layer.set_software_zero_from_feedback(
            self.last_feedback_by_joint,
            active_joints=active_joints,
            target_value=target_value,
        )
        for joint_name in updated:
            index = self.joint_index_by_name.get(joint_name)
            if index is not None:
                self.q_current[index] = target_value
                self.qd_current[index] = 0.0
            feedback = self.last_feedback_by_joint.get(joint_name)
            if feedback is not None:
                feedback["joint_position"] = target_value
                feedback["position_branch_offset"] = 0.0
                feedback["position_unwrapped"] = float(feedback["position"])
                feedback["velocity"] = 0.0
                feedback["joint_velocity"] = 0.0
        return updated, missing
