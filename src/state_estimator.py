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
      - base_lin_vel_b fixed to [0, 0, 0] for the 48-slot policy contract
    """

    def __init__(self, q_initial, imu_sensor=None, imu_filter_cfg=None):
        self.q_current = np.asarray(q_initial, dtype=np.float32).copy()
        self.qd_current = np.zeros_like(self.q_current, dtype=np.float32)

        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.imu_sensor = imu_sensor
        self.imu_filter_cfg = dict(imu_filter_cfg or {})
        self._filtered_base_ang_vel_b = None
        self.last_imu_timestamp = None
        self.last_imu_update_time = None
        self.last_imu_reading = None

    def _as_vector3(self, value, default):
        arr = np.asarray(value if value is not None else default, dtype=np.float32)
        if arr.shape == ():
            arr = np.repeat(arr, 3).astype(np.float32)
        arr = arr.reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"IMU policy filter value must have 3 entries, got {arr.shape}")
        return arr

    def _filter_policy_gyro(self, gyro):
        gyro = np.asarray(gyro, dtype=np.float32).reshape(3)
        cfg = self.imu_filter_cfg
        if not bool(cfg.get("enabled", False)):
            self._filtered_base_ang_vel_b = gyro.copy()
            return gyro

        clip_abs = self._as_vector3(cfg.get("gyro_clip_abs", 0.0), [0.0, 0.0, 0.0])
        if np.any(clip_abs > 0.0):
            lower = np.where(clip_abs > 0.0, -clip_abs, -np.inf)
            upper = np.where(clip_abs > 0.0, clip_abs, np.inf)
            gyro = np.clip(gyro, lower, upper)

        alpha = float(np.clip(cfg.get("gyro_lowpass_alpha", 1.0), 0.0, 1.0))
        if self._filtered_base_ang_vel_b is None or alpha >= 1.0:
            filtered = gyro.copy()
        elif alpha <= 0.0:
            filtered = self._filtered_base_ang_vel_b.copy()
        else:
            filtered = (
                alpha * gyro
                + (1.0 - alpha) * self._filtered_base_ang_vel_b
            ).astype(np.float32)
        self._filtered_base_ang_vel_b = filtered.copy()
        return filtered

    def refresh_imu(self):
        if self.imu_sensor is None:
            return False

        reading = self.imu_sensor.read()
        if reading is None:
            return False

        # The deployed policy is run with base linear velocity fixed at zero.
        # Keep this observation term identical for every IMU source.
        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = self._filter_policy_gyro(reading.base_ang_vel_b)
        self.projected_gravity_b = np.asarray(reading.projected_gravity_b, dtype=np.float32).copy()
        self.last_imu_timestamp = float(reading.timestamp)
        self.last_imu_update_time = time.monotonic()
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

    def read_cached(self, refresh_imu=True):
        if refresh_imu:
            self.refresh_imu()
        return (
            self.q_current.copy(),
            self.qd_current.copy(),
            self.base_lin_vel_b.copy(),
            self.base_ang_vel_b.copy(),
            self.projected_gravity_b.copy(),
        )

    def dry_update_as_if_robot_followed(self, q_target, dt):
        q_target = np.asarray(q_target, dtype=np.float32)
        self.qd_current = (q_target - self.q_current) / float(dt)
        self.q_current = q_target.copy()


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
        imu_filter_cfg=None,
        pose_references=None,
        pose_snap_tolerance=0.0,
        joint_velocity_source="mit",
    ):
        super().__init__(
            q_initial=q_initial,
            imu_sensor=imu_sensor,
            imu_filter_cfg=imu_filter_cfg,
        )

        self.policy_order = list(policy_order)
        self.motor_layer = motor_layer
        self.bus = bus

        self.joint_index_by_name = {
            name: index for index, name in enumerate(self.policy_order)
        }
        self.pose_references_by_joint = self._build_pose_references(pose_references)
        self.pose_reference_arrays = self._build_pose_reference_arrays(pose_references)
        self.pose_snap_tolerance = max(0.0, float(pose_snap_tolerance))
        self.joint_velocity_source = str(joint_velocity_source).strip().lower()
        if self.joint_velocity_source not in ("mit", "finite-difference"):
            raise ValueError("joint_velocity_source must be mit or finite-difference")
        self._previous_position_by_joint = {}
        self._previous_timestamp_by_joint = {}
        self._filtered_fd_velocity_by_joint = {}
        self.joint_name_by_bus_motor_id = {
            (self.motor_layer.joint_can_bus.get(joint_name, "front"), int(motor_id)): joint_name
            for joint_name, motor_id in motor_ids.items()
        }

        self.last_feedback_by_joint = {}
        self.last_feedback_count = 0
        self.last_command_send_timestamp = None
        self.last_refresh_received_bus_motor_ids = set()
        self.last_refresh_current_bus_motor_ids = set()
        self.last_refresh_frame_count = 0
        self.last_refresh_current_feedback_count = 0

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
        # A command cycle can produce more than one response from a motor. Keep
        # only the newest response per routed joint so diagnostics and safety
        # counts can never report more joints than physically exist.
        frames = list(frames)
        feedback_by_joint = {}
        received_bus_motor_ids = set()
        current_bus_motor_ids = set()
        command_send_timestamp = self.last_command_send_timestamp

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
            timestamp = float(getattr(frame, "timestamp", time.monotonic()))
            bus_motor_id = (bus_name, motor_id)
            received_bus_motor_ids.add(bus_motor_id)
            if (
                command_send_timestamp is not None
                and timestamp >= float(command_send_timestamp)
            ):
                current_bus_motor_ids.add(bus_motor_id)
            feedback_by_joint[joint_name] = (
                joint_name,
                index,
                feedback,
                timestamp,
                bus_name,
            )

        count = 0
        for joint_name, index, feedback, timestamp, bus_name in feedback_by_joint.values():
            offset = float(self.motor_layer.joint_offsets[joint_name])
            direction = float(self.motor_layer.joint_directions[joint_name])
            position_raw = float(feedback["position"])
            velocity_raw = float(feedback["velocity"])
            torque_raw = float(feedback["torque"])

            q_joint = motor_position_to_joint_angle(
                position_raw,
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

            qd_mit = float(qd_joint)
            previous_q = self._previous_position_by_joint.get(joint_name)
            previous_t = self._previous_timestamp_by_joint.get(joint_name)
            qd_fd = qd_mit
            if previous_q is not None and previous_t is not None:
                sample_dt = timestamp - float(previous_t)
                if 1.0e-4 <= sample_dt <= 0.25:
                    raw_fd = (q_joint - float(previous_q)) / sample_dt
                    old_fd = self._filtered_fd_velocity_by_joint.get(joint_name, raw_fd)
                    qd_fd = 0.35 * raw_fd + 0.65 * old_fd
            self._previous_position_by_joint[joint_name] = q_joint
            self._previous_timestamp_by_joint[joint_name] = timestamp
            self._filtered_fd_velocity_by_joint[joint_name] = float(qd_fd)

            qd_selected = qd_mit if self.joint_velocity_source == "mit" else float(qd_fd)
            self.q_current[index] = q_joint
            self.qd_current[index] = qd_selected
            feedback = dict(feedback)
            feedback["timestamp"] = timestamp
            feedback["bus_name"] = bus_name
            feedback["received_after_command"] = bool(
                command_send_timestamp is not None
                and timestamp >= float(command_send_timestamp)
            )
            feedback["position_raw"] = position_raw
            feedback["velocity_raw"] = velocity_raw
            feedback["torque_raw"] = torque_raw
            feedback["motor_position"] = motor_position
            feedback["motor_velocity"] = motor_velocity
            feedback["motor_torque"] = motor_torque
            feedback["joint_position"] = q_joint
            feedback["joint_velocity_mit"] = qd_mit
            feedback["joint_velocity_finite_difference"] = float(qd_fd)
            feedback["joint_velocity_source"] = self.joint_velocity_source
            feedback["joint_velocity"] = qd_selected
            feedback["joint_torque"] = tau_joint
            feedback["position"] = q_joint
            feedback["velocity"] = qd_selected
            feedback["torque"] = tau_joint
            feedback["joint_direction"] = direction
            feedback["transmission_jacobian"] = transmission_jacobian
            feedback["transmission_efficiency"] = transmission_efficiency
            feedback["transmission_enabled"] = transmission_enabled
            self.last_feedback_by_joint[joint_name] = feedback
            count += 1

        self.last_feedback_count = count
        self.last_refresh_received_bus_motor_ids = received_bus_motor_ids
        self.last_refresh_current_bus_motor_ids = current_bus_motor_ids
        self.last_refresh_frame_count = len(frames)
        self.last_refresh_current_feedback_count = len(current_bus_motor_ids)
        return count

    def expected_feedback_bus_motor_ids(self, active_joints=None):
        joint_names = list(active_joints or self.motor_layer.active_joints)
        expected = set()
        for joint_name in joint_names:
            motor_id = int(self.motor_layer.motor_ids[joint_name])
            bus_name = self.motor_layer.joint_can_bus.get(joint_name, "front")
            expected.add((bus_name, motor_id))
        return expected

    def mark_command_sent(self, timestamp=None):
        self.last_command_send_timestamp = (
            time.monotonic() if timestamp is None else float(timestamp)
        )

    def refresh_from_bus(self, timeout=0.0, expected_bus_motor_ids=None):
        expected_bus_motor_ids = (
            self.expected_feedback_bus_motor_ids()
            if expected_bus_motor_ids is None
            else expected_bus_motor_ids
        )
        frames = MotorCommandLayer.read_all_frames(
            self.bus,
            timeout=timeout,
            expected_bus_motor_ids=expected_bus_motor_ids,
            proto=self.motor_layer.proto,
        )
        return self.update_from_frames(frames)

    def read(self):
        self.refresh_from_bus(timeout=0.0)
        return self.read_cached(refresh_imu=True)

    def read_cached(self, refresh_imu=True):
        return super().read_cached(refresh_imu=refresh_imu)

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
                feedback["position"] = target_value
                feedback["velocity"] = 0.0
                feedback["joint_velocity"] = 0.0
        return updated, missing
