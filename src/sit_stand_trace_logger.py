#!/usr/bin/env python3
"""Dedicated 200 Hz sit/stand command and raw-feedback trace."""

import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

from motor_command_layer import decode_mit_feedback_frame


TRACE_JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "BL_hip_joint", "BL_thigh_joint", "BL_calf_joint",
    "BR_hip_joint", "BR_thigh_joint", "BR_calf_joint",
)


def _label(joint_name):
    return str(joint_name).removesuffix("_joint")


class SitStandTraceLogger:
    """Write one non-blocking row for each 200 Hz CAN scheduler cycle."""

    GLOBAL_FIELDS = (
        "timestamp", "wall_time", "can_cycle", "command_generation",
        "controller_phase", "transition_progress", "stand_progress",
        "transition_duration_s", "control_frequency_hz", "can_frequency_hz",
        "imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z",
        "imu_roll_deg", "imu_pitch_deg", "imu_yaw_deg",
        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
        "base_height", "FL_foot_contact", "FR_foot_contact",
        "BL_foot_contact", "BR_foot_contact", "battery_voltage",
    )
    JOINT_FIELDS = (
        "q_des", "q_meas", "dq_des", "dq_meas", "dq_meas_mit",
        "kp_cmd", "kd_cmd", "tau_ff_cmd", "position_error",
        "velocity_error", "tau_p_predicted", "tau_d_predicted",
        "tau_pd_predicted", "tau_total_predicted", "tau_command_estimated",
        "tau_est", "motor_current", "motor_voltage", "motor_temperature",
        "feedback_age_ms", "fault_bits", "mode_status", "raw_position",
        "raw_velocity", "raw_torque", "raw_current", "raw_CAN_id",
        "raw_CAN_payload", "bus_name", "motor_id",
    )

    def __init__(
        self,
        log_dir,
        motor_layer,
        metadata,
        queue_size=20000,
    ):
        self.motor_layer = motor_layer
        self.start_monotonic = time.monotonic()
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        directory = Path(log_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"sit_stand_200hz_trace_{stamp}.csv"
        self.metadata_path = self.path.with_suffix(".metadata.json")
        self.fieldnames = list(self.GLOBAL_FIELDS)
        for joint_name in TRACE_JOINT_ORDER:
            prefix = _label(joint_name)
            self.fieldnames.extend(f"{prefix}_{field}" for field in self.JOINT_FIELDS)

        metadata = dict(metadata or {})
        metadata.update({
            "trace_csv": str(self.path),
            "timestamp_clock": "time.monotonic seconds",
            "joint_order": list(TRACE_JOINT_ORDER),
            "motor_current_available": False,
            "motor_voltage_available": False,
            "availability_note": (
                "The active RS04 MIT feedback frame contains position, velocity, "
                "torque, temperature, mode, and fault bits; it contains no motor "
                "current or voltage fields. Blank CSV values are intentional."
            ),
            "created_at": datetime.now().astimezone().isoformat(),
        })
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()
        self._queue = queue.Queue(maxsize=max(1000, int(queue_size)))
        self._stop = threading.Event()
        self._context_lock = threading.Lock()
        self._context = {}
        self._feedback = {}
        self._previous_position = {}
        self._previous_timestamp = {}
        self._filtered_velocity = {}
        self.rows_written = 0
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="sit-stand-200hz-csv",
            daemon=True,
        )
        self._thread.start()

        self._joint_by_bus_motor = {}
        for joint_name in TRACE_JOINT_ORDER:
            bus_name = str(motor_layer.joint_can_bus.get(joint_name, "front"))
            motor_id = int(motor_layer.motor_ids[joint_name])
            self._joint_by_bus_motor[(bus_name, motor_id)] = joint_name

    def update_context(
        self,
        estimator,
        controller_phase,
        transition_progress,
        stand_progress,
        transition_duration_s,
        control_frequency_hz,
        can_frequency_hz,
        battery_voltage=None,
    ):
        reading = getattr(estimator, "last_imu_reading", None)
        quaternion = getattr(reading, "quaternion_wxyz", None)
        rpy = getattr(reading, "rpy_abs_deg", None)
        angular_velocity = getattr(estimator, "base_ang_vel_b", None)

        def values(source, count):
            if source is None:
                return [""] * count
            try:
                result = [float(value) for value in source]
            except (TypeError, ValueError):
                return [""] * count
            return result if len(result) == count else [""] * count

        quat = values(quaternion, 4)
        rpy_values = values(rpy, 3)
        gyro = values(angular_velocity, 3)
        context = {
            "controller_phase": str(controller_phase),
            "transition_progress": float(transition_progress),
            "stand_progress": float(stand_progress),
            "transition_duration_s": float(transition_duration_s),
            "control_frequency_hz": float(control_frequency_hz),
            "can_frequency_hz": float(can_frequency_hz),
            "imu_quat_w": quat[0], "imu_quat_x": quat[1],
            "imu_quat_y": quat[2], "imu_quat_z": quat[3],
            "imu_roll_deg": rpy_values[0], "imu_pitch_deg": rpy_values[1],
            "imu_yaw_deg": rpy_values[2],
            "angular_velocity_x": gyro[0], "angular_velocity_y": gyro[1],
            "angular_velocity_z": gyro[2],
            "base_height": "",
            "FL_foot_contact": "", "FR_foot_contact": "",
            "BL_foot_contact": "", "BR_foot_contact": "",
            "battery_voltage": "" if battery_voltage is None else float(battery_voltage),
        }
        with self._context_lock:
            self._context = context

    def _update_feedback(self, frames):
        for frame in frames or ():
            decoded = decode_mit_feedback_frame(
                frame.can_id,
                frame.data,
                self.motor_layer.proto,
            )
            if decoded is None:
                continue
            bus_name = getattr(frame, "bus_name", None)
            joint_name = self._joint_by_bus_motor.get(
                (str(bus_name), int(decoded["motor_id"]))
            )
            if joint_name is None:
                continue
            timestamp = float(getattr(frame, "timestamp", time.monotonic()))
            if hasattr(self.motor_layer, "decode_joint_feedback"):
                mapped = self.motor_layer.decode_joint_feedback(
                    joint_name=joint_name,
                    position_raw=float(decoded["position"]),
                    velocity_raw=float(decoded["velocity"]),
                    torque_raw=float(decoded["torque"]),
                )
            else:
                direction = float(self.motor_layer.joint_directions[joint_name])
                offset = float(self.motor_layer.joint_offsets[joint_name])
                mapped = {
                    "joint_position": direction * (float(decoded["position"]) - offset),
                    "joint_velocity": direction * float(decoded["velocity"]),
                    "joint_torque": direction * float(decoded["torque"]),
                }
            q_meas = float(mapped["joint_position"])
            dq_mit = float(mapped["joint_velocity"])
            dq_meas = dq_mit
            previous_q = self._previous_position.get(joint_name)
            previous_t = self._previous_timestamp.get(joint_name)
            if previous_q is not None and previous_t is not None:
                dt = timestamp - previous_t
                if 1.0e-4 <= dt <= 0.25:
                    raw_fd = (q_meas - previous_q) / dt
                    old = self._filtered_velocity.get(joint_name, raw_fd)
                    dq_meas = 0.35 * raw_fd + 0.65 * old
            self._previous_position[joint_name] = q_meas
            self._previous_timestamp[joint_name] = timestamp
            self._filtered_velocity[joint_name] = dq_meas
            self._feedback[joint_name] = {
                "timestamp": timestamp,
                "q_meas": q_meas,
                "dq_meas": float(dq_meas),
                "dq_meas_mit": dq_mit,
                "tau_est": float(mapped["joint_torque"]),
                "motor_temperature": float(decoded["temperature_c"]),
                "fault_bits": int(decoded["fault_bits"]),
                "mode_status": int(decoded["mode_status"]),
                "raw_position": float(decoded["position"]),
                "raw_velocity": float(decoded["velocity"]),
                "raw_torque": float(decoded["torque"]),
                "raw_CAN_id": f"0x{int(frame.can_id):08X}",
                "raw_CAN_payload": bytes(frame.data).hex(),
                "bus_name": str(bus_name),
                "motor_id": int(decoded["motor_id"]),
            }

    def record_can_cycle(
        self,
        timestamp,
        cycle_index,
        generation,
        commands,
        received_frames,
    ):
        self._update_feedback(received_frames)
        with self._context_lock:
            context = dict(self._context)
        row = {field: "" for field in self.fieldnames}
        row.update(context)
        row.update({
            "timestamp": float(timestamp),
            "wall_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "can_cycle": int(cycle_index),
            "command_generation": int(generation),
        })
        command_by_joint = {
            item.get("joint_name"): item for item in commands or ()
        }
        for joint_name in TRACE_JOINT_ORDER:
            prefix = _label(joint_name)
            command = command_by_joint.get(joint_name, {})
            feedback = self._feedback.get(joint_name, {})
            q_des = command.get("q_des")
            dq_des = command.get("joint_v_des", command.get("v_des"))
            kp = command.get("kp_effective", command.get("kp"))
            kd = command.get("kd_effective", command.get("kd"))
            tau_ff = command.get("tau_ff")
            q_meas = feedback.get("q_meas")
            dq_meas = feedback.get("dq_meas")
            values = {
                "q_des": q_des, "q_meas": q_meas,
                "dq_des": dq_des, "dq_meas": dq_meas,
                "dq_meas_mit": feedback.get("dq_meas_mit"),
                "kp_cmd": kp, "kd_cmd": kd, "tau_ff_cmd": tau_ff,
                "tau_command_estimated": command.get("tau_pd_est"),
                "tau_est": feedback.get("tau_est"),
                "motor_current": "", "motor_voltage": "", "raw_current": "",
            }
            if all(value is not None for value in (q_des, q_meas)):
                values["position_error"] = float(q_des) - float(q_meas)
            if all(value is not None for value in (dq_des, dq_meas)):
                values["velocity_error"] = float(dq_des) - float(dq_meas)
            if kp is not None and values.get("position_error") is not None:
                values["tau_p_predicted"] = float(kp) * values["position_error"]
            if kd is not None and values.get("velocity_error") is not None:
                values["tau_d_predicted"] = float(kd) * values["velocity_error"]
            if all(values.get(key) is not None for key in ("tau_p_predicted", "tau_d_predicted")):
                values["tau_pd_predicted"] = (
                    values["tau_p_predicted"] + values["tau_d_predicted"]
                )
                values["tau_total_predicted"] = values["tau_pd_predicted"] + float(tau_ff or 0.0)
            for key, value in feedback.items():
                if key == "timestamp":
                    continue
                values.setdefault(key, value)
            if feedback.get("timestamp") is not None:
                values["feedback_age_ms"] = max(
                    0.0, 1000.0 * (float(timestamp) - float(feedback["timestamp"]))
                )
            for key, value in values.items():
                field = f"{prefix}_{key}"
                if field in row and value is not None:
                    row[field] = value
        try:
            self._queue.put_nowait(row)
        except queue.Full as exc:
            raise RuntimeError("sit/stand 200 Hz trace queue is full") from exc

    def _writer_loop(self):
        last_flush = time.monotonic()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.05)
            except queue.Empty:
                row = None
            if row is not None:
                self._writer.writerow(row)
                self.rows_written += 1
                self._queue.task_done()
            now = time.monotonic()
            if now - last_flush >= 0.5:
                self._file.flush()
                last_flush = now
        self._file.flush()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._file.flush()
        self._file.close()
