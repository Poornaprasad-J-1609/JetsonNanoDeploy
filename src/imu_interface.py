#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_imu_config():
    return load_yaml(ROOT / "config" / "imu.yaml")["imu"]


def load_external_module(module_path, module_name):
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"IMU helper module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import IMU helper module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ImuReading:
    base_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    base_lin_vel_b: np.ndarray
    timestamp: float
    quaternion_wxyz: np.ndarray | None = None
    rpy_abs_deg: np.ndarray | None = None
    axes_world: np.ndarray | None = None
    det_r: float | None = None
    cross_err: float | None = None


def as_vector3(value, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 values")
    return arr


def normalize_vector(vec, fallback):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def reorder_quaternion(quat, order):
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError("quaternion must have exactly 4 values")

    order = str(order).lower()
    if order == "wxyz":
        w, x, y, z = quat
    elif order == "xyzw":
        x, y, z, w = quat
    else:
        raise ValueError(f"Unsupported quaternion_order: {order}")

    norm = math.sqrt(float(w * w + x * x + y * y + z * z))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([w / norm, x / norm, y / norm, z / norm], dtype=np.float32)


def quat_to_rotation_matrix(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotmat_to_rpy_deg(rotation):
    rotation = np.asarray(rotation, dtype=np.float32)
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.sqrt(float(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)),
    )
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array(
        [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
        dtype=np.float32,
    )


def unit_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


def force_right_hand_x_forward_z_up(rotation_raw):
    """
    Same frame cleanup as xsens_axis_viewer_v3_absolute.py.

    The absolute Xsens rotation is treated as world-from-IMU. We keep +X as
    forward and +Z as up, then rebuild +Y and +Z so X cross Y equals Z.
    """
    rotation_raw = np.asarray(rotation_raw, dtype=np.float32)
    x_axis = unit_vector(rotation_raw[:, 0])
    z_axis = unit_vector(rotation_raw[:, 2])

    z_axis = unit_vector(z_axis - float(np.dot(z_axis, x_axis)) * x_axis)
    y_axis = unit_vector(np.cross(z_axis, x_axis))
    z_axis = unit_vector(np.cross(x_axis, y_axis))

    rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    if float(np.linalg.det(rotation)) < 0.0:
        y_axis = -y_axis
        z_axis = unit_vector(np.cross(x_axis, y_axis))
        rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    return rotation


def projected_gravity_absolute_xsens(quat_wxyz):
    """
    Match xsens_axis_viewer_v3_absolute.py exactly:
      R_use = force_right_hand_x_forward_z_up(R_world_from_imu)
      projected_gravity = R_use.T @ [0, 0, -1]
    """
    quat_wxyz = reorder_quaternion(quat_wxyz, "wxyz")
    rotation_raw = quat_to_rotation_matrix(quat_wxyz)
    rotation = force_right_hand_x_forward_z_up(rotation_raw)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = rotation.T @ gravity_world
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = rotation[:, 2]
    return {
        "rotation": rotation,
        "projected_gravity": normalize_vector(projected_gravity, [0.0, 0.0, -1.0]),
        "rpy_abs_deg": rotmat_to_rpy_deg(rotation),
        "axes_world": rotation.copy(),
        "det_r": float(np.linalg.det(rotation)),
        "cross_err": float(np.linalg.norm(np.cross(x_axis, y_axis) - z_axis)),
    }


def quat_array(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.shape != (4,):
        raise ValueError("quaternion must have exactly 4 values")
    return quat_wxyz


def quat_conj(quat_wxyz):
    quat_wxyz = quat_array(quat_wxyz)
    return np.array(
        [quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]],
        dtype=np.float32,
    )


def quat_mul(a, b):
    aw, ax, ay, az = [float(v) for v in quat_array(a)]
    bw, bx, by, bz = [float(v) for v in quat_array(b)]
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_rotate(quat_wxyz, vec):
    quat_wxyz = reorder_quaternion(quat_wxyz, "wxyz")
    vec = np.asarray(vec, dtype=np.float32)
    vec_quat = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float32)
    return quat_mul(quat_mul(quat_wxyz, vec_quat), quat_conj(quat_wxyz))[1:]


def quat_rotate_inverse(quat_wxyz, vec):
    return quat_rotate(quat_conj(quat_wxyz), vec)


def projected_gravity_from_quaternion(quat, order="wxyz", frame="world_from_body"):
    quat_wxyz = reorder_quaternion(quat, order)
    rotation = quat_to_rotation_matrix(quat_wxyz)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    frame = str(frame).lower()
    if frame == "world_from_body":
        gravity_body = rotation.T @ gravity_world
    elif frame == "body_from_world":
        gravity_body = rotation @ gravity_world
    else:
        raise ValueError(f"Unsupported quaternion_frame: {frame}")

    return normalize_vector(gravity_body, fallback=[0.0, 0.0, -1.0])


def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


MID_MTDATA2 = 0x36
XDI_QUATERNION = 0x2010
XDI_RATE_OF_TURN = 0x8020
XDI_ACCELERATION = 0x4020
XDI_FREE_ACCELERATION = 0x4030


def extract_xbus_packets(buffer):
    packets = []
    while True:
        try:
            start = buffer.index(0xFA)
        except ValueError:
            buffer.clear()
            break

        if start > 0:
            del buffer[:start]

        if len(buffer) < 5:
            break

        bus_id = buffer[1]
        mid = buffer[2]
        length_byte = buffer[3]
        if length_byte == 0xFF:
            if len(buffer) < 7:
                break
            payload_len = (buffer[4] << 8) | buffer[5]
            header_len = 6
        else:
            payload_len = length_byte
            header_len = 4

        total_len = header_len + payload_len + 1
        if len(buffer) < total_len:
            break

        frame = bytes(buffer[:total_len])
        del buffer[:total_len]
        if (sum(frame[1:]) & 0xFF) != 0:
            continue
        packets.append((bus_id, mid, frame[header_len:-1]))
    return packets


def parse_mtdata2(payload):
    parsed = {}
    i = 0
    while i + 3 <= len(payload):
        data_id = int.from_bytes(payload[i:i + 2], "big")
        size = payload[i + 2]
        i += 3
        if i + size > len(payload):
            break

        data = payload[i:i + size]
        i += size
        data_id_masked = data_id & 0xFFF0

        if data_id_masked == XDI_QUATERNION and size >= 16:
            parsed["quaternion_wxyz"] = np.array(
                struct.unpack(">4f", data[:16]),
                dtype=np.float32,
            )
        elif data_id_masked == XDI_RATE_OF_TURN and size >= 12:
            parsed["gyro_sensor"] = np.array(
                struct.unpack(">3f", data[:12]),
                dtype=np.float32,
            )
        elif data_id_masked == XDI_FREE_ACCELERATION and size >= 12:
            parsed["free_acc_sensor"] = np.array(
                struct.unpack(">3f", data[:12]),
                dtype=np.float32,
            )
        elif data_id_masked == XDI_ACCELERATION and size >= 12:
            parsed["acc_sensor"] = np.array(
                struct.unpack(">3f", data[:12]),
                dtype=np.float32,
            )
    return parsed


def send_xsens_measurement_mode(serial_port):
    serial_port.reset_input_buffer()
    serial_port.write(bytes.fromhex("FA FF 10 00 F1"))
    serial_port.flush()
    time.sleep(0.3)


class FakeImuSensor:
    source_name = "fake"
    required = False

    def __init__(self, cfg=None):
        self.stale_timeout = float((cfg or {}).get("stale_timeout", 0.25))

    def open(self):
        return self

    def close(self):
        return None

    def read(self):
        return ImuReading(
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            timestamp=time.monotonic(),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            rpy_abs_deg=np.zeros(3, dtype=np.float32),
            axes_world=np.eye(3, dtype=np.float32),
            det_r=1.0,
            cross_err=0.0,
        )


class XsensBinaryImuSensor:
    source_name = "xsens"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.required = bool(self.cfg.get("require_live", True))
        self.port = self.cfg.get("port", "/dev/ttyUSB0")
        self.baud = int(self.cfg.get("baud", 115200))
        self.timeout = float(self.cfg.get("timeout", 0.001))
        self.stale_timeout = float(self.cfg.get("stale_timeout", 0.25))
        self.force_zero_base_linear_velocity = bool(
            self.cfg.get("force_zero_base_linear_velocity", True)
        )
        self.estimate_linear_velocity = bool(
            self.cfg.get("estimate_linear_velocity", False)
        )
        self.velocity_decay = float(self.cfg.get("velocity_decay", 0.98))
        self.sensor_to_base_rotation = np.asarray(
            self.cfg.get("sensor_to_base_rotation", np.eye(3)),
            dtype=np.float32,
        )
        if self.sensor_to_base_rotation.shape != (3, 3):
            raise ValueError("sensor_to_base_rotation must be a 3x3 matrix")
        if not np.all(np.isfinite(self.sensor_to_base_rotation)):
            raise ValueError("sensor_to_base_rotation contains non-finite values")
        orthogonality_error = float(
            np.linalg.norm(
                self.sensor_to_base_rotation @ self.sensor_to_base_rotation.T
                - np.eye(3, dtype=np.float32)
            )
        )
        determinant = float(np.linalg.det(self.sensor_to_base_rotation))
        if orthogonality_error > 1e-4 or abs(determinant - 1.0) > 1e-4:
            raise ValueError(
                "sensor_to_base_rotation must be a proper rotation matrix "
                f"(orthogonality_error={orthogonality_error:.3e}, det={determinant:.6f})"
            )
        self.gyro_units = str(self.cfg.get("gyro_units", "rad_s")).lower()
        self.gyro_axis_signs = as_vector3(
            self.cfg.get("gyro_axis_signs", [1.0, 1.0, 1.0]),
            "gyro_axis_signs",
        )
        self.gravity_axis_signs = as_vector3(
            self.cfg.get("gravity_axis_signs", [1.0, 1.0, 1.0]),
            "gravity_axis_signs",
        )
        self.flip_gravity = bool(self.cfg.get("flip_gravity", False))

        self.ser = None
        self.buffer = bytearray()
        self.latest_quaternion_wxyz = None
        self.latest_gyro_sensor = None
        self.latest_free_acc_sensor = None
        self.latest_acc_sensor = None
        self.latest_abs = None
        self.velocity_world = np.zeros(3, dtype=np.float32)
        self.last_velocity_time = None
        self.last_packet_time = None
        self.latest_orientation_time = None
        self.latest_gyro_time = None

    def open(self):
        import serial

        if not os.path.exists(self.port):
            raise FileNotFoundError(f"IMU port not found: {self.port}")

        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(0.1)
        send_xsens_measurement_mode(self.ser)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _read_packets(self):
        waiting = int(getattr(self.ser, "in_waiting", 0))
        if waiting > 0:
            data = self.ser.read(max(512, waiting))
            if data:
                self.buffer.extend(data)

        updated = False
        for _, mid, payload in extract_xbus_packets(self.buffer):
            if mid != MID_MTDATA2:
                continue

            parsed = parse_mtdata2(payload)
            packet_time = time.monotonic()
            if "quaternion_wxyz" in parsed:
                self.latest_quaternion_wxyz = reorder_quaternion(
                    parsed["quaternion_wxyz"],
                    "wxyz",
                )
                self.latest_abs = projected_gravity_absolute_xsens(
                    self.latest_quaternion_wxyz
                )
                self.latest_orientation_time = packet_time
                updated = True
            if "gyro_sensor" in parsed:
                self.latest_gyro_sensor = np.asarray(parsed["gyro_sensor"], dtype=np.float32)
                self.latest_gyro_time = packet_time
                updated = True
            if "free_acc_sensor" in parsed:
                self.latest_free_acc_sensor = np.asarray(
                    parsed["free_acc_sensor"],
                    dtype=np.float32,
                )
                updated = True
            if "acc_sensor" in parsed:
                self.latest_acc_sensor = np.asarray(parsed["acc_sensor"], dtype=np.float32)
                updated = True

        if updated:
            self.last_packet_time = time.monotonic()

    def _gyro_in_base_frame(self):
        gyro = self.sensor_to_base_rotation @ self.latest_gyro_sensor
        gyro = np.asarray(gyro, dtype=np.float32) * self.gyro_axis_signs
        if self.gyro_units in ("deg_s", "deg/s", "dps"):
            gyro = np.deg2rad(gyro).astype(np.float32)
        elif self.gyro_units not in ("rad_s", "rad/s", "rps"):
            raise ValueError(f"Unsupported gyro_units: {self.gyro_units}")
        if not np.all(np.isfinite(gyro)):
            raise ValueError(f"Xsens gyro contains non-finite values: {gyro}")
        return gyro

    def estimate_base_linear_velocity(self):
        if (
            self.force_zero_base_linear_velocity
            or not self.estimate_linear_velocity
            or self.latest_quaternion_wxyz is None
        ):
            return np.zeros(3, dtype=np.float32)

        now = time.monotonic()
        if self.last_velocity_time is None:
            self.last_velocity_time = now
            return np.zeros(3, dtype=np.float32)

        dt = now - self.last_velocity_time
        self.last_velocity_time = now

        if self.latest_free_acc_sensor is not None and 0.0 < dt <= 0.2:
            free_acc_world = quat_rotate(
                self.latest_quaternion_wxyz,
                self.latest_free_acc_sensor,
            )
            self.velocity_world += np.asarray(free_acc_world, dtype=np.float32) * dt
            self.velocity_world *= self.velocity_decay

        velocity_sensor = quat_rotate_inverse(
            self.latest_quaternion_wxyz,
            self.velocity_world,
        )
        velocity_base = self.sensor_to_base_rotation @ np.asarray(
            velocity_sensor,
            dtype=np.float32,
        )
        return np.asarray(velocity_base, dtype=np.float32)

    def read(self):
        if self.ser is None:
            raise RuntimeError("IMU serial port is not open")

        self._read_packets()
        if (
            self.latest_quaternion_wxyz is None
            or self.latest_abs is None
            or self.latest_gyro_sensor is None
            or self.latest_orientation_time is None
            or self.latest_gyro_time is None
        ):
            return None

        projected_gravity_sensor = np.asarray(
            self.latest_abs["projected_gravity"],
            dtype=np.float32,
        )
        projected_gravity = self.sensor_to_base_rotation @ projected_gravity_sensor
        projected_gravity = (
            np.asarray(projected_gravity, dtype=np.float32)
            * self.gravity_axis_signs
        )
        if self.flip_gravity:
            projected_gravity = -projected_gravity
        base_ang_vel = self._gyro_in_base_frame()

        rotation_world_from_sensor = np.asarray(
            self.latest_abs["rotation"],
            dtype=np.float32,
        )
        rotation_world_from_base = (
            rotation_world_from_sensor @ self.sensor_to_base_rotation.T
        )

        base_lin_vel = self.estimate_base_linear_velocity()

        return ImuReading(
            base_ang_vel_b=np.asarray(base_ang_vel, dtype=np.float32),
            projected_gravity_b=normalize_vector(
                projected_gravity,
                fallback=[0.0, 0.0, -1.0],
            ),
            base_lin_vel_b=base_lin_vel,
            timestamp=min(self.latest_orientation_time, self.latest_gyro_time),
            quaternion_wxyz=self.latest_quaternion_wxyz.copy(),
            rpy_abs_deg=rotmat_to_rpy_deg(rotation_world_from_base),
            axes_world=rotation_world_from_base.copy(),
            det_r=float(np.linalg.det(rotation_world_from_base)),
            cross_err=float(
                np.linalg.norm(
                    np.cross(
                        rotation_world_from_base[:, 0],
                        rotation_world_from_base[:, 1],
                    )
                    - rotation_world_from_base[:, 2]
                )
            ),
        )


class SerialImuSensor:
    def __init__(self, cfg, source_name):
        self.cfg = dict(cfg)
        self.source_name = source_name
        self.required = bool(self.cfg.get("require_live", True))
        self.port = self.cfg.get("port", "/dev/ttyUSB1")
        self.baud = int(self.cfg.get("baud", 115200))
        self.timeout = float(self.cfg.get("timeout", 0.001))
        self.stale_timeout = float(self.cfg.get("stale_timeout", 0.25))
        self.gyro_units = str(self.cfg.get("gyro_units", "rad_s")).lower()
        self.gyro_axis_signs = as_vector3(self.cfg.get("gyro_axis_signs", [1.0, 1.0, 1.0]), "gyro_axis_signs")
        self.gravity_axis_signs = as_vector3(
            self.cfg.get("gravity_axis_signs", [1.0, 1.0, 1.0]),
            "gravity_axis_signs",
        )
        self.csv_fields = list(self.cfg.get("csv_fields", ["gx", "gy", "gz", "qw", "qx", "qy", "qz"]))
        self.quaternion_order = self.cfg.get("quaternion_order", "wxyz")
        self.quaternion_frame = self.cfg.get("quaternion_frame", "world_from_body")
        self.force_zero_base_linear_velocity = bool(
            self.cfg.get("force_zero_base_linear_velocity", True)
        )
        self.ser = None

    def open(self):
        import serial

        self.ser = serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        time.sleep(0.05)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def scale_gyro(self, gyro):
        gyro = as_vector3(gyro, "gyro") * self.gyro_axis_signs
        if self.gyro_units in ("deg_s", "deg/s", "dps"):
            gyro = np.deg2rad(gyro)
        elif self.gyro_units not in ("rad_s", "rad/s", "rps"):
            raise ValueError(f"Unsupported gyro_units: {self.gyro_units}")
        return gyro.astype(np.float32)

    def parse_json_line(self, line):
        msg = json.loads(line)
        gyro = first_present(msg, ["base_ang_vel", "ang_vel", "angular_velocity", "gyro", "gyr"])
        if gyro is None:
            raise ValueError("JSON IMU line missing gyro/angular velocity")

        projected_gravity = first_present(
            msg,
            ["projected_gravity", "projected_gravity_b", "gravity", "gravity_b", "grav"],
        )
        if projected_gravity is not None:
            projected_gravity = normalize_vector(
                as_vector3(projected_gravity, "projected_gravity") * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )
        else:
            quat = first_present(msg, ["quat", "quaternion", "orientation"])
            if quat is None:
                raise ValueError("JSON IMU line needs projected_gravity or quaternion")
            projected_gravity = projected_gravity_from_quaternion(
                quat,
                order=self.quaternion_order,
                frame=self.quaternion_frame,
            )
            projected_gravity = normalize_vector(
                projected_gravity * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )

        base_lin_vel = first_present(msg, ["base_lin_vel", "lin_vel", "linear_velocity"])
        if self.force_zero_base_linear_velocity or base_lin_vel is None:
            base_lin_vel = np.zeros(3, dtype=np.float32)
        else:
            base_lin_vel = as_vector3(base_lin_vel, "base_lin_vel")

        return ImuReading(
            base_ang_vel_b=self.scale_gyro(gyro),
            projected_gravity_b=projected_gravity,
            base_lin_vel_b=base_lin_vel.astype(np.float32),
            timestamp=time.monotonic(),
        )

    def parse_csv_line(self, line):
        values = [float(part.strip()) for part in line.split(",")]
        if len(values) != len(self.csv_fields):
            raise ValueError(
                f"CSV IMU line has {len(values)} values, expected {len(self.csv_fields)}"
            )
        msg = dict(zip(self.csv_fields, values))

        gyro = [msg["gx"], msg["gy"], msg["gz"]]
        if all(key in msg for key in ("grav_x", "grav_y", "grav_z")):
            projected_gravity = normalize_vector(
                np.array([msg["grav_x"], msg["grav_y"], msg["grav_z"]], dtype=np.float32)
                * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )
        else:
            quat_keys = ("qw", "qx", "qy", "qz")
            if not all(key in msg for key in quat_keys):
                raise ValueError("CSV IMU line needs qw,qx,qy,qz or grav_x,grav_y,grav_z")
            projected_gravity = projected_gravity_from_quaternion(
                [msg["qw"], msg["qx"], msg["qy"], msg["qz"]],
                order="wxyz",
                frame=self.quaternion_frame,
            )
            projected_gravity = normalize_vector(
                projected_gravity * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )

        base_lin_vel = np.zeros(3, dtype=np.float32)
        if not self.force_zero_base_linear_velocity and all(key in msg for key in ("vx", "vy", "vz")):
            base_lin_vel = np.array([msg["vx"], msg["vy"], msg["vz"]], dtype=np.float32)

        return ImuReading(
            base_ang_vel_b=self.scale_gyro(gyro),
            projected_gravity_b=projected_gravity,
            base_lin_vel_b=base_lin_vel,
            timestamp=time.monotonic(),
        )

    def parse_line(self, line):
        line = line.strip()
        if not line:
            return None
        if self.source_name == "serial_json":
            return self.parse_json_line(line)
        if self.source_name == "serial_csv":
            return self.parse_csv_line(line)
        raise ValueError(f"Unsupported serial IMU source: {self.source_name}")

    def read(self):
        if self.ser is None:
            raise RuntimeError("IMU serial port is not open")

        latest = None
        reads = max(1, int(getattr(self.ser, "in_waiting", 0) > 0) * 32)
        for _ in range(reads):
            raw = self.ser.readline()
            if not raw:
                break
            try:
                line = raw.decode("utf-8", errors="ignore")
                reading = self.parse_line(line)
            except Exception:
                continue
            if reading is not None:
                latest = reading

        return latest


def create_imu_sensor(source="auto", port=None, baud=None):
    cfg = load_imu_config()
    if source == "auto":
        source = cfg.get("source", "fake")

    source = str(source).replace("-", "_").lower()
    if port is not None:
        cfg["port"] = port
    if baud is not None:
        cfg["baud"] = int(baud)

    if source in ("none", "off"):
        return None
    if source == "fake":
        return FakeImuSensor(cfg).open()
    if source in ("xsens", "xsens_binary", "mtdata2"):
        return XsensBinaryImuSensor(cfg).open()
    if source in ("serial_json", "serial_csv"):
        return SerialImuSensor(cfg, source_name=source).open()

    raise ValueError(f"Unknown IMU source: {source}")


def main():
    cfg = load_imu_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["auto", "fake", "none", "xsens", "serial-json", "serial-csv"],
        default="auto",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--hz", "--rate", type=float, default=20.0)
    args = parser.parse_args()

    sensor = create_imu_sensor(source=args.source, port=args.port, baud=args.baud)
    if sensor is None:
        print("IMU source disabled.")
        return 0

    dt = 1.0 / max(1e-6, args.hz)
    steps = int(args.seconds * args.hz)
    print("IMU source:", sensor.source_name)
    print("IMU port:", getattr(sensor, "port", "none"))
    print("IMU baud:", getattr(sensor, "baud", "none"))
    print("Config source:", cfg.get("source", "fake"))

    try:
        for step in range(steps):
            reading = sensor.read()
            if reading is None:
                print(f"step={step:04d} imu=no_new_data")
            else:
                quat = reading.quaternion_wxyz
                rpy = reading.rpy_abs_deg
                det_r = reading.det_r
                cross_err = reading.cross_err
                quat_text = (
                    f" q_abs=[{quat[0]:+.5f}, {quat[1]:+.5f}, {quat[2]:+.5f}, {quat[3]:+.5f}]"
                    if quat is not None
                    else ""
                )
                rpy_text = (
                    f" rpy_abs_deg=[{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}]"
                    if rpy is not None
                    else ""
                )
                quality_text = (
                    f" det_R={det_r:+.3f} cross_err={cross_err:.2e}"
                    if det_r is not None and cross_err is not None
                    else ""
                )
                print(
                    f"step={step:04d} "
                    f"gyro={reading.base_ang_vel_b} "
                    f"gravity={reading.projected_gravity_b} "
                    f"lin_vel={reading.base_lin_vel_b}"
                    f"{quat_text}{rpy_text}{quality_text}"
                )
            time.sleep(dt)
    finally:
        sensor.close()

    return 0


if __name__ == "__main__":
    main()
