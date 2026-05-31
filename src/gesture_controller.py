#!/usr/bin/env python3
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def smoothstep(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


class KeyboardGestureReader:
    """Small non-blocking keyboard reader for number-key gesture triggers."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.fd = None
        self.old_settings = None
        self.active = False

        if not self.enabled or not sys.stdin.isatty():
            return

        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.active = True

    def read_key(self):
        if not self.active:
            return None

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None

        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return key

    def close(self):
        if self.active and self.old_settings is not None and self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        self.active = False


class GestureState:
    def __init__(self, name, segments, start_time):
        self.name = str(name)
        self.segments = list(segments)
        self.segment_index = 0
        self.segment_start_time = float(start_time)
        self.segment_start_q = self.segments[0]["start"].copy()
        self.finished = False

    def target_at(self, now):
        now = float(now)
        while self.segment_index < len(self.segments):
            segment = self.segments[self.segment_index]
            duration = max(1.0e-6, float(segment["duration_s"]))
            hold_s = max(0.0, float(segment.get("hold_s", 0.0)))
            elapsed = now - self.segment_start_time

            if elapsed <= duration:
                alpha = smoothstep(elapsed / duration)
                target = (1.0 - alpha) * self.segment_start_q + alpha * segment["target"]
                return target.astype(np.float32), False

            if elapsed <= duration + hold_s:
                return segment["target"].copy(), False

            self.segment_start_q = segment["target"].copy()
            self.segment_start_time = now
            self.segment_index += 1

        self.finished = True
        return self.segment_start_q.copy(), True


class GestureLibrary:
    def __init__(self, policy_order, path=None, enabled=True):
        self.policy_order = list(policy_order)
        self.path = Path(path) if path is not None else ROOT / "config" / "gestures.yaml"
        self.enabled = bool(enabled)
        self.cfg = load_yaml(self.path) if self.path.exists() else {}
        self.enabled = self.enabled and bool(self.cfg.get("enabled", True))
        self.defaults = self.cfg.get("defaults", {}) or {}
        self.gestures = self.cfg.get("gestures", {}) or {}

        keyboard_cfg = self.cfg.get("keyboard", {}) or {}
        self.keyboard_enabled = bool(keyboard_cfg.get("enabled", False))
        self.key_map = {
            str(key): str(value)
            for key, value in (keyboard_cfg.get("mapping", {}) or {}).items()
        }

        joystick_cfg = self.cfg.get("joystick", {}) or {}
        self.joystick_enabled = bool(joystick_cfg.get("enabled", False))
        self.button_map = {
            int(key): str(value)
            for key, value in (joystick_cfg.get("mapping", {}) or {}).items()
        }

    def key_hint(self):
        if not self.enabled or not self.keyboard_enabled or not self.key_map:
            return "disabled"
        return ", ".join(
            f"{key}={name}"
            for key, name in sorted(self.key_map.items(), key=lambda item: item[0])
        )

    def button_hint(self):
        if not self.enabled or not self.joystick_enabled or not self.button_map:
            return "disabled"
        return ", ".join(
            f"btn{btn}={name}"
            for btn, name in sorted(self.button_map.items())
        )

    def gesture_for_key(self, key):
        if key is None or not self.enabled or not self.keyboard_enabled:
            return None
        return self.key_map.get(str(key))

    def gesture_for_button(self, button):
        if button is None or not self.enabled or not self.joystick_enabled:
            return None
        return self.button_map.get(int(button))

    def has_gesture(self, name):
        return str(name) in self.gestures

    def allowed_in_zero_frame(self, name, zero_frame):
        gesture = self.gestures[str(name)]
        allowed = gesture.get(
            "allowed_zero_frames",
            self.defaults.get("allowed_zero_frames", ["crouch", "stand"]),
        )
        allowed = [str(item).lower() for item in allowed]
        return "any" in allowed or str(zero_frame).lower() in allowed

    def build_state(self, name, zero_frame, q_start, now):
        name = str(name)
        if name not in self.gestures:
            raise KeyError(f"Gesture {name!r} is not defined in {self.path}")

        gesture = self.gestures[name] or {}
        waypoints = gesture.get("waypoints", []) or []
        if not waypoints:
            raise ValueError(f"Gesture {name!r} has no waypoints in {self.path}")

        q_start = np.asarray(q_start, dtype=np.float32).copy()
        previous = q_start.copy()
        segments = []

        default_duration = float(gesture.get("duration_s", self.defaults.get("duration_s", 1.0)))
        default_hold = float(gesture.get("hold_s", self.defaults.get("hold_s", 0.0)))

        for waypoint in waypoints:
            target = self._waypoint_target(
                waypoint=waypoint or {},
                zero_frame=zero_frame,
                base=previous,
            )
            segments.append({
                "start": previous.copy(),
                "target": target,
                "duration_s": float(waypoint.get("duration_s", default_duration)),
                "hold_s": float(waypoint.get("hold_s", default_hold)),
            })
            previous = target.copy()

        return_to_start = bool(
            gesture.get("return_to_start", self.defaults.get("return_to_start", True))
        )
        if return_to_start:
            segments.append({
                "start": previous.copy(),
                "target": q_start.copy(),
                "duration_s": float(gesture.get("return_duration_s", default_duration)),
                "hold_s": 0.0,
            })

        return GestureState(name=name, segments=segments, start_time=now)

    def _waypoint_target(self, waypoint, zero_frame, base):
        base = np.asarray(base, dtype=np.float32).copy()
        pose = self._select_pose(waypoint, zero_frame)
        relative = bool(waypoint.get("relative", self.defaults.get("relative", False)))

        if pose is None:
            return base
        if not isinstance(pose, dict):
            raise ValueError("Gesture waypoint pose must be a joint-angle mapping")

        target = base.copy()
        index_by_name = {name: index for index, name in enumerate(self.policy_order)}
        for joint_name, value in pose.items():
            if joint_name not in index_by_name:
                raise KeyError(f"Unknown joint {joint_name!r} in gesture pose")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Gesture joint {joint_name!r} has non-finite value {value}")
            index = index_by_name[joint_name]
            target[index] = target[index] + value if relative else value
        return target.astype(np.float32)

    @staticmethod
    def _select_pose(waypoint, zero_frame):
        zero_frame = str(zero_frame).lower()
        frame_keys = [
            f"pose_when_{zero_frame}_zero",
            f"pose_{zero_frame}",
            "pose",
        ]
        for key in frame_keys:
            if key in waypoint:
                return waypoint[key] or {}
        return {}
