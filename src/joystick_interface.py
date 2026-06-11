#!/usr/bin/env python3
from pathlib import Path
import argparse
from collections import deque
import select
import sys
import termios
import time
import tty
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_control_limits():
    return load_yaml(ROOT / "config" / "control_limits.yaml")


def apply_deadzone(x, deadzone):
    x = float(x)
    deadzone = float(deadzone)
    if deadzone >= 1.0:
        return 0.0
    if abs(x) < deadzone:
        return 0.0
    sign = 1.0 if x >= 0.0 else -1.0
    return sign * (abs(x) - deadzone) / (1.0 - deadzone)


def expo_curve(x, expo):
    return (1.0 - expo) * x + expo * (x ** 3)


def load_command_limits():
    control_cfg = load_control_limits()
    velocity_cfg = control_cfg.get("velocity_command", {})
    if bool(velocity_cfg.get("enabled", True)):
        lim = velocity_cfg
    else:
        cfg = load_yaml(ROOT / "config" / "joint_map.yaml")
        lim = cfg["command_limits"]

    limits = (
        float(lim["vx_min"]),
        float(lim["vx_max"]),
        float(lim["vy_min"]),
        float(lim["vy_max"]),
        float(lim["yaw_min"]),
        float(lim["yaw_max"]),
    )
    if limits[0] > limits[1] or limits[2] > limits[3] or limits[4] > limits[5]:
        raise ValueError("Invalid velocity command limits: min value is greater than max")
    return limits


def load_joystick_defaults():
    return load_yaml(ROOT / "config" / "joystick.yaml")


def load_speed_scale_defaults():
    cfg = load_control_limits().get("joystick_speed_scale", {})
    speed = {
        "initial": float(cfg.get("initial", 0.5)),
        "min": float(cfg.get("min", 0.2)),
        "max": float(cfg.get("max", 1.0)),
        "step": float(cfg.get("step", 0.1)),
    }
    if speed["min"] > speed["max"]:
        raise ValueError("Invalid joystick speed scale: min is greater than max")
    if speed["step"] < 0.0:
        raise ValueError("Invalid joystick speed scale: step must be >= 0")
    return speed


def init_pygame_for_joystick():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    pygame.joystick.init()
    return pygame


def print_joystick_devices(wait_seconds=0.0):
    pygame = init_pygame_for_joystick()
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    count = pygame.joystick.get_count()
    while count <= 0 and time.monotonic() < deadline:
        time.sleep(0.25)
        pygame.joystick.quit()
        pygame.joystick.init()
        count = pygame.joystick.get_count()

    print(f"Joystick count: {count}")
    for joystick_index in range(count):
        joy = pygame.joystick.Joystick(joystick_index)
        joy.init()
        print(
            f"  index={joystick_index} "
            f"name={joy.get_name()} "
            f"axes={joy.get_numaxes()} "
            f"buttons={joy.get_numbuttons()} "
            f"hats={joy.get_numhats()}"
        )
        joy.quit()

    pygame.joystick.quit()
    pygame.display.quit()


def clip_command(command, limits):
    vx_min, vx_max, vy_min, vy_max, yaw_min, yaw_max = limits
    command = np.asarray(command, dtype=np.float32).copy()
    command[0] = np.clip(command[0], vx_min, vx_max)
    command[1] = np.clip(command[1], vy_min, vy_max)
    command[2] = np.clip(command[2], yaw_min, yaw_max)
    return command


class FixedCommandSource:
    def __init__(self, vx=0.0, vy=0.0, yaw=0.0):
        self.command_limits = load_command_limits()
        self.command = clip_command([vx, vy, yaw], self.command_limits)

    def read(self):
        self.command_limits = load_command_limits()
        self.command = clip_command(self.command, self.command_limits)
        return self.command.copy()

    def get_mode_request(self):
        return None

    def get_emergency_stop_request(self):
        return None

    def get_speed_scale(self):
        return 1.0

    def get_calibration_request(self):
        return None

    def raw_state(self):
        return None

    def close(self):
        pass


class KeyboardCommandSource:
    """
    Non-blocking terminal keyboard drive source.

    Keys:
      w/a/s/d -> movement command while the key repeats
      c -> sit/crouch
      space -> stand
      h -> hold current position
      x -> emergency stop
    """

    def __init__(
        self,
        max_vx=0.15,
        max_vy=0.10,
        max_yaw=0.30,
        speed_scale_initial=0.5,
        speed_scale_min=0.2,
        speed_scale_max=1.0,
        command_timeout_s=0.35,
    ):
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_yaw = float(max_yaw)
        self.speed_scale_min = float(max(0.0, speed_scale_min))
        self.speed_scale_max = float(max(self.speed_scale_min, speed_scale_max))
        self.speed_scale = float(
            np.clip(speed_scale_initial, self.speed_scale_min, self.speed_scale_max)
        )
        self.command_timeout_s = float(max(0.05, command_timeout_s))
        self.command_deadline = -1.0
        self.command = np.zeros(3, dtype=np.float32)
        self.command_limits = load_command_limits()
        self.key_queue = deque()
        self.fd = None
        self.old_settings = None
        self.active = False

        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.active = True

        print("Terminal keyboard command source:")
        print("  w -> forward")
        print("  s -> backward")
        print("  a -> left")
        print("  d -> right")
        print("  c -> crouch/sit")
        print("  space -> stand")
        print("  h -> hold current position")
        print("  x -> emergency stop")
        print(f"  speed scale: {self.speed_scale:.2f}")
        if not self.active:
            print("  WARNING: stdin is not an interactive terminal; keyboard input is inactive.")

    def _poll_keys(self):
        if not self.active:
            return
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                break
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            self.key_queue.append(key.lower())

    def _process_movement_keys(self):
        self._poll_keys()
        if not self.key_queue:
            return

        now = time.monotonic()
        kept = deque()
        while self.key_queue:
            key = self.key_queue.popleft()
            consumed = True
            if key == "w":
                self.command = np.array([self.max_vx, 0.0, 0.0], dtype=np.float32)
                self.command_deadline = now + self.command_timeout_s
            elif key == "s":
                self.command = np.array([-self.max_vx, 0.0, 0.0], dtype=np.float32)
                self.command_deadline = now + self.command_timeout_s
            elif key == "a":
                self.command = np.array([0.0, self.max_vy, 0.0], dtype=np.float32)
                self.command_deadline = now + self.command_timeout_s
            elif key == "d":
                self.command = np.array([0.0, -self.max_vy, 0.0], dtype=np.float32)
                self.command_deadline = now + self.command_timeout_s
            else:
                consumed = False

            if not consumed:
                kept.append(key)
        self.key_queue = kept

    def _clear_motion_command(self):
        self.command = np.zeros(3, dtype=np.float32)
        self.command_deadline = -1.0

    def _pop_matching_key(self, mapping):
        self._poll_keys()
        mapping = mapping or {}
        for index, key in enumerate(self.key_queue):
            if key in mapping:
                del self.key_queue[index]
                if mapping[key] in ("stand", "sit", "hold"):
                    self._clear_motion_command()
                return mapping[key]
        return None

    def read(self):
        self._process_movement_keys()
        if time.monotonic() > self.command_deadline:
            self.command = np.zeros(3, dtype=np.float32)
        self.command_limits = load_command_limits()
        return clip_command(self.command * self.speed_scale, self.command_limits)

    def get_mode_request(self):
        return self._pop_matching_key({
            " ": "stand",
            "c": "sit",
            "h": "hold",
        })

    def get_emergency_stop_request(self):
        reason = self._pop_matching_key({"x": "terminal keyboard emergency stop key x"})
        return reason

    def get_speed_scale(self):
        return self.speed_scale

    def get_calibration_request(self):
        return None

    def raw_state(self):
        self._poll_keys()
        return {
            "pending_keys": list(self.key_queue),
            "command": [float(x) for x in self.command],
        }

    def close(self):
        if self.active and self.old_settings is not None and self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        self.active = False


class JoystickCommandSource:
    """
    Default Xbox / Logitech-like mapping:
      left stick Y  -> vx
      left stick X  -> vy
      right stick X -> yaw

    Buttons:
      button 4       -> stop walking and sit/crouch
      button 5       -> stand
      button 3       -> emergency stop when config/joystick.yaml remaps 0-2
      button 6/7     -> optional speed scale
    """

    def __init__(
        self,
        max_vx=0.15,
        max_vy=0.10,
        max_yaw=0.30,
        axis_vx=1,
        axis_vy=0,
        axis_yaw=3,
        invert_vx=True,
        invert_vy=False,
        invert_yaw=False,
        deadzone=0.08,
        expo=0.35,
        smoothing=0.20,
        use_hat_fallback=True,
        joystick_index=0,
        joystick_wait_seconds=5.0,
        button_stand=5,
        button_sit=4,
        button_policy=-1,
        button_speed_down=6,
        button_speed_up=7,
        button_zero_calibration=-1,
        zero_calibration_hat_index=0,
        zero_calibration_hat_direction=None,
        zero_calibration_hat_any=False,
        zero_calibration_axis=1,
        zero_calibration_axis_direction=1.0,
        zero_calibration_axis_threshold=0.8,
        zero_calibration_cooldown_s=1.0,
        emergency_stop_buttons=None,
        speed_scale_initial=0.5,
        speed_scale_min=0.2,
        speed_scale_max=1.0,
        speed_scale_step=0.1,
    ):
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_yaw = float(max_yaw)
        self.command_limits = load_command_limits()

        self.axis_vx = int(axis_vx)
        self.axis_vy = int(axis_vy)
        self.axis_yaw = int(axis_yaw)

        self.invert_vx = bool(invert_vx)
        self.invert_vy = bool(invert_vy)
        self.invert_yaw = bool(invert_yaw)

        self.deadzone = float(np.clip(deadzone, 0.0, 0.95))
        self.expo = float(np.clip(expo, 0.0, 1.0))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.use_hat_fallback = bool(use_hat_fallback)

        self.button_stand = int(button_stand)
        self.button_sit = int(button_sit)
        self.button_policy = int(button_policy)
        self.button_speed_down = int(button_speed_down)
        self.button_speed_up = int(button_speed_up)
        self.button_zero_calibration = int(button_zero_calibration)
        self.zero_calibration_hat_index = int(zero_calibration_hat_index)
        if zero_calibration_hat_direction is None:
            zero_calibration_hat_direction = [0, -1]
        self.zero_calibration_hat_direction = (
            int(zero_calibration_hat_direction[0]),
            int(zero_calibration_hat_direction[1]),
        )
        self.zero_calibration_hat_any = bool(zero_calibration_hat_any)
        self.zero_calibration_axis = int(zero_calibration_axis)
        self.zero_calibration_axis_direction = (
            1.0 if float(zero_calibration_axis_direction) >= 0.0 else -1.0
        )
        self.zero_calibration_axis_threshold = float(
            np.clip(abs(float(zero_calibration_axis_threshold)), 0.0, 1.0)
        )
        self.zero_calibration_cooldown_s = float(max(0.1, zero_calibration_cooldown_s))
        self.last_zero_calibration_time = -1.0e9
        if emergency_stop_buttons is None:
            emergency_stop_buttons = [0, 1, 2, 3]
        self.emergency_stop_buttons = [int(button_id) for button_id in emergency_stop_buttons]
        self.joystick_wait_seconds = float(max(0.0, joystick_wait_seconds))

        self.speed_scale_min = float(max(0.0, speed_scale_min))
        self.speed_scale_max = float(max(self.speed_scale_min, speed_scale_max))
        self.speed_scale_step = float(max(0.0, speed_scale_step))
        self.speed_scale = float(
            np.clip(speed_scale_initial, self.speed_scale_min, self.speed_scale_max)
        )

        self.command = np.zeros(3, dtype=np.float32)
        self.prev_buttons = {}
        self.prev_hats = {}

        try:
            pygame = init_pygame_for_joystick()
        except ImportError as exc:
            raise ImportError("Install pygame first: pip3 install pygame") from exc

        self.pygame = pygame

        deadline = time.monotonic() + self.joystick_wait_seconds
        count = pygame.joystick.get_count()
        while count <= 0 and time.monotonic() < deadline:
            time.sleep(0.25)
            pygame.joystick.quit()
            pygame.joystick.init()
            count = pygame.joystick.get_count()

        if count <= 0:
            raise RuntimeError(
                "No joystick found. Check that the receiver is plugged in, "
                "the controller is powered on, and Linux sees /dev/input/js*. "
                "Useful checks: lsusb; ls -l /dev/input/js* /dev/input/event*"
            )

        if joystick_index >= count:
            raise RuntimeError(f"joystick_index={joystick_index} but only {count} joystick(s) found")

        self.joy = pygame.joystick.Joystick(joystick_index)
        self.joy.init()
        pygame.event.pump()

        edge_buttons = [
            self.button_stand,
            self.button_sit,
            self.button_policy,
            self.button_speed_down,
            self.button_speed_up,
            self.button_zero_calibration,
        ]
        for button_id in edge_buttons:
            self.prev_buttons[button_id] = self._button(button_id)
        for hat_id in range(self.joy.get_numhats()):
            self.prev_hats[hat_id] = self.joy.get_hat(hat_id)
        self.prev_zero_axis_active = self._zero_axis_active()

        print("Joystick connected:")
        print("  name:", self.joy.get_name())
        print("  axes:", self.joy.get_numaxes())
        print("  buttons:", self.joy.get_numbuttons())
        print("  hats:", self.joy.get_numhats())
        print("Button mapping:")
        print(f"  stand      button: {self.button_stand}")
        print(f"  sit/stop   button: {self.button_sit}")
        print(f"  policy     button: {self.button_policy}")
        print(f"  speed down button: {self.button_speed_down}")
        print(f"  speed up   button: {self.button_speed_up}")
        print(f"  zero-cal   button: {self.button_zero_calibration}")
        if self.zero_calibration_hat_any:
            print(
                "  zero-cal   dpad: "
                f"hat {self.zero_calibration_hat_index} any non-center direction"
            )
        else:
            print(
                "  zero-cal   dpad: "
                f"hat {self.zero_calibration_hat_index} {self.zero_calibration_hat_direction}"
            )
        print(
            "  zero-cal   axis: "
            f"axis {self.zero_calibration_axis} "
            f"direction {self.zero_calibration_axis_direction:+.1f} "
            f"threshold {self.zero_calibration_axis_threshold:.2f}"
        )
        print(f"  emergency buttons: {self.emergency_stop_buttons}")
        print(f"  speed scale: {self.speed_scale:.2f}")

    def _axis(self, axis_id):
        if axis_id < 0 or axis_id >= self.joy.get_numaxes():
            return 0.0
        return float(self.joy.get_axis(axis_id))

    def _button(self, button_id):
        if button_id < 0 or button_id >= self.joy.get_numbuttons():
            return False
        return bool(self.joy.get_button(button_id))

    def _button_rising_edge(self, button_id):
        now = self._button(button_id)
        prev = self.prev_buttons.get(button_id, False)
        self.prev_buttons[button_id] = now
        return now and not prev

    def _hat_rising_edge(self, hat_id, direction):
        if hat_id < 0 or hat_id >= self.joy.get_numhats():
            return False
        now = self.joy.get_hat(hat_id)
        prev = self.prev_hats.get(hat_id, (0, 0))
        self.prev_hats[hat_id] = now
        return now == tuple(direction) and prev != tuple(direction)

    def _hat_active(self, hat_id, direction):
        if hat_id < 0 or hat_id >= self.joy.get_numhats():
            return False
        hat = self.joy.get_hat(hat_id)
        if self.zero_calibration_hat_any:
            return hat != (0, 0)
        return hat == tuple(direction)

    def _zero_axis_active(self):
        axis_id = self.zero_calibration_axis
        if axis_id < 0 or axis_id >= self.joy.get_numaxes():
            return False
        value = self._axis(axis_id)
        return (
            self.zero_calibration_axis_direction * value
            >= self.zero_calibration_axis_threshold
        )

    def _zero_axis_rising_edge(self):
        now = self._zero_axis_active()
        prev = getattr(self, "prev_zero_axis_active", False)
        self.prev_zero_axis_active = now
        return now and not prev

    def _update_speed_scale(self):
        changed = False

        if self._button_rising_edge(self.button_speed_down):
            self.speed_scale = max(
                self.speed_scale_min,
                self.speed_scale - self.speed_scale_step,
            )
            changed = True

        if self._button_rising_edge(self.button_speed_up):
            self.speed_scale = min(
                self.speed_scale_max,
                self.speed_scale + self.speed_scale_step,
            )
            changed = True

        if changed:
            print(f"[JOYSTICK] speed_scale={self.speed_scale:.2f}")

    def read(self):
        self.pygame.event.pump()
        self._update_speed_scale()
        self.command_limits = load_command_limits()

        raw_vx = self._axis(self.axis_vx)
        raw_vy = self._axis(self.axis_vy)
        raw_yaw = self._axis(self.axis_yaw)

        if self.use_hat_fallback and self.joy.get_numhats() > 0:
            hat_x, hat_y = self.joy.get_hat(0)
            if abs(raw_vx) < self.deadzone and hat_y != 0:
                raw_vx = -float(hat_y)
            if abs(raw_vy) < self.deadzone and hat_x != 0:
                raw_vy = float(hat_x)

        if self.invert_vx:
            raw_vx = -raw_vx
        if self.invert_vy:
            raw_vy = -raw_vy
        if self.invert_yaw:
            raw_yaw = -raw_yaw

        vx = (
            expo_curve(apply_deadzone(raw_vx, self.deadzone), self.expo)
            * self.max_vx
            * self.speed_scale
        )
        vy = (
            expo_curve(apply_deadzone(raw_vy, self.deadzone), self.expo)
            * self.max_vy
            * self.speed_scale
        )
        yaw = (
            expo_curve(apply_deadzone(raw_yaw, self.deadzone), self.expo)
            * self.max_yaw
            * self.speed_scale
        )

        target = clip_command([vx, vy, yaw], self.command_limits)
        self.command = (1.0 - self.smoothing) * self.command + self.smoothing * target
        self.command = clip_command(self.command, self.command_limits)

        return self.command.copy()

    def get_mode_request(self):
        self.pygame.event.pump()

        if self._button_rising_edge(self.button_stand):
            return "stand"

        if self._button_rising_edge(self.button_sit):
            return "sit"

        if self._button_rising_edge(self.button_policy):
            return "policy"

        return None

    def get_emergency_stop_request(self):
        self.pygame.event.pump()

        for button_id in self.emergency_stop_buttons:
            if self._button(button_id):
                return f"joystick emergency stop button {button_id}"

        return None

    def get_speed_scale(self):
        return self.speed_scale

    def get_calibration_request(self):
        self.pygame.event.pump()

        if self._button_rising_edge(self.button_zero_calibration):
            self.last_zero_calibration_time = time.monotonic()
            print("[JOYSTICK] zero calibration requested by button")
            return "zero_current_pose"

        now = time.monotonic()
        cooldown_ok = (
            now - self.last_zero_calibration_time
            >= self.zero_calibration_cooldown_s
        )

        if (
            cooldown_ok
            and self._hat_active(
                self.zero_calibration_hat_index,
                self.zero_calibration_hat_direction,
            )
        ):
            self.last_zero_calibration_time = now
            print("[JOYSTICK] zero calibration requested by D-pad")
            return "zero_current_pose"

        if cooldown_ok and self._zero_axis_active():
            self.last_zero_calibration_time = now
            print("[JOYSTICK] zero calibration requested by axis")
            return "zero_current_pose"

        return None

    def raw_state(self):
        self.pygame.event.pump()
        return {
            "axes": [self._axis(i) for i in range(self.joy.get_numaxes())],
            "buttons": [int(self._button(i)) for i in range(self.joy.get_numbuttons())],
            "hats": [self.joy.get_hat(i) for i in range(self.joy.get_numhats())],
        }

    def close(self):
        self.joy.quit()
        self.pygame.joystick.quit()
        self.pygame.display.quit()


class CommandSource:
    def __init__(self, source="fixed", **kwargs):
        if source == "fixed":
            self.impl = FixedCommandSource(
                vx=kwargs.get("vx", 0.0),
                vy=kwargs.get("vy", 0.0),
                yaw=kwargs.get("yaw", 0.0),
            )
        elif source == "keyboard":
            defaults = load_joystick_defaults()
            speed_defaults = load_speed_scale_defaults()
            self.impl = KeyboardCommandSource(
                max_vx=kwargs.get("max_vx", defaults["speed_limits"]["max_vx"]),
                max_vy=kwargs.get("max_vy", defaults["speed_limits"]["max_vy"]),
                max_yaw=kwargs.get("max_yaw", defaults["speed_limits"]["max_yaw"]),
                speed_scale_initial=kwargs.get("speed_scale_initial", speed_defaults["initial"]),
                speed_scale_min=kwargs.get("speed_scale_min", speed_defaults["min"]),
                speed_scale_max=kwargs.get("speed_scale_max", speed_defaults["max"]),
                command_timeout_s=kwargs.get("keyboard_command_timeout", 0.35),
            )
        elif source == "joystick":
            defaults = load_joystick_defaults()
            speed_defaults = load_speed_scale_defaults()
            self.impl = JoystickCommandSource(
                max_vx=kwargs.get("max_vx", defaults["speed_limits"]["max_vx"]),
                max_vy=kwargs.get("max_vy", defaults["speed_limits"]["max_vy"]),
                max_yaw=kwargs.get("max_yaw", defaults["speed_limits"]["max_yaw"]),
                axis_vx=kwargs.get("axis_vx", defaults["axes"]["vx_axis"]),
                axis_vy=kwargs.get("axis_vy", defaults["axes"]["vy_axis"]),
                axis_yaw=kwargs.get("axis_yaw", defaults["axes"]["yaw_axis"]),
                invert_vx=kwargs.get("invert_vx", defaults["invert"]["vx"]),
                invert_vy=kwargs.get("invert_vy", defaults["invert"]["vy"]),
                invert_yaw=kwargs.get("invert_yaw", defaults["invert"]["yaw"]),
                deadzone=kwargs.get("deadzone", defaults["filter"]["deadzone"]),
                expo=kwargs.get("expo", defaults["filter"]["expo"]),
                smoothing=kwargs.get("smoothing", defaults["filter"]["smoothing"]),
                use_hat_fallback=kwargs.get(
                    "use_hat_fallback",
                    defaults["filter"].get("use_hat_fallback", True),
                ),
                joystick_index=kwargs.get("joystick_index", 0),
                joystick_wait_seconds=kwargs.get(
                    "joystick_wait_seconds",
                    defaults.get("connection", {}).get("wait_seconds", 5.0),
                ),
                button_stand=kwargs.get("button_stand", defaults["buttons"]["stand"]),
                button_sit=kwargs.get("button_sit", defaults["buttons"]["sit"]),
                button_policy=kwargs.get("button_policy", defaults["buttons"]["policy"]),
                button_speed_down=kwargs.get(
                    "button_speed_down",
                    defaults["buttons"].get("speed_down", 6),
                ),
                button_speed_up=kwargs.get(
                    "button_speed_up",
                    defaults["buttons"].get("speed_up", 7),
                ),
                button_zero_calibration=kwargs.get(
                    "button_zero_calibration",
                    defaults["buttons"].get("zero_calibration", -1),
                ),
                zero_calibration_hat_index=kwargs.get(
                    "zero_calibration_hat_index",
                    defaults.get("dpad", {}).get("zero_calibration_hat_index", 0),
                ),
                zero_calibration_hat_direction=kwargs.get(
                    "zero_calibration_hat_direction",
                    defaults.get("dpad", {}).get("zero_calibration_hat_direction", [0, -1]),
                ),
                zero_calibration_hat_any=kwargs.get(
                    "zero_calibration_hat_any",
                    defaults.get("dpad", {}).get("zero_calibration_hat_any", False),
                ),
                zero_calibration_axis=kwargs.get(
                    "zero_calibration_axis",
                    defaults.get("dpad", {}).get("zero_calibration_axis", 1),
                ),
                zero_calibration_axis_direction=kwargs.get(
                    "zero_calibration_axis_direction",
                    defaults.get("dpad", {}).get("zero_calibration_axis_direction", 1.0),
                ),
                zero_calibration_axis_threshold=kwargs.get(
                    "zero_calibration_axis_threshold",
                    defaults.get("dpad", {}).get("zero_calibration_axis_threshold", 0.8),
                ),
                zero_calibration_cooldown_s=kwargs.get(
                    "zero_calibration_cooldown_s",
                    defaults.get("dpad", {}).get("zero_calibration_cooldown_s", 1.0),
                ),
                emergency_stop_buttons=kwargs.get(
                    "emergency_stop_buttons",
                    defaults["buttons"].get("emergency_stop", [0, 1, 2, 3]),
                ),
                speed_scale_initial=kwargs.get("speed_scale_initial", speed_defaults["initial"]),
                speed_scale_min=kwargs.get("speed_scale_min", speed_defaults["min"]),
                speed_scale_max=kwargs.get("speed_scale_max", speed_defaults["max"]),
                speed_scale_step=kwargs.get("speed_scale_step", speed_defaults["step"]),
            )
        else:
            raise ValueError(f"Unknown command source: {source}")

    def read(self):
        return self.impl.read()

    def get_mode_request(self):
        return self.impl.get_mode_request()

    def get_emergency_stop_request(self):
        return self.impl.get_emergency_stop_request()

    def get_speed_scale(self):
        return self.impl.get_speed_scale()

    def get_calibration_request(self):
        if hasattr(self.impl, "get_calibration_request"):
            return self.impl.get_calibration_request()
        return None

    def raw_state(self):
        if hasattr(self.impl, "raw_state"):
            return self.impl.raw_state()
        return None

    def close(self):
        self.impl.close()


def main():
    defaults = load_joystick_defaults()
    speed_defaults = load_speed_scale_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["fixed", "joystick", "keyboard"], default="fixed")

    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=float(defaults["speed_limits"]["max_vx"]))
    parser.add_argument("--max-vy", type=float, default=float(defaults["speed_limits"]["max_vy"]))
    parser.add_argument("--max-yaw", type=float, default=float(defaults["speed_limits"]["max_yaw"]))

    parser.add_argument("--axis-vx", type=int, default=int(defaults["axes"]["vx_axis"]))
    parser.add_argument("--axis-vy", type=int, default=int(defaults["axes"]["vy_axis"]))
    parser.add_argument("--axis-yaw", type=int, default=int(defaults["axes"]["yaw_axis"]))
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument(
        "--joystick-wait-seconds",
        type=float,
        default=float(defaults.get("connection", {}).get("wait_seconds", 5.0)),
    )

    parser.add_argument("--invert-vx", action=argparse.BooleanOptionalAction, default=bool(defaults["invert"]["vx"]))
    parser.add_argument("--invert-vy", action=argparse.BooleanOptionalAction, default=bool(defaults["invert"]["vy"]))
    parser.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=bool(defaults["invert"]["yaw"]))

    parser.add_argument("--button-stand", type=int, default=int(defaults["buttons"]["stand"]))
    parser.add_argument("--button-sit", "--button-sit-stop", type=int, default=int(defaults["buttons"]["sit"]))
    parser.add_argument("--button-policy", type=int, default=int(defaults["buttons"]["policy"]))
    parser.add_argument(
        "--button-speed-down",
        type=int,
        default=int(defaults["buttons"].get("speed_down", 6)),
    )
    parser.add_argument(
        "--button-speed-up",
        type=int,
        default=int(defaults["buttons"].get("speed_up", 7)),
    )
    parser.add_argument(
        "--button-zero-calibration",
        type=int,
        default=int(defaults["buttons"].get("zero_calibration", -1)),
    )
    parser.add_argument(
        "--zero-calibration-hat-index",
        type=int,
        default=int(defaults.get("dpad", {}).get("zero_calibration_hat_index", 0)),
    )
    parser.add_argument(
        "--zero-calibration-hat-direction",
        type=int,
        nargs=2,
        default=[
            int(v)
            for v in defaults.get("dpad", {}).get(
                "zero_calibration_hat_direction",
                [0, -1],
            )
        ],
    )
    parser.add_argument(
        "--zero-calibration-hat-any",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("dpad", {}).get("zero_calibration_hat_any", False)),
        help="use any non-centered D-pad/hat direction as the software-zero trigger",
    )
    parser.add_argument(
        "--zero-calibration-axis",
        type=int,
        default=int(defaults.get("dpad", {}).get("zero_calibration_axis", 1)),
        help="axis used as a software-zero trigger; -1 disables axis trigger",
    )
    parser.add_argument(
        "--zero-calibration-axis-direction",
        type=float,
        default=float(defaults.get("dpad", {}).get("zero_calibration_axis_direction", 1.0)),
        help="axis sign that triggers software zero: +1 or -1",
    )
    parser.add_argument(
        "--zero-calibration-axis-threshold",
        type=float,
        default=float(defaults.get("dpad", {}).get("zero_calibration_axis_threshold", 0.8)),
        help="absolute axis threshold for software-zero trigger",
    )
    parser.add_argument(
        "--button-emergency-stop",
        type=int,
        nargs="+",
        default=[int(button_id) for button_id in defaults["buttons"].get("emergency_stop", [0, 1, 2, 3])],
    )

    parser.add_argument("--deadzone", type=float, default=float(defaults["filter"]["deadzone"]))
    parser.add_argument("--expo", type=float, default=float(defaults["filter"]["expo"]))
    parser.add_argument("--smoothing", type=float, default=float(defaults["filter"]["smoothing"]))
    parser.add_argument(
        "--use-hat-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["filter"].get("use_hat_fallback", True)),
    )
    parser.add_argument("--speed-scale-initial", type=float, default=speed_defaults["initial"])
    parser.add_argument("--speed-scale-min", type=float, default=speed_defaults["min"])
    parser.add_argument("--speed-scale-max", type=float, default=speed_defaults["max"])
    parser.add_argument("--speed-scale-step", type=float, default=speed_defaults["step"])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument(
        "--list-joysticks",
        action="store_true",
        help="list pygame joystick indexes and exit",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="also print all raw axes/buttons/hats while testing",
    )

    args = parser.parse_args()

    if args.list_joysticks:
        print_joystick_devices(wait_seconds=args.joystick_wait_seconds)
        return 0

    try:
        source = CommandSource(
            source=args.source,
            vx=args.vx,
            vy=args.vy,
            yaw=args.yaw,
            max_vx=args.max_vx,
            max_vy=args.max_vy,
            max_yaw=args.max_yaw,
            axis_vx=args.axis_vx,
            axis_vy=args.axis_vy,
            axis_yaw=args.axis_yaw,
            invert_vx=args.invert_vx,
            invert_vy=args.invert_vy,
            invert_yaw=args.invert_yaw,
            joystick_index=args.joystick_index,
            joystick_wait_seconds=args.joystick_wait_seconds,
            button_stand=args.button_stand,
            button_sit=args.button_sit,
            button_policy=args.button_policy,
            button_speed_down=args.button_speed_down,
            button_speed_up=args.button_speed_up,
            button_zero_calibration=args.button_zero_calibration,
            zero_calibration_hat_index=args.zero_calibration_hat_index,
            zero_calibration_hat_direction=args.zero_calibration_hat_direction,
            zero_calibration_hat_any=args.zero_calibration_hat_any,
            zero_calibration_axis=args.zero_calibration_axis,
            zero_calibration_axis_direction=args.zero_calibration_axis_direction,
            zero_calibration_axis_threshold=args.zero_calibration_axis_threshold,
            emergency_stop_buttons=args.button_emergency_stop,
            speed_scale_initial=args.speed_scale_initial,
            speed_scale_min=args.speed_scale_min,
            speed_scale_max=args.speed_scale_max,
            speed_scale_step=args.speed_scale_step,
            deadzone=args.deadzone,
            expo=args.expo,
            smoothing=args.smoothing,
            use_hat_fallback=args.use_hat_fallback,
        )
    except (ImportError, RuntimeError) as exc:
        print("ERROR:", exc)
        return 1

    dt = 1.0 / args.hz
    steps = int(args.seconds * args.hz)

    print("Testing command source:", args.source)
    print("Press stand/sit/policy buttons if using joystick.")
    try:
        for i in range(steps):
            emergency_reason = source.get_emergency_stop_request()
            if emergency_reason is not None:
                print(f"emergency_stop={emergency_reason}")
                break

            cmd = source.read()
            mode_req = source.get_mode_request()
            calib_req = source.get_calibration_request()
            print(
                f"step={i:04d} "
                f"vx={cmd[0]: .3f} vy={cmd[1]: .3f} yaw={cmd[2]: .3f} "
                f"speed_scale={source.get_speed_scale():.2f} "
                f"mode_request={mode_req} calibration_request={calib_req}"
            )
            if args.show_raw:
                raw = source.raw_state()
                if raw is not None:
                    axes = " ".join(f"{axis_id}:{value:+.3f}" for axis_id, value in enumerate(raw["axes"]))
                    buttons = " ".join(f"{button_id}:{value}" for button_id, value in enumerate(raw["buttons"]))
                    hats = " ".join(f"{hat_id}:{value}" for hat_id, value in enumerate(raw["hats"]))
                    print(f"  raw_axes=[{axes}] raw_buttons=[{buttons}] raw_hats=[{hats}]")
            time.sleep(dt)
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
