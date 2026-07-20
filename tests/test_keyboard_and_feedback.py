#!/usr/bin/env python3
import contextlib
import io
import time
import unittest

import numpy as np

from joystick_interface import KeyboardCommandSource
from main_controller import encoder_safety_stop_reason
from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor


class DummyEstimator:
    def __init__(self, q_current, feedback_by_joint=None):
        self.q_current = np.asarray(q_current, dtype=np.float32)
        self.last_feedback_by_joint = dict(feedback_by_joint or {})


class KeyboardAndFeedbackTests(unittest.TestCase):
    def make_keyboard(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return KeyboardCommandSource(
                max_vx=1.8,
                max_vy=0.8,
                max_yaw=0.8,
                speed_scale_initial=1.0,
                speed_scale_min=0.5,
                speed_scale_max=1.2,
                speed_scale_step=0.1,
                command_timeout_s=0.2,
            )

    def make_latched_keyboard(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return KeyboardCommandSource(
                max_vx=1.8,
                max_vy=0.8,
                max_yaw=0.8,
                speed_scale_initial=1.0,
                speed_scale_min=0.5,
                speed_scale_max=1.2,
                speed_scale_step=0.1,
                command_timeout_s=0.05,
                control_mode="latched",
                latched_combo_window_s=0.25,
            )

    def command_for_keys(self, *keys):
        source = self.make_keyboard()
        now = time.monotonic()
        for key in keys:
            source.movement_key_deadlines[key] = now + 1.0
        source._update_command_from_active_keys(now)
        command = source.read()
        source.close()
        return command

    def test_keyboard_movement_axes(self):
        np.testing.assert_allclose(self.command_for_keys("w"), [1.8, 0.0, 0.0])
        np.testing.assert_allclose(self.command_for_keys("s"), [-1.8, 0.0, 0.0])
        np.testing.assert_allclose(self.command_for_keys("a"), [0.0, 0.8, 0.0])
        np.testing.assert_allclose(self.command_for_keys("d"), [0.0, -0.8, 0.0])
        np.testing.assert_allclose(self.command_for_keys("q"), [0.0, 0.0, 0.8])
        np.testing.assert_allclose(self.command_for_keys("e"), [0.0, 0.0, -0.8])

    def test_keyboard_diagonal_and_yaw_combinations(self):
        np.testing.assert_allclose(self.command_for_keys("w", "a"), [1.8, 0.8, 0.0])
        np.testing.assert_allclose(self.command_for_keys("w", "d"), [1.8, -0.8, 0.0])
        np.testing.assert_allclose(self.command_for_keys("s", "a"), [-1.8, 0.8, 0.0])
        np.testing.assert_allclose(self.command_for_keys("s", "d"), [-1.8, -0.8, 0.0])
        np.testing.assert_allclose(self.command_for_keys("w", "q"), [1.8, 0.0, 0.8])

    def test_keyboard_pose_keys_clear_motion(self):
        source = self.make_keyboard()
        now = time.monotonic()
        source.movement_key_deadlines["w"] = now + 1.0
        source._update_command_from_active_keys(now)
        source.key_queue.append(" ")
        self.assertEqual(source.get_mode_request(), "stand")
        np.testing.assert_allclose(source.read(), [0.0, 0.0, 0.0])

        source.key_queue.append("c")
        self.assertEqual(source.get_mode_request(), "sit")
        source.key_queue.append("h")
        self.assertEqual(source.get_mode_request(), "hold")
        source.close()

    def test_keyboard_speed_arrows_and_emergency(self):
        source = self.make_keyboard()
        source.key_queue.append("arrow_up")
        source.read()
        self.assertAlmostEqual(source.get_speed_scale(), 1.1)
        source.key_queue.append("arrow_down")
        source.read()
        self.assertAlmostEqual(source.get_speed_scale(), 1.0)
        source.key_queue.append("x")
        self.assertEqual(
            source.get_emergency_stop_request(),
            "terminal keyboard emergency stop key x",
        )
        source.close()

    def test_latched_keyboard_w_remains_active_without_repeat(self):
        source = self.make_latched_keyboard()
        source.key_queue.append("w")
        np.testing.assert_allclose(source.read(), [1.8, 0.0, 0.0])
        time.sleep(0.08)
        np.testing.assert_allclose(source.read(), [1.8, 0.0, 0.0])
        source.close()

    def test_latched_keyboard_pose_keys_clear_motion(self):
        source = self.make_latched_keyboard()
        source.key_queue.append("w")
        source.read()
        source.key_queue.append("h")
        self.assertEqual(source.get_mode_request(), "hold")
        np.testing.assert_allclose(source.read(), [0.0, 0.0, 0.0])
        source.key_queue.append("w")
        source.read()
        source.key_queue.append(" ")
        self.assertEqual(source.get_mode_request(), "stand")
        np.testing.assert_allclose(source.read(), [0.0, 0.0, 0.0])
        source.key_queue.append("c")
        self.assertEqual(source.get_mode_request(), "sit")
        source.key_queue.append("x")
        self.assertEqual(
            source.get_emergency_stop_request(),
            "terminal keyboard emergency stop key x",
        )
        source.close()

    def test_latched_keyboard_speed_change_keeps_command(self):
        source = self.make_latched_keyboard()
        source.key_queue.append("w")
        np.testing.assert_allclose(source.read(), [1.8, 0.0, 0.0])
        source.key_queue.append("arrow_down")
        np.testing.assert_allclose(source.read(), [1.62, 0.0, 0.0])
        source.close()

    def test_latched_keyboard_prints_on_change_not_every_read(self):
        source = self.make_latched_keyboard()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            source.key_queue.append("w")
            source.read()
            source.read()
            source.read()
        lines = [
            line for line in output.getvalue().splitlines()
            if "[KEYBOARD] latched command" in line
        ]
        self.assertEqual(len(lines), 1)
        source.close()

    def test_encoder_safety_skips_fake_dry_mode(self):
        runner = PolicyRunner()
        safety = SafetyMonitor(runner.policy_order)
        estimator = type(
            "FakeLikeEstimator",
            (),
            {"q_current": np.full(len(runner.policy_order), 99.0, dtype=np.float32)},
        )()
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=estimator,
            active_joints=runner.policy_order,
            mode="print",
        )
        self.assertIsNone(reason)

    def test_encoder_safety_requires_fresh_mit_feedback(self):
        runner = PolicyRunner()
        safety = SafetyMonitor(runner.policy_order)
        active = runner.policy_order[:2]
        q_current = np.zeros(len(runner.policy_order), dtype=np.float32)

        missing_estimator = DummyEstimator(q_current, {})
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=missing_estimator,
            active_joints=active,
            mode="mit-signal",
        )
        self.assertIn("missing MIT encoder feedback", reason)

        stale_time = time.monotonic() - 10.0
        stale_feedback = {
            name: {"timestamp": stale_time, "fault_bits": 0, "joint_torque": 0.0}
            for name in active
        }
        stale_estimator = DummyEstimator(q_current, stale_feedback)
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=stale_estimator,
            active_joints=active,
            mode="mit-signal",
        )
        self.assertIn("stale MIT encoder feedback", reason)

        fresh_time = time.monotonic()
        fresh_feedback = {
            name: {"timestamp": fresh_time, "fault_bits": 0, "joint_torque": 0.0}
            for name in active
        }
        fresh_estimator = DummyEstimator(q_current, fresh_feedback)
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=fresh_estimator,
            active_joints=active,
            mode="mit-signal",
        )
        self.assertIsNone(reason)

        fault_feedback = dict(fresh_feedback)
        fault_feedback[active[0]] = dict(fault_feedback[active[0]], fault_bits=1)
        fault_estimator = DummyEstimator(q_current, fault_feedback)
        reason = encoder_safety_stop_reason(
            safety=safety,
            estimator=fault_estimator,
            active_joints=active,
            mode="mit-signal",
        )
        self.assertIn("MOTOR FEEDBACK FAULT", reason)


if __name__ == "__main__":
    unittest.main()
