#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml

from four_bar_transmission import (
    FourBarTransmissionSet,
    LookupProfile,
    TransmissionConfigurationError,
    TransmissionRangeError,
)


class FourBarTransmissionTests(unittest.TestCase):
    def make_config(self, enabled=True):
        return {
            "four_bar_transmission": {
                "enabled": enabled,
                "require_feedback_for_commands": True,
                "clamp_policy_to_hard_limits": True,
                "profiles": {
                    "p": {
                        # q = 0.5 * theta + 0.1
                        "motor_angle_rad": [0.0, 1.0, 2.0, 3.0],
                        "knee_angle_rad": [0.1, 0.6, 1.1, 1.6],
                        "efficiency": 0.8,
                        "min_abs_jacobian": 0.05,
                        "motor_torque_limit_nm": 120.0,
                    }
                },
                "joints": {
                    "L_calf_joint": {
                        "profile": "p",
                        "virtual_sign": 1,
                    },
                    "R_calf_joint": {
                        "profile": "p",
                        "virtual_sign": -1,
                    },
                },
            }
        }

    def transmission_set(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "four_bar.yaml"
        path.write_text(yaml.safe_dump(self.make_config()), encoding="utf-8")
        return FourBarTransmissionSet.from_yaml(
            path,
            policy_order=["L_calf_joint", "R_calf_joint"],
        )

    def test_forward_inverse_and_jacobian(self):
        tx = self.transmission_set()
        self.assertAlmostEqual(
            tx.virtual_from_motor("L_calf_joint", 1.4), 0.8, places=7
        )
        self.assertAlmostEqual(
            tx.motor_from_virtual("L_calf_joint", 0.8), 1.4, places=7
        )
        self.assertAlmostEqual(
            tx.jacobian_from_motor("L_calf_joint", 1.4), 0.5, places=7
        )

    def test_right_side_virtual_sign(self):
        tx = self.transmission_set()
        self.assertAlmostEqual(
            tx.virtual_from_motor("R_calf_joint", 1.4), -0.8, places=7
        )
        self.assertAlmostEqual(
            tx.motor_from_virtual("R_calf_joint", -0.8), 1.4, places=7
        )
        self.assertAlmostEqual(
            tx.jacobian_from_motor("R_calf_joint", 1.4), -0.5, places=7
        )

    def test_velocity_torque_and_gain_mapping(self):
        tx = self.transmission_set()
        self.assertAlmostEqual(
            tx.virtual_velocity_from_motor("L_calf_joint", 1.0, 4.0),
            2.0,
            places=7,
        )
        self.assertAlmostEqual(
            tx.virtual_torque_from_motor("L_calf_joint", 1.0, 10.0),
            16.0,
            places=7,
        )
        self.assertAlmostEqual(
            tx.motor_torque_from_virtual("L_calf_joint", 1.0, 16.0),
            10.0,
            places=7,
        )
        kp_m, kd_m = tx.motor_gains_from_virtual(
            "L_calf_joint", 1.0, 100.0, 4.0
        )
        self.assertAlmostEqual(kp_m, 31.25, places=7)
        self.assertAlmostEqual(kd_m, 1.25, places=7)

    def test_feedback_decode_returns_virtual_joint_state(self):
        tx = self.transmission_set()
        decoded = tx.decode_feedback(
            joint_name="L_calf_joint",
            position_raw=1.5,
            velocity_raw=4.0,
            torque_raw=10.0,
            offset=0.5,
            direction=1.0,
        )
        self.assertAlmostEqual(decoded["motor_position"], 1.0)
        self.assertAlmostEqual(decoded["joint_position"], 0.6)
        self.assertAlmostEqual(decoded["joint_velocity"], 2.0)
        self.assertAlmostEqual(decoded["joint_torque"], 16.0)

    def test_out_of_range_refuses_extrapolation(self):
        tx = self.transmission_set()
        with self.assertRaises(TransmissionRangeError):
            tx.virtual_from_motor("L_calf_joint", 5.0)

    def test_endpoint_tolerance_clamps_small_capture_noise(self):
        tx = self.transmission_set()
        self.assertAlmostEqual(
            tx.virtual_from_motor("L_calf_joint", 3.005),
            1.6,
            places=7,
        )
        with self.assertRaises(TransmissionRangeError):
            tx.virtual_from_motor("L_calf_joint", 3.05)

    def test_disabled_configuration_is_identity(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "off.yaml"
        path.write_text(
            yaml.safe_dump({"four_bar_transmission": {"enabled": False}}),
            encoding="utf-8",
        )
        tx = FourBarTransmissionSet.from_yaml(path)
        self.assertEqual(tx.virtual_from_motor("anything", 1.2), 1.2)
        self.assertEqual(tx.motor_from_virtual("anything", -0.7), -0.7)
        self.assertEqual(tx.jacobian_from_motor("anything", 2.0), 1.0)

    def test_nonmonotonic_table_is_rejected(self):
        with self.assertRaises(TransmissionConfigurationError):
            LookupProfile.from_config(
                "bad",
                {
                    "motor_angle_rad": [0.0, 1.0, 2.0],
                    "knee_angle_rad": [0.0, 1.0, 0.5],
                },
            )


class FourBarCommandAdapterTests(unittest.TestCase):
    """Exercise the wrapper against a small MotorCommandLayer-compatible stub."""

    def setUp(self):
        self.old_module = sys.modules.get("motor_command_layer")
        stub = types.ModuleType("motor_command_layer")

        class BaseMotorCommandLayer:
            def __init__(
                self,
                policy_order,
                motor_ids,
                active_joints=None,
                joint_can_bus=None,
            ):
                self.policy_order = list(policy_order)
                self.motor_ids = dict(motor_ids)
                self.active_joints = list(active_joints or policy_order)
                self.joint_can_bus = dict(joint_can_bus or {})
                self.policy_index_by_joint = {
                    name: i for i, name in enumerate(self.policy_order)
                }
                self.hard_joint_limits = {"L_calf_joint": (0.1, 1.6)}
                self.policy_target_limits = {"L_calf_joint": (0.1, 1.6)}
                self.joint_offsets = {"L_calf_joint": 0.5}
                self.joint_directions = {"L_calf_joint": 1.0}
                self.joint_coordinate_shifts = {"L_calf_joint": 0.0}
                self.proto = {
                    "p_min": -12.5,
                    "p_max": 12.5,
                    "v_min": -30.0,
                    "v_max": 30.0,
                    "tau_min": -120.0,
                    "tau_max": 120.0,
                }
                self.command_proto = dict(self.proto)
                self.command_proto.update(
                    {
                        "kp_min": 0.0,
                        "kp_max": 500.0,
                        "kd_min": 0.0,
                        "kd_max": 20.0,
                    }
                )

            def apply_hard_joint_limit(self, joint_name, q_des, phase=None):
                low, high = self.hard_joint_limits[joint_name]
                return max(low, min(high, float(q_des)))

            def apply_hard_joint_limit_to_motor_position(
                self, joint_name, p_des, offset, direction=1.0
            ):
                q = float(direction) * (float(p_des) - float(offset))
                q = self.apply_hard_joint_limit(joint_name, q)
                return float(offset) + float(direction) * q, q

            def reload_joint_limits(self, force=False):
                return True

            def command_proto_for_phase(self, phase):
                return self.command_proto

            def _effective_unsigned_wire_value(self, value, field, command_proto=None):
                return float(value)

            def _effective_signed_wire_value(self, value, field, command_proto=None):
                return float(value)

            def apply_mit_parameter_limits(self, p_des, v_des, kp, kd, tau_ff):
                return (
                    max(-12.5, min(12.5, float(p_des))),
                    max(-30.0, min(30.0, float(v_des))),
                    max(0.0, min(500.0, float(kp))),
                    max(0.0, min(20.0, float(kd))),
                    max(-120.0, min(120.0, float(tau_ff))),
                )

            def build_mit_commands(
                self,
                q_target,
                phase="policy",
                feedback_by_joint=None,
                joint_velocity_target=None,
            ):
                return [
                    {
                        "joint_name": "L_calf_joint",
                        "motor_id": 7,
                        "bus_name": "front",
                        "phase": phase,
                        "command_encoding": "test",
                        "q_des": float(q_target[0]),
                        "q_requested": float(q_target[0]),
                        "q_before_torque_limit": float(q_target[0]),
                        "torque_limited": False,
                        "tau_pd_est": None,
                        "offset": 0.5,
                        "direction": 1.0,
                        "p_des": 1.3,
                        "p_base": 1.3,
                        "p_limit_adjustment": 0.0,
                        "joint_v_des": 0.0,
                        "joint_v_des_requested": 0.0,
                        "v_des": 0.0,
                        "kp": 100.0,
                        "kd": 4.0,
                        "kp_effective": 100.0,
                        "kd_effective": 4.0,
                        "joint_tau_ff": 8.0,
                        "joint_tau_ff_effective": 8.0,
                        "tau_ff": 8.0,
                        "can_id": 0,
                        "data": bytes(8),
                    }
                ]

        def mit_can_id(motor_id, proto, tau_ff=0.0):
            return int(motor_id) << 8

        def pack_mit_command(p_des, v_des, kp, kd, proto):
            return bytes(8)

        stub.MotorCommandLayer = BaseMotorCommandLayer
        stub.mit_can_id = mit_can_id
        stub.pack_mit_command = pack_mit_command
        sys.modules["motor_command_layer"] = stub

    def tearDown(self):
        sys.modules.pop("four_bar_motor_command_layer_test", None)
        if self.old_module is None:
            sys.modules.pop("motor_command_layer", None)
        else:
            sys.modules["motor_command_layer"] = self.old_module

    def load_wrapper(self):
        source = Path(__file__).resolve().parents[1] / "src" / "four_bar_motor_command_layer.py"
        spec = importlib.util.spec_from_file_location(
            "four_bar_motor_command_layer_test", source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.FourBarMotorCommandLayer

    def test_command_is_rewritten_in_motor_space(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg_path = Path(tmp.name) / "four_bar.yaml"
        cfg = {
            "four_bar_transmission": {
                "enabled": True,
                "require_feedback_for_commands": True,
                "clamp_policy_to_hard_limits": True,
                "profiles": {
                    "p": {
                        "motor_angle_rad": [0.0, 1.0, 2.0, 3.0],
                        "knee_angle_rad": [0.1, 0.6, 1.1, 1.6],
                        "efficiency": 0.8,
                        "min_abs_jacobian": 0.05,
                        "motor_torque_limit_nm": 120.0,
                    }
                },
                "joints": {
                    "L_calf_joint": {
                        "profile": "p",
                        "virtual_sign": 1,
                    }
                },
            }
        }
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        Wrapper = self.load_wrapper()
        layer = Wrapper(
            policy_order=["L_calf_joint"],
            motor_ids={"L_calf_joint": 7},
            transmission_config_path=str(cfg_path),
        )
        feedback = {
            "L_calf_joint": {
                "joint_position": 0.6,
                "joint_velocity": 0.2,
                "motor_position": 1.0,
                "motor_velocity": 0.4,
            }
        }
        command = layer.build_mit_commands(
            q_target=[0.8],
            phase="policy",
            feedback_by_joint=feedback,
        )[0]

        # q=0.8 -> theta=1.4 and raw p=offset+theta=1.9.
        self.assertAlmostEqual(command["motor_position_des"], 1.4, places=7)
        self.assertAlmostEqual(command["p_des"], 1.9, places=7)
        self.assertAlmostEqual(command["transmission_jacobian"], 0.5, places=7)
        self.assertAlmostEqual(command["kp_virtual"], 100.0, places=7)
        self.assertAlmostEqual(command["kd_virtual"], 4.0, places=7)
        self.assertAlmostEqual(command["kp"], 31.25, places=7)
        self.assertAlmostEqual(command["kd"], 1.25, places=7)
        self.assertAlmostEqual(command["tau_ff"], 5.0, places=7)
        self.assertAlmostEqual(command["q_des"], 0.8, places=7)
        self.assertTrue(command["transmission_enabled"])


if __name__ == "__main__":
    unittest.main()
