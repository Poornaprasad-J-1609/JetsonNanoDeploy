#!/usr/bin/env python3
"""
Direct MIT joint motion test.

This is only a hardware diagnostic. It bypasses policy, joystick, and IMU so we
can prove whether a selected motor tracks MIT position commands at all.
"""
import argparse
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from motor_command_layer import (
    MotorCommandLayer,
    motor_position_to_joint_angle,
    print_mit_commands,
)
from policy_runner import PolicyRunner
from robstride_can_interface import ATUsbCan
from state_estimator import MitFeedbackStateEstimator


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_motor_ids():
    return load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]


def load_joint_can_bus():
    return load_yaml(ROOT / "config" / "motor_ids.yaml").get("joint_can_bus", {})


def print_joint_feedback(step, commands, estimator):
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {})
    for cmd in commands:
        joint_name = cmd["joint_name"]
        feedback = feedback_by_joint.get(joint_name)
        if feedback is None:
            print(
                f"step={step:05d} {joint_name:16s} "
                f"id=0x{cmd['motor_id']:02X} q_des={cmd['q_des']:+.3f} feedback=missing"
            )
            continue

        q_fb = float(feedback.get(
            "joint_position",
            motor_position_to_joint_angle(
                feedback["position"],
                offset=float(cmd["offset"]),
                direction=float(cmd.get("direction", 1.0)),
                reference=float(cmd["q_des"]),
            ),
        ))
        print(
            f"step={step:05d} {joint_name:16s} "
            f"id=0x{cmd['motor_id']:02X} "
            f"q_des={cmd['q_des']:+.3f} "
            f"q_fb={q_fb:+.3f} "
            f"q_err={cmd['q_des'] - q_fb:+.3f} "
            f"vel={float(feedback['velocity']):+.3f} "
            f"tau={float(feedback['torque']):+.3f} "
            f"fault=0x{int(feedback['fault_bits']):02X} "
            f"mode={int(feedback['mode_status'])}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB1",
                        help="fallback USB-CAN port used when --port-front / --port-back are not given")
    parser.add_argument("--port-front", default=None,
                        help="USB-CAN port for front legs (FL/FR); overrides --port")
    parser.add_argument("--port-back", default=None,
                        help="USB-CAN port for back legs (BL/BR); overrides --port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--joints",
        nargs="+",
        required=True,
        help="joint names to move, for example FR_thigh_joint FR_calf_joint",
    )
    parser.add_argument("--amplitude", type=float, default=0.20)
    parser.add_argument("--frequency", type=float, default=0.35)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--show-hex", action="store_true")
    parser.add_argument(
        "--start-from-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start sine motion around measured feedback pose",
    )
    parser.add_argument(
        "--hold-before-seconds",
        type=float,
        default=1.0,
        help="hold measured pose before moving",
    )
    args = parser.parse_args()
    args.log_every = max(1, args.log_every)

    port_front = args.port_front if args.port_front is not None else args.port
    port_back  = args.port_back  if args.port_back  is not None else args.port

    runner = PolicyRunner()
    motor_ids = load_motor_ids()
    joint_can_bus = load_joint_can_bus()
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=args.joints,
        joint_can_bus=joint_can_bus,
    )

    print("==== DIRECT MIT JOINT MOTION TEST ====")
    print("This bypasses policy, joystick, and IMU.")
    print("Joints:", ", ".join(args.joints))
    print("Port front (FL/FR):", port_front)
    print("Port back  (BL/BR):", port_back)
    print("Amplitude:", args.amplitude)
    print("Frequency:", args.frequency)
    print()

    buses = None
    try:
        bus_front = ATUsbCan(port=port_front, baud=args.baud).open()
        print(f"USB-CAN front ({port_front}) opened.")
        try:
            if port_back != port_front:
                bus_back = ATUsbCan(port=port_back, baud=args.baud).open()
                print(f"USB-CAN back ({port_back}) opened.")
            else:
                bus_back = bus_front
                print(f"USB-CAN back shares front port ({port_front}).")
        except Exception:
            bus_front.close()
            raise
        buses = {"front": bus_front, "back": bus_back}

        estimator = MitFeedbackStateEstimator(
            q_initial=runner.q_stand,
            policy_order=runner.policy_order,
            motor_ids=motor_ids,
            motor_layer=layer,
            bus=buses,
            imu_sensor=None,
            pose_references={
                "default": runner.q_default,
                "stand": runner.q_stand,
                "crouch": runner.q_crouch,
            },
            pose_snap_tolerance=0.35,
        )

        print("Polling feedback...")
        layer.send_raw_commands(buses, layer.build_feedback_poll_commands())
        estimator.refresh_from_bus(timeout=0.20)

        print("Sending enable frames...")
        layer.send_raw_commands(buses, layer.build_enable_commands())

        q_center = runner.q_stand.copy()
        if args.start_from_feedback:
            q_current, _, _, _, _ = estimator.read()
            q_center = q_current.copy()

        print("Center pose:")
        for joint_name in args.joints:
            index = runner.policy_order.index(joint_name)
            print(f"  {joint_name:16s} center={q_center[index]:+.3f} id=0x{int(motor_ids[joint_name]):02X}")

        dt = runner.control_dt
        hold_steps = max(0, int(args.hold_before_seconds / dt))
        total_steps = max(1, int(args.seconds / dt))

        for step in range(hold_steps + total_steps):
            moving_step = step - hold_steps
            q_target = q_center.copy()

            if moving_step >= 0:
                phase = 2.0 * np.pi * float(args.frequency) * moving_step * dt
                for joint_number, joint_name in enumerate(args.joints):
                    index = runner.policy_order.index(joint_name)
                    joint_phase = phase + joint_number * 0.5 * np.pi
                    q_target[index] = q_center[index] + float(args.amplitude) * np.sin(joint_phase)

            commands = layer.build_mit_commands(
                q_target,
                phase="policy",
                feedback_by_joint=getattr(estimator, "last_feedback_by_joint", None),
            )
            layer.send_signal_commands(buses, commands)
            estimator.refresh_from_bus(timeout=0.02)

            if step % args.log_every == 0:
                print_joint_feedback(step, commands, estimator)
                if args.show_hex:
                    print_mit_commands(commands, show_hex=True)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt: stopping.")
    finally:
        if buses is not None:
            try:
                print("\nSending stop frames...")
                layer.send_raw_commands(buses, layer.build_stop_commands())
                print("Stop frames sent.")
            except Exception as exc:
                print("WARNING: failed to send stop frames:", exc)
            closed_ids = set()
            for b in buses.values():
                if id(b) not in closed_ids:
                    b.close()
                    closed_ids.add(id(b))
            print("USB-CAN closed.")

    print("Direct MIT joint motion test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
