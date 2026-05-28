#!/usr/bin/env python3
"""
Pure-policy walking test for partial hardware.

This script always computes the full 12-joint policy action. By default it also
sends MIT packets for all configured motor IDs; motors that are not connected
simply will not receive/respond. Use --joints only to choose which connected
joints are highlighted in the terminal.
"""
import argparse
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from imu_interface import create_imu_sensor, load_imu_config
from joystick_interface import CommandSource, load_joystick_defaults, load_speed_scale_defaults
from motor_command_layer import MotorCommandLayer, print_mit_commands
from policy_runner import PolicyRunner
from robstride_can_interface import ATUsbCan
from safety_monitor import SafetyMonitor
from state_estimator import FakeStateEstimator, MitFeedbackStateEstimator


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_motor_ids():
    return load_yaml(ROOT / "config" / "motor_ids.yaml")["motor_ids"]


def load_joint_can_bus():
    return load_yaml(ROOT / "config" / "motor_ids.yaml").get("joint_can_bus", {})


def fake_start_pose_array(runner, name):
    if name == "stand":
        return runner.q_stand.copy()
    if name == "crouch":
        return runner.q_crouch.copy()
    raise ValueError(f"Unknown fake start pose: {name}")


def joystick_walk_requested(command, threshold):
    command = np.asarray(command, dtype=np.float32)
    return bool(np.max(np.abs(command)) > float(threshold))


def refresh_estimator_feedback(estimator, timeout=0.0):
    if hasattr(estimator, "refresh_from_bus"):
        return estimator.refresh_from_bus(timeout=timeout)
    return 0


def feedback_torque_stats(estimator):
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {})
    if not feedback_by_joint:
        return None

    tau = np.asarray(
        [float(feedback["torque"]) for feedback in feedback_by_joint.values()],
        dtype=np.float32,
    )
    return float(np.abs(tau).mean()), float(np.abs(tau).max())


def estimator_imu_status(estimator):
    if hasattr(estimator, "imu_status"):
        return estimator.imu_status()
    return "none"


def command_speed_scale(command_source):
    if hasattr(command_source, "get_speed_scale"):
        return float(command_source.get_speed_scale())
    return 1.0


def joint_group_gain(joint_name, args):
    if "hip" in joint_name:
        return float(args.hip_output_gain)
    if "thigh" in joint_name:
        return float(args.thigh_output_gain)
    if "calf" in joint_name:
        return float(args.calf_output_gain)
    return 1.0


def apply_policy_output_gains(runner, q_policy_target, args):
    q_policy_target = np.asarray(q_policy_target, dtype=np.float32)
    gains = np.array(
        [
            float(args.policy_output_gain) * joint_group_gain(joint_name, args)
            for joint_name in runner.policy_order
        ],
        dtype=np.float32,
    )
    if np.any(gains < 0.0):
        raise ValueError("Policy output gains must be >= 0")

    return runner.q_default + (q_policy_target - runner.q_default) * gains


def safety_filter_for_test(safety, q_policy_target, q_previous_target, use_rate_limit):
    if use_rate_limit:
        return safety.safety_filter(q_policy_target, q_previous_target)

    safety.reload_control_limits()
    safety.reload_joint_limits()
    return safety.clip_q_target(q_policy_target).astype(np.float32)


def apply_test_step_limit(q_target, q_previous_target, max_target_step):
    max_target_step = float(max_target_step)
    if max_target_step <= 0.0:
        return np.asarray(q_target, dtype=np.float32)

    q_target = np.asarray(q_target, dtype=np.float32)
    q_previous_target = np.asarray(q_previous_target, dtype=np.float32)
    dq = np.clip(q_target - q_previous_target, -max_target_step, max_target_step)
    return (q_previous_target + dq).astype(np.float32)


def print_test_joint_debug(joints, commands, estimator, runner=None, action=None, q_policy_target=None):
    command_by_joint = {cmd["joint_name"]: cmd for cmd in commands}
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {})
    q_current = getattr(estimator, "q_current", None)
    qd_current = getattr(estimator, "qd_current", None)
    index_by_joint = getattr(estimator, "joint_index_by_name", {})
    policy_index_by_joint = {}
    if runner is not None:
        policy_index_by_joint = {
            joint_name: index for index, joint_name in enumerate(runner.policy_order)
        }

    for joint_name in joints:
        cmd = command_by_joint.get(joint_name)
        if cmd is None:
            print(f"  joint={joint_name:16s} no command generated")
            continue

        index = index_by_joint.get(joint_name)
        q_fb = None if q_current is None or index is None else float(q_current[index])
        qd_fb = None if qd_current is None or index is None else float(qd_current[index])
        tau_fb = None
        if joint_name in feedback_by_joint:
            tau_fb = float(feedback_by_joint[joint_name]["torque"])

        line = (
            f"  joint={joint_name:16s} "
            f"id=0x{cmd['motor_id']:02X} "
        )
        policy_index = policy_index_by_joint.get(joint_name)
        if action is not None and policy_index is not None:
            line += f"act={float(action[policy_index]):+.3f} "
        if q_policy_target is not None and policy_index is not None:
            line += f"q_policy={float(q_policy_target[policy_index]):+.3f} "
        line += (
            f"q_req={cmd['q_requested']:+.3f} "
            f"q_des={cmd['q_des']:+.3f}"
        )
        if q_fb is not None:
            line += f" q_fb={q_fb:+.3f} q_err={cmd['q_des'] - q_fb:+.3f}"
        if qd_fb is not None:
            line += f" qd_fb={qd_fb:+.3f}"
        if tau_fb is not None:
            line += f" tau_fb={tau_fb:+.3f}"
        else:
            line += " feedback=missing"
        print(line)


def make_command_source(args):
    return CommandSource(
        source=args.command_source,
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
        use_hat_fallback=args.use_hat_fallback,
        speed_scale_initial=args.speed_scale_initial,
        speed_scale_min=args.speed_scale_min,
        speed_scale_max=args.speed_scale_max,
        speed_scale_step=args.speed_scale_step,
        deadzone=args.deadzone,
        expo=args.expo,
        smoothing=args.smoothing,
    )


def main():
    joystick_defaults = load_joystick_defaults()
    speed_defaults = load_speed_scale_defaults()
    imu_defaults = load_imu_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["print", "mit-signal"], default="mit-signal")
    parser.add_argument("--port", default="/dev/ttyUSB1",
                        help="fallback USB-CAN port used when --port-front / --port-back are not given")
    parser.add_argument("--port-front", default=None,
                        help="USB-CAN port for front legs (FL/FR); overrides --port")
    parser.add_argument("--port-back", default=None,
                        help="USB-CAN port for back legs (BL/BR); overrides --port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--joints",
        "--test-joints",
        nargs="+",
        default=None,
        help="connected joints to highlight in telemetry; default highlights all",
    )
    parser.add_argument(
        "--send-all-policy-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send commands for every motor ID in motor_ids.yaml; unconnected motors ignore them",
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--policy-activation", choices=["elu", "relu", "tanh", "identity", "none"], default="elu")

    parser.add_argument("--command-source", choices=["fixed", "joystick"], default="joystick")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--max-vx", type=float, default=float(joystick_defaults["speed_limits"]["max_vx"]))
    parser.add_argument("--max-vy", type=float, default=float(joystick_defaults["speed_limits"]["max_vy"]))
    parser.add_argument("--max-yaw", type=float, default=float(joystick_defaults["speed_limits"]["max_yaw"]))
    parser.add_argument("--axis-vx", type=int, default=int(joystick_defaults["axes"]["vx_axis"]))
    parser.add_argument("--axis-vy", type=int, default=int(joystick_defaults["axes"]["vy_axis"]))
    parser.add_argument("--axis-yaw", type=int, default=int(joystick_defaults["axes"]["yaw_axis"]))
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument(
        "--joystick-wait-seconds",
        type=float,
        default=float(joystick_defaults.get("connection", {}).get("wait_seconds", 5.0)),
    )
    parser.add_argument("--invert-vx", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["vx"]))
    parser.add_argument("--invert-vy", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["vy"]))
    parser.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["yaw"]))
    parser.add_argument(
        "--use-hat-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(joystick_defaults["filter"].get("use_hat_fallback", True)),
    )
    parser.add_argument("--deadzone", type=float, default=float(joystick_defaults["filter"]["deadzone"]))
    parser.add_argument("--expo", type=float, default=float(joystick_defaults["filter"]["expo"]))
    parser.add_argument("--smoothing", type=float, default=float(joystick_defaults["filter"]["smoothing"]))
    parser.add_argument("--speed-scale-initial", type=float, default=speed_defaults["initial"])
    parser.add_argument("--speed-scale-min", type=float, default=speed_defaults["min"])
    parser.add_argument("--speed-scale-max", type=float, default=speed_defaults["max"])
    parser.add_argument("--speed-scale-step", type=float, default=speed_defaults["step"])

    parser.add_argument("--imu-source", choices=["auto", "fake", "none", "xsens", "serial-json", "serial-csv"], default="auto")
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument("--feedback-timeout", type=float, default=0.05)
    parser.add_argument("--policy-steps", type=int, default=0, help="0 or negative runs until Ctrl+C/emergency")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--show-hex", action="store_true")
    parser.add_argument("--fake-start", choices=["stand", "crouch"], default="stand")
    parser.add_argument("--walk-command-threshold", type=float, default=0.02)
    parser.add_argument(
        "--mit-phase",
        choices=["startup", "policy"],
        default="startup",
        help="MIT gains to use while sending policy radians; startup is much softer",
    )
    parser.add_argument(
        "--max-target-step",
        type=float,
        default=0.01,
        help="extra partial-test target slew cap in rad/control-step; <=0 disables it",
    )
    parser.add_argument(
        "--policy-output-gain",
        type=float,
        default=1.0,
        help="tester-only multiplier for policy radians around default pose",
    )
    parser.add_argument(
        "--hip-output-gain",
        type=float,
        default=1.0,
        help="extra tester-only multiplier for hip policy radians",
    )
    parser.add_argument(
        "--thigh-output-gain",
        type=float,
        default=1.0,
        help="extra tester-only multiplier for thigh policy radians",
    )
    parser.add_argument(
        "--calf-output-gain",
        type=float,
        default=1.0,
        help="extra tester-only multiplier for calf policy radians",
    )
    parser.add_argument(
        "--rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply target slew-rate limiting; --no-rate-limit keeps hard angle limits only",
    )

    args = parser.parse_args()
    args.log_every = max(1, args.log_every)
    steps = None if args.policy_steps <= 0 else args.policy_steps

    port_front = args.port_front if args.port_front is not None else args.port
    port_back  = args.port_back  if args.port_back  is not None else args.port

    active_imu_source = args.imu_source
    if active_imu_source == "auto":
        active_imu_source = imu_defaults.get("source", "fake")
    active_imu_source = str(active_imu_source).replace("-", "_").lower()
    active_imu_port = args.imu_port if args.imu_port is not None else imu_defaults.get("port")
    if (
        args.mode == "mit-signal"
        and active_imu_source in ("xsens", "xsens_binary", "mtdata2", "serial_json", "serial_csv")
        and active_imu_port in {port_front, port_back}
    ):
        print("ERROR: IMU port", active_imu_port, "conflicts with a CAN bus port.")
        print("CAN ports in use:", sorted({port_front, port_back}))
        print("Use a separate device, e.g. --imu-port /dev/ttyUSB2")
        return 1

    runner = PolicyRunner(policy_path=args.policy_path, policy_activation=args.policy_activation)
    safety = SafetyMonitor(runner.policy_order)
    motor_ids = load_motor_ids()
    joint_can_bus = load_joint_can_bus()
    test_joints = list(args.joints or runner.policy_order)
    send_joints = runner.policy_order if args.send_all_policy_joints else test_joints

    unknown = [joint for joint in test_joints if joint not in runner.policy_order]
    if unknown:
        raise KeyError(f"Unknown test joint(s): {unknown}")

    motor_layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=send_joints,
        joint_can_bus=joint_can_bus,
    )
    command_source = make_command_source(args)
    q_fake_start = fake_start_pose_array(runner, args.fake_start)

    buses = None
    imu_sensor = None
    previous_action = np.zeros(len(runner.policy_order), dtype=np.float32)

    print("==== PURE POLICY PARTIAL-HARDWARE TEST ====")
    print("Mode:", args.mode)
    print("Port front (FL/FR):", port_front)
    print("Port back  (BL/BR):", port_back)
    print("Policy:", runner.policy_path)
    print("Policy obs/actions:", runner.observation_dim, runner.action_dim)
    print("Control dt:", runner.control_dt)
    print("Command source:", args.command_source)
    print("Initial command:", command_source.read())
    print("Base linear velocity input: [0, 0, 0]")
    print(
        "Policy output gains:",
        f"all={args.policy_output_gain:.2f}",
        f"hip={args.hip_output_gain:.2f}",
        f"thigh={args.thigh_output_gain:.2f}",
        f"calf={args.calf_output_gain:.2f}",
    )
    print("MIT command phase:", args.mit_phase)
    print("Extra max target step:", f"{args.max_target_step:.4f} rad/step")
    print("Rate limit:", "on" if args.rate_limit else "off; hard angle limits still on")
    print("Sending commands to:", "all policy joints" if args.send_all_policy_joints else ", ".join(send_joints))
    print("Telemetry joints:", ", ".join(test_joints))
    print("No gait assist. No hard-coded walking overlay.")
    print()

    try:
        imu_sensor = create_imu_sensor(
            source=args.imu_source,
            port=args.imu_port,
            baud=args.imu_baud,
        )
        if imu_sensor is None:
            print("IMU disabled; gyro/gravity observations use estimator fallback.")
        else:
            print(
                "IMU opened:",
                getattr(imu_sensor, "source_name", "imu"),
                "port:",
                getattr(imu_sensor, "port", "none"),
            )

        if args.mode == "mit-signal":
            print("Opening USB-CAN ports...")
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
                q_initial=q_fake_start,
                policy_order=runner.policy_order,
                motor_ids=motor_ids,
                motor_layer=motor_layer,
                bus=buses,
                imu_sensor=imu_sensor,
            )
            print("Polling feedback and enabling sent joints...")
            motor_layer.send_raw_commands(buses, motor_layer.build_feedback_poll_commands())
            estimator.refresh_from_bus(timeout=0.10)
            motor_layer.send_raw_commands(buses, motor_layer.build_enable_commands())
        else:
            estimator = FakeStateEstimator(q_initial=q_fake_start, imu_sensor=imu_sensor)

        refresh_estimator_feedback(estimator, timeout=args.feedback_timeout)
        q_current, _, _, _, _ = estimator.read()
        q_previous_target = q_current.copy()
        has_motion_target = False
        step = 0

        print("Running. Move joystick to run policy; release to hold. Ctrl+C stops.")
        while steps is None or step < steps:
            action = np.zeros(len(runner.policy_order), dtype=np.float32)
            q_policy_target = q_previous_target.copy()
            (
                q_current,
                qd_current,
                _base_lin_vel_b,
                base_ang_vel_b,
                projected_gravity_b,
            ) = estimator.read()

            if hasattr(estimator, "imu_stale") and estimator.imu_stale():
                print("\nSTOP: IMU data missing or stale")
                break

            stop, reason = safety.emergency_stop_check(
                projected_gravity_b=projected_gravity_b,
                base_ang_vel_b=base_ang_vel_b,
            )
            if stop:
                print("\nSTOP:", reason)
                break

            emergency_reason = command_source.get_emergency_stop_request()
            if emergency_reason is not None:
                print("\nSTOP:", emergency_reason)
                break

            command = command_source.read()
            walk_requested = joystick_walk_requested(command, args.walk_command_threshold)

            if walk_requested:
                obs = runner.build_observation(
                    base_lin_vel_b=np.zeros(3, dtype=np.float32),
                    base_ang_vel_b=base_ang_vel_b,
                    projected_gravity_b=projected_gravity_b,
                    command=command,
                    q_current=q_current,
                    qd_current=qd_current,
                    previous_action=previous_action,
                )
                action = runner.infer_action(obs)
                q_policy_target = runner.action_to_q_target(action)
                q_policy_target = apply_policy_output_gains(
                    runner=runner,
                    q_policy_target=q_policy_target,
                    args=args,
                )
                q_safe_target = safety_filter_for_test(
                    safety=safety,
                    q_policy_target=q_policy_target,
                    q_previous_target=q_previous_target,
                    use_rate_limit=args.rate_limit,
                )
                q_safe_target = apply_test_step_limit(
                    q_target=q_safe_target,
                    q_previous_target=q_previous_target,
                    max_target_step=args.max_target_step,
                )
                commands = motor_layer.build_mit_commands(q_safe_target, phase=args.mit_phase)
                previous_action = action.copy()
                has_motion_target = True
                active_mode = "policy"
            else:
                q_safe_target = q_previous_target.copy()
                commands = (
                    motor_layer.build_mit_commands(q_safe_target, phase="startup")
                    if has_motion_target
                    else []
                )
                previous_action = np.zeros(len(runner.policy_order), dtype=np.float32)
                active_mode = "hold"

            if args.mode == "mit-signal":
                motor_layer.send_signal_commands(buses, commands)

            refresh_estimator_feedback(estimator, timeout=args.feedback_timeout)

            if step % args.log_every == 0:
                command = np.asarray(command, dtype=np.float32)
                tau_fb = feedback_torque_stats(estimator)
                line = (
                    f"step={step:06d} mode={active_mode:6s} "
                    f"vx={command[0]:+.3f} vy={command[1]:+.3f} yaw={command[2]:+.3f} "
                    f"speed={command_speed_scale(command_source):.2f} "
                    f"imu={estimator_imu_status(estimator)} "
                    f"act_max={float(np.max(np.abs(action))):+.3f} "
                    f"cmds={len(commands):02d}"
                )
                if tau_fb is None:
                    line += " tau_fb=na tau_fb_max=na"
                else:
                    line += f" tau_fb={tau_fb[0]:+.3f} tau_fb_max={tau_fb[1]:+.3f}"
                print(line)
                if args.show_hex and commands:
                    print_mit_commands([cmd for cmd in commands if cmd["joint_name"] in test_joints], show_hex=True)
                print_test_joint_debug(
                    test_joints,
                    commands,
                    estimator,
                    runner=runner,
                    action=action,
                    q_policy_target=q_policy_target,
                )

            q_previous_target = q_safe_target.copy()
            step += 1
            time.sleep(runner.control_dt)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt: stopping.")
    finally:
        command_source.close()
        if imu_sensor is not None:
            imu_sensor.close()
        if buses is not None:
            try:
                print("\nSending stop frames...")
                motor_layer.send_raw_commands(buses, motor_layer.build_stop_commands())
                print("Stop frames sent.")
            except Exception as exc:
                print("WARNING: failed to send stop frames:", exc)
            closed_ids = set()
            for b in buses.values():
                if id(b) not in closed_ids:
                    b.close()
                    closed_ids.add(id(b))
            print("USB-CAN closed.")

    print("Policy walk joint test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
