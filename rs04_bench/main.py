from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from .analysis.plant_identification import identify_plant, recommend_kd
from .analysis.step_response import analyze_step_response
from .config import load_config
from .control.loop import BenchController
from .experiments.manager import ExperimentSpec
from .motor.mock import MockMotorInterface
from .motor.robstride import RobStrideMotorInterface


def parser():
    result = argparse.ArgumentParser(description="RS04 200 Hz actuator characterization bench")
    result.add_argument("--config", help="optional YAML overriding rs04_bench/config.yaml")
    result.add_argument("--mock", action="store_true", help="simulate a second-order motor; never accesses CAN")
    result.add_argument("--interface", help="SocketCAN interface, for example slcan0")
    result.add_argument("--motor-id", type=lambda value: int(value, 0), help="RS04 CAN motor ID")
    result.add_argument("--bitrate", type=int, help="CAN bitrate")
    result.add_argument("--no-gui", action="store_true", help="run a headless mock/demo or offline analysis")
    result.add_argument("--demo-step", action="store_true", help="run a safe 0.1 rad mock step and exit")
    result.add_argument("--analyze", metavar="CSV", help="offline step and plant analysis")
    result.add_argument("--torque-source", choices=("auto", "measured", "estimated", "commanded"), default="auto")
    result.add_argument("--desired-kp", type=float, default=80.0)
    result.add_argument("--desired-zeta", type=float, default=0.7)
    return result


def build_config(args):
    override = {"motor": {}}
    if args.interface:
        override["motor"]["interface"] = args.interface
    if args.motor_id is not None:
        override["motor"]["id"] = args.motor_id
    if args.bitrate is not None:
        override["motor"]["bitrate"] = args.bitrate
    return load_config(args.config, override if override["motor"] else None)


def offline_analysis(args, config):
    output = {}
    try:
        output["step_response"] = analyze_step_response(
            args.analyze, config.analysis.settling_band_fraction
        )
    except Exception as exc:
        output["step_response"] = {"valid": False, "reason": str(exc)}
    try:
        plant = identify_plant(
            args.analyze, vars(config.pendulum), config.analysis.filter_window,
            config.analysis.filter_polynomial_order, config.analysis.velocity_deadband_rad_s,
            config.analysis.minimum_identification_samples,
            config.analysis.minimum_excitation_velocity_rad_s,
            args.torque_source,
        )
        plant.pop("signals", None)
        output["plant_identification"] = plant
        if plant["valid"]:
            output["gain_starting_estimate"] = recommend_kd(
                plant["estimated_inertia_kg_m2"],
                plant["estimated_viscous_damping_nm_s_rad"],
                args.desired_kp, args.desired_zeta,
            )
    except Exception as exc:
        output["plant_identification"] = {"valid": False, "reason": str(exc)}
    print(json.dumps(output, indent=2, default=str))
    return 0


def demo_step(controller, config):
    controller.start()
    controller.connect()
    controller.enable()
    spec = ExperimentSpec(
        mode="step", kp=config.control.initial_kp, kd=config.control.initial_kd,
        parameters={"initial_position": 0.0, "step_amplitude": 0.1,
                    "pre_hold_s": 0.5, "post_duration_s": 2.0},
        notes="headless mock validation step",
    )
    controller.start_experiment(spec)
    deadline = time.monotonic() + 5.0
    while controller.snapshot().experiment_active and time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if snapshot.safety_event:
            raise RuntimeError(snapshot.safety_event)
        time.sleep(0.02)
    while controller.logger.active and time.monotonic() < deadline + 3.0:
        time.sleep(0.02)
    path = controller.snapshot().last_csv_path
    controller.shutdown()
    if not path:
        raise RuntimeError("mock step did not produce a CSV")
    result = analyze_step_response(path, config.analysis.settling_band_fraction)
    print("Mock step CSV:", path)
    print(json.dumps(result, indent=2, default=str))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    config = build_config(args)
    if args.analyze:
        return offline_analysis(args, config)
    if args.no_gui and not args.demo_step:
        parser().error("--no-gui requires --demo-step or --analyze")
    if args.demo_step and not args.mock:
        parser().error("--demo-step is intentionally restricted to --mock")
    motor = (
        MockMotorInterface(config.mock)
        if args.mock else
        RobStrideMotorInterface(config.motor.interface, config.motor.id, config.motor.bitrate)
    )
    controller = BenchController(motor, config)
    if args.demo_step:
        return demo_step(controller, config)
    try:
        import tkinter as tk
        from .gui.main_window import BenchMainWindow
    except ImportError as exc:
        raise SystemExit(f"GUI dependency unavailable: {exc}. Install python3-tk and matplotlib.")
    root = tk.Tk()
    controller.start()
    BenchMainWindow(root, controller, config)
    try:
        root.mainloop()
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
