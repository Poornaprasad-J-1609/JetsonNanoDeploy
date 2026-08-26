#!/usr/bin/env python3
"""Standalone entry point for the guarded 12-motor pure-PD pose test."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import main_controller


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone Grallator RS04 pure-PD sit/stand hardware test"
    )
    parser.add_argument("--can-front", default="slcan0")
    parser.add_argument("--can-back", default="slcan1")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def build_controller_args(args):
    return [
        "--mode", "mit-signal",
        "--can-count", "2",
        "--can-ports", args.can_front, args.can_back,
        "--can-backend", "socketcan",
        "--can-bitrate", "1000000",
        "--feedback-source", "mit",
        "--joint-velocity-source", "finite-difference",
        "--command-source", "keyboard",
        "--imu-source", "xsens",
        "--imu-port", args.imu_port,
        "--imu-stale-timeout", "0.10",
        "--control-hz", "50",
        "--can-command-hz", "200",
        "--pose-test-only",
        "--pose-gains-config", str(ROOT / "config" / "sit_stand_test_gains.yaml"),
        "--sit-stand-trace-200hz",
        "--robot-mass-kg", "50",
        "--pose-test-max-temperature-c", "75",
        "--start-control-mode", "idle",
        "--startup-action", "hold",
        "--initial-zero-frame", "stand",
        "--no-auto-zero-on-startup",
        "--no-auto-stand-zero",
        "--no-auto-sit-zero",
        "--no-auto-policy-after-stand",
        "--no-stand-policy-stabilization",
        "--no-imu-stabilization",
        "--no-gait-assist",
        "--pose-transition-speed-rad-s", "0.40",
        "--pose-transition-min-seconds", "1.50",
        "--feedback-timeout", "0.05",
        "--fresh-feedback-max-age", "0.08",
        "--policy-steps", "0",
        "--log-every", "1",
        "--print-every", "10",
        "--log-prefix", "standalone_pure_pd_sit_stand",
        "--no-auto-push-log",
    ]


def main(argv=None):
    args = parse_args(argv)
    controller_args = build_controller_args(args)
    if args.print_command:
        print(shlex.join([sys.executable, str(ROOT / "src" / "main_controller.py"), *controller_args]))
        return 0
    sys.argv = [str(ROOT / "src" / "main_controller.py"), *controller_args]
    return int(main_controller.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
