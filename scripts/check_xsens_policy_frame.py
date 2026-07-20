#!/usr/bin/env python3
"""Display Xsens raw and policy-frame IMU values for mounting validation."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imu_interface import (  # noqa: E402
    create_imu_sensor,
    imu_reading_quality,
    policy_frame_roll_pitch_from_gravity,
)


def fmt(values, precision=4):
    if values is None:
        return "none"
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in arr) + "]"


def print_expected_signs():
    print("Expected policy-frame signs with the current imu.yaml transform:")
    print("  level:          projected_gravity ~= [+0, +0, -1], roll ~= 0, pitch ~= 0")
    print("  raise front:    projected_gravity_x positive, pitch positive")
    print("  lower front:    projected_gravity_x negative, pitch negative")
    print("  roll left high: projected_gravity_y negative, roll positive")
    print("  roll right high: projected_gravity_y positive, roll negative")
    print("  yaw +Z:         gyro_z positive while rotating; gravity mostly unchanged")
    print("Only edit imu.yaml mounting signs/rotation after measuring a real sign mismatch.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--hz", "--rate", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--max-roll-pitch-deg", type=float, default=75.0)
    args = parser.parse_args()

    sensor = create_imu_sensor(source="xsens", port=args.port, baud=args.baud)
    dt = 1.0 / max(1.0e-6, float(args.hz))
    deadline = (
        time.monotonic() + float(args.seconds)
        if float(args.seconds) > 0.0
        else None
    )
    last_timestamp = None
    last_rate_time = None
    sample_rate = 0.0

    print_expected_signs()
    print("source=xsens port=", getattr(sensor, "port", args.port), sep="")
    print("Press Ctrl+C to stop.")

    try:
        while deadline is None or time.monotonic() < deadline:
            reading = sensor.read()
            now = time.monotonic()
            if reading is None:
                print("imu=no_new_data")
                time.sleep(dt)
                continue

            timestamp = float(reading.timestamp)
            if last_timestamp is not None and timestamp > last_timestamp:
                elapsed = timestamp - last_timestamp
                if elapsed > 1.0e-6:
                    instant_rate = 1.0 / elapsed
                    sample_rate = (
                        instant_rate
                        if last_rate_time is None
                        else 0.85 * sample_rate + 0.15 * instant_rate
                    )
                    last_rate_time = now
            last_timestamp = timestamp

            roll, pitch = policy_frame_roll_pitch_from_gravity(reading.projected_gravity_b)
            rpy = getattr(reading, "rpy_abs_deg", None)
            yaw = float(rpy[2]) if rpy is not None and len(rpy) >= 3 else float("nan")
            ok, reason = imu_reading_quality(
                reading,
                stale_timeout=float(getattr(sensor, "stale_timeout", 0.25)),
                max_roll_pitch_deg=float(args.max_roll_pitch_deg),
            )
            raw_quat = getattr(sensor, "latest_quaternion_wxyz", None)
            raw_gyro = getattr(sensor, "latest_gyro_sensor", None)
            age_ms = 1000.0 * (now - timestamp)
            print(
                "raw_quat="
                f"{fmt(raw_quat, 5)} "
                "raw_gyro="
                f"{fmt(raw_gyro, 5)} "
                "policy_gyro="
                f"{fmt(reading.base_ang_vel_b, 5)} "
                "projected_gravity="
                f"{fmt(reading.projected_gravity_b, 5)} "
                f"roll={np.degrees(roll):+.2f}deg "
                f"pitch={np.degrees(pitch):+.2f}deg "
                f"yaw={yaw:+.2f}deg "
                f"age={age_ms:.1f}ms "
                f"rate={sample_rate:.1f}Hz "
                f"valid={ok} "
                f"reason={reason}"
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sensor.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
