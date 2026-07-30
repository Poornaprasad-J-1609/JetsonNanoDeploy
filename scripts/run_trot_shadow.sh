#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

IMU_PORT="${IMU_PORT:-/dev/ttyUSB0}"
CLOCK_FORMULA="${CLOCK_FORMULA:-sin-cos-zero}"
CLOCK_HZ="${CLOCK_HZ:-1.5}"
POLICY_STEPS="${POLICY_STEPS:-500}"

exec /usr/bin/python3 src/main_controller.py \
  --mode print \
  --feedback-source fake \
  --command-source fixed \
  --imu-source xsens \
  --imu-port "$IMU_PORT" \
  --imu-stale-timeout 0.10 \
  --base-lin-vel-source zero \
  --control-hz 50 \
  --policy-shadow-mode \
  --unverified-shadow-clock "$CLOCK_FORMULA" \
  --unverified-shadow-clock-hz "$CLOCK_HZ" \
  --policy-steps "$POLICY_STEPS" \
  --log-every 5 \
  --no-auto-push-log
