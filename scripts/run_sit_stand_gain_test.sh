#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAN_FRONT="${CAN_FRONT:-slcan0}"
CAN_BACK="${CAN_BACK:-slcan1}"
IMU_PORT="${IMU_PORT:-/dev/ttyUSB0}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec /usr/bin/python3 src/main_controller.py \
  --mode mit-signal \
  --can-count 2 \
  --can-ports "$CAN_FRONT" "$CAN_BACK" \
  --can-backend socketcan \
  --can-bitrate 1000000 \
  --feedback-source mit \
  --joint-velocity-source finite-difference \
  --command-source keyboard \
  --imu-source xsens \
  --imu-port "$IMU_PORT" \
  --imu-stale-timeout 0.10 \
  --control-hz 50 \
  --can-command-hz 200 \
  --pose-test-only \
  --pose-gains-config "$ROOT_DIR/config/sit_stand_test_gains.yaml" \
  --sit-stand-trace-200hz \
  --robot-mass-kg 50 \
  --start-control-mode idle \
  --startup-action hold \
  --initial-zero-frame stand \
  --no-auto-zero-on-startup \
  --no-auto-stand-zero \
  --no-auto-sit-zero \
  --no-auto-policy-after-stand \
  --no-stand-policy-stabilization \
  --no-imu-stabilization \
  --no-gait-assist \
  --pose-transition-speed-rad-s 0.40 \
  --pose-transition-min-seconds 1.50 \
  --feedback-timeout 0.05 \
  --fresh-feedback-max-age 0.08 \
  --policy-steps 0 \
  --log-every 1 \
  --print-every 10 \
  --log-prefix sit_stand_gain_test \
  --no-auto-push-log \
  "$@"
