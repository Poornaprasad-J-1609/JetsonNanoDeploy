#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CAN_FRONT="${CAN_FRONT:-slcan0}"
CAN_BACK="${CAN_BACK:-slcan1}"
IMU_PORT="${IMU_PORT:-/dev/ttyUSB0}"

for interface in "$CAN_FRONT" "$CAN_BACK"; do
    if ! ip link show "$interface" >/dev/null 2>&1; then
        echo "ERROR: SocketCAN interface $interface does not exist." >&2
        exit 1
    fi
done

if [[ ! -e "$IMU_PORT" ]]; then
    echo "ERROR: Xsens IMU device $IMU_PORT does not exist." >&2
    exit 1
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
    "$ROOT_DIR/src/main_controller.py"
    --mode mit-signal
    --can-count 2
    --can-ports "$CAN_FRONT" "$CAN_BACK"
    --can-backend socketcan
    --can-bitrate 1000000
    --feedback-source mit
    --command-source keyboard
    --imu-source xsens
    --imu-port "$IMU_PORT"
    --imu-stale-timeout 0.25
    --control-hz 50
    --can-command-hz 200
    --can-command-stale-timeout 0.25
    --base-lin-vel-source zero
    --start-control-mode idle
    --startup-action hold
    --initial-zero-frame stand
    --no-auto-zero-on-startup
    --no-auto-stand-zero
    --no-auto-sit-zero
    --no-stand-policy-stabilization
    --no-imu-stabilization
    --no-gait-assist
    --walk-command-threshold 0.02
    --max-vx 1.80
    --max-vy 0.80
    --max-yaw 0
    --speed-scale-initial 0.08
    --speed-scale-min 0.04
    --speed-scale-max 0.12
    --speed-scale-step 0.01
    --keyboard-command-timeout 0.20
    --walk-command-grace-seconds 0.20
    --policy-command-gain 1.5
    --policy-command-vx-max 0.20
    --policy-command-vy-max 0.12
    --policy-command-yaw-max 0
    --policy-action-clip 3.2
    --policy-action-smoothing 0.35
    --policy-action-delta-limit 0.20
    --policy-entry-ramp-seconds 2.0
    --torque-profile-stage stage14
    --pose-transition-speed-rad-s 0.55
    --pose-transition-min-seconds 1.2
    --stand-ready-error-rad 0.25
    --pose-pd-torque-limit 16
    --feedback-timeout 0.05
    --fresh-feedback-max-age 0.08
    --policy-steps 0
    --log-every 5
    --no-auto-push-log
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "$PYTHON_BIN" "${args[@]}" "$@"
    printf '\n'
    exit 0
fi

exec "$PYTHON_BIN" "${args[@]}" "$@"
