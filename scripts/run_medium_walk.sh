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
    --joint-velocity-source finite-difference
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
    # Restore the command envelope from 0c17450. The actor still passes
    # through the current clip, EMA, delta, and entry-ramp safety pipeline.
    --policy-command-gain 1.5
    --policy-command-vx-max 0.20
    --policy-command-vy-max 0.12
    --policy-command-yaw-max 0
    --policy-action-clip 3.2
    --policy-action-smoothing 0.35
    --policy-action-delta-limit 0.20
    --policy-entry-ramp-seconds 2.0
    # Real feedback caused the unbounded actor to reach |action|=10.34 after
    # entry. Keep the trained 0.25 action scale, but activate the configured
    # clip/smoothing/delta pipeline before targets reach loaded hardware.
    --no-exact-policy-after-entry
    # Loaded ground profile. The successful stand log needs 23-26 Nm steady
    # rear-leg support and reached 50.8 Nm transiently. Enter policy at 30 Nm
    # and ramp toward the bounded 40 Nm software ceiling only while clean.
    --torque-profile-stage stage40
    --policy-pd-torque-limit-start 30
    --policy-pd-torque-limit-final 40
    --policy-torque-ramp-max-measured-torque 40
    --acknowledge-40nm-suspension-test
    --pose-transition-speed-rad-s 0.55
    --pose-transition-min-seconds 1.2
    --stand-ready-error-rad 0.25
    --stand-ready-velocity-rad-s 0.15
    # Pose phases use the proven e9a4a13 legacy packet path; measured hardware
    # safety remains active while host-side pose torque rewriting is disabled.
    --feedback-timeout 0.05
    --fresh-feedback-max-age 0.08
    --policy-steps 0
    --log-every 20
    --no-auto-push-log
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "$PYTHON_BIN" "${args[@]}" "$@"
    printf '\n'
    exit 0
fi

exec "$PYTHON_BIN" "${args[@]}" "$@"
