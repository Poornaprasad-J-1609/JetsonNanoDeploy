#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CAN_FRONT="${CAN_FRONT:-slcan0}"
CAN_BACK="${CAN_BACK:-slcan1}"
IMU_PORT="${IMU_PORT:-/dev/ttyUSB0}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
    "$ROOT_DIR/src/main_controller.py"
    --mode mit-signal
    --can-count 2
    --can-ports "$CAN_FRONT" "$CAN_BACK"
    --can-backend socketcan
    --can-bitrate 1000000
    --feedback-source mit
    # Restore the 0c17450 observation path. Its loaded runs held 50 Hz and
    # produced a more balanced gait than direct MIT velocity feedback.
    --joint-velocity-source finite-difference
    --command-source keyboard
    --imu-source xsens
    --imu-port "$IMU_PORT"
    --imu-stale-timeout 0.10
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
    # SPACE settles in stand; W/A/S/D/Q/E then provide the trained velocity
    # command observation and trigger locomotion-policy takeover.
    --walk-command-threshold 0.02
    --max-vx 1.80
    --max-vy 0.80
    --max-yaw 0
    --speed-scale-initial 0.08
    --speed-scale-min 0.04
    --speed-scale-max 0.12
    --speed-scale-step 0.01
    # Latched movement commands remain active until a pose, hold, or e-stop key.
    --keyboard-control-mode latched
    --keyboard-command-timeout 0.20
    --walk-command-grace-seconds 0.20
    # Restore the command envelope from 0c17450.
    --policy-command-gain 1.5
    --policy-command-vx-max 0.20
    --policy-command-vy-max 0.12
    --policy-command-yaw-max 0
    --policy-action-clip 3.2
    # The Aug 1 loaded runs showed 0.50-0.62 rad real hip travel versus
    # 0.24-0.36 rad in simulation. Reduce only the motor-facing hip action;
    # raw actions and previous_action observations remain unchanged.
    --policy-hip-action-scale 0.30
    --policy-action-smoothing 0.35
    --policy-action-delta-limit 0.20
    --policy-entry-ramp-seconds 2.0
    # Real telemetry left the trained state envelope and drove raw actor values
    # to 25-35 (simulation max: 7.77). Keep previous raw action in the policy
    # observation, but bound only the motor-facing action through the clip,
    # smoothing, and delta controls above.
    --no-exact-policy-after-entry
    # Loaded support profile from the July 31 telemetry: hips remain bounded at
    # 18 Nm while thighs/calves use a fixed 30 Nm ceiling. The old 24 -> 30 Nm
    # ramp never advanced and left most sagittal packets torque-limited.
    --torque-profile-stage stage30
    --policy-pd-torque-profile "$ROOT_DIR/config/policy_torque_loaded.yaml"
    --pose-transition-speed-rad-s 0.55
    --pose-transition-min-seconds 1.2
    # A crouch request from a loaded policy state reached 117 Nm in the July
    # 31 telemetry. Preserve the synchronized pose target while bounding the
    # effective legacy impedance to a value that still supports the robot.
    --pose-pd-torque-limit 40
    --stand-ready-error-rad 0.25
    --stand-ready-velocity-rad-s 0.15
    # Pose phases retain the proven e9a4a13 target/gain path. The finite torque
    # ceiling above scales impedance only when its estimate exceeds 40 Nm.
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

exec "$PYTHON_BIN" "${args[@]}" "$@"
