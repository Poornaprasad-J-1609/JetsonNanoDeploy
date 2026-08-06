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
    # Preserve the validated actor-to-target equation after policy entry.
    # A two-second position blend handles takeover; physical joint and torque
    # guards remain the final protection layer.
    --policy-action-clip 0
    --policy-hip-action-scale 1.0
    --policy-action-smoothing 0
    --policy-action-delta-limit 0
    --policy-entry-ramp-seconds 2.0
    --exact-policy-after-entry
    # Use the requested common 100 Nm authority ceiling for every joint. This
    # does not command 100 Nm continuously; Kp/Kd and tracking error determine
    # actual torque. The independent measured-feedback stop remains at 110 Nm.
    --torque-profile-stage stage100
    --acknowledge-100nm-loaded-ground-test
    --policy-pd-torque-profile "$ROOT_DIR/config/policy_torque_loaded.yaml"
    --policy-absolute-torque-ceiling 100
    --policy-torque-ramp-delay-seconds 2.0
    --policy-torque-ramp-seconds 8.0
    # Match the measured loaded-ground and MuJoCo tracking envelope. The ramp
    # pauses on larger errors or torque and backs off after sustained measured
    # torque violations; hard encoder/tilt/torque safety remains independent.
    --policy-torque-ramp-max-tracking-error-rad 1.20
    --policy-torque-ramp-max-measured-torque 100.0
    --policy-torque-ramp-max-feedback-age 0.060
    --policy-torque-ramp-max-cycle-work-ms 28.0
    --measured-torque-soft-hip 100.0
    --measured-torque-soft-thigh 100.0
    --measured-torque-soft-calf 100.0
    # The Aug 5 pose log showed clean monotonic targets but underdamped loaded
    # tracking. Keep the shared transition phase and reduce its peak speed.
    --pose-transition-speed-rad-s 0.40
    --pose-transition-min-seconds 1.5
    --pose-pd-torque-limit 100
    --stand-ready-error-rad 0.25
    --stand-ready-velocity-rad-s 0.15
    # Pose phases use the same finite authority ceiling as walking.
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
