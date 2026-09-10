#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "EXPERIMENTAL OPEN-LOOP SIMULATION GAIT"
echo "First test must be suspended or substantially unloaded with a spotter."
echo "Hold W=forward or S=backward; release the key to return to stand."
echo "SPACE=stand, C=sit, X=emergency stop. Return to stand before reversing."

if [[ "${GAIT_ACKNOWLEDGE_100NM_SUSPENSION:-}" != "YES" ]]; then
  echo "ERROR: this high-authority test requires a fully suspended robot." >&2
  echo "Set GAIT_ACKNOWLEDGE_100NM_SUSPENSION=YES only after checking the rig and E-stop." >&2
  exit 2
fi

exec "$ROOT_DIR/scripts/run_medium_walk.sh" \
  --hardcoded-gait-config "$ROOT_DIR/config/hardcoded_gait_july13_minimal.yaml" \
  --hardcoded-gait-amplitude-scale "${GAIT_AMPLITUDE_SCALE:-0.50}" \
  --hardcoded-gait-frequency-scale "${GAIT_FREQUENCY_SCALE:-0.50}" \
  --max-vx 1.0 \
  --max-vy 0 \
  --max-yaw 0 \
  --speed-scale-initial 0.25 \
  --speed-scale-min 0.25 \
  --speed-scale-max 0.25 \
  --policy-command-gain 1.0 \
  --policy-command-vx-max 0.25 \
  --policy-command-vy-max 0 \
  --policy-command-yaw-max 0 \
  --keyboard-control-mode repeat \
  --keyboard-command-timeout 0.20 \
  --walk-command-grace-seconds 0.10 \
  --walk-stop-confirm-seconds 0.10 \
  --policy-entry-ramp-seconds 3.0 \
  --policy-kp-override "${GAIT_KP:-250}" \
  --policy-kd-override "${GAIT_KD:-4}" \
  --pose-gains-config "$ROOT_DIR/config/hardcoded_sit_stand_kp250_kd4.yaml" \
  --torque-profile-stage stage100 \
  --acknowledge-100nm-suspension-test \
  --policy-pd-torque-profile "$ROOT_DIR/config/policy_torque_suspension_100.yaml" \
  --policy-absolute-torque-ceiling 100 \
  --pose-pd-torque-limit 100 \
  --policy-torque-ramp-max-measured-torque 95 \
  --measured-torque-soft-hip 95 \
  --measured-torque-soft-thigh 95 \
  --measured-torque-soft-calf 95 \
  "$@"
