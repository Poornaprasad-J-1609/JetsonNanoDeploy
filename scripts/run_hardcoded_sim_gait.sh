#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "EXPERIMENTAL OPEN-LOOP SIMULATION GAIT"
echo "First test must be suspended or substantially unloaded with a spotter."
echo "W=forward, S=backward, SPACE=stand, C=sit, X=emergency stop."
echo "Return to SPACE before changing direction."

exec "$ROOT_DIR/scripts/run_medium_walk.sh" \
  --hardcoded-gait-config "$ROOT_DIR/config/hardcoded_gait_sim_vx0p5.yaml" \
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
  --policy-entry-ramp-seconds 3.0 \
  --torque-profile-stage stage40 \
  --acknowledge-40nm-suspension-test \
  --policy-pd-torque-profile "$ROOT_DIR/config/policy_torque_suspension_40.yaml" \
  --policy-absolute-torque-ceiling 40 \
  --pose-pd-torque-limit 40 \
  --policy-torque-ramp-max-measured-torque 35 \
  --measured-torque-soft-hip 35 \
  --measured-torque-soft-thigh 35 \
  --measured-torque-soft-calf 35 \
  "$@"
