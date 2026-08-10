#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep Grallator's policy/observation contract and hardware safety boundary,
# while selecting the simple MEVIUS2 motor-facing walking path:
#   q_target = q_policy_reference + 0.25 * raw_action
#   MIT: v_des=0, Kp=50, Kd=2, tau_ff=0
# The 100 Nm value is a dormant final authority guard, not commanded torque.
exec "$ROOT_DIR/scripts/run_medium_walk.sh" \
  --exact-policy-after-entry \
  --policy-action-clip 100 \
  --policy-hip-action-clip 0 \
  --policy-hip-action-scale 1 \
  --policy-action-smoothing 0 \
  --policy-action-delta-limit 0 \
  --policy-entry-ramp-seconds 3.0 \
  --policy-kp-override 50 \
  --policy-kd-override 2 \
  "$@"
