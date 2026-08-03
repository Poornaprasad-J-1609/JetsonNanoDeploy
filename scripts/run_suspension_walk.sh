#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Reuse the fully validated two-CAN/50 Hz policy command, then override only
# the first suspended-test envelope. Duplicate argparse options intentionally
# use the final value supplied here. The 2026-08-03 suspension run inherited
# the loaded-ground 0.55 rad/s pose transition and accumulated 0.37 rad calf
# error before vibrating, so pose motion is limited to 0.20 rad/s here.
exec "$ROOT_DIR/scripts/run_medium_walk.sh" \
    --torque-profile-stage stage14 \
    --policy-pd-torque-profile "" \
    --policy-hip-action-scale 0.50 \
    --policy-action-smoothing 0.20 \
    --policy-action-delta-limit 0.30 \
    --speed-scale-initial 0.04 \
    --speed-scale-min 0.04 \
    --speed-scale-max 0.04 \
    --policy-command-vx-max 0.12 \
    --policy-command-vy-max 0.08 \
    --policy-command-yaw-max 0 \
    --pose-pd-torque-limit 14 \
    --pose-transition-speed-rad-s 0.20 \
    --pose-transition-min-seconds 2.0 \
    --stand-ready-error-rad 0.08 \
    --stand-ready-velocity-rad-s 0.10 \
    --fresh-feedback-max-age 0.030 \
    --feedback-snapshot-max-skew-ms 20 \
    --policy-torque-ramp-max-feedback-age 0.030 \
    --suspension-status-seconds 1.0 \
    "$@"
