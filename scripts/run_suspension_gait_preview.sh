#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "FULL-SUSPENSION GAIT PREVIEW"
echo "This mode overrides actor motor targets with a bounded diagonal trot."
echo "Do not use it with feet touching the ground or carrying body weight."

exec "$ROOT_DIR/scripts/run_medium_walk.sh" \
    --torque-profile-stage stage14 \
    --policy-pd-torque-limit-start 10 \
    --policy-pd-torque-limit-final 14 \
    --policy-torque-ramp-delay-seconds 3 \
    --policy-torque-ramp-seconds 6 \
    --no-exact-policy-after-entry \
    --gait-assist \
    --speed-scale-initial 0.04 \
    --speed-scale-min 0.04 \
    --speed-scale-max 0.04 \
    --policy-command-vx-max 0.12 \
    --policy-command-vy-max 0.08 \
    --policy-action-clip 2.8 \
    --policy-hip-action-clip 1.6 \
    --policy-hip-action-scale 0.65 \
    --policy-action-smoothing 0.35 \
    --policy-action-delta-limit 0.15 \
    --policy-entry-ramp-seconds 3.0 \
    --pose-transition-speed-rad-s 0.35 \
    --pose-transition-min-seconds 2.0 \
    --suspension-status-seconds 0.5 \
    "$@"
