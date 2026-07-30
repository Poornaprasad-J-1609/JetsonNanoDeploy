#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PATH="$ROOT_DIR/policy/policy.pt"
EXPECTED_SHA256="48b3d7c7e10fd0d27a053fdf3af56bcd9190481c35798b585f0a0ff0478cf8b3"

if [[ ! -f "$POLICY_PATH" ]]; then
    echo "ERROR: trot policy is missing: $POLICY_PATH" >&2
    exit 1
fi

ACTUAL_SHA256="$(sha256sum "$POLICY_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: trot policy SHA256 mismatch." >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  actual:   $ACTUAL_SHA256" >&2
    exit 1
fi

echo "Trot actor: model_12357 (48 observations, 12 actions)"
echo "Press SPACE and wait for measured stand readiness, then press W once."
echo "W enters pure policy control; H, SPACE, C, or X leaves walking control."

# Keep the existing measured-good motor, feedback, timing, and safety profile.
# This wrapper only pins the new actor and a conservative forward command.
exec "$ROOT_DIR/scripts/run_medium_walk.sh" \
    --policy-path "$POLICY_PATH" \
    --max-vx 1.80 \
    --max-vy 0.00 \
    --max-yaw 0.00 \
    --speed-scale-initial 0.06 \
    --speed-scale-min 0.04 \
    --speed-scale-max 0.08 \
    --policy-command-vx-max 0.16 \
    --policy-command-vy-max 0.00 \
    --policy-command-yaw-max 0.00 \
    --no-gait-assist \
    "$@"
