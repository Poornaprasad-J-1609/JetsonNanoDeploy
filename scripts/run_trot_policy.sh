#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
POLICY_PATH="$ROOT_DIR/policy/model_12357_actor.pt"
EXPECTED_SHA256="139dc25e7ad44628cebfea12e96095781d8fc8e070d4419487dfb88a240f79d3"

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

if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/check_deployment_readiness.py" \
    --policy-path "$POLICY_PATH"; then
    echo "ERROR: model_12357 hardware deployment is not qualified." >&2
    echo "Provide the exact Isaac training/export source and independent golden vectors." >&2
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
