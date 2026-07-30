#!/usr/bin/env python3
"""Hardware-free artifact and semantic-readiness verification."""

import argparse

import numpy as np
import torch

from policy_runner import (
    EXPECTED_ACTION_DIM,
    EXPECTED_OBSERVATION_DIM,
    EXPECTED_POLICY_JOINT_ORDER,
    EXPECTED_POLICY_SHA256,
    PolicyRunner,
)
from deployment_readiness import evaluate_deployment_readiness


def model_accepts_shape(model, width):
    try:
        with torch.no_grad():
            model(torch.zeros(1, int(width), dtype=torch.float32))
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--allow-policy-hash-mismatch",
        action="store_true",
        help="allow an unrecognized policy SHA256 after explicit artifact verification",
    )
    args = parser.parse_args()

    runner = PolicyRunner(
        policy_path=args.policy_path,
        allow_policy_hash_mismatch=args.allow_policy_hash_mismatch,
    )

    with torch.no_grad():
        output = runner.policy(
            torch.zeros(1, EXPECTED_OBSERVATION_DIM, dtype=torch.float32)
        )

    replay_observation = np.zeros(EXPECTED_OBSERVATION_DIM, dtype=np.float32)
    replay_observation[0:3] = [0.2, -0.4, 0.6]
    replay_observation[8] = -1.0
    replay_action = runner.infer_action(replay_observation)
    readiness = evaluate_deployment_readiness(runner.root, runner.policy_path)

    failures = []
    if runner.policy_sha256 != EXPECTED_POLICY_SHA256:
        failures.append("policy SHA256 does not match the approved artifact")
    if runner.observation_dim != EXPECTED_OBSERVATION_DIM:
        failures.append("runner observation dimension is not 48")
    if runner.action_dim != EXPECTED_ACTION_DIM:
        failures.append("runner action dimension is not 12")
    if list(runner.policy_order) != EXPECTED_POLICY_JOINT_ORDER:
        failures.append("configured policy joint order is internally inconsistent")
    if tuple(output.shape) != (1, EXPECTED_ACTION_DIM):
        failures.append(f"zero [1,48] output shape is {list(output.shape)}, expected [1,12]")
    if not bool(torch.isfinite(output).all()):
        failures.append("zero [1,48] output contains NaN or Inf")
    if model_accepts_shape(runner.policy, 34):
        failures.append("policy unexpectedly accepted [1,34]")
    if model_accepts_shape(runner.policy, 45):
        failures.append("policy unexpectedly accepted [1,45]")
    if replay_action.shape != (EXPECTED_ACTION_DIM,) or not np.all(
        np.isfinite(replay_action)
    ):
        failures.append("nonzero-clock actor replay did not return finite [12]")
    if not readiness.policy_ready:
        failures.append("model_12357 semantic contract is not qualified")

    print("Policy:", runner.policy_path)
    print("SHA256:", runner.policy_sha256)
    print("Expected SHA256:", EXPECTED_POLICY_SHA256)
    print("Hash verified:", runner.policy_hash_matches)
    print("Format:", runner.policy_format)
    print("Observation/action dimensions:", runner.observation_dim, runner.action_dim)
    print("Joint order:", ", ".join(runner.policy_order))
    print("[1,48] output:", list(output.shape), "finite=", bool(torch.isfinite(output).all()))
    print("[1,34] rejected:", not model_accepts_shape(runner.policy, 34))
    print("[1,45] rejected:", not model_accepts_shape(runner.policy, 45))
    print("Nonzero clock replay:", replay_observation[0:3].tolist())
    print("Semantic policy readiness:", readiness.policy_ready)
    if not readiness.policy_ready:
        print("\n".join(readiness.lines()))

    if failures:
        print("\nPOLICY CONTRACT FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1

    print("\nPOLICY CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
