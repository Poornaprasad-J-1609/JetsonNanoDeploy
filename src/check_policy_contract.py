#!/usr/bin/env python3
"""Hardware-free verification of the fixed 48-observation policy contract."""

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

    obs = runner.build_observation(
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
        q_current=runner.q_default.copy(),
        qd_current=np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32),
        previous_action=np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32),
    )

    failures = []
    if runner.policy_sha256 != EXPECTED_POLICY_SHA256:
        failures.append("policy SHA256 does not match the approved artifact")
    if runner.observation_dim != EXPECTED_OBSERVATION_DIM:
        failures.append("runner observation dimension is not 48")
    if runner.action_dim != EXPECTED_ACTION_DIM:
        failures.append("runner action dimension is not 12")
    if list(runner.policy_order) != EXPECTED_POLICY_JOINT_ORDER:
        failures.append("policy joint order does not match the verified IsaacLab order")
    if tuple(output.shape) != (1, EXPECTED_ACTION_DIM):
        failures.append(f"zero [1,48] output shape is {list(output.shape)}, expected [1,12]")
    if not bool(torch.isfinite(output).all()):
        failures.append("zero [1,48] output contains NaN or Inf")
    if model_accepts_shape(runner.policy, 34):
        failures.append("policy unexpectedly accepted [1,34]")
    if model_accepts_shape(runner.policy, 45):
        failures.append("policy unexpectedly accepted [1,45]")
    if not np.array_equal(obs[0:3], np.zeros(3, dtype=np.float32)):
        failures.append("build_observation did not force obs[0:3] to exact zeros")

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
    print("obs[0:3]:", obs[0:3].tolist())

    if failures:
        print("\nPOLICY CONTRACT FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1

    print("\nPOLICY CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
