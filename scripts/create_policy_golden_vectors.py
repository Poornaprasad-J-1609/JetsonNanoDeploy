#!/usr/bin/env python3
"""Create deterministic Grallator policy golden vectors."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_qualification import create_golden_vectors  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "tests" / "data" / "grallator_policy_golden_vectors.npz"),
    )
    parser.add_argument("--allow-policy-hash-mismatch", action="store_true")
    args = parser.parse_args()
    runner = PolicyRunner(
        policy_path=args.policy_path,
        allow_policy_hash_mismatch=args.allow_policy_hash_mismatch,
    )
    path = create_golden_vectors(runner, args.output)
    print("Policy SHA256:", runner.policy_sha256)
    print("Golden vectors written:", path)


if __name__ == "__main__":
    main()
