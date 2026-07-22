#!/usr/bin/env python3
"""Check the loaded policy against deterministic golden observations."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_qualification import check_golden_vectors  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--vectors",
        default=str(ROOT / "tests" / "data" / "grallator_policy_golden_vectors.npz"),
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--allow-policy-hash-mismatch", action="store_true")
    args = parser.parse_args()
    runner = PolicyRunner(
        policy_path=args.policy_path,
        allow_policy_hash_mismatch=args.allow_policy_hash_mismatch,
    )
    result = check_golden_vectors(runner, args.vectors, tolerance=args.tolerance)
    print("Policy SHA256:", runner.policy_sha256)
    print("Cases:", result.get("case_count", 0))
    print("Maximum absolute output difference:", result["maximum_absolute_error"])
    print("Golden policy vector test:", "PASS" if result["passed"] else "FAIL")
    for error in result["errors"]:
        print("ERROR:", error)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
