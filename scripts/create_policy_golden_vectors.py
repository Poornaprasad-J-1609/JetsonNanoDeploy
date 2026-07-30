#!/usr/bin/env python3
"""Package independent Isaac observation/action rows as golden vectors."""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_qualification import create_golden_vectors_from_isaac_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Isaac CSV containing exact obs_000..047 and reference actor actions",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "tests" / "data" / "grallator_policy_golden_vectors.npz"),
    )
    parser.add_argument("--policy-sha256", default=None)
    args = parser.parse_args()

    policy_hash = args.policy_sha256
    if not policy_hash:
        with (ROOT / "config" / "policy_contract.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            contract = yaml.safe_load(stream) or {}
        policy_hash = str(
            contract["policy_contract"]["artifact"]["sha256"]
        )
    path = create_golden_vectors_from_isaac_csv(
        args.source_csv,
        args.output,
        policy_hash,
    )
    print("Policy SHA256:", policy_hash)
    print("Independent Isaac source:", Path(args.source_csv).expanduser().resolve())
    print("Golden vectors written:", path)


if __name__ == "__main__":
    main()
