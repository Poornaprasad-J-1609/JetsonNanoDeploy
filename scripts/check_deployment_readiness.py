#!/usr/bin/env python3
"""Report whether model_12357 may enter policy-controlled hardware mode."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deployment_readiness import evaluate_deployment_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default=str(ROOT / "policy" / "policy.pt"))
    args = parser.parse_args()

    report = evaluate_deployment_readiness(ROOT, Path(args.policy_path))
    print("\n".join(report.lines()))
    return 0 if report.hardware_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
