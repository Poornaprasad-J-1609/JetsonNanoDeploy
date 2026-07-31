#!/usr/bin/env python3
"""Export and verify the deterministic actor from an RSL-RL checkpoint."""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_runner import (
    EXPECTED_ACTION_DIM,
    EXPECTED_OBSERVATION_DIM,
    build_actor_from_state_dict,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    return checkpoint


def rejects_width(actor, width):
    try:
        actor(torch.zeros((1, int(width)), dtype=torch.float32))
    except Exception:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="RSL-RL checkpoint path")
    parser.add_argument("output", help="output TorchScript actor path")
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--verification-cases", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13108)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.verification_cases < 1:
        raise ValueError("--verification-cases must be positive")

    checkpoint = load_checkpoint(source)
    actor = build_actor_from_state_dict(
        checkpoint["model_state_dict"],
        activation=args.activation,
    ).eval()
    scripted = torch.jit.script(actor).eval()

    generator = torch.Generator().manual_seed(int(args.seed))
    observations = torch.randn(
        (int(args.verification_cases), EXPECTED_OBSERVATION_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        expected = actor(observations)
        scripted_output = scripted(observations)
    if tuple(scripted_output.shape) != (
        int(args.verification_cases),
        EXPECTED_ACTION_DIM,
    ):
        raise RuntimeError(
            f"actor output shape is {list(scripted_output.shape)}, expected "
            f"[{args.verification_cases}, {EXPECTED_ACTION_DIM}]"
        )
    if not bool(torch.isfinite(scripted_output).all()):
        raise RuntimeError("exported actor produced NaN or Inf")
    maximum_error = float(torch.max(torch.abs(expected - scripted_output)))
    if maximum_error != 0.0:
        raise RuntimeError(
            f"TorchScript actor differs from checkpoint actor by {maximum_error}"
        )
    if not rejects_width(scripted, 34) or not rejects_width(scripted, 45):
        raise RuntimeError("exported actor accepted an invalid observation width")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".policy-export-", dir=str(output.parent))
    )
    # TorchScript uses the archive basename internally. Keep it identical to
    # the final name so repeated exports have the same bytes and SHA256.
    temporary = temporary_directory / output.name
    try:
        torch.jit.save(scripted, str(temporary))
        reloaded = torch.jit.load(str(temporary), map_location="cpu").eval()
        with torch.inference_mode():
            reloaded_output = reloaded(observations)
        reload_error = float(torch.max(torch.abs(expected - reloaded_output)))
        if reload_error != 0.0:
            raise RuntimeError(
                f"reloaded actor differs from checkpoint actor by {reload_error}"
            )
        os.replace(temporary, output)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)

    print("Source:", source)
    print("Source iteration:", checkpoint.get("iter"))
    print("Source SHA256:", sha256_file(source))
    print("Actor:", output)
    print("Actor SHA256:", sha256_file(output))
    print(
        "Observation/action dimensions:",
        EXPECTED_OBSERVATION_DIM,
        EXPECTED_ACTION_DIM,
    )
    print("Verification cases:", args.verification_cases)
    print("Maximum absolute export error:", maximum_error)
    print("Invalid widths 34 and 45 rejected: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
