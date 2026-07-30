#!/usr/bin/env python3
"""Export and verify the deterministic actor from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_runner import (  # noqa: E402
    EXPECTED_ACTION_DIM,
    EXPECTED_OBSERVATION_DIM,
    build_actor_from_state_dict,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary, got {type(checkpoint)!r}")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain model_state_dict")
    return checkpoint


def model_rejects_width(model, width: int) -> bool:
    try:
        with torch.inference_mode():
            model(torch.zeros(1, int(width), dtype=torch.float32))
    except Exception:
        return True
    return False


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        os.execve(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "policy" / "policy.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "policy" / "model_12357_actor.pt"),
    )
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--comparison-cases", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=12357)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if int(args.comparison_cases) < 1:
        parser.error("--comparison-cases must be >= 1")
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")

    with (ROOT / "config" / "policy_contract.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        contract = yaml.safe_load(stream) or {}
    artifact = contract["policy_contract"]["artifact"]
    expected_source_hash = str(
        artifact.get("source_checkpoint_sha256", artifact.get("sha256", ""))
    ).lower()
    source_hash = sha256_file(checkpoint_path)
    if source_hash != expected_source_hash:
        raise RuntimeError(
            "Source checkpoint SHA256 mismatch: "
            f"expected={expected_source_hash} actual={source_hash}"
        )

    checkpoint = load_checkpoint(checkpoint_path)
    actor = build_actor_from_state_dict(
        checkpoint["model_state_dict"],
        activation=args.activation,
    )
    actor.eval()
    scripted = torch.jit.script(actor)
    scripted.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    observations = torch.randn(
        int(args.comparison_cases),
        EXPECTED_OBSERVATION_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        source_actions = actor(observations)
        exported_actions = scripted(observations)
        repeat_actions = scripted(observations)

    expected_shape = (int(args.comparison_cases), EXPECTED_ACTION_DIM)
    if tuple(exported_actions.shape) != expected_shape:
        raise RuntimeError(
            f"Exported actor returned {tuple(exported_actions.shape)}, "
            f"expected {expected_shape}"
        )
    if not bool(torch.isfinite(exported_actions).all()):
        raise RuntimeError("Exported actor returned NaN or Inf")
    maximum_error = float(
        torch.max(torch.abs(source_actions - exported_actions)).item()
    )
    repeat_error = float(
        torch.max(torch.abs(exported_actions - repeat_actions)).item()
    )
    if maximum_error > float(args.tolerance):
        raise RuntimeError(
            f"Checkpoint/export maximum error {maximum_error:.9g} exceeds "
            f"{float(args.tolerance):.9g}"
        )
    if repeat_error != 0.0:
        raise RuntimeError(
            f"Exported actor is not deterministic; repeat error={repeat_error:.9g}"
        )
    if not model_rejects_width(scripted, 34) or not model_rejects_width(
        scripted, 45
    ):
        raise RuntimeError("Exported actor accepted an invalid observation width")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_actor = io.BytesIO()
    torch.jit.save(scripted, serialized_actor)
    actor_bytes = serialized_actor.getvalue()
    with tempfile.NamedTemporaryFile(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(actor_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary_path, output_path)
        output_path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)

    loaded = torch.jit.load(str(output_path), map_location="cpu")
    loaded.eval()
    with torch.inference_mode():
        loaded_actions = loaded(observations)
    loaded_error = float(
        torch.max(torch.abs(source_actions - loaded_actions)).item()
    )
    if loaded_error > float(args.tolerance):
        raise RuntimeError(
            f"Saved actor maximum error {loaded_error:.9g} exceeds "
            f"{float(args.tolerance):.9g}"
        )

    print("Source checkpoint:", checkpoint_path)
    print("Source SHA256:", source_hash)
    print("Checkpoint iteration:", checkpoint.get("iter", "unknown"))
    print("Exported actor:", output_path)
    print("Exported SHA256:", sha256_file(output_path))
    print("Observation/action dimensions:", EXPECTED_OBSERVATION_DIM, EXPECTED_ACTION_DIM)
    print("Comparison cases:", int(args.comparison_cases))
    print("Maximum checkpoint/export error:", f"{loaded_error:.9g}")
    print("Deterministic repeat error:", f"{repeat_error:.9g}")
    print("Invalid widths 34/45 rejected: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
