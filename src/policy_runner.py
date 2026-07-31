#!/usr/bin/env python3
import hashlib
import math
import os
from pathlib import Path
import numpy as np
import torch

from joint_mapping import POLICY_JOINT_ORDER

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_POLICY_SHA256 = "139dc25e7ad44628cebfea12e96095781d8fc8e070d4419487dfb88a240f79d3"
EXPECTED_OBSERVATION_DIM = 48
EXPECTED_ACTION_DIM = 12
EXPECTED_POLICY_JOINT_ORDER = list(POLICY_JOINT_ORDER)
EXPECTED_OBSERVATION_LAYOUT = {
    "marching_clock": list(range(0, 3)),
    "base_ang_vel": list(range(3, 6)),
    "projected_gravity": list(range(6, 9)),
    "command": list(range(9, 12)),
    "joint_pos_relative": list(range(12, 24)),
    "joint_vel": list(range(24, 36)),
    "previous_action": list(range(36, 48)),
}


def configure_torch_for_realtime():
    """Keep tiny actor inference off Torch's high-overhead CPU thread pool."""
    thread_count = max(1, int(os.environ.get("GRALLATOR_TORCH_THREADS", "1")))
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work.
        pass
    return thread_count


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as policy_file:
        for chunk in iter(lambda: policy_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activation_from_name(name):
    name = str(name).lower()
    if name == "elu":
        return torch.nn.ELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name in ("identity", "none"):
        return torch.nn.Identity()
    raise ValueError(f"Unknown policy activation: {name}")


def resolve_policy_path(root, policy_path=None):
    if policy_path is not None:
        return Path(policy_path)

    preferred = root / "policy" / "model_12357_actor.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(
        (root / "policy").glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt policy found in {root / 'policy'}")
    return candidates[0]


def build_actor_from_state_dict(state_dict, activation="elu"):
    actor_indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("actor.") and key.endswith(".weight")
    )
    if not actor_indices:
        raise ValueError("Checkpoint does not contain actor.*.weight tensors")

    modules = []
    for layer_number, actor_index in enumerate(actor_indices):
        weight = state_dict[f"actor.{actor_index}.weight"]
        bias = state_dict[f"actor.{actor_index}.bias"]
        out_features, in_features = weight.shape

        layer = torch.nn.Linear(in_features, out_features)
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        modules.append(layer)

        if layer_number < len(actor_indices) - 1:
            modules.append(activation_from_name(activation))

    actor = torch.nn.Sequential(*modules)
    actor.eval()
    return actor


def load_policy_model(policy_path, activation="elu"):
    policy_path = Path(policy_path)

    try:
        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()
        return policy, "torchscript"
    except RuntimeError:
        pass

    try:
        checkpoint = torch.load(policy_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Jetson releases with older PyTorch do not expose weights_only yet.
        checkpoint = torch.load(policy_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported policy checkpoint type: {type(checkpoint)!r}")

    actor = build_actor_from_state_dict(state_dict, activation=activation)
    return actor, "checkpoint_actor"


def infer_linear_io_dims(policy):
    first_linear = None
    last_linear = None
    for module in policy.modules():
        if isinstance(module, torch.nn.Linear):
            if first_linear is None:
                first_linear = module
            last_linear = module

    if first_linear is None or last_linear is None:
        return None, None

    return int(first_linear.in_features), int(last_linear.out_features)


def layout_indices(spec, values_len):
    spec = list(spec)
    if len(spec) == values_len:
        return spec
    if len(spec) == 2 and spec[1] >= spec[0]:
        indices = list(range(int(spec[0]), int(spec[1]) + 1))
        if len(indices) == values_len:
            return indices
    raise ValueError(f"Observation layout {spec} cannot hold {values_len} value(s)")


class PolicyRunner:
    def __init__(
        self,
        policy_path=None,
        policy_activation="elu",
        allow_policy_hash_mismatch=False,
        expected_policy_sha256=EXPECTED_POLICY_SHA256,
    ):
        self.root = ROOT
        self.torch_thread_count = configure_torch_for_realtime()

        self.joint_cfg = load_yaml(self.root / "config" / "joint_map.yaml")
        self.pose_cfg = load_yaml(self.root / "config" / "default_pose.yaml")

        self.policy_order = self.joint_cfg["policy_to_real_order"]
        if list(self.policy_order) != EXPECTED_POLICY_JOINT_ORDER:
            raise ValueError(
                "policy_to_real_order does not match the verified IsaacLab log order. "
                f"Required {EXPECTED_POLICY_JOINT_ORDER}, got {list(self.policy_order)}"
            )
        configured_policy_signs = self.joint_cfg.get("policy_joint_signs", {}) or {}
        self.policy_joint_signs = np.asarray(
            [
                float(configured_policy_signs.get(joint_name, 1.0))
                for joint_name in self.policy_order
            ],
            dtype=np.float32,
        )
        invalid_signs = [
            joint_name
            for joint_name, sign in zip(self.policy_order, self.policy_joint_signs)
            if sign not in (-1.0, 1.0)
        ]
        if invalid_signs:
            raise ValueError(
                "policy_joint_signs must contain only +1 or -1 for: "
                + ", ".join(invalid_signs)
            )
        self.action_scale = float(self.joint_cfg["policy_action_scale"])
        policy_contract = self.joint_cfg.get("policy_contract", {}) or {}
        self.autonomous_march = bool(policy_contract.get("autonomous_march", False))
        self.marching_clock_frequency_hz = float(
            policy_contract.get("marching_clock_frequency_hz", 0.0)
        )
        self.marching_clock_formula = str(
            policy_contract.get("marching_clock_formula", "")
        ).strip().lower()
        self.policy_command = np.asarray(
            policy_contract.get("command", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        configured_action_scales = self.joint_cfg.get("policy_action_scales", {}) or {}
        missing_action_scales = [
            joint_name
            for joint_name in self.policy_order
            if joint_name not in configured_action_scales
        ]
        if missing_action_scales:
            raise ValueError(
                "policy_action_scales is missing required joint(s): "
                + ", ".join(missing_action_scales)
            )
        self.action_scale_by_joint = np.asarray(
            [
                float(configured_action_scales[joint_name])
                for joint_name in self.policy_order
            ],
            dtype=np.float32,
        )
        self.control_dt = float(self.joint_cfg["control_dt"])
        if not np.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError("policy_action_scale must be finite and > 0")
        if (
            self.action_scale_by_joint.shape != (EXPECTED_ACTION_DIM,)
            or not np.all(np.isfinite(self.action_scale_by_joint))
            or np.any(self.action_scale_by_joint == 0.0)
        ):
            raise ValueError(
                "policy_action_scales must provide 12 finite, nonzero signed values"
            )
        if self.autonomous_march:
            if (
                not np.isfinite(self.marching_clock_frequency_hz)
                or self.marching_clock_frequency_hz <= 0.0
            ):
                raise ValueError(
                    "autonomous marching requires marching_clock_frequency_hz > 0"
                )
            if self.marching_clock_formula != "sin_cos_one":
                raise ValueError(
                    "autonomous marching requires marching_clock_formula: "
                    "sin_cos_one"
                )
            if self.policy_command.shape != (3,) or not np.array_equal(
                self.policy_command,
                np.zeros(3, dtype=np.float32),
            ):
                raise ValueError(
                    "autonomous marching requires command [0, 0, 0]"
                )
        if not np.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt must be finite and > 0")

        self.q_default = self.pose_to_array(self.pose_cfg["default_pose"])
        self.q_stand = self.pose_to_array(self.pose_cfg["stand_pose"])
        self.q_crouch = self.pose_to_array(self.pose_cfg["crouch_pose"])

        self.observation_layout = self.joint_cfg["observation_layout"]
        self.policy_path = resolve_policy_path(self.root, policy_path=policy_path)
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")
        self.policy_sha256 = sha256_file(self.policy_path)
        self.expected_policy_sha256 = str(expected_policy_sha256).strip().lower()
        self.policy_hash_matches = self.policy_sha256 == self.expected_policy_sha256
        if not self.policy_hash_matches and not bool(allow_policy_hash_mismatch):
            raise RuntimeError(
                "Policy SHA256 mismatch. "
                f"Expected {self.expected_policy_sha256}, got {self.policy_sha256} "
                f"for {self.policy_path}. Refusing to run. Pass "
                "--allow-policy-hash-mismatch only after verifying the artifact."
            )
        if not self.policy_hash_matches:
            print(
                "WARNING: policy SHA256 mismatch explicitly allowed: "
                f"expected={self.expected_policy_sha256} actual={self.policy_sha256}"
            )
        self.policy, self.policy_format = load_policy_model(
            self.policy_path,
            activation=policy_activation,
        )
        self.policy.eval()

        inferred_observation_dim, inferred_action_dim = infer_linear_io_dims(self.policy)
        if (
            inferred_observation_dim is not None
            and inferred_observation_dim != EXPECTED_OBSERVATION_DIM
        ):
            raise ValueError(
                f"Policy expects {inferred_observation_dim} observations; "
                f"deployment requires exactly {EXPECTED_OBSERVATION_DIM}"
            )
        if inferred_action_dim is not None and inferred_action_dim != EXPECTED_ACTION_DIM:
            raise ValueError(
                f"Policy outputs {inferred_action_dim} actions; "
                f"deployment requires exactly {EXPECTED_ACTION_DIM}"
            )

        self.observation_dim = EXPECTED_OBSERVATION_DIM
        self.action_dim = EXPECTED_ACTION_DIM
        self._validate_observation_layout()
        self._validate_policy_tensor_contract()

        if len(self.policy_order) != EXPECTED_ACTION_DIM:
            raise ValueError(
                f"policy_order has {len(self.policy_order)} joints; "
                f"deployment requires exactly {EXPECTED_ACTION_DIM}"
            )

    def _validate_observation_layout(self):
        for field_name, expected_indices in EXPECTED_OBSERVATION_LAYOUT.items():
            if field_name not in self.observation_layout:
                raise ValueError(f"Observation layout is missing required field {field_name}")
            actual_indices = layout_indices(
                self.observation_layout[field_name],
                len(expected_indices),
            )
            if actual_indices != expected_indices:
                raise ValueError(
                    f"Observation layout for {field_name} is {actual_indices}; "
                    f"required {expected_indices}"
                )

    def _validate_policy_tensor_contract(self):
        probe = torch.zeros(1, EXPECTED_OBSERVATION_DIM, dtype=torch.float32)
        try:
            with torch.no_grad():
                output = self.policy(probe)
        except Exception as exc:
            raise ValueError(
                f"Policy does not accept required input shape [1, {EXPECTED_OBSERVATION_DIM}]"
            ) from exc
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Policy returned {type(output)!r}; expected torch.Tensor")
        if tuple(output.shape) != (1, EXPECTED_ACTION_DIM):
            raise ValueError(
                f"Policy returned shape {list(output.shape)}; "
                f"required [1, {EXPECTED_ACTION_DIM}]"
            )
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Policy returned NaN or Inf for a zero [1, 48] observation")

    def pose_to_array(self, pose_dict):
        missing = [name for name in self.policy_order if name not in pose_dict]
        if missing:
            raise KeyError(f"Pose is missing required joint(s): {missing}")
        pose = np.array([pose_dict[name] for name in self.policy_order], dtype=np.float32)
        if pose.shape != (EXPECTED_ACTION_DIM,):
            raise ValueError(
                f"Pose has shape {list(pose.shape)}; required [{EXPECTED_ACTION_DIM}]"
            )
        if not np.all(np.isfinite(pose)):
            raise ValueError("Pose contains NaN or Inf")
        return pose

    def build_observation(
        self,
        base_ang_vel_b,
        projected_gravity_b,
        command,
        q_current,
        qd_current,
        previous_action,
        marching_clock,
    ):
        obs = np.zeros(EXPECTED_OBSERVATION_DIM, dtype=np.float32)

        fields = {
            "marching_clock": np.asarray(marching_clock, dtype=np.float32),
            "base_ang_vel": np.asarray(base_ang_vel_b, dtype=np.float32),
            "projected_gravity": np.asarray(projected_gravity_b, dtype=np.float32),
            "command": np.asarray(command, dtype=np.float32),
            "joint_pos_relative": (
                self.policy_joint_signs
                * (np.asarray(q_current, dtype=np.float32) - self.q_default)
            ),
            "joint_vel": self.policy_joint_signs * np.asarray(qd_current, dtype=np.float32),
            "previous_action": np.asarray(previous_action, dtype=np.float32),
        }

        for field_name, values in fields.items():
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            expected_length = len(EXPECTED_OBSERVATION_LAYOUT[field_name])
            if values.shape != (expected_length,):
                raise ValueError(
                    f"Observation field {field_name} has shape {list(values.shape)}; "
                    f"required [{expected_length}]"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Observation field {field_name} contains NaN or Inf")
            indices = layout_indices(self.observation_layout[field_name], len(values))
            if max(indices) >= self.observation_dim:
                raise ValueError(
                    f"Observation layout for {field_name} reaches index {max(indices)}, "
                    f"but policy observation dimension is {self.observation_dim}"
                )
            obs[indices] = values

        return obs

    def marching_clock(self, elapsed_s, march_enabled=True):
        elapsed_s = float(elapsed_s)
        if not np.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("marching-clock elapsed time must be finite and >= 0")
        if not bool(march_enabled):
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        phase = (
            elapsed_s * self.marching_clock_frequency_hz
        ) % 1.0
        angle = 2.0 * math.pi * phase
        return np.array(
            [math.sin(angle), math.cos(angle), 1.0],
            dtype=np.float32,
        )

    def infer_action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (EXPECTED_OBSERVATION_DIM,):
            raise ValueError(
                f"Policy observation has shape {list(obs.shape)}; "
                f"required [{EXPECTED_OBSERVATION_DIM}]"
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError("Policy observation contains NaN or Inf")
        clock = obs[0:3]
        clock_norm = float(np.dot(clock[0:2], clock[0:2]))
        if not np.isclose(clock_norm, 1.0, atol=1.0e-4):
            raise ValueError(
                "Policy marching-clock sin/cos slots must have unit magnitude"
            )
        if float(clock[2]) not in (0.0, 1.0):
            raise ValueError("Policy marching-clock mode slot must be 0 or 1")

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy(obs_t).squeeze(0).cpu().numpy()
        action = action.astype(np.float32)
        if action.shape != (EXPECTED_ACTION_DIM,):
            raise ValueError(
                f"Policy returned shape {list(action.shape)}, "
                f"required [{EXPECTED_ACTION_DIM}]"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy action contains NaN or Inf")
        return action

    def action_to_q_target(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (EXPECTED_ACTION_DIM,):
            raise ValueError(
                f"Policy action has shape {list(action.shape)}; "
                f"required [{EXPECTED_ACTION_DIM}]"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy action contains NaN or Inf")
        return (
            self.q_default
            + self.policy_joint_signs * self.action_scale_by_joint * action
        )

    def array_to_joint_dict(self, q):
        q = np.asarray(q, dtype=np.float32)
        return {name: float(q[i]) for i, name in enumerate(self.policy_order)}


if __name__ == "__main__":
    runner = PolicyRunner()
    print("Loaded policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Policy format:", runner.policy_format)
    print("Observation dim:", runner.observation_dim)
    print("Action dim:", runner.action_dim)
    print("Control dt:", runner.control_dt)
    print("Legacy action scale:", runner.action_scale)
    print("Signed action scales:", runner.action_scale_by_joint)
    print("Autonomous march:", runner.autonomous_march)
    print("Marching clock frequency:", runner.marching_clock_frequency_hz)
    print("Joint order:")
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name}")
    print("Q_DEFAULT:", runner.q_default)
    print("Q_STAND:", runner.q_stand)
    print("Q_CROUCH:", runner.q_crouch)
