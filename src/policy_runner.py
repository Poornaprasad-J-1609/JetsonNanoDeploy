#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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

    preferred = root / "policy" / "policy.pt"
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
    def __init__(self, policy_path=None, policy_activation="elu"):
        self.root = ROOT

        self.joint_cfg = load_yaml(self.root / "config" / "joint_map.yaml")
        self.pose_cfg = load_yaml(self.root / "config" / "default_pose.yaml")

        self.policy_order = self.joint_cfg["policy_to_real_order"]
        self.action_scale = float(self.joint_cfg["policy_action_scale"])
        self.control_dt = float(self.joint_cfg["control_dt"])

        self.q_default = self.pose_to_array(self.pose_cfg["default_pose"])
        self.q_stand = self.pose_to_array(self.pose_cfg["stand_pose"])
        self.q_crouch = self.pose_to_array(self.pose_cfg["crouch_pose"])

        self.observation_layout = self.joint_cfg["observation_layout"]
        self.policy_path = resolve_policy_path(self.root, policy_path=policy_path)
        self.policy, self.policy_format = load_policy_model(
            self.policy_path,
            activation=policy_activation,
        )
        self.policy.eval()

        self.observation_dim, self.action_dim = infer_linear_io_dims(self.policy)
        if self.observation_dim is None:
            self.observation_dim = self._probe_observation_dim()
        if self.action_dim is None:
            self.action_dim = len(self.policy_order)

        if self.action_dim != len(self.policy_order):
            raise ValueError(
                f"Policy outputs {self.action_dim} actions, "
                f"but policy_order has {len(self.policy_order)} joints"
            )

    def _probe_observation_dim(self):
        for obs_dim in (48, 45, 51, 60, 63, 66, 69, 72, 235, 240, 252):
            obs_t = torch.zeros(1, obs_dim, dtype=torch.float32)
            try:
                with torch.no_grad():
                    self.policy(obs_t)
                return obs_dim
            except Exception:
                pass
        raise ValueError("Could not infer policy observation dimension")

    def pose_to_array(self, pose_dict):
        return np.array([pose_dict[name] for name in self.policy_order], dtype=np.float32)

    def build_observation(
        self,
        base_lin_vel_b,
        base_ang_vel_b,
        projected_gravity_b,
        command,
        q_current,
        qd_current,
        previous_action,
    ):
        obs = np.zeros(self.observation_dim, dtype=np.float32)

        fields = {
            "base_lin_vel": np.asarray(base_lin_vel_b, dtype=np.float32),
            "base_ang_vel": np.asarray(base_ang_vel_b, dtype=np.float32),
            "projected_gravity": np.asarray(projected_gravity_b, dtype=np.float32),
            "command": np.asarray(command, dtype=np.float32),
            "joint_pos_relative": np.asarray(q_current, dtype=np.float32) - self.q_default,
            "joint_vel": np.asarray(qd_current, dtype=np.float32),
            "previous_action": np.asarray(previous_action, dtype=np.float32),
        }

        for field_name, values in fields.items():
            if field_name not in self.observation_layout:
                continue
            indices = layout_indices(self.observation_layout[field_name], len(values))
            if max(indices) >= self.observation_dim:
                raise ValueError(
                    f"Observation layout for {field_name} reaches index {max(indices)}, "
                    f"but policy observation dimension is {self.observation_dim}"
                )
            obs[indices] = values

        return obs

    def infer_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_t).squeeze(0).cpu().numpy()
        action = action.astype(np.float32)
        if action.shape[0] != len(self.policy_order):
            raise ValueError(
                f"Policy returned {action.shape[0]} actions, "
                f"expected {len(self.policy_order)}"
            )
        return action

    def action_to_q_target(self, action):
        action = np.asarray(action, dtype=np.float32)
        return self.q_default + self.action_scale * action

    def array_to_joint_dict(self, q):
        q = np.asarray(q, dtype=np.float32)
        return {name: float(q[i]) for i, name in enumerate(self.policy_order)}


if __name__ == "__main__":
    runner = PolicyRunner()
    print("Loaded policy:", runner.policy_path)
    print("Policy format:", runner.policy_format)
    print("Observation dim:", runner.observation_dim)
    print("Action dim:", runner.action_dim)
    print("Control dt:", runner.control_dt)
    print("Action scale:", runner.action_scale)
    print("Joint order:")
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name}")
    print("Q_DEFAULT:", runner.q_default)
    print("Q_STAND:", runner.q_stand)
    print("Q_CROUCH:", runner.q_crouch)
