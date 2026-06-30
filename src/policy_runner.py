#!/usr/bin/env python3
import hashlib
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
        candidate = Path(policy_path).expanduser()
        if not candidate.is_absolute() and not candidate.exists():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Policy file not found: {candidate}")
        return candidate.resolve()

    preferred = root / "policy" / "policy.pt"
    if preferred.exists():
        return preferred.resolve()

    candidates = sorted(
        (root / "policy").glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt policy found in {root / 'policy'}")
    return candidates[0].resolve()


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

        self.policy_order = list(self.joint_cfg["policy_to_real_order"])
        action_scale_value = self.joint_cfg["policy_action_scale"]
        if isinstance(action_scale_value, bool):
            raise TypeError("policy_action_scale must be numeric, not boolean")
        self.action_scale = float(action_scale_value)
        self.control_dt = float(self.joint_cfg["control_dt"])
        self.policy_contract = dict(self.joint_cfg["policy_contract"])
        force_zero_value = self.policy_contract["force_zero_base_linear_velocity"]
        if not isinstance(force_zero_value, bool):
            raise TypeError(
                "policy_contract.force_zero_base_linear_velocity must be a YAML boolean"
            )
        self.force_zero_base_linear_velocity = force_zero_value
        self.normalize_projected_gravity = bool(
            self.policy_contract.get("normalize_projected_gravity", True)
        )
        self.observation_scales = dict(
            self.policy_contract.get("observation_scales", {})
        )

        self._validate_static_config()
        self.q_default = self.pose_to_array(
            self.pose_cfg["default_pose"], "default_pose"
        )
        self.q_stand = self.pose_to_array(
            self.pose_cfg["stand_pose"], "stand_pose"
        )
        self.q_crouch = self.pose_to_array(
            self.pose_cfg["crouch_pose"], "crouch_pose"
        )
        self.q_sit_when_stand_zero = self.pose_to_array(
            self.pose_cfg["sit_pose_when_stand_zero"],
            "sit_pose_when_stand_zero",
        )
        self.q_stand_when_sit_zero = self.pose_to_array(
            self.pose_cfg["stand_pose_when_sit_zero"],
            "stand_pose_when_sit_zero",
        )
        expected_sit_delta = -self.q_stand_when_sit_zero
        if not np.allclose(
            self.q_sit_when_stand_zero,
            expected_sit_delta,
            rtol=0.0,
            atol=1e-7,
        ):
            mismatch = {
                joint_name: {
                    "configured": float(self.q_sit_when_stand_zero[index]),
                    "required": float(expected_sit_delta[index]),
                }
                for index, joint_name in enumerate(self.policy_order)
                if abs(
                    float(self.q_sit_when_stand_zero[index])
                    - float(expected_sit_delta[index])
                ) > 1e-7
            }
            raise ValueError(
                "sit_pose_when_stand_zero must be the exact negative of "
                f"stand_pose_when_sit_zero; mismatches={mismatch}"
            )

        self.observation_layout = self.joint_cfg["observation_layout"]
        self.policy_path = resolve_policy_path(self.root, policy_path=policy_path)
        self.policy_sha256 = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
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
        self._validate_policy_contract()

    def _validate_static_config(self):
        if len(self.policy_order) != 12:
            raise ValueError(
                "policy_to_real_order must contain exactly 12 joints; "
                f"got {len(self.policy_order)}"
            )
        if len(set(self.policy_order)) != 12:
            raise ValueError("policy_to_real_order must contain 12 unique joint names")
        if not all(isinstance(name, str) and name for name in self.policy_order):
            raise ValueError("policy_to_real_order contains an invalid joint name")
        if not np.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError(
                "policy_action_scale must exist and be a finite value greater than zero"
            )
        if not np.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt must be a finite value greater than zero")
        if not self.force_zero_base_linear_velocity:
            raise ValueError(
                "policy_contract.force_zero_base_linear_velocity must be true "
                "for policy/policy.pt"
            )

        required_poses = (
            "default_pose",
            "stand_pose",
            "crouch_pose",
            "sit_pose_when_stand_zero",
            "stand_pose_when_sit_zero",
        )
        expected_joints = set(self.policy_order)
        for pose_name in required_poses:
            if pose_name not in self.pose_cfg:
                raise KeyError(f"Missing required pose '{pose_name}' in default_pose.yaml")
            pose = self.pose_cfg[pose_name]
            if not isinstance(pose, dict):
                raise TypeError(f"{pose_name} must be a YAML mapping")
            actual_joints = set(pose)
            missing = sorted(expected_joints - actual_joints)
            extra = sorted(actual_joints - expected_joints)
            if missing or extra or len(pose) != 12:
                raise ValueError(
                    f"{pose_name} must contain exactly the 12 policy joints; "
                    f"missing={missing}, extra={extra}"
                )
            for joint_name in self.policy_order:
                try:
                    value = float(pose[joint_name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{pose_name}.{joint_name} must be numeric"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"{pose_name}.{joint_name} must be finite")

    def _validate_policy_contract(self):
        expected_observation_dim = int(self.policy_contract["observation_dim"])
        expected_action_dim = int(self.policy_contract["action_dim"])
        if self.observation_dim != expected_observation_dim:
            raise ValueError(
                f"Policy expects {self.observation_dim} observations, but the "
                f"deployment contract requires {expected_observation_dim}"
            )
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f"Policy outputs {self.action_dim} actions, but the deployment "
                f"contract requires {expected_action_dim}"
            )
        expected_joint_order = list(self.policy_contract["action_joint_order"])
        if self.policy_order != expected_joint_order:
            raise ValueError(
                "policy_to_real_order does not match policy.pt action order: "
                f"configured={self.policy_order}, expected={expected_joint_order}"
            )

        expected_layout = {
            "base_lin_vel": [0, 1, 2],
            "base_ang_vel": [3, 4, 5],
            "projected_gravity": [6, 7, 8],
            "command": [9, 10, 11],
            "joint_pos_relative": list(range(12, 24)),
            "joint_vel": list(range(24, 36)),
            "previous_action": list(range(36, 48)),
        }
        value_lengths = {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "command": 3,
            "joint_pos_relative": len(self.policy_order),
            "joint_vel": len(self.policy_order),
            "previous_action": len(self.policy_order),
        }
        occupied = []
        for field_name, expected_indices in expected_layout.items():
            if field_name not in self.observation_layout:
                raise ValueError(f"Missing policy observation field: {field_name}")
            actual_indices = layout_indices(
                self.observation_layout[field_name],
                value_lengths[field_name],
            )
            if actual_indices != expected_indices:
                raise ValueError(
                    f"Observation field {field_name} uses indices {actual_indices}; "
                    f"policy.pt requires {expected_indices}"
                )
            occupied.extend(actual_indices)

        if sorted(occupied) != list(range(self.observation_dim)):
            raise ValueError("Policy observation layout must cover indices 0..47 exactly once")

    @staticmethod
    def _finite_vector(values, expected_length, field_name):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_length,):
            raise ValueError(
                f"{field_name} must have shape ({expected_length},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{field_name} contains NaN or infinite values: {values}")
        return values

    def _scale_observation_field(self, field_name, values):
        scale = np.asarray(
            self.observation_scales.get(field_name, 1.0),
            dtype=np.float32,
        )
        if scale.ndim == 0:
            return values * float(scale)
        if scale.shape != values.shape:
            raise ValueError(
                f"Observation scale for {field_name} has shape {scale.shape}, "
                f"expected {values.shape}"
            )
        return values * scale

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

    def pose_to_array(self, pose_dict, pose_name="pose"):
        values = np.array(
            [float(pose_dict[name]) for name in self.policy_order],
            dtype=np.float32,
        )
        return self._finite_vector(values, len(self.policy_order), pose_name)

    def assert_observation_contract(self, obs):
        obs = self._finite_vector(obs, self.observation_dim, "policy observation")
        if self.force_zero_base_linear_velocity and not np.array_equal(
            obs[0:3], np.zeros(3, dtype=np.float32)
        ):
            raise ValueError(
                "Policy observation contract violated: indices 0:3 must be "
                f"exactly [0, 0, 0], got {obs[0:3].tolist()}"
            )
        return obs

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

        base_lin_vel_b = self._finite_vector(base_lin_vel_b, 3, "base_lin_vel")
        if self.force_zero_base_linear_velocity:
            base_lin_vel_b = np.zeros(3, dtype=np.float32)

        base_ang_vel_b = self._finite_vector(base_ang_vel_b, 3, "base_ang_vel")
        projected_gravity_b = self._finite_vector(
            projected_gravity_b,
            3,
            "projected_gravity",
        )
        if self.normalize_projected_gravity:
            gravity_norm = float(np.linalg.norm(projected_gravity_b))
            if gravity_norm < 1e-6:
                raise ValueError("projected_gravity norm is zero; IMU orientation is invalid")
            projected_gravity_b = projected_gravity_b / gravity_norm

        command = self._finite_vector(command, 3, "command")
        q_current = self._finite_vector(
            q_current,
            len(self.policy_order),
            "joint_pos",
        )
        qd_current = self._finite_vector(
            qd_current,
            len(self.policy_order),
            "joint_vel",
        )
        previous_action = self._finite_vector(
            previous_action,
            len(self.policy_order),
            "previous_action",
        )

        fields = {
            "base_lin_vel": base_lin_vel_b,
            "base_ang_vel": base_ang_vel_b,
            "projected_gravity": projected_gravity_b,
            "command": command,
            "joint_pos_relative": q_current - self.q_default,
            "joint_vel": qd_current,
            "previous_action": previous_action,
        }

        for field_name, values in fields.items():
            if field_name not in self.observation_layout:
                continue
            values = self._scale_observation_field(field_name, values)
            indices = layout_indices(self.observation_layout[field_name], len(values))
            if max(indices) >= self.observation_dim:
                raise ValueError(
                    f"Observation layout for {field_name} reaches index {max(indices)}, "
                    f"but policy observation dimension is {self.observation_dim}"
                )
            obs[indices] = values

        return self.assert_observation_contract(obs)

    def infer_action(self, obs):
        obs = self.assert_observation_contract(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_t).squeeze(0).cpu().numpy()
        action = action.astype(np.float32)
        if action.shape[0] != len(self.policy_order):
            raise ValueError(
                f"Policy returned {action.shape[0]} actions, "
                f"expected {len(self.policy_order)}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Policy returned NaN or infinite action values: {action}")
        return action

    def observation_summary(self, obs):
        obs = self._finite_vector(obs, self.observation_dim, "policy observation")
        return (
            f"base_lin={obs[0:3].tolist()} "
            f"gyro={np.round(obs[3:6], 4).tolist()} "
            f"gravity={np.round(obs[6:9], 4).tolist()} "
            f"command={np.round(obs[9:12], 4).tolist()} "
            f"q_rel_max={float(np.max(np.abs(obs[12:24]))):.4f} "
            f"qd_max={float(np.max(np.abs(obs[24:36]))):.4f}"
        )

    def action_to_q_target(self, action):
        action = self._finite_vector(action, self.action_dim, "policy action")
        q_target = self.q_default + self.action_scale * action
        return self._finite_vector(
            q_target,
            len(self.policy_order),
            "policy q_target",
        )

    def array_to_joint_dict(self, q):
        q = np.asarray(q, dtype=np.float32)
        return {name: float(q[i]) for i, name in enumerate(self.policy_order)}


if __name__ == "__main__":
    runner = PolicyRunner()
    print("Loaded policy:", runner.policy_path)
    print("Policy format:", runner.policy_format)
    print("Policy SHA256:", runner.policy_sha256)
    print("Observation dim:", runner.observation_dim)
    print("Action dim:", runner.action_dim)
    print("Control dt:", runner.control_dt)
    print("Action scale:", runner.action_scale)
    print("Force zero base linear velocity:", runner.force_zero_base_linear_velocity)
    print("Joint order:")
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name}")
    print("Q_DEFAULT:", runner.q_default)
    print("Q_STAND:", runner.q_stand)
    print("Q_CROUCH:", runner.q_crouch)
