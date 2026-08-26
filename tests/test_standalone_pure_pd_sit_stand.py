import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "standalone_pure_pd_sit_stand.py"


def load_script():
    spec = importlib.util.spec_from_file_location("standalone_pose_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_entry_uses_guarded_dual_can_pose_only_path():
    module = load_script()
    args = module.parse_args([])
    command = module.build_controller_args(args)
    joined = " ".join(command)
    assert "--can-count 2" in joined
    assert "--can-ports slcan0 slcan1" in joined
    assert "--control-hz 50" in joined
    assert "--can-command-hz 200" in joined
    assert "--pose-test-only" in command
    assert "--no-auto-policy-after-stand" in command
    assert "--sit-stand-trace-200hz" in command


def test_standalone_pose_profile_is_uniform_250_and_4():
    with (ROOT / "config" / "sit_stand_test_gains.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        profile = yaml.safe_load(stream)["sit_stand_gain_test"]
    for phase in ("sit", "stand"):
        for group in ("hip", "thigh", "calf"):
            assert profile["gains"][phase][group] == {"kp": 250.0, "kd": 4.0}
