import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "configure_xsens_policy_output.py"
SPEC = importlib.util.spec_from_file_location("configure_xsens_policy_output", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_xbus_measurement_frame_matches_known_command():
    assert MODULE.xbus_frame(MODULE.MID_GOTO_MEASUREMENT).hex() == "faff1000f1"


def test_output_configuration_round_trip():
    entries = [(MODULE.XDI_QUATERNION, 200), (MODULE.XDI_RATE_OF_TURN, 200)]
    payload = MODULE.configuration_payload(entries)
    assert payload.hex() == "201000c8802000c8"
    assert MODULE.parse_output_configuration(payload) == entries
