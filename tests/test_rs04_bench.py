import csv
import json
import math
import time

import numpy as np
import pytest

from rs04_bench.analysis.plant_identification import identify_plant, recommend_kd
from rs04_bench.analysis.step_response import analyze_step_response
from rs04_bench.config import load_config
from rs04_bench.control.loop import BenchController
from rs04_bench.control.trajectories import ChirpTrajectory, ConstantSpeedTrajectory
from rs04_bench.experiments.manager import ExperimentManager, ExperimentSpec
from rs04_bench.motor.mock import MockMotorInterface
from rs04_bench.motor.robstride import official_rs04_protocol


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("kind", ["linear", "logarithmic"])
def test_chirp_analytical_velocity_matches_position_derivative(kind):
    chirp = ChirpTrajectory(0.1, 0.03, 0.2, 3.0, 8.0, kind)
    for t in np.linspace(0.2, 7.8, 30):
        epsilon = 1e-6
        numerical = (chirp.sample(t + epsilon).q - chirp.sample(t - epsilon).q) / (2 * epsilon)
        assert chirp.sample(t).qd == pytest.approx(numerical, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize("target", [0.8, -0.8])
def test_manual_constant_speed_trajectory_moves_in_selected_direction(target):
    trajectory = ConstantSpeedTrajectory(0.0, target, 0.2)
    halfway = trajectory.sample(2.0)
    assert halfway.q == pytest.approx(math.copysign(0.4, target))
    assert halfway.qd == pytest.approx(math.copysign(0.2, target))
    finished = trajectory.sample(10.0)
    assert finished.q == pytest.approx(target)
    assert finished.qd == pytest.approx(0.0)


def test_dial_clockwise_is_positive_and_anticlockwise_is_negative():
    from rs04_bench.gui.angle_gauge import AngleGauge

    targets = []
    gauge = object.__new__(AngleGauge)
    gauge.size = 330
    gauge.center = 165.0
    gauge.radius = 128.7
    gauge.limit_rad = 2.5
    gauge._actual = 0.0
    gauge._desired = 0.0
    gauge._command = targets.append
    gauge._draw = lambda: None
    event = type("Event", (), {})()
    event.x, event.y = gauge.center + gauge.radius, gauge.center
    gauge._select_target(event)
    assert targets[-1] == pytest.approx(math.pi / 2.0)
    event.x = gauge.center - gauge.radius
    gauge._select_target(event)
    assert targets[-1] == pytest.approx(-math.pi / 2.0)


def test_controller_slews_manual_target_at_configured_speed(tmp_path):
    config = load_config(overrides={
        "logging": {"directory": str(tmp_path)},
        "control": {"manual_speed_rad_s": 0.2},
    })
    controller = BenchController(MockMotorInterface(config.mock), config)
    try:
        controller.start()
        controller.connect()
        controller.enable()
        controller.set_manual_target(0.5)
        time.sleep(0.12)
        snapshot = controller.snapshot()
        assert snapshot.requested_position_rad == pytest.approx(0.5)
        assert 0.005 < snapshot.command.q_des < 0.05
        assert snapshot.command.qd_des == pytest.approx(0.2)
    finally:
        controller.shutdown()


def test_mock_controller_measures_near_200_hz_and_logs_blank_unavailable_signals(tmp_path):
    config = load_config(overrides={"logging": {"directory": str(tmp_path)}})
    controller = BenchController(MockMotorInterface(config.mock), config)
    try:
        controller.start()
        controller.connect()
        controller.enable()
        controller.start_experiment(ExperimentSpec(
            "step", {"initial_position": 0.0, "step_amplitude": 0.05,
                     "pre_hold_s": 0.15, "post_duration_s": 0.45},
            kp=30.0, kd=2.0,
        ))
        deadline = time.monotonic() + 2.0
        while controller.snapshot().experiment_active and time.monotonic() < deadline:
            time.sleep(0.01)
        while controller.logger.active and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = controller.snapshot()
        assert 180.0 <= snapshot.timing.average_hz <= 220.0
        assert snapshot.safety_event == ""
        path = tmp_path / next(item.name for item in tmp_path.glob("rs04_step_*.csv"))
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) >= 80
        assert all(row["motor_current_a"] == "" for row in rows)
        assert all(row["motor_voltage_v"] == "" for row in rows)
        assert any(row["tau_measured_nm"] != "" for row in rows)
        metadata = json.loads(next(tmp_path.glob("*_metadata.json")).read_text(encoding="utf-8"))
        assert metadata["availability"]["current"] == "unavailable"
    finally:
        controller.shutdown()


def test_step_response_reports_standard_metrics_without_fake_damping(tmp_path):
    dt = 0.005
    t = np.arange(0.0, 5.0, dt)
    step_at = 1.0
    zeta, wn = 0.35, 8.0
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    q = np.zeros_like(t)
    post = t >= step_at
    tp = t[post] - step_at
    q[post] = 0.1 * (1.0 - np.exp(-zeta * wn * tp) * (
        np.cos(wd * tp) + zeta / math.sqrt(1-zeta*zeta) * np.sin(wd * tp)
    ))
    qd = np.where(post, 0.1, 0.0)
    rows = [{
        "experiment_time": ti, "q_des_rad": target, "q_actual_rad": actual,
        "qd_actual_rad_s": velocity, "tau_commanded_nm": 0.0,
        "tau_measured_nm": "", "tau_estimated_nm": "", "motor_current_a": "",
    } for ti, target, actual, velocity in zip(t, qd, q, np.gradient(q, dt))]
    path = tmp_path / "step.csv"
    write_csv(path, rows)
    result = analyze_step_response(path)
    assert result["overshoot_percent"] > 5.0
    assert result["damping_ratio_from_overshoot"] == pytest.approx(zeta, abs=0.06)
    assert result["rise_time_s"] > 0
    assert result["settling_time_s"] > 0


def test_robust_plant_identification_recovers_clean_synthetic_parameters(tmp_path):
    dt = 0.005
    t = np.arange(0.0, 12.0, dt)
    q = 0.30 * np.sin(2*math.pi*0.55*t) + 0.10 * np.sin(2*math.pi*1.35*t)
    qd = 0.30*2*math.pi*0.55*np.cos(2*math.pi*0.55*t) + 0.10*2*math.pi*1.35*np.cos(2*math.pi*1.35*t)
    qdd = -0.30*(2*math.pi*0.55)**2*np.sin(2*math.pi*0.55*t) - 0.10*(2*math.pi*1.35)**2*np.sin(2*math.pi*1.35*t)
    inertia, damping, coulomb = 0.032, 0.18, 0.22
    tau = inertia*qdd + damping*qd + coulomb*np.sign(qd)
    rows = [{
        "experiment_time": ti, "q_actual_rad": qi, "q_des_rad": qi,
        "qd_actual_rad_s": vi, "tau_measured_nm": torque,
        "tau_commanded_nm": torque, "tau_estimated_nm": "",
    } for ti, qi, vi, torque in zip(t, q, qd, tau)]
    path = tmp_path / "chirp.csv"
    write_csv(path, rows)
    result = identify_plant(path, filter_window=21, filter_order=3)
    assert result["valid"]
    assert result["estimated_inertia_kg_m2"] == pytest.approx(inertia, rel=0.12)
    assert result["estimated_viscous_damping_nm_s_rad"] == pytest.approx(damping, rel=0.20)
    assert result["estimated_coulomb_friction_nm"] == pytest.approx(coulomb, rel=0.20)
    assert result["r_squared"] > 0.90


def test_model_based_kd_is_labeled_as_starting_estimate():
    result = recommend_kd(0.02, 0.1, 80.0, 0.7)
    assert result["model_based_starting_kd"] > 0
    assert result["label"] == "model-based starting estimate"
    assert "Validate" in result["warning"]


def test_official_rs04_protocol_has_expected_direct_impedance_fields():
    protocol = official_rs04_protocol()
    assert protocol["comm_type_mit_control"] == 1
    assert protocol["comm_type_enable"] == 3
    assert protocol["comm_type_stop"] == 4
    assert protocol["kp_max"] >= 500.0
    assert protocol["kd_max"] >= 5.0
    assert protocol["tau_max"] == pytest.approx(120.0)


def test_live_gain_update_preserves_experiment_clock_and_trajectory():
    manager = ExperimentManager()
    spec = ExperimentSpec(
        "chirp", {"center_position": 0.0, "amplitude": 0.05,
                  "f_start_hz": 0.5, "f_end_hz": 2.0, "duration_s": 5.0,
                  "kind": "linear"}, kp=20.0, kd=1.0,
    )
    manager.start(spec, now=100.0)
    before = manager.sample(101.0)
    manager.update_gains(40.0, 3.0)
    after = manager.sample(101.0)
    assert before[2] == pytest.approx(after[2])
    assert before[4].q == pytest.approx(after[4].q)
    assert after[0].kp == pytest.approx(40.0)
    assert after[0].kd == pytest.approx(3.0)
    assert after[5] == "gain_update"


def test_commanded_torque_limit_stops_before_unsafe_step_is_sent(tmp_path):
    config = load_config(overrides={"logging": {"directory": str(tmp_path)}})

    class CountingMock(MockMotorInterface):
        def __init__(self, mock_config):
            super().__init__(mock_config)
            self.sent_targets = []

        def send_command(self, command):
            self.sent_targets.append(command.q_des)
            super().send_command(command)

    motor = CountingMock(config.mock)
    controller = BenchController(motor, config)
    try:
        controller.start()
        controller.connect()
        controller.enable()
        controller.start_experiment(ExperimentSpec(
            "step", {"initial_position": 0.0, "step_amplitude": 0.20,
                     "pre_hold_s": 0.05, "post_duration_s": 0.3},
            kp=300.0, kd=2.0,
        ))
        deadline = time.monotonic() + 1.0
        while not controller.snapshot().safety_event and time.monotonic() < deadline:
            time.sleep(0.005)
        assert "commanded impedance torque exceeds limit" in controller.snapshot().safety_event
        assert not motor.enabled
        assert not any(math.isclose(target, 0.20) for target in motor.sent_targets)
    finally:
        controller.shutdown()
