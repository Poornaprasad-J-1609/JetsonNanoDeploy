from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime

from ..experiments.manager import ExperimentManager, ExperimentSpec
from ..logging.data_logger import AsyncDataLogger
from ..motor.interface import MotorCommand, MotorState
from .safety import SafetyMonitor


@dataclass(frozen=True)
class TimingStatistics:
    instantaneous_hz: float = 0.0
    average_hz: float = 0.0
    minimum_hz: float = 0.0
    maximum_hz: float = 0.0
    mean_dt_s: float = 0.0
    std_dt_s: float = 0.0
    missed_cycles: int = 0
    late_cycles: int = 0
    cycle_count: int = 0


@dataclass(frozen=True)
class ControlSnapshot:
    timestamp_monotonic: float = 0.0
    state: MotorState | None = None
    command: MotorCommand = MotorCommand()
    requested_position_rad: float = 0.0
    torque_commanded_nm: float = 0.0
    torque_estimated_nm: float | None = None
    feedback_age_s: float = math.inf
    connected: bool = False
    enabled: bool = False
    experiment_mode: str = "idle"
    experiment_id: str = ""
    experiment_time: float = 0.0
    experiment_active: bool = False
    safety_event: str = ""
    experiment_event: str = ""
    timing: TimingStatistics = TimingStatistics()
    last_csv_path: str = ""


class BenchController:
    """Owns motor I/O in one measured absolute-deadline 200 Hz thread."""

    def __init__(self, motor, config):
        self.motor = motor
        self.config = config
        self.safety = SafetyMonitor(config.safety)
        self.experiments = ExperimentManager()
        self.logger = AsyncDataLogger(
            config.logging.directory,
            config.logging.queue_size,
            config.logging.flush_interval_s,
        )
        self._lock = threading.RLock()
        self._snapshot = ControlSnapshot()
        self._thread = None
        self._stop = threading.Event()
        self._estop = threading.Event()
        self._latest_state = None
        self._manual_kp = config.control.initial_kp
        self._manual_kd = config.control.initial_kd
        self._manual_tau_ff = 0.0
        self._manual_target = 0.0
        self._manual_speed = config.control.manual_speed_rad_s
        self._experiment_finishing = False
        self._history = deque(maxlen=max(1000, int(30 * config.control.frequency_hz)))

    def connect(self):
        self.motor.connect()
        state = None
        if hasattr(self.motor, "passive_poll"):
            state = self.motor.passive_poll()
        else:
            state = self.motor.read_state(timeout_s=0.0)
        if state is not None:
            self._latest_state = state
            self._manual_target = state.position
            self.experiments.set_manual_target(
                state.position, self._manual_kp, self._manual_kd, self._manual_tau_ff
            )
        with self._lock:
            self._snapshot = replace(
                self._snapshot, state=state, connected=True,
                feedback_age_s=0.0 if state else math.inf,
            )

    def disconnect(self):
        self.emergency_stop("operator disconnect")
        self.motor.disconnect()
        with self._lock:
            self._snapshot = replace(self._snapshot, connected=False, enabled=False)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rs04-motor-200hz", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        try:
            if self.motor.connected:
                self.motor.disable()
                self.motor.disconnect()
        finally:
            if self.logger.active:
                self.logger.stop({"shutdown": True})

    def enable(self):
        if not self.motor.connected:
            raise RuntimeError("connect before enabling the motor")
        state = self.motor.read_state(timeout_s=0.01) or self._latest_state
        if state is None:
            raise RuntimeError("cannot enable without fresh motor position feedback")
        self._estop.clear()
        self._latest_state = state
        self._manual_target = state.position
        self.experiments.set_manual_target(
            state.position, self._manual_kp, self._manual_kd, self._manual_tau_ff
        )
        self.motor.enable()

    def disable(self):
        self.experiments.stop()
        if self.motor.connected:
            self.motor.disable()
        self._finish_experiment({"stopped_by": "disable"})

    def emergency_stop(self, reason="operator emergency stop"):
        self._estop.set()
        self.experiments.stop()
        try:
            if self.motor.connected:
                self.motor.disable()
        finally:
            with self._lock:
                self._snapshot = replace(self._snapshot, enabled=False, safety_event=str(reason))
            self._finish_experiment({"safety_event": str(reason)})

    def update_gains(self, kp=None, kd=None):
        kp = self._manual_kp if kp is None else float(kp)
        kd = self._manual_kd if kd is None else float(kd)
        c = self.config.control
        if not c.kp_min <= kp <= c.kp_max:
            raise ValueError(f"Kp must be within [{c.kp_min}, {c.kp_max}]")
        if not c.kd_min <= kd <= c.kd_max:
            raise ValueError(f"Kd must be within [{c.kd_min}, {c.kd_max}]")
        self._manual_kp, self._manual_kd = kp, kd
        self.experiments.update_gains(kp, kd)

    def set_manual_target(self, target):
        target = float(target)
        limits = self.config.safety
        if not limits.min_position_rad <= target <= limits.max_position_rad:
            raise ValueError("target exceeds configured position limits")
        self._manual_target = target
        start = self._snapshot.command.q_des
        self.experiments.set_manual_target(
            target, self._manual_kp, self._manual_kd, self._manual_tau_ff,
            initial_position=start, speed_rad_s=self._manual_speed,
        )

    def update_manual_speed(self, speed_rad_s):
        speed = float(speed_rad_s)
        if not math.isfinite(speed) or not 0 < speed <= self.config.safety.max_velocity_rad_s:
            raise ValueError(
                f"manual speed must be within (0, {self.config.safety.max_velocity_rad_s}] rad/s"
            )
        self._manual_speed = speed
        return speed

    def hold_current_position(self):
        if self._latest_state is None:
            raise RuntimeError("cannot capture hold target without motor feedback")
        target = float(self._latest_state.position)
        self._manual_target = target
        self.experiments.start(ExperimentSpec(
            mode="hold", parameters={"position": target},
            kp=self._manual_kp, kd=self._manual_kd, tau_ff=self._manual_tau_ff,
        ))
        return target

    def start_manual_logging(self):
        if self.logger.active:
            raise RuntimeError("a log session is already active")
        spec, _, _ = self.experiments.snapshot()
        if spec.mode not in {"manual", "hold"}:
            raise RuntimeError("manual logging is available only in manual/hold mode")
        self.logger.start(spec.mode, self._metadata(spec))
        self._experiment_finishing = False

    def stop_logging(self):
        self._finish_experiment({"stopped_by": "operator log stop"})

    def nudge(self, direction):
        self.set_manual_target(
            self._manual_target + float(direction) * self.config.control.manual_step_rad
        )

    def start_experiment(self, spec: ExperimentSpec):
        if self.logger.active:
            raise RuntimeError("finish the active experiment before starting another")
        if spec.mode != "free_decay" and not self.motor.enabled:
            raise RuntimeError("enable the motor before starting this experiment")
        self.update_gains(spec.kp, spec.kd)
        if not self.config.control.tau_ff_min_nm <= float(spec.tau_ff) <= self.config.control.tau_ff_max_nm:
            raise ValueError(
                f"tau_ff must be within [{self.config.control.tau_ff_min_nm}, "
                f"{self.config.control.tau_ff_max_nm}] Nm"
            )
        self._validate_experiment_targets(spec)
        metadata = self._metadata(spec)
        self.logger.start(spec.mode, metadata)
        self._experiment_finishing = False
        experiment_id = self.experiments.start(spec)
        if spec.mode == "free_decay" and spec.parameters.get("disabled", True):
            self.motor.disable()
        return experiment_id

    def _validate_experiment_targets(self, spec):
        p = spec.parameters
        candidates = []
        if spec.mode == "step":
            candidates = [p["initial_position"], p["initial_position"] + p["step_amplitude"]]
        elif spec.mode == "chirp":
            candidates = [p["center_position"] - abs(p["amplitude"]), p["center_position"] + abs(p["amplitude"])]
        elif spec.mode in {"manual", "hold"}:
            candidates = [p["position"]]
        for value in candidates:
            if not self.config.safety.min_position_rad <= value <= self.config.safety.max_position_rad:
                raise ValueError(f"experiment target {value:+.4f} rad exceeds position limits")

    def _metadata(self, spec):
        return {
            "motor_model": self.config.motor.model,
            "motor_id": self.config.motor.id,
            "can_interface": self.config.motor.interface,
            "can_bitrate": self.config.motor.bitrate,
            "control_rate_hz": self.config.control.frequency_hz,
            "kp": spec.kp, "kd": spec.kd, "tau_ff_nm": spec.tau_ff,
            "experiment_type": spec.mode,
            "test_parameters": spec.parameters,
            "pendulum": self.config.pendulum,
            "safety_limits": self.config.safety,
            "torque_feedback_source": self.config.motor.torque_feedback_source,
            "current_source": self.config.motor.current_source,
            "voltage_source": self.config.motor.voltage_source,
            "current_calibration": {
                "torque_constant_nm_per_a": self.config.motor.torque_constant_nm_per_a,
                "formula": "tau_estimated_nm = torque_constant_nm_per_a * motor_current_a",
            },
            "notes": spec.notes,
        }

    def _finish_experiment(self, metadata=None):
        with self._lock:
            if not self.logger.active or self._experiment_finishing:
                return
            self._experiment_finishing = True
        def finish():
            path = self.logger.stop(metadata or {})
            with self._lock:
                self._snapshot = replace(self._snapshot, last_csv_path=str(path or ""))
                self._experiment_finishing = False
        threading.Thread(target=finish, name="rs04-experiment-finalize", daemon=True).start()

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def history(self):
        with self._lock:
            return list(self._history)

    def _run(self):
        period_ns = int(round(1_000_000_000.0 / self.config.control.frequency_hz))
        next_deadline = time.perf_counter_ns()
        previous_start = None
        dts = deque(maxlen=4000)
        missed = 0
        late = 0
        consecutive_late = 0
        cycle = 0
        last_passive_poll = 0.0
        while not self._stop.is_set():
            now_ns = time.perf_counter_ns()
            if now_ns < next_deadline:
                time.sleep((next_deadline - now_ns) / 1e9)
            cycle_start_ns = time.perf_counter_ns()
            cycle_start = cycle_start_ns / 1e9
            lateness_s = max(0.0, (cycle_start_ns - next_deadline) / 1e9)
            if previous_start is None:
                dt = self.config.period_s
            else:
                dt = (cycle_start_ns - previous_start) / 1e9
                dts.append(dt)
            previous_start = cycle_start_ns
            if lateness_s > self.config.period_s * 0.25:
                late += 1
                consecutive_late += 1
            else:
                consecutive_late = 0
            skipped = max(0, int((cycle_start_ns - next_deadline) // period_ns))
            missed += skipped
            next_deadline += (skipped + 1) * period_ns

            spec, exp_id, elapsed, active, point, event = self.experiments.sample(cycle_start)
            command = MotorCommand(point.q, point.qd, spec.kp, spec.kd, spec.tau_ff)
            state = None
            safety_event = ""
            previous_state = self._latest_state
            previous_feedback_age = (
                math.inf if previous_state is None
                else max(0.0, cycle_start - previous_state.timestamp_monotonic)
            )
            if self.motor.enabled:
                safety_event = self.safety.check(
                    previous_state, command, previous_feedback_age
                ) or ""
            try:
                if (
                    self.motor.connected and self.motor.enabled
                    and not self._estop.is_set() and not safety_event
                ):
                    self.motor.send_command(command)
                    state = self.motor.read_state(timeout_s=min(0.002, self.config.period_s * 0.4))
                elif self.motor.connected and (
                    (active and spec.mode == "free_decay")
                    or cycle_start - last_passive_poll >= 0.05
                ):
                    if hasattr(self.motor, "passive_poll"):
                        state = self.motor.passive_poll()
                    else:
                        state = self.motor.read_state(timeout_s=0.0)
                    last_passive_poll = cycle_start
            except Exception as exc:
                safety_event = f"communication failure: {exc}"
            if state is not None:
                self._latest_state = state
            state = self._latest_state
            feedback_age = math.inf if state is None else max(0.0, cycle_start - state.timestamp_monotonic)
            if not safety_event and self.motor.enabled:
                safety_event = self.safety.check(state, command, feedback_age) or ""
            if consecutive_late >= self.config.safety.max_consecutive_late_cycles:
                safety_event = f"control loop missed {consecutive_late} consecutive timing deadlines"
            if self.logger.error is not None:
                safety_event = f"CSV writer failed: {self.logger.error}"
            if safety_event:
                self._estop.set()
                try:
                    if self.motor.connected:
                        self.motor.disable()
                except Exception:
                    pass
                self.experiments.stop()

            tau_commanded = 0.0
            tau_estimated = None
            if state is not None:
                tau_commanded = command.kp * (command.q_des - state.position) + command.kd * (
                    command.qd_des - state.velocity
                ) + command.tau_ff
                if (
                    state.current is not None
                    and self.config.motor.torque_constant_nm_per_a is not None
                ):
                    tau_estimated = (
                        float(state.current)
                        * float(self.config.motor.torque_constant_nm_per_a)
                    )
            timing = self._timing(dts, dt, missed, late, cycle + 1)
            snapshot = ControlSnapshot(
                timestamp_monotonic=cycle_start, state=state, command=command,
                requested_position_rad=(
                    self._manual_target if spec.mode in {"manual", "hold"} else command.q_des
                ),
                torque_commanded_nm=tau_commanded, torque_estimated_nm=tau_estimated,
                feedback_age_s=feedback_age, connected=self.motor.connected,
                enabled=self.motor.enabled, experiment_mode=spec.mode,
                experiment_id=exp_id, experiment_time=elapsed,
                experiment_active=active, safety_event=safety_event,
                experiment_event=event, timing=timing,
                last_csv_path=self._snapshot.last_csv_path,
            )
            with self._lock:
                self._snapshot = snapshot
                self._history.append(snapshot)
            if self.logger.active:
                try:
                    self.logger.log(self._row(snapshot, cycle, lateness_s))
                except Exception as exc:
                    self.emergency_stop(str(exc))
            if not active and self.logger.active and spec.mode not in {"manual", "hold"}:
                self._finish_experiment({"completed_normally": not bool(safety_event)})
            cycle += 1

    @staticmethod
    def _timing(dts, current_dt, missed, late, cycles):
        finite = [value for value in dts if value > 0]
        mean = statistics.fmean(finite) if finite else current_dt
        std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
        frequencies = [1.0 / value for value in finite] or [0.0]
        return TimingStatistics(
            instantaneous_hz=0.0 if current_dt <= 0 else 1.0 / current_dt,
            average_hz=0.0 if mean <= 0 else 1.0 / mean,
            minimum_hz=min(frequencies), maximum_hz=max(frequencies),
            mean_dt_s=mean, std_dt_s=std, missed_cycles=missed,
            late_cycles=late, cycle_count=cycles,
        )

    @staticmethod
    def _row(snapshot, cycle, lateness_s):
        state, command = snapshot.state, snapshot.command
        return {
            "timestamp_wall": datetime.now().astimezone().isoformat(),
            "timestamp_monotonic": snapshot.timestamp_monotonic,
            "experiment_time": snapshot.experiment_time,
            "control_dt": snapshot.timing.mean_dt_s if cycle == 0 else (
                0.0 if snapshot.timing.instantaneous_hz <= 0 else 1.0 / snapshot.timing.instantaneous_hz
            ),
            "control_frequency": snapshot.timing.instantaneous_hz,
            "experiment_mode": snapshot.experiment_mode,
            "q_des_rad": command.q_des,
            "q_target_requested_rad": snapshot.requested_position_rad,
            "q_actual_rad": None if state is None else state.position,
            "position_error_rad": None if state is None else command.q_des - state.position,
            "qd_des_rad_s": command.qd_des,
            "qd_actual_rad_s": None if state is None else state.velocity,
            "velocity_error_rad_s": None if state is None else command.qd_des - state.velocity,
            "kp": command.kp, "kd": command.kd, "tau_ff_nm": command.tau_ff,
            "tau_commanded_nm": snapshot.torque_commanded_nm,
            "tau_measured_nm": None if state is None else state.torque_measured,
            "tau_estimated_nm": snapshot.torque_estimated_nm,
            "motor_current_a": None if state is None else state.current,
            "motor_voltage_v": None if state is None else state.voltage,
            "motor_temperature_c": None if state is None else state.temperature,
            "motor_enabled": int(snapshot.enabled), "safety_event": snapshot.safety_event,
            "experiment_event": snapshot.experiment_event, "experiment_id": snapshot.experiment_id,
            "cycle_index": cycle, "deadline_lateness_s": lateness_s,
            "missed_cycles_total": snapshot.timing.missed_cycles,
            "feedback_age_s": snapshot.feedback_age_s,
            "motor_fault_bits": None if state is None else state.fault_bits,
            "motor_mode_status": None if state is None else state.mode_status,
        }
