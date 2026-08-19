from __future__ import annotations

import math

import numpy as np

from .signal_processing import estimate_delay_ms, load_numeric_csv


def _finite_max(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return None if not len(values) else float(np.max(np.abs(values)))


def analyze_step_response(path, settling_band_fraction=0.02):
    data = load_numeric_csv(path)
    required = ("experiment_time", "q_des_rad", "q_actual_rad")
    if any(key not in data for key in required):
        raise ValueError("CSV lacks step-response columns")
    t, qd, q = (data[key] for key in required)
    mask = np.isfinite(t) & np.isfinite(qd) & np.isfinite(q)
    t, qd, q = t[mask], qd[mask], q[mask]
    if len(t) < 50:
        raise ValueError("insufficient samples for step analysis")
    changes = np.abs(np.diff(qd))
    if not len(changes) or float(np.max(changes)) < 1e-5:
        raise ValueError("no position step found")
    step_index = int(np.argmax(changes)) + 1
    pre = q[max(0, step_index - max(10, step_index // 3)):step_index]
    tail_count = max(10, int(0.10 * len(q)))
    initial = float(np.median(pre))
    target = float(qd[step_index])
    final = float(np.median(q[-tail_count:]))
    amplitude = target - float(qd[step_index - 1])
    response_amplitude = final - initial
    direction = 1.0 if response_amplitude >= 0 else -1.0
    normalized = direction * (q - initial)
    final_norm = abs(response_amplitude)
    if final_norm < 1e-5:
        raise ValueError("step response has negligible measured movement")
    after = np.arange(step_index, len(t))
    def first_crossing(level):
        indices = after[normalized[after] >= level * final_norm]
        return None if not len(indices) else float(t[indices[0]] - t[step_index])
    t10, t90 = first_crossing(0.10), first_crossing(0.90)
    rise_time = None if t10 is None or t90 is None else t90 - t10
    peak_local = int(np.argmax(normalized[after]))
    peak_index = int(after[peak_local])
    peak = float(q[peak_index])
    overshoot_ratio = max(0.0, (normalized[peak_index] - final_norm) / final_norm)
    band = max(abs(amplitude) * settling_band_fraction, 1e-4)
    outside = np.flatnonzero(np.abs(q[step_index:] - final) > band)
    settling = None
    if len(outside) and outside[-1] + step_index + 1 < len(t):
        settling = float(t[outside[-1] + step_index + 1] - t[step_index])
    elif not len(outside):
        settling = 0.0
    zeta = None
    # Below 1% the value is usually encoder/filter quantization or a monotonic
    # tail, not a defensible logarithmic-decrement observation.
    if 0.01 <= overshoot_ratio < 1.0:
        log_mp = math.log(overshoot_ratio)
        zeta = -log_mp / math.sqrt(math.pi ** 2 + log_mp ** 2)
    centered = direction * (q[step_index:] - final)
    peaks = []
    for index in range(1, len(centered) - 1):
        if centered[index] > centered[index - 1] and centered[index] >= centered[index + 1] and centered[index] > band:
            peaks.append(index + step_index)
    damped_frequency_hz = None
    if len(peaks) >= 2:
        periods = np.diff(t[peaks])
        if np.all(periods > 0):
            damped_frequency_hz = float(1.0 / np.median(periods))
    delay = estimate_delay_ms(t, qd, q)
    result = {
        "step_amplitude_rad": float(amplitude),
        "initial_position_rad": initial,
        "steady_state_position_rad": final,
        "steady_state_error_rad": target - final,
        "rise_time_s": rise_time,
        "peak_time_s": float(t[peak_index] - t[step_index]),
        "peak_position_rad": peak,
        "overshoot_percent": 100.0 * overshoot_ratio,
        "settling_time_s": settling,
        "damped_frequency_hz": damped_frequency_hz,
        "damping_ratio_from_overshoot": zeta,
        "damping_estimate_valid": zeta is not None,
        "rms_position_error_rad": float(np.sqrt(np.mean((qd[step_index:] - q[step_index:]) ** 2))),
        "maximum_velocity_rad_s": _finite_max(data.get("qd_actual_rad_s", [])),
        "maximum_commanded_torque_nm": _finite_max(data.get("tau_commanded_nm", [])),
        "maximum_measured_torque_nm": _finite_max(data.get("tau_measured_nm", [])),
        "maximum_estimated_torque_nm": _finite_max(data.get("tau_estimated_nm", [])),
        "maximum_current_a": _finite_max(data.get("motor_current_a", [])),
        "delay": delay,
        "quality_note": (
            "Damping ratio is meaningful only for an underdamped response with measurable overshoot."
            if zeta is not None else
            "Response was not sufficiently underdamped; damping ratio from overshoot was not reported."
        ),
    }
    return result
