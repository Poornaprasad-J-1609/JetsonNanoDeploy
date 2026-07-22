#!/usr/bin/env python3
"""Frequency, phase, and diagonal-pair analysis for quadruped gait signals."""

from dataclasses import dataclass

import numpy as np

from joint_mapping import POLICY_JOINT_ORDER


LEG_BY_JOINT = {name: name.split("_", 1)[0] for name in POLICY_JOINT_ORDER}
TYPE_BY_JOINT = {name: name.split("_", 2)[1] for name in POLICY_JOINT_ORDER}
DIAGONAL_A = {"FL", "BR"}
DIAGONAL_B = {"FR", "BL"}


@dataclass(frozen=True)
class SignalEstimate:
    frequency_hz: float
    amplitude: float
    phase_rad: float
    periodic: bool
    sufficient_amplitude: bool


def estimate_signal(signal, sample_rate_hz, minimum_amplitude=0.01):
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    values = values[valid]
    if values.size < 20 or sample_rate_hz <= 0.0:
        return SignalEstimate(0.0, 0.0, 0.0, False, False)
    centered = values - np.mean(values)
    amplitude = 0.5 * float(np.max(centered) - np.min(centered))
    sufficient = amplitude >= float(minimum_amplitude)
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / float(sample_rate_hz))
    band = (frequencies >= 0.25) & (frequencies <= min(8.0, 0.45 * sample_rate_hz))
    if not np.any(band):
        return SignalEstimate(0.0, amplitude, 0.0, False, sufficient)
    indices = np.flatnonzero(band)
    dominant_index = int(indices[np.argmax(np.abs(spectrum[indices]))])
    dominant_power = float(np.abs(spectrum[dominant_index]) ** 2)
    total_power = float(np.sum(np.abs(spectrum[indices]) ** 2))
    periodic = bool(sufficient and total_power > 0.0 and dominant_power / total_power >= 0.35)
    return SignalEstimate(
        frequency_hz=float(frequencies[dominant_index]),
        amplitude=amplitude,
        phase_rad=float(np.angle(spectrum[dominant_index])),
        periodic=periodic,
        sufficient_amplitude=sufficient,
    )


def _wrap_phase(value):
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def pairwise_phase_rows(signals, sample_rate_hz, minimum_amplitude=0.01):
    signals = np.asarray(signals, dtype=np.float64)
    estimates = [
        estimate_signal(signals[:, i], sample_rate_hz, minimum_amplitude)
        for i in range(signals.shape[1])
    ]
    rows = []
    for i, joint_a in enumerate(POLICY_JOINT_ORDER):
        for j, joint_b in enumerate(POLICY_JOINT_ORDER):
            if TYPE_BY_JOINT[joint_a] != TYPE_BY_JOINT[joint_b]:
                continue
            a = signals[:, i] - np.nanmean(signals[:, i])
            b = signals[:, j] - np.nanmean(signals[:, j])
            correlation = (
                float(np.corrcoef(a, b)[0, 1])
                if np.std(a) > 1.0e-9 and np.std(b) > 1.0e-9 else np.nan
            )
            phase = _wrap_phase(estimates[j].phase_rad - estimates[i].phase_rad)
            rows.append({
                "joint_a": joint_a,
                "joint_b": joint_b,
                "joint_type": TYPE_BY_JOINT[joint_a],
                "correlation": correlation,
                "phase_difference_rad": phase,
                "phase_difference_deg": float(np.degrees(phase)),
                "frequency_a_hz": estimates[i].frequency_hz,
                "frequency_b_hz": estimates[j].frequency_hz,
                "amplitude_a": estimates[i].amplitude,
                "amplitude_b": estimates[j].amplitude,
                "periodic_a": int(estimates[i].periodic),
                "periodic_b": int(estimates[j].periodic),
            })
    return estimates, rows


def classify_diagonal_trot(signals, sample_rate_hz, minimum_amplitude=0.01):
    estimates, rows = pairwise_phase_rows(
        signals, sample_rate_hz, minimum_amplitude=minimum_amplitude
    )
    periodic_count = sum(item.periodic for item in estimates)
    if periodic_count < 8:
        return "UNKNOWN" if periodic_count < 4 else "FAIL", estimates, rows

    pair_lookup = {(row["joint_a"], row["joint_b"]): row for row in rows}
    same_diagonal = []
    opposite_diagonal = []
    for joint_type in ("hip", "thigh", "calf"):
        by_leg = {
            LEG_BY_JOINT[name]: name
            for name in POLICY_JOINT_ORDER
            if TYPE_BY_JOINT[name] == joint_type
        }
        for leg_a, leg_b in (("FL", "BR"), ("FR", "BL")):
            same_diagonal.append(pair_lookup[(by_leg[leg_a], by_leg[leg_b])])
        for leg_a, leg_b in (("FL", "FR"), ("BR", "BL")):
            opposite_diagonal.append(pair_lookup[(by_leg[leg_a], by_leg[leg_b])])

    same_ok = sum(
        abs(row["phase_difference_deg"]) <= 70.0 or row["correlation"] >= 0.30
        for row in same_diagonal
    ) >= 4
    opposite_ok = sum(
        abs(abs(row["phase_difference_deg"]) - 180.0) <= 80.0
        or row["correlation"] <= -0.20
        for row in opposite_diagonal
    ) >= 3
    return ("PASS" if same_ok and opposite_ok else "FAIL"), estimates, rows
