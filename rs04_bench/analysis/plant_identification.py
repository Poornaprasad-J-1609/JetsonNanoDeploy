from __future__ import annotations

import math

import numpy as np

from .signal_processing import (
    estimate_delay_ms,
    load_numeric_csv,
    local_polynomial_filter,
    robust_linear_regression,
)


def gravity_torque(q, pendulum):
    mass_moment = (
        float(pendulum.get("attached_mass_kg", 0.0)) * float(pendulum.get("mass_com_radius_m", 0.0))
        + float(pendulum.get("lever_mass_kg", 0.0)) * float(pendulum.get("lever_com_radius_m", 0.0))
    )
    magnitude = mass_moment * float(pendulum.get("gravity_m_s2", 9.80665))
    convention = pendulum.get("angle_zero_convention", "downward_vertical")
    if convention == "downward_vertical":
        return magnitude * np.sin(q)
    if convention == "horizontal":
        return magnitude * np.cos(q)
    raise ValueError("unknown pendulum angle zero convention")


def identify_plant(
    path,
    pendulum=None,
    filter_window=21,
    filter_order=3,
    velocity_deadband=0.03,
    minimum_samples=200,
    minimum_excitation_velocity=0.15,
    torque_source="auto",
):
    data = load_numeric_csv(path)
    pendulum = pendulum or {}
    if "experiment_time" not in data or "q_actual_rad" not in data:
        raise ValueError("CSV lacks position/time data")
    sources = {
        "measured": "tau_measured_nm",
        "estimated": "tau_estimated_nm",
        "commanded": "tau_commanded_nm",
    }
    selected = None
    if torque_source == "auto":
        for candidate in ("measured", "estimated", "commanded"):
            values = data.get(sources[candidate])
            if values is not None and np.isfinite(values).sum() >= minimum_samples:
                selected = candidate
                break
    elif torque_source in sources:
        selected = torque_source
    if selected is None or sources[selected] not in data:
        raise ValueError("no usable measured, estimated, or commanded torque signal")
    t, raw_q, filtered_q, velocity, acceleration = local_polynomial_filter(
        data["experiment_time"], data["q_actual_rad"], filter_window, filter_order
    )
    raw_t = data["experiment_time"]
    raw_tau = data[sources[selected]]
    finite_tau = np.isfinite(raw_t) & np.isfinite(raw_tau)
    if finite_tau.sum() < minimum_samples:
        raise ValueError("insufficient finite torque samples")
    torque = np.interp(t, raw_t[finite_tau], raw_tau[finite_tau])
    moving = np.abs(velocity) >= float(velocity_deadband)
    finite = moving & np.isfinite(acceleration) & np.isfinite(torque)
    if finite.sum() < minimum_samples:
        raise ValueError("insufficient moving samples after friction deadband filtering")
    if float(np.percentile(np.abs(velocity[finite]), 90)) < minimum_excitation_velocity:
        raise ValueError("excitation is insufficient for defensible inertia/damping identification")
    gravity = gravity_torque(filtered_q, pendulum)
    target = torque - gravity
    design = np.column_stack((acceleration, velocity, np.sign(velocity)))
    beta, rmse, r_squared, stderr, regression_mask = robust_linear_regression(
        design[finite], target[finite]
    )
    inertia, damping, coulomb = (float(value) for value in beta)
    warnings = []
    if selected == "commanded":
        warnings.append("Fit used commanded impedance torque, not measured torque; actuator saturation/delay can bias estimates.")
    if inertia <= 0:
        warnings.append("Estimated inertia is non-positive; reject this fit.")
    if damping < 0:
        warnings.append("Estimated viscous damping is negative; excitation/model quality is inadequate.")
    if r_squared < 0.5:
        warnings.append("R-squared is below 0.5; do not use these parameters for gain selection.")
    delay = estimate_delay_ms(
        data["experiment_time"],
        data.get("tau_commanded_nm", data["q_des_rad"]),
        data.get("qd_actual_rad_s", data["q_actual_rad"]),
    )
    known_load_inertia = (
        float(pendulum.get("known_lever_inertia_kg_m2", 0.0))
        + float(pendulum.get("attached_mass_kg", 0.0)) * float(pendulum.get("mass_com_radius_m", 0.0)) ** 2
    )
    return {
        "estimated_inertia_kg_m2": inertia,
        "estimated_viscous_damping_nm_s_rad": damping,
        "estimated_coulomb_friction_nm": coulomb,
        "standard_error": {
            "inertia": float(stderr[0]), "damping": float(stderr[1]), "coulomb": float(stderr[2])
        },
        "known_load_inertia_kg_m2": known_load_inertia,
        "estimated_motor_reflected_inertia_kg_m2": inertia - known_load_inertia,
        "fit_rmse_nm": rmse,
        "r_squared": r_squared,
        "samples_used": int(finite.sum()),
        "torque_source": selected,
        "delay": delay,
        "valid": inertia > 0 and damping >= 0 and r_squared >= 0.5,
        "warnings": warnings,
        "signals": {
            "time_s": t, "position_raw_rad": raw_q, "position_filtered_rad": filtered_q,
            "velocity_filtered_rad_s": velocity, "acceleration_filtered_rad_s2": acceleration,
            "torque_nm": torque, "gravity_torque_nm": gravity,
        },
    }


def recommend_kd(inertia, damping, kp, desired_zeta):
    values = (inertia, damping, kp, desired_zeta)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gain recommendation inputs must be finite")
    if inertia <= 0 or kp <= 0 or desired_zeta <= 0:
        raise ValueError("inertia, Kp, and desired damping ratio must be positive")
    kd = 2.0 * desired_zeta * math.sqrt(inertia * kp) - damping
    return {
        "kp": float(kp), "desired_damping_ratio": float(desired_zeta),
        "model_based_starting_kd": max(0.0, float(kd)),
        "unclamped_kd": float(kd),
        "natural_frequency_rad_s": math.sqrt(kp / inertia),
        "label": "model-based starting estimate",
        "warning": "Validate with a small hardware step; saturation, delay, compliance and discrete-time effects are not represented.",
    }
