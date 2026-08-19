from __future__ import annotations

import csv
import math

import numpy as np


def load_numeric_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns = {}
    if not rows:
        return columns
    for key in rows[0]:
        values = []
        for row in rows:
            text = row.get(key, "")
            try:
                values.append(float(text) if text not in ("", None) else math.nan)
            except (TypeError, ValueError):
                values.append(math.nan)
        columns[key] = np.asarray(values, dtype=float)
    columns["_raw_rows"] = rows
    return columns


def local_polynomial_filter(time_s, values, window=21, order=3):
    """Savitzky-Golay-style local polynomial smooth and derivatives.

    Data is interpolated onto its median-rate uniform grid before applying
    least-squares polynomial convolution. This avoids differentiating noisy
    encoder values directly and does not require SciPy on the Jetson.
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(x)
    if mask.sum() < max(window, order + 2):
        raise ValueError("insufficient finite samples for local polynomial filtering")
    t_valid, x_valid = t[mask], x[mask]
    order_idx = np.argsort(t_valid)
    t_valid, x_valid = t_valid[order_idx], x_valid[order_idx]
    unique = np.r_[True, np.diff(t_valid) > 1e-9]
    t_valid, x_valid = t_valid[unique], x_valid[unique]
    dt = float(np.median(np.diff(t_valid)))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid sample timestamps")
    grid = t_valid[0] + np.arange(int((t_valid[-1] - t_valid[0]) / dt) + 1) * dt
    uniform = np.interp(grid, t_valid, x_valid)
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(grid) - (1 - len(grid) % 2))
    if window <= order:
        raise ValueError("filter window is too short for polynomial order")
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=float) * dt
    design = np.vander(offsets, N=order + 1, increasing=True)
    pinv = np.linalg.pinv(design)
    padded = np.pad(uniform, (half, half), mode="reflect")
    smooth = np.convolve(padded, pinv[0][::-1], mode="valid")
    velocity = np.convolve(padded, pinv[1][::-1], mode="valid")
    acceleration = np.convolve(padded, (2.0 * pinv[2])[::-1], mode="valid")
    return grid, uniform, smooth, velocity, acceleration


def robust_linear_regression(design, target, max_iterations=30, huber_delta=1.5):
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.shape[0] <= x.shape[1] * 10:
        raise ValueError("insufficient finite samples for robust regression")
    weights = np.ones(len(y))
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iterations):
        residual = y - x @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-12
        normalized = np.abs(residual) / scale
        weights = np.where(normalized <= huber_delta, 1.0, huber_delta / normalized)
        weighted_x = x * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
        if np.linalg.norm(updated - beta) < 1e-10:
            beta = updated
            break
        beta = updated
    predicted = x @ beta
    residual = y - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum(residual ** 2) / ss_total) if ss_total > 1e-12 else math.nan
    covariance = np.linalg.pinv(x.T @ (weights[:, None] * x)) * rmse ** 2
    stderr = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return beta, rmse, r_squared, stderr, mask


def estimate_delay_ms(time_s, command, response, max_delay_s=0.2):
    t = np.asarray(time_s, dtype=float)
    u = np.asarray(command, dtype=float)
    y = np.asarray(response, dtype=float)
    mask = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    if mask.sum() < 50:
        return None
    t, u, y = t[mask], u[mask], y[mask]
    dt = float(np.median(np.diff(t)))
    du = np.gradient(u, dt)
    dy = np.gradient(y, dt)
    du -= np.mean(du)
    dy -= np.mean(dy)
    max_lag = min(int(max_delay_s / dt), len(du) // 3)
    correlations = []
    for lag in range(max_lag + 1):
        a = du[: len(du) - lag or None]
        b = dy[lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        correlations.append(-math.inf if denom <= 1e-12 else float(np.dot(a, b) / denom))
    lag = int(np.argmax(correlations))
    if not math.isfinite(correlations[lag]) or correlations[lag] < 0.15:
        return None
    return {"delay_ms": 1000.0 * lag * dt, "delay_cycles": lag, "correlation": correlations[lag]}
