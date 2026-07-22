#!/usr/bin/env python3
"""Small deterministic diagnostics shared by runtime and offline audits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingErrors:
    actor_to_feedback: np.ndarray
    actor_to_transmitted: np.ndarray
    transmitted_to_feedback: np.ndarray

    @property
    def tracking_error_max(self):
        return float(np.max(np.abs(self.transmitted_to_feedback)))

    @property
    def policy_authority_loss_max(self):
        return float(np.max(np.abs(self.actor_to_transmitted)))


def calculate_tracking_errors(actor_q_target, q_des_transmitted, q_feedback):
    actor = np.asarray(actor_q_target, dtype=np.float64).reshape(-1)
    transmitted = np.asarray(q_des_transmitted, dtype=np.float64).reshape(-1)
    feedback = np.asarray(q_feedback, dtype=np.float64).reshape(-1)
    if actor.shape != transmitted.shape or actor.shape != feedback.shape:
        raise ValueError(
            "actor, transmitted and feedback joint vectors must have identical shapes"
        )
    if not np.all(np.isfinite(np.concatenate((actor, transmitted, feedback)))):
        raise ValueError("tracking vectors contain NaN or Inf")
    return TrackingErrors(
        actor_to_feedback=actor - feedback,
        actor_to_transmitted=actor - transmitted,
        transmitted_to_feedback=transmitted - feedback,
    )


def command_targets_in_policy_order(base_target, commands, policy_index_by_joint):
    target = np.asarray(base_target, dtype=np.float32).copy()
    for command in commands or []:
        index = policy_index_by_joint.get(command.get("joint_name"))
        if index is not None and command.get("q_des") is not None:
            target[index] = float(command["q_des"])
    return target


def validate_joint_velocity_arrays(reported, finite_difference):
    """Compare reported and finite-difference qd in policy order."""
    reported = np.asarray(reported, dtype=np.float64)
    finite_difference = np.asarray(finite_difference, dtype=np.float64)
    if reported.shape != finite_difference.shape or reported.ndim != 2:
        raise ValueError("velocity arrays must have matching [samples, joints] shapes")
    rows = []
    passed = True
    for index in range(reported.shape[1]):
        a = reported[:, index]
        b = finite_difference[:, index]
        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]
        moving = (np.abs(a) > 0.03) | (np.abs(b) > 0.03)
        a = a[moving]
        b = b[moving]
        if a.size < 20 or np.std(a) < 1.0e-5 or np.std(b) < 1.0e-5:
            row = {
                "index": index,
                "correlation": float("nan"),
                "gain_ratio": float("nan"),
                "sign_agreement": float("nan"),
                "rms_difference": float("nan"),
                "passed": False,
            }
            passed = False
        else:
            correlation = float(np.corrcoef(a, b)[0, 1])
            gain_ratio = float(
                np.sqrt(np.mean(a * a))
                / max(1.0e-9, np.sqrt(np.mean(b * b)))
            )
            sign_agreement = float(np.mean(np.sign(a) == np.sign(b)))
            rms = float(np.sqrt(np.mean((a - b) ** 2)))
            joint_passed = bool(
                correlation >= 0.80
                and 0.5 <= gain_ratio <= 2.0
                and sign_agreement >= 0.95
            )
            passed = passed and joint_passed
            row = {
                "index": index,
                "correlation": correlation,
                "gain_ratio": gain_ratio,
                "sign_agreement": sign_agreement,
                "rms_difference": rms,
                "passed": joint_passed,
            }
        rows.append(row)
    return passed, rows
