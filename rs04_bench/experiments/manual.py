from .manager import ExperimentSpec


def manual_spec(position, kp, kd, tau_ff=0.0, notes=""):
    return ExperimentSpec("manual", {"position": float(position)}, float(kp), float(kd), float(tau_ff), notes)


def hold_spec(position, kp, kd, tau_ff=0.0, notes=""):
    return ExperimentSpec("hold", {"position": float(position)}, float(kp), float(kd), float(tau_ff), notes)
