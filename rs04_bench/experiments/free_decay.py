from .manager import ExperimentSpec


def free_decay_spec(duration, hold_position=0.0, disabled=True, notes=""):
    return ExperimentSpec("free_decay", {
        "duration_s": float(duration), "hold_position": float(hold_position),
        "disabled": bool(disabled),
    }, 0.0, 0.0, notes=notes)
