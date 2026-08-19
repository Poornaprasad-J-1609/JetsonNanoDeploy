from .manager import ExperimentSpec


def step_spec(initial_position, amplitude, kp, kd, pre_hold_s=2.0, post_duration_s=5.0, notes=""):
    return ExperimentSpec("step", {
        "initial_position": float(initial_position), "step_amplitude": float(amplitude),
        "pre_hold_s": float(pre_hold_s), "post_duration_s": float(post_duration_s),
    }, float(kp), float(kd), notes=notes)
