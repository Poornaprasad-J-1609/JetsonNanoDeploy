from .manager import ExperimentSpec


def chirp_spec(center, amplitude, f_start, f_end, duration, kp, kd, kind="linear", notes=""):
    return ExperimentSpec("chirp", {
        "center_position": float(center), "amplitude": float(amplitude),
        "f_start_hz": float(f_start), "f_end_hz": float(f_end),
        "duration_s": float(duration), "kind": str(kind),
    }, float(kp), float(kd), notes=notes)
