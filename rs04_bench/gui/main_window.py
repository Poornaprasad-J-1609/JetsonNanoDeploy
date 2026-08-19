from __future__ import annotations

import json
import math
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..analysis.plant_identification import identify_plant, recommend_kd
from ..analysis.step_response import analyze_step_response
from ..experiments.gain_sweep import GainSweepRunner
from ..experiments.manager import ExperimentSpec
from .angle_gauge import AngleGauge
from .plots import RollingPlots


class BenchMainWindow:
    def __init__(self, root, controller, config):
        self.root, self.controller, self.config = root, controller, config
        self.root.title("RS04 Actuator Characterization Bench")
        self.root.geometry("1480x920")
        self.root.minsize(1180, 760)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.status = tk.StringVar(value="Disconnected. Motor commands are disabled.")
        self.safety = tk.StringVar(value="SAFE / DISABLED")
        self.telemetry = {name: tk.StringVar(value="--") for name in (
            "position", "desired_position", "velocity", "desired_velocity", "position_error",
            "requested_position",
            "velocity_error", "tau_commanded", "tau_measured", "tau_estimated", "current",
            "voltage", "temperature", "kp", "kd", "control_hz", "gui_hz", "elapsed",
            "timing", "feedback_age",
        )}
        self.kp_entry = tk.StringVar(value=str(config.control.initial_kp))
        self.kd_entry = tk.StringVar(value=str(config.control.initial_kd))
        self.target_entry = tk.StringVar(value="0.0")
        self.speed_entry = tk.StringVar(value=str(config.control.manual_speed_rad_s))
        self.result_text = None
        self._last_gui = time.perf_counter()
        self._gui_dts = []
        self._last_analyzed_path = ""
        self._sweep_was_running = False
        self.sweep = GainSweepRunner(controller, callback=self._sweep_result)
        self._build()
        self.root.bind("<Left>", lambda event: self._nudge(-1))
        self.root.bind("<Right>", lambda event: self._nudge(1))
        self._tick()

    def _build(self):
        style = ttk.Style()
        style.configure("Emergency.TButton", foreground="#b91c1c", font=("TkDefaultFont", 11, "bold"))
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        for text, command in (
            ("CONNECT", self._connect), ("ENABLE MOTOR", self._enable),
            ("DISABLE MOTOR", self._disable), ("EMERGENCY STOP", self._estop),
        ):
            ttk.Button(top, text=text, command=command, style="Emergency.TButton" if "EMERGENCY" in text else None).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT, padx=18)
        ttk.Label(top, textvariable=self.safety, foreground="#b91c1c", font=("TkDefaultFont", 11, "bold")).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=2)
        body.add(right, weight=3)
        self.gauge = AngleGauge(
            left, size=330,
            limit_rad=max(abs(self.config.safety.min_position_rad), abs(self.config.safety.max_position_rad)),
            command=self._dial_target,
        )
        self.gauge.pack(pady=4)
        self._telemetry_panel(left).pack(fill=tk.X, pady=5)
        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)
        self._manual_tab(notebook)
        self._experiment_tab(notebook)
        self._analysis_tab(notebook)
        self.plots = RollingPlots(right, self.controller)
        self.plots.pack(fill=tk.BOTH, expand=True)

    def _telemetry_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Live actuator telemetry", padding=6)
        rows = [
            ("Position", "position"), ("Requested position", "requested_position"),
            ("Transmitted position", "desired_position"),
            ("Velocity", "velocity"), ("Desired velocity", "desired_velocity"),
            ("Position error", "position_error"), ("Velocity error", "velocity_error"),
            ("Commanded torque", "tau_commanded"), ("Measured torque", "tau_measured"),
            ("Estimated torque", "tau_estimated"), ("Current", "current"),
            ("Bus voltage", "voltage"), ("Temperature", "temperature"),
            ("Kp / Kd", "kp"), ("Control / GUI rate", "control_hz"),
            ("Experiment elapsed", "elapsed"), ("Feedback age", "feedback_age"),
            ("Timing", "timing"),
        ]
        for index, (label, key) in enumerate(rows):
            ttk.Label(frame, text=label + ":").grid(row=index, column=0, sticky="w", padx=3)
            ttk.Label(frame, textvariable=self.telemetry[key]).grid(row=index, column=1, sticky="w", padx=5)
        return frame

    def _manual_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="Manual / Hold")
        ttk.Label(frame, text=f"Kp [{self.config.control.kp_min}, {self.config.control.kp_max}]").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.kp_entry, width=10).grid(row=0, column=1)
        ttk.Button(frame, text="Update Kp", command=lambda: self._update_gains("kp")).grid(row=0, column=2, padx=3)
        ttk.Label(frame, text=f"Kd [{self.config.control.kd_min}, {self.config.control.kd_max}]").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.kd_entry, width=10).grid(row=1, column=1)
        ttk.Button(frame, text="Update Kd", command=lambda: self._update_gains("kd")).grid(row=1, column=2, padx=3)
        ttk.Button(frame, text="Update Both", command=lambda: self._update_gains("both")).grid(row=2, column=2, padx=3, pady=4)
        ttk.Label(frame, text="Target [rad]").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.target_entry, width=10).grid(row=3, column=1)
        ttk.Button(frame, text="Set Target", command=self._set_target).grid(row=3, column=2, padx=3)
        ttk.Label(frame, text="Speed [rad/s]").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.speed_entry, width=10).grid(row=4, column=1)
        ttk.Button(frame, text="Set Speed", command=self._set_speed).grid(row=4, column=2, padx=3)
        ttk.Button(frame, text="−", command=lambda: self._nudge(-1)).grid(row=5, column=0, pady=5)
        ttk.Button(frame, text="HOLD", command=self._hold).grid(row=5, column=1, pady=5)
        ttk.Button(frame, text="+", command=lambda: self._nudge(1)).grid(row=5, column=2, pady=5)
        ttk.Label(frame, text=f"Arrow step: {self.config.control.manual_step_rad:.6f} rad ({math.degrees(self.config.control.manual_step_rad):.2f} deg)").grid(row=6, column=0, columnspan=3, sticky="w")
        ttk.Button(frame, text="START MANUAL LOG", command=lambda: self._guard(self.controller.start_manual_logging)).grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(frame, text="STOP LOG", command=self.controller.stop_logging).grid(row=7, column=2, sticky="ew", pady=4)

    def _field(self, parent, row, label, default):
        variable = tk.StringVar(value=str(default))
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="w")
        return variable

    def _experiment_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="Experiments")
        self.step_initial = self._field(frame, 0, "Step initial [rad]", 0.0)
        self.step_amplitude = self._field(frame, 1, "Step amplitude [rad]", 0.10)
        self.step_pre = self._field(frame, 2, "Pre-hold [s]", 2.0)
        self.step_duration = self._field(frame, 3, "Post-step [s]", 5.0)
        ttk.Button(frame, text="RUN STEP", command=self._run_step).grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Separator(frame).grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)
        self.chirp_center = self._field(frame, 6, "Chirp center [rad]", 0.0)
        self.chirp_amplitude = self._field(frame, 7, "Amplitude [rad]", 0.05)
        self.chirp_start = self._field(frame, 8, "Start frequency [Hz]", 0.2)
        self.chirp_end = self._field(frame, 9, "End frequency [Hz]", 5.0)
        self.chirp_duration = self._field(frame, 10, "Duration [s]", 15.0)
        self.chirp_kind = tk.StringVar(value="linear")
        ttk.Combobox(frame, textvariable=self.chirp_kind, values=("linear", "logarithmic"), state="readonly", width=12).grid(row=11, column=1)
        ttk.Label(frame, text="Chirp type").grid(row=11, column=0, sticky="w")
        ttk.Button(frame, text="RUN CHIRP", command=self._run_chirp).grid(row=12, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Separator(frame).grid(row=13, column=0, columnspan=2, sticky="ew", pady=6)
        self.decay_duration = self._field(frame, 14, "Free-decay duration [s]", 10.0)
        ttk.Button(frame, text="RECORD DISABLED FREE DECAY", command=self._run_decay).grid(row=15, column=0, columnspan=2, sticky="ew", pady=5)

    def _analysis_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="Results / Identification")
        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Analyze CSV", command=self._select_analysis).pack(side=tk.LEFT, padx=3)
        self.recommend_kp = tk.StringVar(value="80")
        self.recommend_zeta = tk.StringVar(value="0.7")
        ttk.Entry(controls, textvariable=self.recommend_kp, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="Kp").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.recommend_zeta, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="desired zeta").pack(side=tk.LEFT)
        sweep = ttk.LabelFrame(frame, text="Safe gain sweep", padding=4)
        sweep.pack(fill=tk.X, pady=5)
        self.sweep_kp = self._field(sweep, 0, "Kp values (comma-separated)", "20,40,60,80")
        self.sweep_kd = self._field(sweep, 1, "Kd values (comma-separated)", "1,2,3,4")
        self.sweep_step = self._field(sweep, 2, "Small step [rad]", 0.05)
        ttk.Button(sweep, text="START SAFE SWEEP", command=self._start_sweep).grid(row=3, column=0, sticky="ew", pady=3)
        ttk.Button(sweep, text="ABORT SWEEP", command=self.sweep.abort).grid(row=3, column=1, sticky="ew", pady=3)
        self.result_text = tk.Text(frame, height=23, width=58, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=6)

    def _guard(self, callback):
        try:
            callback()
        except Exception as exc:
            messagebox.showerror("RS04 bench", str(exc))
            self.status.set(str(exc))

    def _connect(self): self._guard(lambda: (self.controller.connect(), self.status.set("Connected; motor remains disabled.")))
    def _enable(self): self._guard(lambda: (self.controller.enable(), self.status.set("Motor enabled at measured current position.")))
    def _disable(self): self._guard(lambda: (self.controller.disable(), self.status.set("Motor disabled.")))
    def _estop(self): self.controller.emergency_stop(); self.status.set("EMERGENCY STOP")

    def _update_gains(self, which):
        def update():
            kp = float(self.kp_entry.get()) if which in {"kp", "both"} else None
            kd = float(self.kd_entry.get()) if which in {"kd", "both"} else None
            self.controller.update_gains(kp, kd)
        self._guard(update)

    def _apply_manual_target(self, target):
        self.controller.update_manual_speed(float(self.speed_entry.get()))
        self.controller.set_manual_target(float(target))
        self.target_entry.set(f"{float(target):.6f}")

    def _set_target(self): self._guard(lambda: self._apply_manual_target(float(self.target_entry.get())))
    def _set_speed(self): self._guard(lambda: self.controller.update_manual_speed(float(self.speed_entry.get())))
    def _dial_target(self, target): self._guard(lambda: self._apply_manual_target(target))
    def _nudge(self, direction):
        self._guard(lambda: (
            self.controller.update_manual_speed(float(self.speed_entry.get())),
            self.controller.nudge(direction),
        ))
    def _hold(self):
        def hold():
            target = self.controller.hold_current_position()
            self.target_entry.set(f"{target:.6f}")
        self._guard(hold)

    def _spec_gains(self): return float(self.kp_entry.get()), float(self.kd_entry.get())
    def _run_step(self):
        def run():
            kp, kd = self._spec_gains()
            self.controller.start_experiment(ExperimentSpec("step", {
                "initial_position": float(self.step_initial.get()), "step_amplitude": float(self.step_amplitude.get()),
                "pre_hold_s": float(self.step_pre.get()), "post_duration_s": float(self.step_duration.get()),
            }, kp, kd))
        self._guard(run)

    def _run_chirp(self):
        def run():
            kp, kd = self._spec_gains()
            self.controller.start_experiment(ExperimentSpec("chirp", {
                "center_position": float(self.chirp_center.get()), "amplitude": float(self.chirp_amplitude.get()),
                "f_start_hz": float(self.chirp_start.get()), "f_end_hz": float(self.chirp_end.get()),
                "duration_s": float(self.chirp_duration.get()), "kind": self.chirp_kind.get(),
            }, kp, kd))
        self._guard(run)

    def _run_decay(self):
        if not messagebox.askyesno("Free decay", "This explicitly disables MIT impedance control. Confirm the pendulum is mechanically safe and the travel is clear."):
            return
        self._guard(lambda: self.controller.start_experiment(ExperimentSpec(
            "free_decay", {"duration_s": float(self.decay_duration.get()), "disabled": True,
                           "hold_position": self.controller.snapshot().state.position if self.controller.snapshot().state else 0.0}, 0.0, 0.0,
            notes="Operator explicitly confirmed disabled free-decay procedure",
        )))

    def _select_analysis(self):
        path = filedialog.askopenfilename(filetypes=(("RS04 CSV", "*.csv"), ("All files", "*")))
        if path: self._analyze_async(path)

    def _analyze_async(self, path):
        def work():
            output = []
            identification_signals = None
            try:
                try:
                    output.append(("STEP RESPONSE", analyze_step_response(path, self.config.analysis.settling_band_fraction)))
                except Exception as exc:
                    output.append(("STEP RESPONSE", {"note": str(exc)}))
                try:
                    plant = identify_plant(
                        path, vars(self.config.pendulum), self.config.analysis.filter_window,
                        self.config.analysis.filter_polynomial_order, self.config.analysis.velocity_deadband_rad_s,
                        self.config.analysis.minimum_identification_samples,
                        self.config.analysis.minimum_excitation_velocity_rad_s,
                    )
                    plant_display = {key: value for key, value in plant.items() if key != "signals"}
                    identification_signals = plant.get("signals")
                    output.append(("PLANT IDENTIFICATION", plant_display))
                    if plant["valid"]:
                        output.append(("KD STARTING ESTIMATE", recommend_kd(
                            plant["estimated_inertia_kg_m2"], plant["estimated_viscous_damping_nm_s_rad"],
                            float(self.recommend_kp.get()), float(self.recommend_zeta.get()),
                        )))
                except Exception as exc:
                    output.append(("PLANT IDENTIFICATION", {"note": str(exc)}))
            except Exception as exc:
                output = [("ANALYSIS ERROR", {"error": str(exc)})]
            self.root.after(0, lambda: (
                self._show_results(path, output),
                self._show_identification_signals(identification_signals)
                if identification_signals is not None else None,
            ))
        threading.Thread(target=work, name="rs04-offline-analysis", daemon=True).start()

    def _show_results(self, path, sections):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"TEST RESULT\n{'=' * 48}\n{path}\n\n")
        for title, values in sections:
            self.result_text.insert(tk.END, title + "\n" + "-" * 48 + "\n")
            self.result_text.insert(tk.END, json.dumps(values, indent=2, default=str) + "\n\n")

    def _show_identification_signals(self, signals):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            return
        window = tk.Toplevel(self.root)
        window.title("Identification preprocessing: raw and filtered signals")
        figure = Figure(figsize=(9, 7), dpi=90, tight_layout=True)
        axes = figure.subplots(4, 1, sharex=True)
        t = signals["time_s"]
        axes[0].plot(t, signals["position_raw_rad"], color="#94a3b8", label="raw/interpolated", linewidth=0.8)
        axes[0].plot(t, signals["position_filtered_rad"], color="#06b6d4", label="local polynomial", linewidth=1.2)
        axes[0].set_ylabel("q [rad]")
        axes[0].legend()
        axes[1].plot(t, signals["velocity_filtered_rad_s"], color="#22c55e")
        axes[1].set_ylabel("qd [rad/s]")
        axes[2].plot(t, signals["acceleration_filtered_rad_s2"], color="#f59e0b")
        axes[2].set_ylabel("qdd [rad/s2]")
        axes[3].plot(t, signals["torque_nm"], label="selected torque", color="#ef4444")
        axes[3].plot(t, signals["gravity_torque_nm"], label="gravity model", color="#8b5cf6")
        axes[3].set_ylabel("tau [Nm]")
        axes[3].set_xlabel("time [s]")
        axes[3].legend()
        for axis in axes:
            axis.grid(True, alpha=0.25)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def _sweep_result(self, result):
        self.root.after(0, lambda: self.status.set(f"Sweep Kp={result.kp:g} Kd={result.kd:g}: cost={result.cost:.4f}"))

    def _start_sweep(self):
        def run():
            kp = [float(value.strip()) for value in self.sweep_kp.get().split(",") if value.strip()]
            kd = [float(value.strip()) for value in self.sweep_kd.get().split(",") if value.strip()]
            if not kp or not kd:
                raise ValueError("Kp and Kd sweep lists cannot be empty")
            snapshot = self.controller.snapshot()
            initial = snapshot.state.position if snapshot.state else 0.0
            self.sweep.start(kp, kd, initial, float(self.sweep_step.get()))
            self._sweep_was_running = True
        self._guard(run)

    @staticmethod
    def _fmt(value, unit, digits=4):
        return "unavailable" if value is None or not math.isfinite(float(value)) else f"{float(value):+.{digits}f} {unit}"

    def _tick(self):
        now = time.perf_counter()
        dt = now - self._last_gui
        self._last_gui = now
        if dt > 0: self._gui_dts.append(dt)
        self._gui_dts = self._gui_dts[-200:]
        gui_hz = 0.0 if not self._gui_dts else 1.0 / (sum(self._gui_dts) / len(self._gui_dts))
        snapshot = self.controller.snapshot()
        state, command = snapshot.state, snapshot.command
        actual = 0.0 if state is None else state.position
        self.gauge.update_angles(actual, snapshot.requested_position_rad)
        self.telemetry["position"].set(self._fmt(None if state is None else state.position, "rad") + ("" if state is None else f" / {math.degrees(state.position):+.2f} deg"))
        self.telemetry["requested_position"].set(self._fmt(snapshot.requested_position_rad, "rad") + f" / {math.degrees(snapshot.requested_position_rad):+.2f} deg")
        self.telemetry["desired_position"].set(self._fmt(command.q_des, "rad") + f" / {math.degrees(command.q_des):+.2f} deg")
        self.telemetry["velocity"].set(self._fmt(None if state is None else state.velocity, "rad/s"))
        self.telemetry["desired_velocity"].set(self._fmt(command.qd_des, "rad/s"))
        self.telemetry["position_error"].set(self._fmt(None if state is None else command.q_des-state.position, "rad"))
        self.telemetry["velocity_error"].set(self._fmt(None if state is None else command.qd_des-state.velocity, "rad/s"))
        self.telemetry["tau_commanded"].set(self._fmt(snapshot.torque_commanded_nm, "Nm", 2))
        self.telemetry["tau_measured"].set(self._fmt(None if state is None else state.torque_measured, "Nm", 2))
        self.telemetry["tau_estimated"].set(self._fmt(snapshot.torque_estimated_nm, "Nm", 2))
        self.telemetry["current"].set(self._fmt(None if state is None else state.current, "A", 2))
        self.telemetry["voltage"].set(self._fmt(None if state is None else state.voltage, "V", 2))
        self.telemetry["temperature"].set(self._fmt(None if state is None else state.temperature, "C", 1))
        self.telemetry["kp"].set(f"{command.kp:.2f} / {command.kd:.2f}")
        self.telemetry["control_hz"].set(f"{snapshot.timing.instantaneous_hz:.1f} / {gui_hz:.1f} Hz")
        self.telemetry["elapsed"].set(f"{snapshot.experiment_time:.3f} s ({snapshot.experiment_mode})")
        self.telemetry["feedback_age"].set("unavailable" if not math.isfinite(snapshot.feedback_age_s) else f"{1000*snapshot.feedback_age_s:.2f} ms")
        self.telemetry["timing"].set(
            f"avg {snapshot.timing.average_hz:.2f}, min/max "
            f"{snapshot.timing.minimum_hz:.1f}/{snapshot.timing.maximum_hz:.1f} Hz, "
            f"mean dt {1000*snapshot.timing.mean_dt_s:.3f} ms, "
            f"jitter {1000*snapshot.timing.std_dt_s:.3f} ms, "
            f"late/missed {snapshot.timing.late_cycles}/{snapshot.timing.missed_cycles}"
        )
        if snapshot.safety_event:
            self.safety.set("STOPPED: " + snapshot.safety_event)
        else:
            self.safety.set("ENABLED" if snapshot.enabled else "SAFE / DISABLED")
        if snapshot.last_csv_path and snapshot.last_csv_path != self._last_analyzed_path:
            self._last_analyzed_path = snapshot.last_csv_path
            if "step" in Path(snapshot.last_csv_path).name:
                self._analyze_async(snapshot.last_csv_path)
        if self._sweep_was_running and not self.sweep.running:
            self._sweep_was_running = False
            candidates = [
                {"kp": item.kp, "kd": item.kd, "cost": item.cost, "safe": item.safe,
                 "note": item.note, "csv": item.csv_path}
                for item in self.sweep.best_candidates()
            ]
            self._show_results("gain sweep", [
                ("BEST SAFE CANDIDATES", {"candidates": candidates, "sweep_error": self.sweep.error,
                                          "warning": "Review traces and safety margins; lowest cost is not uniquely perfect."})
            ])
        self.plots.update_plot(now)
        self.root.after(max(10, int(1000.0 / self.config.control.gui_frequency_hz)), self._tick)

    def _close(self):
        if messagebox.askokcancel("Quit", "Disable the actuator and close the bench application?"):
            self.controller.shutdown()
            self.root.destroy()
