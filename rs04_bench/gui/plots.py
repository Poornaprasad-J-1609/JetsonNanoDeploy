from __future__ import annotations

import math
import tkinter as tk


class RollingPlots(tk.Frame):
    def __init__(self, parent, controller, window_s=10.0, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        self.window_s = float(window_s)
        self.available = False
        self._last_draw = 0.0
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            tk.Label(self, text="Live plots require matplotlib (pip install matplotlib).", fg="#f59e0b").pack(padx=20, pady=20)
            return
        self.figure = Figure(figsize=(8.5, 7.0), dpi=90, tight_layout=True)
        self.axes = self.figure.subplots(5, 1, sharex=True)
        labels = ["Position [rad]", "Velocity [rad/s]", "Torque [Nm]", "Position error [rad]", "Control rate [Hz]"]
        self.lines = []
        series = [("q_des", "q"), ("qd_des", "qd"), ("tau_cmd", "tau_meas", "tau_est"), ("error",), ("hz",)]
        colors = ("#f59e0b", "#06b6d4", "#ef4444")
        for axis, label, names in zip(self.axes, labels, series):
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
            axis_lines = []
            for index, name in enumerate(names):
                line, = axis.plot([], [], label=name, color=colors[index % len(colors)], linewidth=1.2)
                axis_lines.append(line)
            axis.legend(loc="upper right", fontsize=7, ncol=len(names))
            self.lines.append(axis_lines)
        self.axes[-1].set_xlabel("Recent time [s]")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.available = True

    def update_plot(self, now):
        if not self.available or now - self._last_draw < 0.10:
            return
        self._last_draw = now
        history = self.controller.history()
        if len(history) < 2:
            return
        cutoff = history[-1].timestamp_monotonic - self.window_s
        history = [sample for sample in history if sample.timestamp_monotonic >= cutoff]
        t0 = history[0].timestamp_monotonic
        t = [sample.timestamp_monotonic - t0 for sample in history]
        def state_value(sample, name):
            return math.nan if sample.state is None else getattr(sample.state, name)
        groups = [
            ([s.command.q_des for s in history], [state_value(s, "position") for s in history]),
            ([s.command.qd_des for s in history], [state_value(s, "velocity") for s in history]),
            ([s.torque_commanded_nm for s in history], [state_value(s, "torque_measured") for s in history], [s.torque_estimated_nm or math.nan for s in history]),
            ([math.nan if s.state is None else s.command.q_des - s.state.position for s in history],),
            ([s.timing.instantaneous_hz for s in history],),
        ]
        for axis, lines, values in zip(self.axes, self.lines, groups):
            for line, y in zip(lines, values):
                line.set_data(t, y)
            axis.relim()
            axis.autoscale_view()
        self.canvas.draw_idle()
