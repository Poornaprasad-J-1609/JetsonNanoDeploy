from __future__ import annotations

import math
import tkinter as tk


class AngleGauge(tk.Canvas):
    def __init__(self, parent, size=330, limit_rad=math.pi, command=None, **kwargs):
        super().__init__(
            parent, width=size, height=size, bg="#111820",
            highlightthickness=0, cursor="hand2", **kwargs,
        )
        self.size = size
        self.center = size / 2
        self.radius = size * 0.39
        self.limit_rad = float(limit_rad)
        self._actual = 0.0
        self._desired = 0.0
        self._command = command
        self.bind("<Button-1>", self._select_target)
        self.bind("<B1-Motion>", self._select_target)
        self._draw()

    def _select_target(self, event):
        dx = float(event.x) - self.center
        dy = float(event.y) - self.center
        if math.hypot(dx, dy) < self.radius * 0.20:
            return
        angle = math.atan2(dy, dx) + math.pi / 2.0
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
        angle = min(max(angle, -self.limit_rad), self.limit_rad)
        self._desired = angle
        self._draw()
        if self._command is not None:
            self._command(angle)

    def _point(self, angle, radius=None):
        radius = self.radius if radius is None else radius
        display = -math.pi / 2 + float(angle)
        return self.center + radius * math.cos(display), self.center + radius * math.sin(display)

    def _draw(self):
        self.delete("all")
        c, r = self.center, self.radius
        self.create_oval(c-r, c-r, c+r, c+r, outline="#64748b", width=3)
        for index in range(25):
            angle = -self.limit_rad + 2 * self.limit_rad * index / 24
            outer = self._point(angle)
            inner = self._point(angle, r - (14 if index % 3 == 0 else 8))
            self.create_line(*inner, *outer, fill="#64748b", width=2 if index % 3 == 0 else 1)
        desired = self._point(self._desired, r + 2)
        self.create_line(c, c, *desired, fill="#f59e0b", width=4, dash=(5, 3), arrow=tk.LAST)
        actual = self._point(self._actual, r - 12)
        self.create_line(c, c, *actual, fill="#22d3ee", width=7, arrow=tk.LAST)
        self.create_oval(c-9, c-9, c+9, c+9, fill="#e2e8f0", outline="")
        self.create_text(c, self.size-43, text=f"actual {self._actual:+.4f} rad  ({math.degrees(self._actual):+.2f} deg)", fill="#22d3ee", font=("TkDefaultFont", 11, "bold"))
        self.create_text(c, self.size-21, text=f"target {self._desired:+.4f} rad  ({math.degrees(self._desired):+.2f} deg)", fill="#f59e0b", font=("TkDefaultFont", 10))

    def update_angles(self, actual, desired):
        self._actual = float(actual or 0.0)
        self._desired = float(desired or 0.0)
        self._draw()
