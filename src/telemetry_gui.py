#!/usr/bin/env python3
"""
Real-time telemetry GUI for GRALLATOR robot controller.

Run automatically via:  python3 main_controller.py --gui ...
Or standalone (connect to a running controller):
    python3 telemetry_gui.py --port 57543
"""
import argparse
import json
import math
import socket
import threading
import time
import tkinter as tk
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

TELEMETRY_PORT = 57543
STALE_TIMEOUT  = 2.0   # seconds before showing "STALE"
ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = ROOT / "assets" / "xera_robotics_india_logo.jpeg"

# ── colour palette ────────────────────────────────────────────────────────────
BG       = "#05070d"
PANEL    = "#101722"
PANEL_2  = "#131e2b"
HEADER   = "#1a2433"
CARD     = "#0b1220"
FG       = "#eef6ff"
FG_DIM   = "#8fa3b8"
GREEN    = "#42e087"
YELLOW   = "#ffd166"
RED      = "#ff5c70"
BLUE     = "#4ea1ff"
CYAN     = "#4de3ff"
ORANGE   = "#ff9f45"
BORDER   = "#27384e"
WHITE    = "#ffffff"

FONT_TITLE  = ("DejaVu Sans", 16, "bold")
FONT_HEADER = ("DejaVu Sans", 10, "bold")
FONT_BODY   = ("DejaVu Sans Mono", 10)
FONT_BIG    = ("DejaVu Sans", 20, "bold")
FONT_SMALL  = ("DejaVu Sans",  9)


def _c(val, lo_warn, lo_err, hi_warn, hi_err):
    """Return a colour string based on value thresholds."""
    if val is None:
        return FG_DIM
    if val <= lo_err or val >= hi_err:
        return RED
    if val <= lo_warn or val >= hi_warn:
        return YELLOW
    return GREEN


class Bar(tk.Canvas):
    """Horizontal progress bar with value label."""
    def __init__(self, parent, width=200, height=16,
                 lo=-1.0, hi=1.0, fg=BLUE, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=PANEL_2, highlightthickness=0, **kw)
        self._lo = lo
        self._hi = hi
        self._fg = fg
        self._bar_width  = width
        self._bar_height = height
        self._track = self.create_rectangle(0, 0, width, height,
                                            fill=CARD, outline=BORDER)
        self._bar   = self.create_rectangle(0, 0, 0, height,
                                            fill=fg, outline="")
        self._zero  = self.create_line(self._zero_x(), 0,
                                       self._zero_x(), height,
                                       fill=FG_DIM, width=1)

    def _zero_x(self):
        span = self._hi - self._lo
        if span == 0:
            return self._bar_width // 2
        return int((0 - self._lo) / span * self._bar_width)

    def set(self, value, color=None):
        if value is None:
            self.coords(self._bar, 0, 0, 0, self._bar_height)
            return
        ratio = (float(value) - self._lo) / (self._hi - self._lo)
        ratio = max(0.0, min(1.0, ratio))
        zx = self._zero_x()
        px = int(ratio * self._bar_width)
        if px >= zx:
            self.coords(self._bar, zx, 1, px, self._bar_height - 1)
        else:
            self.coords(self._bar, px, 1, zx, self._bar_height - 1)
        if color:
            self.itemconfig(self._bar, fill=color)


class JointRow:
    """One row in the joint table."""
    def __init__(self, parent, row, joint_name):
        bg = PANEL if row % 2 == 0 else PANEL_2
        self._bg = bg

        def lbl(col, text, fg=FG, anchor="e", width=None):
            kw = dict(bg=bg, fg=fg, font=FONT_BODY, anchor=anchor,
                      padx=4, pady=2)
            if width:
                kw["width"] = width
            w = tk.Label(parent, text=text, **kw)
            w.grid(row=row, column=col, sticky="nsew")
            return w

        self._name  = lbl(0, joint_name, fg=CYAN, anchor="w", width=18)
        self._bus   = lbl(1, "---", width=5)
        self._id    = lbl(2, "---", width=6)
        self._qdes  = lbl(3, "---", width=8)
        self._qfb   = lbl(4, "---", width=8)
        self._qerr  = lbl(5, "---", width=8)
        self._vel   = lbl(6, "---", width=8)
        self._tcmd  = lbl(7, "---", width=8)
        self._tfb   = lbl(8, "---", width=8)
        self._temp  = lbl(9, "---", width=7)
        self._fault = lbl(10, "---", width=9)

    def set_name(self, joint_name):
        self._name.config(text=joint_name)

    def _lbl_fmt(self, widget, text, color=FG):
        widget.config(text=text, fg=color)

    def update(self, jd):
        self.set_name(jd.get("n", self._name.cget("text")))
        self._lbl_fmt(self._bus, str(jd.get("bus", "---")), FG_DIM)
        self._lbl_fmt(self._id, f"0x{jd['id']:02X}", FG_DIM)

        qd  = jd.get("qd")
        qf  = jd.get("qf")
        vf  = jd.get("vf")
        tc  = jd.get("tc")
        tf  = jd.get("tf")
        temp= jd.get("temp")
        fault= int(jd.get("fault", 0))
        mode = int(jd.get("mode", 0))

        def fmt(v, dec=3):
            return f"{v:+.{dec}f}" if v is not None else "  ---"

        self._lbl_fmt(self._qdes, fmt(qd))
        self._lbl_fmt(self._qfb,  fmt(qf))

        if qd is not None and qf is not None:
            err = qd - qf
            ecol = GREEN if abs(err) < 0.05 else (YELLOW if abs(err) < 0.15 else RED)
            self._lbl_fmt(self._qerr, fmt(err), ecol)
        else:
            self._lbl_fmt(self._qerr, "  ---", FG_DIM)

        vcol = _c(vf, -25, -29, 25, 29) if vf is not None else FG_DIM
        self._lbl_fmt(self._vel, fmt(vf), vcol)

        self._lbl_fmt(self._tcmd, fmt(tc, 2))

        tfcol = FG
        if tf is not None:
            tfcol = GREEN if abs(tf) < 6 else (YELLOW if abs(tf) < 10 else RED)
        self._lbl_fmt(self._tfb, fmt(tf, 2), tfcol)

        if temp is not None:
            tcol = GREEN if temp < 55 else (YELLOW if temp < 70 else RED)
            self._lbl_fmt(self._temp, f"{temp:5.1f}°", tcol)
        else:
            self._lbl_fmt(self._temp, "  ---", FG_DIM)

        if fault != 0:
            self._lbl_fmt(self._fault, f"0x{fault:02X} ERR", RED)
        elif mode == 0:
            self._lbl_fmt(self._fault, "STOPPED", YELLOW)
        elif mode == 2:
            self._lbl_fmt(self._fault, "MIT OK", GREEN)
        else:
            self._lbl_fmt(self._fault, f"m={mode}", FG_DIM)


class VecDisplay(tk.Frame):
    """Three-value vector display with optional bars."""
    def __init__(self, parent, label, lo=-2.0, hi=2.0, unit="", **kw):
        super().__init__(parent, bg=PANEL, **kw)
        tk.Label(self, text=label, bg=PANEL, fg=FG_DIM,
                 font=FONT_SMALL).grid(row=0, column=0, columnspan=3, sticky="w")
        self._bars  = []
        self._vals  = []
        axes = ["X", "Y", "Z"]
        for col, ax in enumerate(axes):
            tk.Label(self, text=ax, bg=PANEL, fg=FG_DIM,
                     font=FONT_SMALL).grid(row=1, column=col*2, sticky="e")
            v = tk.Label(self, text=" 0.000", bg=PANEL, fg=CYAN,
                         font=FONT_BODY, width=7, anchor="e")
            v.grid(row=1, column=col*2+1, sticky="w")
            self._vals.append(v)

    def set(self, vec):
        if vec is None:
            for v in self._vals:
                v.config(text="  ---", fg=FG_DIM)
            return
        for i, val in enumerate(vec[:3]):
            self._vals[i].config(text=f"{val:+.3f}", fg=CYAN)


class AxisViewer(tk.Canvas):
    """Compact absolute Xsens axis viewer: IMU axes in world frame + gravity."""

    def __init__(self, parent, width=380, height=215, **kw):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            **kw,
        )
        self._width = width
        self._height = height
        self._scale = min(width, height) * 0.34
        self._center = (width * 0.50, height * 0.58)
        self._draw_static()

    def _project(self, vec):
        x, y, z = [float(v) for v in vec[:3]]
        cx, cy = self._center
        sx = cx + self._scale * (x - 0.52 * y)
        sy = cy - self._scale * (z - 0.30 * y)
        return sx, sy

    def _draw_arrow(self, tag, vec, color, width=4, dash=None):
        x0, y0 = self._center
        x1, y1 = self._project(vec)
        self.create_line(
            x0,
            y0,
            x1,
            y1,
            fill=color,
            width=width,
            arrow=tk.LAST,
            arrowshape=(10, 12, 5),
            dash=dash,
            tags=tag,
        )

    def _draw_static(self):
        self.delete("all")
        self.create_text(
            10,
            8,
            anchor="nw",
            text="XSENS ABSOLUTE AXES",
            fill=FG_DIM,
            font=FONT_HEADER,
        )
        self._draw_arrow("world", [1.0, 0.0, 0.0], "#a84242", width=1, dash=(3, 3))
        self._draw_arrow("world", [0.0, 1.0, 0.0], "#3e8f53", width=1, dash=(3, 3))
        self._draw_arrow("world", [0.0, 0.0, 1.0], "#466eaa", width=1, dash=(3, 3))
        self.create_text(self._width - 12, 16, anchor="ne", text="world", fill=FG_DIM, font=FONT_SMALL)
        self._text_id = self.create_text(
            10,
            self._height - 64,
            anchor="nw",
            text="waiting for IMU quaternion",
            fill=FG_DIM,
            font=FONT_SMALL,
        )

    def set(self, imu_view):
        self.delete("imu_axis")
        if not imu_view:
            self.itemconfig(self._text_id, text="waiting for IMU quaternion", fill=FG_DIM)
            return

        axes = imu_view.get("axes_world") or []
        gravity = imu_view.get("projected_gravity") or [0.0, 0.0, -1.0]
        quat = imu_view.get("quat_wxyz") or []
        rpy = imu_view.get("rpy_abs_deg") or []
        det_r = imu_view.get("det_r")
        cross_err = imu_view.get("cross_err")

        if len(axes) >= 3:
            self._draw_arrow("imu_axis", axes[0], "#ff4d5e", width=5)
            self._draw_arrow("imu_axis", axes[1], "#3ee86b", width=5)
            self._draw_arrow("imu_axis", axes[2], "#4ea1ff", width=5)
        self._draw_arrow("imu_axis", gravity, "#f6f8ff", width=3)

        lines = []
        if len(quat) >= 4:
            lines.append(
                "q_abs=[%+.3f,%+.3f,%+.3f,%+.3f]"
                % (quat[0], quat[1], quat[2], quat[3])
            )
        if len(rpy) >= 3:
            lines.append("rpy_abs_deg=[%+.1f,%+.1f,%+.1f]" % (rpy[0], rpy[1], rpy[2]))
        if len(gravity) >= 3:
            lines.append(
                "projected_g=[%+.3f,%+.3f,%+.3f]"
                % (gravity[0], gravity[1], gravity[2])
            )
        if det_r is not None and cross_err is not None:
            lines.append("det_R=%+.3f  cross_err=%.1e" % (float(det_r), float(cross_err)))

        self.itemconfig(self._text_id, text="\n".join(lines), fill=CYAN)


class TelemetryGUI:
    def __init__(self, root, port):
        self.root  = root
        self.port  = port
        self._data = None
        self._lock = threading.Lock()
        self._last_recv = None
        self._rx_count  = 0
        self._hz_samples = []
        self._logo_image = None

        root.title("XERA Robotics India — GRALLATOR Telemetry")
        root.configure(bg=BG)
        root.minsize(1160, 760)

        self._build_ui()
        self._start_receiver()
        self._update()

    # ── UI construction ───────────────────────────────────────────────────────

    def _load_logo(self, path):
        if Image is None or ImageTk is None or not Path(path).exists():
            return None

        image = Image.open(path).convert("RGBA")
        image.thumbnail((112, 72), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _build_ui(self):
        root = self.root

        # ── top status bar ────────────────────────────────────────────────────
        top = tk.Frame(root, bg=BG, pady=8, padx=10)
        top.pack(fill="x", padx=0, pady=0)

        brand = tk.Frame(top, bg=BG)
        brand.pack(side="left", fill="y")

        logo_tile = tk.Frame(brand, bg=WHITE, highlightbackground=BORDER,
                             highlightthickness=1, padx=4, pady=4)
        logo_tile.pack(side="left")
        self._logo_image = self._load_logo(LOGO_PATH)
        if self._logo_image is not None:
            tk.Label(logo_tile, image=self._logo_image, bg=WHITE).pack()
        else:
            tk.Label(logo_tile, text="XERA", bg=WHITE, fg="#0361a8",
                     font=("DejaVu Sans", 18, "bold"), width=7, height=2).pack()

        title_block = tk.Frame(brand, bg=BG)
        title_block.pack(side="left", padx=12)
        self._lbl_title = tk.Label(title_block, text="GRALLATOR TELEMETRY",
                                   bg=BG, fg=FG, font=FONT_TITLE, anchor="w")
        self._lbl_title.pack(anchor="w")
        tk.Label(title_block, text="XERA ROBOTICS INDIA  |  LIVE CONTROL CONSOLE",
                 bg=BG, fg=FG_DIM, font=FONT_SMALL, anchor="w").pack(anchor="w", pady=(2, 0))

        status = tk.Frame(top, bg=BG)
        status.pack(side="right", fill="y")

        self._lbl_conn = tk.Label(status, text="○ WAITING",
                                  bg=CARD, fg=FG_DIM, font=FONT_HEADER,
                                  padx=10, pady=5)
        self._lbl_conn.pack(side="right", padx=(6, 0))

        self._lbl_hz = tk.Label(status, text="--- Hz",
                                bg=CARD, fg=FG_DIM, font=FONT_HEADER,
                                padx=10, pady=5)
        self._lbl_hz.pack(side="right", padx=(6, 0))

        self._lbl_safe = tk.Label(status, text="● SAFE",
                                  bg=CARD, fg=GREEN, font=FONT_HEADER,
                                  padx=10, pady=5)
        self._lbl_safe.pack(side="right", padx=(6, 0))

        self._lbl_imu  = tk.Label(status, text="IMU: ---",
                                  bg=CARD, fg=FG_DIM, font=FONT_HEADER,
                                  padx=10, pady=5)
        self._lbl_imu.pack(side="right", padx=(6, 0))

        self._lbl_mode = tk.Label(status, text="mode: ---",
                                  bg=CARD, fg=YELLOW, font=FONT_HEADER,
                                  padx=10, pady=5)
        self._lbl_mode.pack(side="right", padx=(6, 0))

        self._lbl_step = tk.Label(status, text="step: ------",
                                  bg=CARD, fg=FG, font=FONT_HEADER,
                                  padx=10, pady=5)
        self._lbl_step.pack(side="right", padx=(6, 0))

        # ── middle row: command + IMU ─────────────────────────────────────────
        mid = tk.Frame(root, bg=BG)
        mid.pack(fill="x", padx=10, pady=(2, 8))

        # joystick command
        cmd_f = tk.LabelFrame(mid, text=" JOYSTICK COMMAND ",
                               bg=PANEL, fg=CYAN, font=FONT_HEADER,
                               relief="flat", bd=1, padx=8, pady=8,
                               highlightbackground=BORDER, highlightthickness=1)
        cmd_f.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self._cmd_bars  = {}
        self._cmd_vals  = {}
        cfg = [("Vx  (fwd/back)", "vx", -1.8, 1.8, BLUE),
               ("Vy  (left/rgt)", "vy", -0.8, 0.8, CYAN),
               ("Yaw (turn)    ", "yaw",-0.7, 0.7, ORANGE)]
        for r, (label, key, lo, hi, col) in enumerate(cfg):
            tk.Label(cmd_f, text=label, bg=PANEL, fg=FG_DIM,
                     font=FONT_BODY, width=16, anchor="w"
                     ).grid(row=r, column=0, padx=6, pady=4)
            bar = Bar(cmd_f, width=270, height=20, lo=lo, hi=hi, fg=col)
            bar.grid(row=r, column=1, padx=6)
            val = tk.Label(cmd_f, text="+0.000", bg=PANEL, fg=col,
                           font=FONT_BODY, width=7)
            val.grid(row=r, column=2, padx=6)
            self._cmd_bars[key] = bar
            self._cmd_vals[key] = val

        # speed scale
        speed_f = tk.Frame(cmd_f, bg=PANEL)
        speed_f.grid(row=3, column=0, columnspan=3, pady=(8, 2), sticky="w")
        tk.Label(speed_f, text="Speed scale:", bg=PANEL, fg=FG_DIM,
                 font=FONT_BODY).pack(side="left", padx=4)
        self._lbl_speed = tk.Label(speed_f, text="0.50×",
                                   bg=PANEL, fg=GREEN, font=FONT_HEADER)
        self._lbl_speed.pack(side="left")
        tk.Label(speed_f, text="   Action max:", bg=PANEL, fg=FG_DIM,
                 font=FONT_BODY).pack(side="left", padx=(8,4))
        self._lbl_actmax = tk.Label(speed_f, text="---",
                                    bg=PANEL, fg=FG, font=FONT_HEADER)
        self._lbl_actmax.pack(side="left")

        # IMU vectors
        imu_f = tk.LabelFrame(mid, text=" IMU / STATE ESTIMATOR ",
                               bg=PANEL, fg=CYAN, font=FONT_HEADER,
                               relief="flat", bd=1, padx=8, pady=8,
                               highlightbackground=BORDER, highlightthickness=1)
        imu_f.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self._vec_linvel = VecDisplay(imu_f, "Base Linear Velocity (m/s)")
        self._vec_linvel.pack(fill="x", padx=6, pady=4)

        self._vec_angvel = VecDisplay(imu_f, "Base Angular Velocity (rad/s)",
                                       lo=-8, hi=8)
        self._vec_angvel.pack(fill="x", padx=6, pady=4)

        self._vec_grav = VecDisplay(imu_f, "Projected Gravity",
                                     lo=-1.1, hi=1.1)
        self._vec_grav.pack(fill="x", padx=6, pady=4)

        self._axis_viewer = AxisViewer(imu_f)
        self._axis_viewer.pack(fill="x", padx=6, pady=(6, 4))

        roll_f = tk.Frame(imu_f, bg=PANEL)
        roll_f.pack(fill="x", padx=6, pady=2)
        tk.Label(roll_f, text="Roll:", bg=PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left")
        self._lbl_roll = tk.Label(roll_f, text=" ---°", bg=PANEL, fg=CYAN,
                                  font=FONT_BODY)
        self._lbl_roll.pack(side="left", padx=4)
        tk.Label(roll_f, text="Pitch:", bg=PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(8,0))
        self._lbl_pitch = tk.Label(roll_f, text=" ---°", bg=PANEL, fg=CYAN,
                                   font=FONT_BODY)
        self._lbl_pitch.pack(side="left", padx=4)
        tk.Label(roll_f, text="Yaw:", bg=PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(8,0))
        self._lbl_yaw_abs = tk.Label(roll_f, text=" ---°", bg=PANEL, fg=CYAN,
                                     font=FONT_BODY)
        self._lbl_yaw_abs.pack(side="left", padx=4)

        # ── joint table ───────────────────────────────────────────────────────
        tbl_outer = tk.Frame(root, bg=BG)
        tbl_outer.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        tbl_f = tk.LabelFrame(tbl_outer, text=" MOTOR TELEMETRY ",
                               bg=PANEL, fg=CYAN, font=FONT_HEADER,
                               relief="flat", bd=1, padx=6, pady=6,
                               highlightbackground=BORDER, highlightthickness=1)
        tbl_f.pack(fill="both", expand=True)

        headers = [
            ("Joint Name",       18, "w"),
            ("Bus",              5,  "e"),
            ("ID",               6,  "e"),
            ("Q_des(rad)",       8,  "e"),
            ("Q_fb (rad)",       8,  "e"),
            ("Q_err (rad)",      8,  "e"),
            ("Vel(rad/s)",       8,  "e"),
            ("Tau_cmd(Nm)",      8,  "e"),
            ("Tau_fb (Nm)",      8,  "e"),
            ("Temp (°C)",        7,  "e"),
            ("Status",           9,  "e"),
        ]
        for col, (text, width, anchor) in enumerate(headers):
            tk.Label(tbl_f, text=text, bg=HEADER, fg=FG_DIM,
                     font=FONT_HEADER, width=width, anchor=anchor,
                     padx=4, pady=3, relief="flat"
                     ).grid(row=0, column=col, sticky="nsew")
            tbl_f.columnconfigure(col, weight=1)

        self._joint_rows = {}
        for r in range(1, 13):
            name = f"joint_{r:02d}"
            self._joint_rows[name] = JointRow(tbl_f, r, name)

        # ── bottom status bar ─────────────────────────────────────────────────
        bot = tk.Frame(root, bg=CARD, pady=5)
        bot.pack(fill="x")
        self._lbl_bot = tk.Label(bot, text="Waiting for controller...",
                                 bg=CARD, fg=FG_DIM, font=FONT_SMALL)
        self._lbl_bot.pack(side="left", padx=12)
        self._lbl_ts = tk.Label(bot, text="",
                                bg=CARD, fg=FG_DIM, font=FONT_SMALL)
        self._lbl_ts.pack(side="right", padx=12)

    # ── UDP receiver (background thread) ──────────────────────────────────────

    def _start_receiver(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self.port))
        self._sock.settimeout(0.5)

        t = threading.Thread(target=self._recv_loop, daemon=True)
        t.start()

    def _recv_loop(self):
        while True:
            try:
                raw, _ = self._sock.recvfrom(16384)
                data = json.loads(raw.decode("utf-8", errors="ignore"))
                now  = time.monotonic()
                with self._lock:
                    self._data      = data
                    self._last_recv = now
                    self._rx_count += 1
                    self._hz_samples.append(now)
                    cutoff = now - 2.0
                    self._hz_samples = [t for t in self._hz_samples if t >= cutoff]
            except socket.timeout:
                pass
            except Exception:
                pass

    # ── GUI refresh ───────────────────────────────────────────────────────────

    def _update(self):
        with self._lock:
            data      = self._data
            last_recv = self._last_recv
            hz_n      = len(self._hz_samples)
            hz_span   = (self._hz_samples[-1] - self._hz_samples[0]) if len(self._hz_samples) > 1 else None

        now = time.monotonic()

        if last_recv is None:
            age = None
        else:
            age = now - last_recv

        # connection status
        if age is None:
            conn_text  = "○ WAITING"
            conn_color = FG_DIM
        elif age > STALE_TIMEOUT:
            conn_text  = f"○ STALE {age:.1f}s"
            conn_color = RED
        else:
            conn_text  = "● LIVE"
            conn_color = GREEN

        self._lbl_conn.config(text=conn_text, fg=conn_color)

        hz_text = "--- Hz"
        if hz_span and hz_span > 0:
            hz = (hz_n - 1) / hz_span
            hz_text = f"{hz:.1f} Hz"
        self._lbl_hz.config(text=hz_text)

        if data is not None and (age is None or age < STALE_TIMEOUT):
            self._apply_data(data, age)

        self._lbl_ts.config(text=time.strftime("%H:%M:%S"))
        self.root.after(80, self._update)   # ~12 Hz refresh

    def _apply_data(self, d, age):
        step  = d.get("step", 0)
        mode  = str(d.get("mode", "---")).upper()
        speed = float(d.get("speed", 1.0))
        safe  = bool(d.get("safe", True))
        imu_s = str(d.get("imu", "---"))
        actmax= d.get("act_max", None)
        fault_reason = str(d.get("fault_reason", "") or "").strip()

        # ── status bar ────────────────────────────────────────────────────────
        self._lbl_step.config(text=f"step: {step:07d}")

        mode_col = {
            "POLICY": GREEN,
            "STAND":  BLUE,
            "SIT":    ORANGE,
            "HOLD":   YELLOW,
        }.get(mode, FG_DIM)
        self._lbl_mode.config(text=f"mode: {mode}", fg=mode_col)

        imu_col = GREEN if "live" in imu_s else (YELLOW if "fake" in imu_s else RED)
        self._lbl_imu.config(text=f"IMU: {imu_s}", fg=imu_col)

        if safe:
            self._lbl_safe.config(text="● SAFE", fg=GREEN)
        else:
            self._lbl_safe.config(text="● FAULT", fg=RED)

        # ── commands ──────────────────────────────────────────────────────────
        cmd = d.get("cmd", [0.0, 0.0, 0.0])
        for key, idx, lo, hi in [("vx", 0, -1.8, 1.8),
                                  ("vy", 1, -0.8, 0.8),
                                  ("yaw",2, -0.7, 0.7)]:
            val = float(cmd[idx]) if len(cmd) > idx else 0.0
            col = GREEN if abs(val) > 0.02 else FG_DIM
            self._cmd_bars[key].set(val, col)
            self._cmd_vals[key].config(text=f"{val:+.3f}", fg=col)

        self._lbl_speed.config(text=f"{speed:.2f}×")

        if actmax is not None:
            ac = GREEN if actmax < 0.8 else (YELLOW if actmax < 1.5 else RED)
            self._lbl_actmax.config(text=f"{actmax:.3f}", fg=ac)

        # ── IMU vectors ───────────────────────────────────────────────────────
        self._vec_linvel.set(d.get("base_vel"))
        self._vec_angvel.set(d.get("ang_vel"))
        grav = d.get("gravity")
        self._vec_grav.set(grav)

        imu_view = d.get("imu_view", {})
        self._axis_viewer.set(imu_view)
        rpy_abs = imu_view.get("rpy_abs_deg") if isinstance(imu_view, dict) else None

        if rpy_abs and len(rpy_abs) >= 3:
            roll = float(rpy_abs[0])
            pitch = float(rpy_abs[1])
            yaw_abs = float(rpy_abs[2])
            rcol = GREEN if abs(roll) < 10 else (YELLOW if abs(roll) < 25 else RED)
            pcol = GREEN if abs(pitch) < 10 else (YELLOW if abs(pitch) < 25 else RED)
            self._lbl_roll.config(text=f"{roll:+.1f}°", fg=rcol)
            self._lbl_pitch.config(text=f"{pitch:+.1f}°", fg=pcol)
            self._lbl_yaw_abs.config(text=f"{yaw_abs:+.1f}°", fg=CYAN)
        elif grav and len(grav) >= 3:
            gz = float(grav[2])
            gx = float(grav[0])
            gy = float(grav[1])
            down_z = max(1e-6, -gz)
            roll  = math.degrees(math.atan2(gy, down_z))
            pitch = math.degrees(math.atan2(-gx, down_z))
            rcol  = GREEN if abs(roll)  < 10 else (YELLOW if abs(roll)  < 25 else RED)
            pcol  = GREEN if abs(pitch) < 10 else (YELLOW if abs(pitch) < 25 else RED)
            self._lbl_roll.config( text=f"{roll:+.1f}°",  fg=rcol)
            self._lbl_pitch.config(text=f"{pitch:+.1f}°", fg=pcol)
            self._lbl_yaw_abs.config(text=" ---°", fg=FG_DIM)

        # ── joint table ───────────────────────────────────────────────────────
        joints = d.get("joints", [])
        active_names = set()

        for jd in joints:
            name = jd.get("n", "?")
            active_names.add(name)
            if name not in self._joint_rows:
                # first time we see this joint — remap the placeholder row
                # find an unused placeholder row
                for placeholder, row_obj in list(self._joint_rows.items()):
                    if placeholder.startswith("joint_") and placeholder not in active_names:
                        row_obj.set_name(name)
                        self._joint_rows[name] = row_obj
                        del self._joint_rows[placeholder]
                        break
            if name in self._joint_rows:
                self._joint_rows[name].update(jd)

        # ── bottom bar ────────────────────────────────────────────────────────
        n_fb  = sum(1 for jd in joints if jd.get("tf") is not None)
        n_hot = sum(1 for jd in joints if (jd.get("temp") or 0) > 55)
        n_flt = sum(1 for jd in joints if int(jd.get("fault", 0)) != 0)
        age_s = f"{age*1000:.0f} ms ago" if age is not None else "?"
        if fault_reason:
            if len(fault_reason) > 150:
                fault_reason = fault_reason[:147] + "..."
            self._lbl_bot.config(text=f"FAULT: {fault_reason}", fg=RED)
        else:
            self._lbl_bot.config(
                text=(f"joints={len(joints)}  feedback={n_fb}  "
                      f"hot={n_hot}  faults={n_flt}  "
                      f"last packet: {age_s}"),
                fg=(RED if n_flt > 0 else (YELLOW if n_hot > 0 else FG_DIM)),
            )


def main():
    parser = argparse.ArgumentParser(description="GRALLATOR Telemetry GUI")
    parser.add_argument("--port", type=int, default=TELEMETRY_PORT,
                        help="UDP port to receive telemetry on")
    args = parser.parse_args()

    root = tk.Tk()
    TelemetryGUI(root, args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
