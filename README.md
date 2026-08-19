# XERA Quadruped → robot_viewer live bridge

## Single-actuator RS04 bench

The reusable 200 Hz RS04 actuator characterization, system-identification, and
PD tuning application lives in [`rs04_bench/`](rs04_bench/README.md). Start in
motor-free mock mode:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python3 -m rs04_bench --mock
```

Read the bench README before connecting or enabling real hardware.

Mirror your real robot's commanded joint angles onto the URDF in
[fan-ziqi/robot_viewer](https://github.com/fan-ziqi/robot_viewer) in real time.

Tailored to **Poornaprasad-J-1609/JetsonNanoDeploy**. When `main_controller.py`
commands the RS04 actuators (walking, stand, sit, or a set-zero), the same
angles appear live on the URDF in your browser.

```
  Jetson Nano                                          Your PC (browser)
  ┌─────────────────────────────┐                      ┌────────────────────────┐
  │ main_controller.py          │  UDP 57543           │ robot_viewer           │
  │   TelemetrySender ──────────┼──▶ {joints: qd,...}  │  + xera_viewer_client  │
  │ xera_bridge.py (taps 57543) ─┼── ws://:8765 ───────▶│  setJointValue()       │
  └─────────────────────────────┘   {joint angles}     └────────────────────────┘
```

## Why this design
`robot_viewer` is a browser-only Three.js app with no network input of its own,
and your `TelemetrySender` already broadcasts every commanded joint angle (`qd`,
radians) over UDP to `127.0.0.1:57543` each control step. So the bridge just
taps that stream — **no changes to your control code** — and forwards it to the
viewer over WebSocket. Your telemetry is localhost-only, so the bridge runs on
the Jetson; the PC connects over the WebSocket port.

## Confirmed from your repo
- Joint names come from `grallator_isaac_lab.urdf` (NOT `grallator_final.urdf`):
  `FR/FL/BR/BL` × `hip/thigh/calf` `_joint` — matches `config/motor_ids.yaml`,
  `config/joint_map.yaml`, and all pose configs.
- Telemetry port `57543` (`TELEMETRY_PORT` in `main_controller.py`).
- Set-zero is a software re-zero (`set_software_zero_from_feedback`); after it,
  `qd` collapses toward 0 and the viewer follows automatically. No special case.

> Load **`grallator_isaac_lab.urdf`** in the viewer. `grallator_final.urdf` uses
> different names (`*_collar_joint`, `*_knee_joint`) that won't match your code.

## Files
- `xera_bridge.py` — taps UDP telemetry, serves WebSocket. Runs on the Jetson.
- `xera_viewer_client.js` — patches robot_viewer to receive & drive joints.
- `integrate_example.py` — OPTIONAL direct hook (only if you don't use the tap).

## Setup

### 1. Jetson
```bash
pip install websockets
python xera_bridge.py            # taps 57543, serves ws://0.0.0.0:8765
```
Then run your controller as usual in another terminal:
```bash
python src/main_controller.py --mode mit-signal ...
```

### 2. Viewer (PC)
1. Open robot_viewer (live demo http://viewer.robotsfan.com/ or your own
   `pnpm run dev`) and **upload `grallator_isaac_lab.urdf`**.
2. Open DevTools console (F12), paste all of `xera_viewer_client.js`.
3. Connect to the Jetson:
   ```js
   XeraViewer.connect("ws://192.168.1.42:8765");   // <-- your Jetson's LAN IP
   ```
4. Verify joints line up (should report all 12 present):
   ```js
   XeraViewer.listJoints();
   ```

That's it. Every commanded angle now mirrors live in the viewer.

## Two consumers (viewer + telemetry_gui.py at once)
Plain UDP delivers each datagram to only one bound socket, so the bridge and
`telemetry_gui.py` can't both reliably read port 57543 on every platform. Pick one:

- **Easiest:** run the bridge instead of the GUI — it prints the essential
  telemetry (step, mode, sample joint) as it forwards.
- **Both:** use Option B in `integrate_example.py` — create an
  `XeraViewerBridge` inside `main_controller.py`, call `start_ws()` (WebSocket
  only, no UDP tap), and push `commands` to it right after `telemetry.send(...)`.
  The GUI keeps the UDP port to itself.

## Notes
- **Units are radians** throughout (your `qd` is already radians) — no conversion.
- The viewer shows the **commanded** angle (`qd`). To show measured position
  instead, change the bridge to forward `qf` instead of `qd` in `_udp_loop`.
- **Same network:** the PC must reach the Jetson IP on TCP 8765. Find the IP
  with `hostname -I`; open the firewall port if needed.
- The viewer auto-reconnects if the bridge or Jetson restarts.

## Quick test (no robot, no controller)
```bash
python integrate_example.py      # sweeps all 12 joints over WebSocket
```
Connect the viewer to `ws://<jetson-ip>:8765` and watch it move.
