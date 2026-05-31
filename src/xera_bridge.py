"""
xera_bridge.py  (tailored for Poornaprasad-J-1609/JetsonNanoDeploy)
===================================================================
Mirrors the XERA Quadruped's commanded joint angles into robot_viewer LIVE,
WITHOUT changing a single line of your control code.

HOW IT HOOKS IN
---------------
main_controller.py already runs a TelemetrySender that emits a JSON UDP packet
every control step to 127.0.0.1:57543. Each packet contains:

    {
      "mode": "stand" | "policy" | "sit" | ... ,
      "joints": [ { "n": "FR_hip_joint", "qd": <commanded rad>, "qf": <fb rad>, ... }, ... ],
      ...
    }

This bridge listens to that same UDP stream and forwards each joint's commanded
angle "qd" to any robot_viewer connected over WebSocket. So:

  - Movement command  -> qd changes  -> viewer joint moves.
  - Set-zero (software re-zero) -> qd collapses toward 0 next packet -> viewer
    snaps the joints to home, exactly mirroring the robot. No special handling.

Because your telemetry only sends to 127.0.0.1, THIS BRIDGE RUNS ON THE JETSON.
The viewer on your PC connects to the Jetson over the WebSocket port (8765).

PORT SHARING
------------
telemetry_gui.py also binds UDP 57543. Plain UDP delivers a datagram to only
ONE bound socket, so the simplest conflict-free path is to run THIS BRIDGE
INSTEAD of telemetry_gui.py (it prints the essential telemetry too). To run both
the GUI and the viewer, see README "Two consumers".

USAGE (on the Jetson)
---------------------
    pip install websockets
    python xera_bridge.py                  # listens 57543, serves ws://0.0.0.0:8765

Then on your PC: open robot_viewer, load grallator_isaac_lab.urdf, paste
xera_viewer_client.js in the console, and run:
    XeraViewer.connect("ws://<JETSON_IP>:8765")

You can also feed joints directly (no UDP) from your own code if you prefer:
    from xera_bridge import XeraViewerBridge
    bridge = XeraViewerBridge(); bridge.start_ws()
    bridge.set_joint("FR_hip_joint", 0.2)
"""

import asyncio
import json
import socket
import threading
from typing import Dict, Optional

import websockets


# Your URDF revolute joints (grallator_isaac_lab.urdf). The bridge only forwards
# joints whose names appear here, so stray telemetry keys can't break the viewer.
XERA_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "BR_hip_joint", "BR_thigh_joint", "BR_calf_joint",
    "BL_hip_joint", "BL_thigh_joint", "BL_calf_joint",
]

TELEMETRY_PORT = 57543   # matches TelemetrySender in main_controller.py


class XeraViewerBridge:
    def __init__(self, ws_host: str = "0.0.0.0", ws_port: int = 8765):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self._clients = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._state: Dict[str, float] = {}
        self._mode: str = ""
        self._lock = threading.Lock()

    # ---------- WebSocket server ----------

    def start_ws(self):
        if self._ws_thread and self._ws_thread.is_alive():
            return
        self._ws_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._ws_thread.start()
        print(f"[XeraBridge] WebSocket serving on ws://{self.ws_host}:{self.ws_port}")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        async def handler(ws):
            self._clients.add(ws)
            print(f"[XeraBridge] Viewer connected ({len(self._clients)} total)")
            with self._lock:
                snapshot = dict(self._state)
            if snapshot:
                try:
                    await ws.send(json.dumps({"type": "joints", "values": snapshot}))
                except Exception:
                    pass
            try:
                async for _ in ws:
                    pass
            finally:
                self._clients.discard(ws)
                print(f"[XeraBridge] Viewer disconnected ({len(self._clients)} left)")

        await websockets.serve(handler, self.ws_host, self.ws_port)

    def _broadcast(self, msg: dict):
        if not self._loop:
            return
        data = json.dumps(msg)
        asyncio.run_coroutine_threadsafe(self._send_all(data), self._loop)

    async def _send_all(self, data: str):
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # ---------- direct API (optional, if you skip UDP) ----------

    def set_joint(self, name: str, value_rad: float):
        if name not in XERA_JOINTS:
            return
        with self._lock:
            self._state[name] = value_rad
        self._broadcast({"type": "joint", "name": name, "value": value_rad})

    def set_joints(self, values: Dict[str, float]):
        clean = {k: float(v) for k, v in values.items() if k in XERA_JOINTS}
        if not clean:
            return
        with self._lock:
            self._state.update(clean)
        self._broadcast({"type": "joints", "values": clean})

    # ---------- UDP telemetry tap ----------

    def start_udp_tap(self, port: int = TELEMETRY_PORT, verbose: bool = True):
        """Listen to main_controller's telemetry and forward qd -> viewer."""
        t = threading.Thread(target=self._udp_loop, args=(port, verbose), daemon=True)
        t.start()
        print(f"[XeraBridge] Listening to telemetry on udp://127.0.0.1:{port}")

    def _udp_loop(self, port: int, verbose: bool):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.bind(("127.0.0.1", port))
        last_print = 0
        while True:
            try:
                raw, _ = sock.recvfrom(16384)
                packet = json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception:
                continue

            values = {}
            for jd in packet.get("joints", []):
                name = jd.get("n")
                qd = jd.get("qd")
                if name in XERA_JOINTS and qd is not None:
                    values[name] = float(qd)

            mode = str(packet.get("mode", ""))
            if values:
                with self._lock:
                    self._state.update(values)
                    self._mode = mode
                self._broadcast({"type": "joints", "values": values, "mode": mode})

            if verbose:
                step = packet.get("step", 0)
                if isinstance(step, int) and step - last_print >= 25:  # ~0.5 s @ 50 Hz
                    last_print = step
                    fr = values.get("FR_thigh_joint")
                    print(f"[XeraBridge] step={step} mode={mode} "
                          f"FR_thigh_qd={fr if fr is None else round(fr,3)} "
                          f"({len(values)} joints) -> {len(self._clients)} viewer(s)")


if __name__ == "__main__":
    import sys
    _port = 57543
    if "--port" in sys.argv:
        _port = int(sys.argv[sys.argv.index("--port") + 1])

    bridge = XeraViewerBridge(ws_host="0.0.0.0", ws_port=8765)
    bridge.start_ws()
    bridge.start_udp_tap(_port, verbose=True)
    print(f"[XeraBridge] Running, listening on {_port}. Ctrl-C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n[XeraBridge] stopped")
