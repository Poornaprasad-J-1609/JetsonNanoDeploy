"""
integrate_example.py  (for Poornaprasad-J-1609/JetsonNanoDeploy)
================================================================
You almost certainly do NOT need this file. The recommended path is zero code
changes: just run `python xera_bridge.py`, which taps your existing UDP
telemetry on 127.0.0.1:57543.

This file is only for the OPTIONAL case where you want the viewer fed directly
from inside main_controller.py (e.g. you want to keep telemetry_gui.py on the
same UDP port AND drive the viewer without juggling sockets).

--------------------------------------------------------------------
Option A — recommended, no edits:
    Terminal 1 (Jetson):  python xera_bridge.py
    Terminal 2 (Jetson):  python src/main_controller.py ...   # as usual
    PC browser:           XeraViewer.connect("ws://<JETSON_IP>:8765")

--------------------------------------------------------------------
Option B — direct hook inside main_controller.py (shown below):

In main_controller.py, where TelemetrySender is created, also create the bridge:

    from xera_bridge import XeraViewerBridge
    viewer_bridge = XeraViewerBridge(ws_host="0.0.0.0", ws_port=8765)
    viewer_bridge.start_ws()        # WebSocket only; no UDP tap in this mode

Then, wherever you currently call `telemetry.send(..., commands=commands, ...)`,
add one line right after it:

    telemetry.send(step, mode, command, command_source, commands, action=action)
    push_commands_to_viewer(viewer_bridge, commands)     # <-- the one line

The helper below pulls the same q_des the telemetry already reports, so the
viewer shows exactly what telemetry_gui.py shows. Set-zero is automatic: after
a software re-zero the next commands carry q_des≈0 and the viewer follows.
"""


def push_commands_to_viewer(viewer_bridge, commands):
    """
    `commands` is the list returned by MotorCommandLayer.build_mit_commands(...).
    Each item has 'joint_name' and 'q_des' (the commanded joint angle in rad) --
    the exact value TelemetrySender already publishes as 'qd'.
    """
    values = {}
    for cmd in commands:
        name = cmd.get("joint_name")
        q_des = cmd.get("q_des")
        if name is not None and q_des is not None:
            values[name] = float(q_des)
    if values:
        viewer_bridge.set_joints(values)


# ----- standalone demo of the helper with mock command dicts -----
if __name__ == "__main__":
    import math, time
    from xera_bridge import XeraViewerBridge, XERA_JOINTS

    vb = XeraViewerBridge(ws_host="0.0.0.0", ws_port=8765)
    vb.start_ws()
    print("Connect the viewer to ws://<JETSON_IP>:8765, then watch it move.")
    t = 0.0
    while True:
        commands = [
            {"joint_name": j, "q_des": 0.4 * math.sin(t + i * 0.3)}
            for i, j in enumerate(XERA_JOINTS)
        ]
        push_commands_to_viewer(vb, commands)
        t += 0.05
        time.sleep(0.02)
