/*
 * xera_viewer_client.js  (tailored for the XERA Quadruped / grallator URDF)
 * ========================================================================
 * Connects robot_viewer (fan-ziqi/robot_viewer) to xera_bridge.py running on
 * the Jetson, and drives the loaded URDF's joints from your robot's commanded
 * angles in real time.
 *
 * robot_viewer uses gkjohnson's urdf-loader. The loaded model exposes:
 *      robot.setJointValue(jointName, angleRadians)
 *      robot.joints   -> { jointName: URDFJoint, ... }
 *
 * USE
 * ---
 * 1) Open robot_viewer, upload grallator_isaac_lab.urdf.
 * 2) Open DevTools console (F12), paste this whole file.
 * 3) XeraViewer.connect("ws://localhost:8765");
 * 4) XeraViewer.listJoints();   // sanity-check names match the URDF
 *
 * The bridge sends:
 *    {"type":"joints","values":{"FR_hip_joint":0.1,...},"mode":"stand"}
 *    {"type":"joint","name":"FR_hip_joint","value":0.1}
 * (Set-zero needs no special message: qd collapses to ~0 and is sent as joints.)
 */

(function (global) {
  // Expected joints from grallator_isaac_lab.urdf (revolute only).
  const XERA_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "BR_hip_joint", "BR_thigh_joint", "BR_calf_joint",
    "BL_hip_joint", "BL_thigh_joint", "BL_calf_joint",
  ];

  const XeraViewer = {
    socket: null,
    url: null,
    robot: null,
    mode: "",
    autoReconnect: true,
    _reconnectTimer: null,
    _warnedMissing: {},

    findRobot() {
      if (this.robot && this.robot.joints) return this.robot;
      const candidates = [
        global.robot,
        global.viewer && global.viewer.robot,
        global.urdfViewer && global.urdfViewer.robot,
        global.app && global.app.robot,
      ];
      for (const c of candidates) if (c && c.joints) { this.robot = c; return c; }

      const search = (obj) => {
        if (!obj) return null;
        if (obj.joints && obj.setJointValue) return obj;
        if (obj.children) for (const ch of obj.children) {
          const f = search(ch); if (f) return f;
        }
        return null;
      };
      const scenes = [];
      if (global.scene) scenes.push(global.scene);
      if (global.viewer && global.viewer.scene) scenes.push(global.viewer.scene);
      for (const s of scenes) { const f = search(s); if (f) { this.robot = f; return f; } }

      for (const k in global) {
        try {
          const v = global[k];
          if (v && v.joints && typeof v.setJointValue === "function") {
            this.robot = v; return v;
          }
        } catch (e) {}
      }
      return null;
    },

    setJoint(name, value) {
      const r = this.findRobot();
      if (!r) return;
      if (!(name in r.joints)) {
        if (!this._warnedMissing[name]) {
          this._warnedMissing[name] = true;
          console.warn(`[XeraViewer] joint "${name}" not in URDF. ` +
                       `Loaded grallator_final.urdf instead of grallator_isaac_lab.urdf? ` +
                       `URDF joints: ${Object.keys(r.joints).join(", ")}`);
        }
        return;
      }
      r.setJointValue(name, value);
    },

    setJoints(values) {
      const r = this.findRobot();
      if (!r) return;
      for (const name in values) this.setJoint(name, values[name]);
    },

    zeroAll() {
      const r = this.findRobot();
      if (!r) return;
      for (const name of XERA_JOINTS) if (name in r.joints) r.setJointValue(name, 0);
    },

    handleMessage(msg) {
      if (msg.mode !== undefined) this.mode = msg.mode;
      switch (msg.type) {
        case "joint":    this.setJoint(msg.name, msg.value); break;
        case "joints":   this.setJoints(msg.values);         break;
        case "zero":     this.setJoint(msg.name, 0);          break;
        case "zero_all": this.zeroAll();                      break;
        default: console.debug("[XeraViewer] unknown msg", msg);
      }
    },

    connect(url) {
      this.url = url || this.url || `ws://${location.hostname}:8765`;
      console.log("[XeraViewer] connecting to", this.url);
      this.socket = new WebSocket(this.url);
      this.socket.onopen = () =>
        console.log("[XeraViewer] connected. Live joint mirroring is ON.");
      this.socket.onmessage = (ev) => {
        try { this.handleMessage(JSON.parse(ev.data)); }
        catch (e) { console.error("[XeraViewer] bad message", ev.data, e); }
      };
      this.socket.onclose = () => {
        console.warn("[XeraViewer] disconnected.");
        if (this.autoReconnect) {
          clearTimeout(this._reconnectTimer);
          this._reconnectTimer = setTimeout(() => this.connect(this.url), 2000);
        }
      };
      this.socket.onerror = (e) => console.error("[XeraViewer] socket error", e);
    },

    disconnect() {
      this.autoReconnect = false;
      if (this.socket) this.socket.close();
    },

    listJoints() {
      const r = this.findRobot();
      if (!r) { console.warn("[XeraViewer] robot not loaded"); return []; }
      const names = Object.keys(r.joints);
      const missing = XERA_JOINTS.filter((n) => !(n in r.joints));
      console.table(names);
      if (missing.length)
        console.warn("[XeraViewer] expected joints NOT in this URDF:", missing,
                     "\nMake sure you loaded grallator_isaac_lab.urdf.");
      else
        console.log("[XeraViewer] all 12 XERA joints present. Good to go.");
      return names;
    },
  };

  global.XeraViewer = XeraViewer;
  console.log("[XeraViewer] loaded. Call XeraViewer.connect('ws://<JETSON_IP>:8765')");
})(window);
