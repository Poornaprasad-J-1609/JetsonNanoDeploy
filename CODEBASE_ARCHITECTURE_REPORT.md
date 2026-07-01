# GRALLATOR_DEPLOY Codebase Architecture and Logic Report

Audit date: 2026-07-01  
Audited branch: `walk-branch-jun18`  
Audited commit: `7ee1a61` (`walk-fix-2`)

## 1. Scope and current repository state

This report describes the code that is present after the recent Git checkout. It
is an analysis of the current files, not a claim that the hardware behavior has
been validated on a suspended robot.

At the start of the audit, tracked source files were clean. The following local
configuration/policy changes were intentionally present and included in the
baseline analysis:

- `config/default_pose.yaml` is modified.
- `config/joint_limits.yaml` is modified.
- `config/mit_motor_control.yaml` is modified.
- `policy/policy.pt` is a new untracked policy file.

The active policy is `policy/policy.pt`, because `PolicyRunner` explicitly
prefers that filename. It is a valid TorchScript model with 48 observations and
12 actions. It is numerically different from the older checkpoint
`grallator_flat_sim2real_latest (5).pt`.

### Post-audit policy contract enforcement

Following artifact verification, deployment now enforces the approved policy
SHA-256, an exact 48-input/12-output tensor contract, the exact observation
layout below, finite observations/actions, and zero base linear velocity at
indices 0:3. Startup prints the selected hash. A different artifact is refused
unless the operator explicitly passes `--allow-policy-hash-mismatch` after
independent verification. `src/check_policy_contract.py` provides the dedicated
hardware-free acceptance check.

### Simulation-alignment update

The supplied IsaacLab CSV logs subsequently verified the semantic policy order
as BL/BR/FL/FR hips, then thighs, then calves. Exact logged observations replay
through deployment within floating-point tolerance. Deployment now uses a fixed
hardware stand-zero frame, `crouch_pose` as the sole crouch target, no automatic
stand/sit software-zero, continuous zero-command policy inference after stand,
and no deploy-only target slew in policy mode. Sit/stand still use configured
rate limits and all final motor commands retain hard joint limits. These changes
supersede the older zero-frame and policy-rate descriptions later in this audit.

## 2. Executive summary

The deployment is organized around `src/main_controller.py`. It collects a
command, encoder state, and IMU state; builds the 48-value policy observation;
runs the policy; converts the 12 actions into joint targets; applies optional
motion assists and safety filters; converts joint targets into RobStride MIT
packets; and routes those packets over one, two, or four USB-CAN adapters.

The design has several good properties:

- Motor feedback is mapped by `(bus_name, motor_id)`, preventing cross-bus ID
  contamination.
- Policy and safety logic consume converted joint radians, not raw encoder
  radians.
- Every final MIT position command is hard-clipped against
  `config/joint_limits.yaml`.
- One-, two-, and four-CAN routing pass the dry routing tests.
- Software-zero and clockwise/counter-clockwise direction math pass the dry
  routing test.
- The current policy has the expected 48-to-12 tensor dimensions.
- The selected policy hash is checked before the model is loaded.

The current checkout is not internally clean enough to call hardware-ready
without qualification. The highest-priority findings are:

1. `BL_calf_joint=-1.95` in `stand_pose_when_sit_zero` violates its configured
   minimum of `-1.20`. The requested pose is therefore clipped asymmetrically.
2. The startup-to-stand retry path references undefined `q_safe_target` and can
   raise `NameError` when feedback is incomplete.
3. Stale or missing motor feedback causes an emergency stop and motor stop
   frames, not a hold-current-position fallback.
4. Motor fault bits are decoded and displayed but are not checked by the main
   safety monitor.
5. The Xsens stale-data check can classify cached data as live after packets
   stop arriving.
6. The policy's semantic joint order is not encoded in TorchScript; it has now
   been verified separately against the simulation motor log.

These findings are detailed in section 13.

## 3. System architecture

```text
Keyboard / joystick / fixed command
              |
              v
      joystick_interface.py
              |
              | command [vx, vy, yaw] and mode requests
              v
        main_controller.py <---------------- IMU interface
              |                                  |
              |                                  | gyro, gravity, quaternion
              |<--------- state_estimator.py ----+
              |                  ^
              |                  | MIT feedback frames
              v                  |
        policy_runner.py         |
              |                  |
              | 12 policy actions
              v                  |
       action-to-position        |
              |                  |
       optional motion assist    |
              |                  |
        safety_monitor.py        |
              |                  |
              v                  |
     motor_command_layer.py -----+
              |
              | extended CAN IDs and 8-byte MIT payloads
              v
        can_topology.py
              |
              v
   robstride_can_interface.py
              |
              v
       1 / 2 / 4 USB-CAN adapters
              |
              v
       12 RobStride actuators

main_controller.py also sends:
  - terminal telemetry and per-run CSV logs
  - localhost UDP telemetry to telemetry_gui.py
  - optional UDP data consumed by xera_bridge.py
```

## 4. Main runtime flow

### 4.1 Startup

`main_controller.py` performs these steps:

1. Loads command, IMU, motion-assist, policy-deployment, joint, motor, safety,
   and MIT configuration.
2. Loads the policy and infers its input/output sizes.
3. Resolves active joints and the selected one-, two-, or four-CAN topology.
4. Rejects an IMU serial port that exactly matches an active CAN port.
5. Opens the IMU and, for `signal` or `mit-signal`, opens the CAN adapters.
6. Creates either `FakeStateEstimator` or `MitFeedbackStateEstimator`.
7. Optionally applies startup software zero when explicitly enabled.
8. Starts CSV logging and the optional GUI.
9. In `mit-signal` mode, sends the configured number of enable-frame rounds.
10. Runs an optional startup stand transition, then enters the policy/pose loop.

Normal defaults are passive: `start_control_mode=idle`,
`startup_action=hold`, and `auto_zero_on_startup=false`. In idle mode no MIT
position command is generated until a pose, hold, or movement request establishes
a motion target.

### 4.2 Control loop

The loop target period is 20 ms (`50 Hz`). Each iteration:

1. Reads IMU and motor feedback.
2. Checks keyboard/joystick emergency-stop and IMU safety.
3. Handles calibration and mode requests.
4. Reads and clips `[vx, vy, yaw]`.
5. Applies the optional policy command gain and per-axis policy caps.
6. Selects `idle`, `hold`, `stand`, `sit`, or `policy` behavior.
7. Builds a safe joint target and MIT commands.
8. Sends commands, grouped by physical adapter. Distinct adapters send in
   parallel; commands on one adapter send sequentially with the configured gap.
9. Refreshes MIT feedback and runs encoder safety checks.
10. Updates zero-frame transitions, terminal/CSV telemetry, and UDP GUI data.
11. Sleeps for the remainder of the 20 ms cycle.

Any normal exit from real MIT mode sends stop frames to all active motors in the
`finally` block.

## 5. Controller state and pose logic

### Modes

- `idle`: no MIT target is sent until a motion target has been established.
- `hold`: repeats the last target. Pressing hold attempts to capture fresh joint
  feedback, retaining the previous target for any missing joint.
- `stand`: moves toward the frame-dependent stand target using the configured
  per-step limits.
- `sit`: moves toward the frame-dependent crouch target.
- `policy`: runs policy inference while a movement command exceeds the walking
  threshold, or while stand-policy stabilization is enabled in the stand frame.

### Zero frames

The code maintains a software coordinate frame; it does not send a RobStride
hardware set-zero command from `main_controller.py`.

- In `crouch` zero frame, crouch is the calibration value and stand is
  `calibration + stand_pose_when_sit_zero`.
- After stand settles, optional stand auto-zero changes the software offsets so
  stand becomes the RL zero frame.
- In `stand` zero frame, crouch is computed as
  `stand_calibration - stand_pose_when_sit_zero`.
- After crouch settles, optional sit auto-zero changes the software offsets back
  to crouch zero.

Important: `sit_pose_when_stand_zero` is loaded and checked by diagnostics, but
the current runtime does not use it for the sit target. Runtime deliberately uses
the exact inverse of `stand_pose_when_sit_zero`. The status message saying the
sit target uses `sit_pose_when_stand_zero` is therefore inaccurate.

### Pose synchronization

The position target is rate-limited by `dq_max_per_step`. During stand/sit, if
measured joint error exceeds `pose_sync_error_rad`, the next target is not
advanced. This lets slower joints catch up, but it also means one delayed joint
can pause the progression of every joint.

## 6. Policy contract and data units

### Active model

- Path: `policy/policy.pt`
- SHA-256: `330e02f2406e129bb6ed6ec4a6c6bcd5f80d7e7799a2ab087e81eda65d2fae69`
- Format: TorchScript `_TorchPolicyExporter`
- Network: `48 -> 512 -> 256 -> 128 -> 12`, ELU hidden activations
- Normalizer: identity
- Control period: `0.02 s`
- Action scale: `0.25 rad/action-unit`

The old checkpoint remains loadable as a state-dict actor, but it is not selected
while `policy/policy.pt` exists.

### Observation layout

```text
0:3    base linear velocity
3:6    base angular velocity from IMU gyro, rad/s
6:9    projected gravity in body frame, unit vector
9:12   command [vx, vy, yaw_rate]
12:24  joint_position - default_pose, rad
24:36  joint velocity, rad/s
36:48  previous raw policy action
```

The current policy contract requires base linear velocity to be zero. The
controller rejects nonzero base-linear-velocity source selections, and
`PolicyRunner.build_observation()` independently writes exact zeros to indices
0:3 regardless of the supplied estimator value.

### Joint state path

The policy receives joint-space values:

```text
raw MIT position
  -> subtract joint offset
  -> apply joint direction
  -> wrap to [-pi, pi)
  -> q_current joint radians
  -> subtract q_default
  -> policy observation
```

Raw motor position is retained only for diagnostics and selecting the nearest
equivalent motor command branch. It is not passed directly to the policy.

### Action path

```text
policy action
  -> optional action clip and temporal smoothing
  -> q_default + 0.25 * action
  -> optional IMU posture and gait overlays
  -> soft joint-position clip
  -> per-step rate limit
  -> final hard joint clip in MotorCommandLayer
  -> direction/offset conversion
  -> nearest equivalent motor position around live raw feedback
  -> MIT packet
```

The model file proves only tensor dimensions, not semantic joint order. The
configured order has now been verified against
`grallator_sim_motor_log_20260701_065137.csv`:

```text
BL hip, BR hip, FL hip, FR hip,
BL thigh, BR thigh, FL thigh, FR thigh,
BL calf, BR calf, FL calf, FR calf
```

This order exactly reconstructs all 48 logged simulation observations and maps
the corresponding 12 logged policy actions to their named joints.

## 7. Command interfaces

### Keyboard

- `W`: positive `vx`
- `S`: negative `vx`
- `A`: positive `vy`
- `D`: negative `vy`
- Simultaneous active W/S and A/D deadlines create diagonal commands.
- Up/down arrows change speed scale.
- `C`: sit/crouch mode
- Space: stand mode
- `H`: capture/hold the current pose
- `X`: emergency stop

Keyboard movement relies on terminal key repeat. A key remains active for the
configured timeout after each character. Keyboard yaw is not implemented;
`max_yaw` is accepted but the keyboard command always sets yaw to zero.

### Joystick

Default mappings come from `config/joystick.yaml`:

- Left Y (`axis 1`): `vx`
- Left X (`axis 0`): `vy`
- Right X (`axis 2`): yaw
- Button 4: sit
- Button 5: stand
- Buttons 0-3: emergency stop
- Buttons 6/7: speed down/up
- D-pad down: software-zero calibration request

The D-pad calibration trigger uses a level plus cooldown check, not a strict
rising edge, so holding it can issue repeated software-zero requests once per
cooldown interval.

### Fixed source

The fixed source is useful for dry runs. It has no mode or emergency requests
and continually returns the configured command.

## 8. IMU path

`imu_interface.py` supports:

- `fake`: upright gravity and zero angular velocity.
- `xsens`: binary Xbus/MTData2 quaternion, rate-of-turn, and acceleration data.
- `serial-json`: line-oriented JSON.
- `serial-csv`: configured line-oriented CSV fields.
- `none`: no sensor object; estimator fallback values remain active.

The Xsens path requests measurement mode, parses MTData2, converts the
world-from-body quaternion to projected gravity, rotates gyro/gravity through
`sensor_to_base_rotation`, and passes those values into the policy.

There are two different stabilization mechanisms:

- The RL policy always receives gyro and projected gravity in policy mode.
- `--imu-stabilization` adds a separate hand-authored posture correction after
  the policy output. This is not part of the trained policy.
- `--stand-policy-stabilization` runs the RL policy with a zero movement command
  after stand auto-zero completes.

`flip_gravity`, `gyro_units`, and axis-sign fields in `imu.yaml` affect the
generic serial parser but are not all consumed by the Xsens binary class. The
Xsens implementation assumes its MTData2 gyro units and uses the fixed absolute
quaternion conversion plus `sensor_to_base_rotation`.

## 9. Motor, MIT protocol, and CAN routing

### MIT frame construction

`motor_command_layer.py` constructs RobStride/CyberGear-style extended IDs:

```text
bits 28..24: communication type
bits 23..8:  feed-forward torque encoding
bits 7..0:   motor ID
```

The eight payload bytes contain 16-bit position, velocity, Kp, and Kd. Feedback
decoding extracts motor ID, fault bits, mode status, position, velocity, torque,
and temperature.

### Position conversion and wrap handling

Joint feedback uses:

```text
q_joint = wrap_to_pi(direction * (raw_motor_position - offset))
```

The outgoing motor target first computes `offset + direction*q_des`, then picks
the equivalent `2*pi` branch nearest live motor feedback. This avoids commanding
a full revolution merely because feedback booted on an adjacent encoder branch.

### Routing

- 1 CAN: all joints -> `can0`
- 2 CAN: FR/FL -> `front`, BR/BL -> `back`
- 4 CAN: each leg -> its own FR/FL/BR/BL adapter

Commands are grouped by physical adapter. Different adapters are written from a
thread pool so front/back or per-leg batches begin close together. A single
adapter still receives sequential frames and each serial write is flushed.

Feedback is keyed by `(bus_name, motor_id)`. The old motor-ID-only fallback is
not used, which is the correct behavior for reused IDs on separate physical
buses.

The current motor IDs are globally unique (`0x01` through `0x0C`), so duplicate
ID routing is not presently needed. `check_motor_connections.py` validates
duplicate IDs per physical adapter, but `main_controller.py` does not call that
validation helper.

### Hardware versus software zero

- `main_controller.py` uses software offsets only.
- `check_motor_connections.py`, key `S`, sends RobStride communication type 6
  hardware set-zero frames and can update default/stand YAML values.
- `check_motor_connections.py`, key `C`, only saves current converted joint
  radians to `crouch_pose`; it does not hardware-zero motors.

## 10. Safety behavior

### Target safety

There are two layers:

1. `SafetyMonitor` optionally clips position and per-step rate according to
   `control_limits.yaml` and `joint_limits.yaml`.
2. `MotorCommandLayer` always applies its loaded hard joint limits immediately
   before MIT conversion, even when soft position limiting is disabled.

Current `dq_max_per_step=0.04 rad` at 50 Hz corresponds to `2.0 rad/s` for every
joint. The policy diagnostic shows that most or all joints are rate-limited on
the first policy step from the default pose.

MIT p/v/Kp/Kd/torque fields are also clipped to the configured protocol caps.

### State safety

The safety monitor checks:

- projected-gravity Z against the bad-tilt threshold;
- body angular-velocity norm;
- feedback vector length;
- missing and stale active-joint feedback when motion requires feedback;
- finite joint radians;
- absolute joint-angle sanity;
- joint angles against joint limits plus a margin.

The state check uses converted joint radians. It does not reject nonzero MIT
fault bits. Missing/stale feedback currently ends the loop as an emergency stop;
it is not converted into hold mode.

## 11. Configuration ownership

| Configuration | Main consumers | Runtime behavior |
|---|---|---|
| `control_limits.yaml` | joystick, safety, motor layer, main controller | Command limits and speed scale are re-read; safety/MIT limits reload on file mtime. Policy deployment defaults are startup-only. |
| `default_pose.yaml` | policy runner, motor checker | Loaded once at startup. Motor checker can write pose values and creates a backup. |
| `imu.yaml` | IMU interface, controller | Loaded at startup; selects serial settings, transforms, and base-linear-velocity behavior. |
| `joint_limits.yaml` | safety, motor layer, checkers | Safety and final hard motor limits reload when file mtime changes. |
| `joint_map.yaml` | policy runner, joystick defaults, checkers | Defines control dt, action scale, observation indices, and policy joint order. Loaded at startup. |
| `joint_offsets.yaml` | motor layer | Persistent motor offset and direction map, loaded at startup. Runtime software-zero changes the in-memory copy only. |
| `joystick.yaml` | joystick interface | Startup mappings/filter defaults; command limits are separately re-read from `control_limits.yaml`. |
| `mit_motor_control.yaml` | motor layer, controller | Protocol, startup duration, gains, feed-forward, and frame gap. Loaded at startup. |
| `motion_assist.yaml` | main controller | Optional hand-authored IMU posture and gait overlays. Loaded at startup, then CLI flags override enable fields in memory. |
| `motor_ids.yaml` | controller, motor layer, tools | Motor IDs and optional active-joint set. Topology-specific routing is computed from joint names and `--can-count`. |
| `safety_limits.yaml` | safety monitor | Tilt, angular-rate, and encoder checks; reloaded on file mtime. |

## 12. Source file responsibilities

| File | Responsibility |
|---|---|
| `main_controller.py` | Application entry point, mode state machine, zero-frame transitions, policy loop, safety orchestration, telemetry, CSV, GUI process, shutdown. |
| `policy_runner.py` | Policy discovery/loading, pose arrays, observation construction, policy inference, action-to-joint-target conversion. |
| `state_estimator.py` | Fake state for dry runs and MIT-feedback state mapped into policy joint order. Applies software zero through the motor layer. |
| `motor_command_layer.py` | Joint/motor conversion, wrap handling, hard limits, MIT packet encoding/decoding, command generation, per-bus sending. |
| `robstride_can_interface.py` | Serial framing and parsing for the AT USB-CAN adapter. |
| `can_topology.py` | One/two/four adapter argument handling, joint routing, port sharing, opening/closing, duplicate-ID validation helper. |
| `safety_monitor.py` | Hot-reloaded joint/rate limits, tilt/angular-rate emergency checks, encoder sanity. |
| `imu_interface.py` | Fake, Xsens MTData2, serial JSON, and serial CSV IMU readers and coordinate conversion. |
| `joystick_interface.py` | Fixed, terminal keyboard, and pygame joystick command sources, filters, buttons, speed scaling, and mode requests. |
| `check_can_routing.py` | Hardware-free tests of MIT/enable/poll/set-zero/stop routing, shared adapters, duplicate IDs, zero, and direction math. |
| `check_policy_limits.py` | Loads the real policy, checks observation layout and poses, samples commands, and verifies safety/final MIT limits. |
| `check_policy_contract.py` | Verifies approved SHA-256, exact 48/12 dimensions, finite zero-input output, rejection of 34/45 inputs, and forced zero indices 0:3. |
| `check_motor_connections.py` | Live motor polling, strict bus+ID feedback lookup, terminal/GUI status, hardware set-zero, crouch pose capture. |
| `policy_walk_joint_test.py` | Policy test harness that computes the full policy output but can highlight or command selected joints. It has tester-only gains and limits, so it is not identical to main deployment. |
| `test_mit_joint_motion.py` | Direct sine-wave MIT joint test that deliberately bypasses policy, command source, and IMU. |
| `telemetry_gui.py` | Dark Tk telemetry dashboard and Xsens axis viewer using localhost UDP packets. |
| `xera_bridge.py` | Optional UDP-to-WebSocket bridge for an external browser robot viewer. |
| `integrate_example.py` | Optional example for directly feeding the viewer bridge from controller command dictionaries. |

## 13. Findings and risks

### Critical/high priority

#### F1. Edited stand pose is outside the BL calf hard limit

`config/default_pose.yaml` requests `BL_calf_joint=-1.95` in
`stand_pose_when_sit_zero`; `config/joint_limits.yaml` permits only
`[-1.20, +1.42]`. `check_policy_limits.py` fails for this reason. The final
motor target is clipped to `-1.20`, so the robot cannot execute the configured
pose symmetrically and stand settling/auto-zero may not complete as expected.

#### F2. Undefined variable in startup stand feedback retry

`main_controller.py:1195` passes `q_safe_target` to the retry helper, but the
local target is named `q_safe`. If incomplete feedback enters this branch during
`--startup-action stand`, Python raises `NameError` instead of applying the
intended keepalive and safety handling.

#### F3. Stale feedback stops motors instead of holding

`safety_monitor.py:217-251` classifies missing/stale feedback as a safety fault.
`main_controller.py:1966-1989` labels it an emergency stop and breaks the loop;
the `finally` block sends stop frames. This differs from a hold-last-safe-target
strategy and can explain abrupt loss of hold when one motor reply is delayed.

#### F4. MIT motor fault bits are not part of main safety decisions

`motor_command_layer.py` decodes `fault_bits`, and telemetry/check tools display
them. `SafetyMonitor.encoder_sanity_check` does not inspect those fields. A motor
can therefore report a nonzero fault while the controller continues until some
other feedback or state check fails.

#### F5. Xsens cached samples can appear perpetually live

`XsensBinaryImuSensor.read()` returns the last cached reading whenever no new
packet is parsed. `FakeStateEstimator.refresh_imu()` sets
`last_imu_update_time=time.monotonic()` on every such read instead of using or
comparing the reading timestamp. After the first valid packet, a disconnected or
stalled Xsens stream can remain labeled `live` while old orientation is reused.

#### F6. Policy semantic order required external verification (resolved)

TorchScript does not expose joint names, so tensor shapes alone were
insufficient. The simulation motor log now verifies the current
`policy_to_real_order`, and exact observation replay reproduces logged actions
within `1.907e-6` maximum absolute error.

### Medium priority

#### F7. Main controller does not enforce duplicate-ID physical-bus validation

The validation exists in `can_topology.py` and is called by the motor checker,
but not by `main_controller.py`. Current IDs are unique, so the present config is
safe from this specific conflict. Reusing IDs later could make multiple logical
buses sharing one port ambiguous.

#### F8. Pose configuration and runtime status text disagree

The runtime stand-zero sit target ignores `sit_pose_when_stand_zero` and uses the
negative stand delta. The log at `main_controller.py:1766` still says the former
is used. Editing that YAML block will not change the actual stand-zero sit
motion, although the diagnostic checker still validates it.

#### F9. Keyboard lateral signs and yaw are implicit limitations

Actual keyboard mapping is A=`+vy`, D=`-vy`; it is not configurable independently
from code. Keyboard Q/E yaw is absent. This is internally consistent with the
printed labels only if `+vy` means left in the trained command frame.

#### F10. Two logical CAN buses may silently share one default port

With `--can-count 2` and no topology-specific ports, both front and back fall
back to `--port`, normally `/dev/ttyUSB0`. The adapter object is intentionally
shared. Commands are still routed, but this is physically one CAN network, not
two isolated buses. Real commands should always specify both ports explicitly.

#### F11. Current edited policy gains are aggressive relative to the checkout

Local changes set policy Kp to 75 for hip, thigh, and calf and Kd to 1.8. The
checked-out values were Kp 8/10/12 and Kd 0.8/1.0/1.2. This report cannot decide
which gains match the mechanics, but the local change substantially increases
position stiffness and must be treated as a hardware-risk change.

#### F12. Direct non-policy callers still lack a finite motor-target guard

`PolicyRunner` now rejects malformed/non-finite observations and actions. The
motor layer itself still does not explicitly reject NaN/Inf before packing, so
direct callers that bypass `PolicyRunner` should receive the same guard.

### Documentation/tooling drift

#### F13. `requirements.txt` omits `websockets`

`xera_bridge.py` imports `websockets`, but the package is documented as a
separate manual install and is absent from the portability requirements file.

#### F14. README paths do not match the current tree

The README examples run `python xera_bridge.py` and refer to
`xera_viewer_client.js`; the bridge is currently under `src/`, and no viewer
client file exists in this repository. The README is mainly a viewer-bridge
document rather than a system architecture/readme for the deployment itself.

#### F15. Telemetry exceptions are intentionally silent

`TelemetrySender.send()` catches every exception without reporting it, and the
GUI receive loop also ignores all non-timeout exceptions. Controller motion is
not interrupted by GUI failures, but serialization/network defects can be hard
to diagnose.

## 14. Validation performed

The following hardware-free checks were run against the current working tree:

| Check | Result |
|---|---|
| Python compile of all `src/*.py` | PASS |
| Load `policy/policy.pt` as TorchScript | PASS |
| Policy dimensions `48 -> 12` | PASS |
| Approved policy SHA-256 | PASS |
| Zero `[1,48]` inference returns finite `[1,12]` | PASS |
| `[1,34]` and `[1,45]` policy inputs rejected | PASS |
| Observation indices `0:3` forced to exact zero | PASS |
| Unapproved policy hash refused by default | PASS |
| Simulation/deployment semantic joint order | PASS |
| Exact simulation observation/action replay | PASS, maximum error `1.907e-6` |
| Zero-input policy output finite | PASS |
| One-CAN routing | PASS |
| Two-CAN routing | PASS |
| Four-CAN routing | PASS |
| Duplicate-ID feedback isolation test | PASS |
| Software-zero and direction test | PASS |
| Observation layout coverage/uniqueness | PASS |
| Upright Xsens gravity sanity | PASS |
| Dry policy loop, 5 steps, fixed `vx=0.2` | PASS |
| Pose/limit consistency | FAIL: `BL_calf_joint` stand delta |

The dry policy run generated 12 commands per step and routed six to front and
six to back. It does not validate serial timing, physical direction, torque,
mechanical limits, IMU mounting, policy joint semantics, contact behavior, or
real gait quality.

## 15. Recommended correction order

1. Reconcile `BL_calf_joint` stand pose and its hard joint limit based on the
   measured safe mechanical range; rerun `check_policy_limits.py` until it passes.
2. Fix the `q_safe_target` startup typo and add a targeted test for incomplete
   startup feedback.
3. Decide and implement one explicit stale-feedback policy: controlled hold,
   bounded retry followed by stop, or immediate stop. Document the safety
   rationale.
4. Add motor fault-bit handling to main safety behavior.
5. Make Xsens liveness depend on new packet timestamps, not successful reads of
   cached data.
6. Verify and record the exact policy joint order and observation scaling from
   the training source that exported `policy.pt`.
7. Add finite/shape checks around observations, actions, targets, and IMU data.
8. Run the hardware-free checks again, then proceed through connection, suspended
   pose, suspended policy, and finally supported low-speed gait tests.

## 16. Useful audit commands

```bash
cd /path/to/GRALLATOR_DEPLOY
export PYTHONPATH="$PWD/src"

/usr/bin/python3 -m compileall -q src
/usr/bin/python3 src/check_policy_contract.py
/usr/bin/python3 src/check_can_routing.py --can-count 1
/usr/bin/python3 src/check_can_routing.py --can-count 2
/usr/bin/python3 src/check_can_routing.py --can-count 4
/usr/bin/python3 src/check_policy_limits.py --base-lin-vel-source zero

/usr/bin/python3 src/main_controller.py \
  --mode print \
  --feedback-source fake \
  --command-source fixed \
  --vx 0.20 --vy 0.0 --yaw 0.0 \
  --imu-source fake \
  --no-imu-stabilization \
  --no-gait-assist \
  --base-lin-vel-source zero \
  --start-control-mode policy \
  --startup-action hold \
  --initial-zero-frame stand \
  --no-auto-zero-on-startup \
  --policy-steps 100 \
  --log-every 5
```

Do not interpret a dry-run pass as hardware approval. Real MIT commands should
continue to be tested with the robot secured and with verified mechanical joint
limits, CAN ports, motor IDs, directions, IMU mounting, and an accessible
emergency stop.
