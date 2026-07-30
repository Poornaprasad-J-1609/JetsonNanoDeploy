# model_12357 Sim-to-Real Audit

Status: **hardware policy enable is blocked**.

The checkpoint proves a deterministic `48 -> 512 -> 256 -> 128 -> 12` ELU
actor. It does not contain observation names, marching-clock semantics, joint
names, action scales, default pose, or previous-action semantics. Those values
must come from the exact Isaac environment/export source and must pass an
independent golden-vector comparison.

The policy artifact is:

```text
policy/policy.pt
SHA256 48b3d7c7e10fd0d27a053fdf3af56bcd9190481c35798b585f0a0ff0478cf8b3
checkpoint iteration 12357
input [N, 48]
output [N, 12]
```

## Mismatch Table

| Signal or behavior | Isaac requirement | Current deployment value | Mismatch and risk | Required correction |
| --- | --- | --- | --- | --- |
| Observation `0:3` | Exact marching clock | Old path generated zeros | Actor phase is wrong; stationary or asymmetric output | Export clock function, frequency, phase/reset rule, and initial state from the exact training source |
| Observation `3:6` | Base angular velocity, body frame, rad/s | Xsens path exists; frame verification false | Wrong axis/sign rotates stabilization feedback | Record a controlled roll/pitch/yaw fixture test and compare with Isaac |
| Observation `6:9` | Normalized projected gravity in base frame | Xsens quaternion conversion exists; verification false | Wrong quaternion/frame convention destabilizes posture | Compare exact Isaac gravity function against fixture vectors |
| Observation `9:12` | Current Phase-2 command is `[0,0,0]` | `W` triggers policy mode while the actor command is forced to zero | Current behavior matches the supplied stationary contract | Keep the override until a command-tracking actor replaces this policy |
| Joint observation order | Exact Isaac articulation/action order | Candidate `BL,BR,FL,FR`, grouped by joint type | A valid action can drive the wrong motor | Export joint IDs/names from the exact environment |
| Joint observation signs | Training adapter reportedly hip/thigh `-1`, calf `+1` | `joint_map.yaml` policy signs are all `+1`; motor signs are separate | Policy can observe or command mirrored motion | Verify policy-space signs separately from encoder/motor polarity |
| Previous action | Exact raw/clipped/scaled convention | Current runtime stores raw actor output | Recurrent temporal input is inconsistent | Copy Isaac observation term implementation |
| Actor clip | Candidate `[-1,1]` | Current control default clip is `2.8`, with extra hip conditioning | Target amplitudes differ from training | Verify and apply exact clip before scale |
| Action scale | Candidate per leg `[-.25,-.40,+.40]` | Scalar `+0.25` plus policy signs | Magnitude and sign mismatch | Export action term joint order and per-index scale |
| Candidate motor IDs | `[4,1,10,7,5,2,11,8,6,3,12,9]` | Current names produce `[10,7,4,1,11,8,5,2,12,9,6,3]` | Candidate conventions conflict | Resolve from Isaac names, then perform one-joint suspended routing |
| Default pose | Exact Isaac `q_default[12]` | All zeros in hardware stand frame | May be valid physically but is unverified for this actor | Export `default_joint_pos` in actor order and measure stand residual |
| Four-bar | Required position, velocity, torque, and gain conversion | Math and per-leg tables exist; global enable is false | Direct calf coordinates do not match virtual Isaac joints | Revalidate every table, enable globally, then pass endpoint/Jacobian checks |
| Four-bar Jacobian | `|dq/dtheta| >= 0.05`, endpoint tolerance `0.01` | Guards configured per profile | Profiles are measured but not approved for this actor | Repeatability test and signed virtual-joint comparison for every calf |
| Policy rate | 50 Hz, monotonic | Fixed at 50 Hz | Structure matches; timing qualification not approved | Log zero-overrun 30 s shadow run |
| Low-level rate | 200 Hz, 5 ms | Dedicated latest-target sender at 200 Hz | Transport exists; target hold/interpolation semantics unverified | Copy training actuator hold/interpolation behavior and measure jitter |
| IMU freshness | Stop above 100 ms | Configured `0.10 s` | Threshold now matches; sensor rate/frame still unverified | Verify timestamps, sequence count, and fresh rate at 100-200 Hz |
| Tilt shutdown | 15 degrees | Gravity-z and CLI defaults now 15 degrees | Threshold matches; complete fault transition is not certified | Integrate with explicit safety state machine |
| Hardware action gain | Start `0.05`, offset only | Implemented as `q_default + gain*(q_actor-q_default)` | Correct formula; actor action conversion remains unverified | Keep at `0.05` through three clean suspended trials after qualification |
| Safety states | `DISABLED -> DAMPING -> CALIBRATION -> STAND -> POLICY_ARMED -> POLICY_ACTIVE` | Existing modes are idle/hold/sit/stand/policy | Fault behavior is spread across controller paths | Implement and verify the explicit state machine before approval |
| Current watchdog | Measured current, continuous/peak duration | No verified current conversion/logging | Torque estimate cannot enforce electrical limits | Measure current telemetry and specify continuous/peak limits |
| Battery watchdog | Verified undervoltage threshold | No verified battery source or threshold | Brownout can corrupt control and CAN | Add measured pack voltage and latched undervoltage fault |
| Thermal watchdog | Verified motor temperature limit/derating | Temperature feedback exists; approval false | Sustained load can overheat actuators | Define warning/fault thresholds and derating curve |
| Golden vectors | 100-1000 independent Isaac rows, error `<=1e-5` | No independent file present | Actor/export/ordering mistakes remain invisible | Export exact actor input/output CSV and run the checker |

## Exact Contract

The supplied contract is:

```text
obs[0:3]    marching clock (formula/frequency/reset unverified)
obs[3:6]    base angular velocity, body frame, rad/s
obs[6:9]    normalized projected gravity, body frame
obs[9:12]   command; stationary actor expects [0,0,0]
obs[12:24]  q - q_default, exact actor order/sign
obs[24:36]  joint velocity, exact actor order/sign
obs[36:48]  previous action, exact training convention
```

The candidate action path is:

```python
raw_action = actor(obs)
clipped_action = clip(raw_action, -1.0, 1.0)
joint_offset = per_index_action_scale * clipped_action
q_actor_target = q_default + joint_offset
q_hardware_target = q_default + hardware_action_gain * joint_offset
```

The clock, order, signs, clip, scale, `q_default`, and previous-action
convention are not encoded in the checkpoint. They remain blocked in
`config/policy_contract.yaml`.

## Mapping Candidate

The generated table
[`model_12357_joint_mapping_candidate.csv`](model_12357_joint_mapping_candidate.csv)
is derived from the current YAML files, not from model semantics. Every row is
marked unverified.

Current candidate actor order and motor IDs:

| Index | Joint | Motor ID | Encoder/target sign | Conversion | Hard range rad |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | BL hip | 10 | -1 | identity | -0.50 to +0.50 |
| 1 | BR hip | 7 | -1 | identity | -0.50 to +0.50 |
| 2 | FL hip | 4 | -1 | identity | -0.50 to +0.50 |
| 3 | FR hip | 1 | -1 | identity | -0.50 to +0.50 |
| 4 | BL thigh | 11 | -1 | identity | -2.00 to +2.00 |
| 5 | BR thigh | 8 | -1 | identity | -2.00 to +2.00 |
| 6 | FL thigh | 5 | -1 | identity | -2.00 to +2.00 |
| 7 | FR thigh | 2 | -1 | identity | -2.00 to +2.00 |
| 8 | BL calf | 12 | +1 | BL four-bar | 0.00 to +1.36 |
| 9 | BR calf | 9 | +1 | BR four-bar | -1.36 to 0.00 |
| 10 | FL calf | 6 | +1 | FL four-bar | 0.00 to +1.36 |
| 11 | FR calf | 3 | +1 | FR four-bar | -1.36 to 0.00 |

The supplied action scale list is leg-interleaved while this candidate order is
grouped by joint type. It must not be copied by index until the Isaac action
joint IDs are exported.

## Missing Measurements

- Exact training task/environment/action/observation/export source.
- Marching-clock formula, frequency, phase offsets, reset, and episode behavior.
- Isaac actor joint names and IDs in observation and action order.
- Position/velocity sign adapter and action target sign per actor index.
- Exact previous-action value and clipping stage.
- Per-index action scale, actor clipping, and `q_default`.
- Independent Isaac observation/action golden dataset.
- Encoder zero error at physical stand for all 12 joints.
- Four-bar repeatability and virtual-angle error for all four calves.
- Effective step response for hip, thigh, and calf in joint space.
- Current-to-torque calibration and continuous/peak current limits.
- Battery warning/fault voltage and voltage measurement source.
- Motor warning/fault temperature and thermal derating curve.
- Friction, backlash, dead zone, and command/feedback latency.
- 50 Hz overrun, 200 Hz jitter, CAN age, and sensor-to-actuator percentiles.
- Real robot mass, center of mass, payload, and harness load.

## Code Structure

- `src/policy_runner.py`: deterministic checkpoint actor loading and raw
  `[48] -> [12]` inference. It accepts exact nonzero clock vectors for replay.
  Generated observations require an explicit `marching_clock[3]`.
- `src/deployment_readiness.py`: policy and hardware qualification gates.
- `src/main_controller.py`: keyboard input, 50 Hz orchestration, pose modes,
  policy target pipeline, watchdogs, telemetry, and CAN lifecycle.
- `src/motor_command_layer.py`: MIT command construction, routing, gains, hard
  limits, and direct joint/motor conversion.
- `src/four_bar_motor_command_layer.py`: virtual-calf wrapper around the motor
  layer; maps targets, feedback velocity/torque, gains, and limits.
- `src/four_bar_transmission.py`: measured interpolation tables, inverse map,
  Jacobian guards, torque/velocity conversion, and endpoint checks.
- `src/state_estimator.py`: live motor and IMU feedback in policy coordinates.
- `src/safety_monitor.py`: joint/rate/tilt/feedback/torque checks.
- `scripts/check_deployment_readiness.py`: the mandatory fail-closed report.
- `scripts/create_policy_golden_vectors.py`: packages independent Isaac CSV
  rows. It never creates expected outputs from the deployment actor.
- `scripts/check_policy_golden_vectors.py`: compares Jetson actor output with
  Isaac references at tolerance `1e-5`.

The existing controller is not yet the requested production safety state
machine. Current/battery watchdogs and complete 200 Hz electrical telemetry are
also absent. The readiness gate therefore remains blocked even if the actor
hash and tensor dimensions pass.

The implementation names its Jacobian `J = dq_virtual/dtheta_motor`, so it uses
`qdot = J*theta_dot` and `tau_virtual = efficiency*tau_motor/J`. The supplied
text uses the reciprocal convention `dtheta_motor/dq_virtual`. These are
physically equivalent only when each equation consistently uses the same
definition; the current four-bar implementation does so.

## Staged Procedure

No supported stand or policy motion command is authorized yet.

1. **Passive readiness report**

   ```bash
   cd ~/JetsonNanoDeploy
   export PYTHONPATH="$PWD/src"
   python3 scripts/check_deployment_readiness.py
   ```

   Pass criterion: every policy and hardware row says `PASS`.

2. **Independent Isaac golden data**

   ```bash
   python3 scripts/create_policy_golden_vectors.py \
     --source-csv /path/from/isaac/model_12357_actor_io.csv \
     --output tests/data/model_12357_isaac_golden.npz

   python3 scripts/check_policy_golden_vectors.py \
     --vectors tests/data/model_12357_isaac_golden.npz \
     --tolerance 1e-5
   ```

   Pass criterion: 100-1000 rows, finite `[12]` outputs, maximum error
   `<=1e-5`. Then set `semantic_contract.golden_vectors_path` in
   `config/policy_contract.yaml` to that reviewed file.

3. **Exact dry replay, motors unopened**

   ```bash
   python3 src/main_controller.py \
     --mode print \
     --policy-shadow-mode \
     --policy-replay-csv /path/from/isaac/model_12357_actor_io.csv \
     --policy-replay-fixed-50-hz
   ```

   Pass criterion: all rows replay, no NaN/Inf, and raw action error `<=1e-5`.

4. **Passive encoder/four-bar collection**

   ```bash
   python3 src/check_motor_connections.py \
     --can-count 2 \
     --can-ports slcan0 slcan1 \
     --can-backend socketcan \
     --can-bitrate 1000000 \
     --all \
     --rate 10 \
     --disable-set-zero-key \
     --disable-crouch-key
   ```

   Pass criterion: 12/12 expected IDs only, fresh feedback, zero fault bits,
   stand errors within 0.02 rad, and all calves report the verified four-bar
   path after it is enabled.

5. **Suspended single-joint calibration**

   Use the existing `--joint-routing-test` only after selecting exactly one
   `--active-joints` entry, verifying the physical E-stop, and reviewing the
   generated mapping row. Limit motion to 0.02 rad and torque to 20 Nm or less.
   This calibration mode is intentionally separate from policy enable.

6. **Supported stand, suspended policy pulse, and micro-marching**

   These commands remain intentionally unavailable while readiness is blocked.
   After all checks are approved, start at gain `0.05`, two seconds suspended,
   and require three clean trials before each gain increase. The 15-degree
   tilt, 100 ms IMU stale limit, encoder freshness, joint limits, temperature,
   current, battery, saturation, and timing watchdogs must all be active.
