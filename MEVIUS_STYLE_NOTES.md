# Grallator MEVIUS-Style Controller

This worktree isolates the simpler motor-facing controller used during the
earlier MEVIUS-style Grallator tests.

It intentionally keeps Grallator's trained model-8675 policy contract:

- 48 observations and the validated Grallator joint order
- `q_target = q_policy_reference + 0.25 * raw_action`
- motor signs applied downstream by `motor_directions.yaml`
- three editable gain pairs: hip `250/4`, thigh `250/4`, calf `250/4`,
  shared across all operating phases
- physical joint, feedback, thermal, tilt, and absolute torque safety limits
- 50 Hz policy loop and 200 Hz CAN command stream

The MEVIUS-style launch profile follows the upstream split-loop idea: actor
targets are produced at 50 Hz and held by the 200 Hz motor-command loop. After
the entry blend, the motor target is simply:

```text
q_target = q_policy_reference + 0.25 * raw_action
```

There is no routine action EMA, hip multiplier, per-step target limiter,
gravity-support feedforward, or virtual joint-stop preload. The launcher sets
an actor clip of `100`; the exact-policy path bypasses action conditioning,
including this clip. Physical joint clamping still applies. This does
not substitute MEVIUS2's policy, default pose, `0.2` action scale, or symmetry
vector because those belong to a different robot and actor.

The requested `250/4` gains differ from model 8675's training gains: hips
`80/5`, front thighs/calves `110/6.5`, rear thighs/calves `130/8`.
Matching command timing does not establish stability with these different gains.

After stand reaches the configured position and velocity tolerances, policy
takeover starts automatically at a zero locomotion command. A 3-second target
blend handles the transition. Movement keys (`W/A/S/D/Q/E`) then change the
policy command; they are not required to start policy inference.

Hard physical joint limits and fault stops remain mandatory. The 100 Nm
estimated command ceiling is a final authority guard, not a requested torque;
successful nominal operation should not activate it.

## Upstream command architecture

The upstream MEVIUS2 controller runs policy updates at 50 Hz and independent
CAN motor loops at 200 Hz. Each motor loop sends the latest position,
velocity, Kp, Kd, and feedforward torque command directly to the actuator.
This folder preserves that split using Grallator's `CanCommandStreamer` and
official RS04 packet encoding:

- https://github.com/haraduka/mevius2/blob/master/scripts/mevius2_main.py
- https://github.com/haraduka/mevius2/blob/master/scripts/mevius2_utils.py
- https://github.com/haraduka/mevius2/blob/master/scripts/parameters.py

## Qualification status

Current `250/4` verification: 136 repository tests passed. An offline packet
check covered all 12 joints in all six phases (72 commands), confirming
official encoding, effective gains within wire quantization of `250/4`, zero
feedforward, and unchanged small in-range targets. No CAN interface was opened.
These checks establish software behavior, not hardware tracking or gait stability.

The earlier verification below used hip `80/5` and thigh/calf `120/7.25`.
It is historical evidence, not qualification of the current `250/4` settings.
The policy artifact, shape, order, zero base-velocity contract, and `0.25`
target conversion passed, along with 136 repository tests. A 100-trial
logged-feedback replay confirmed zero hip conditioning and zero action
filtering; that replay still applied its own target rate limiter.

The same replay also shows hard physical joint clamping on about 23-30 percent
of samples, depending on command. The model-8675 reference simulation permits
hips to reach +/-0.79 rad and calves to reach +/-1.56 rad, while the measured
hardware configuration permits only +/-0.50 rad hips and +/-1.36 rad calves.
Those hard limits must not be widened without mechanical validation.

A 500-case closed-loop noisy replay kept the entry transition continuous
(about 0.0001 rad target change and 1.16 Nm estimated torque change), but the
unfiltered steady actor later diverged in every synthetic case, with raw action
magnitude reaching about 59.7. That harness uses a simplified first-order
plant, not a validated robot dynamics model, so it cannot establish real gait
stability or certify a supported or unsupported test. Hardware performance
with the current gains remains unverified. Physical joint and torque guards
remain enabled.
