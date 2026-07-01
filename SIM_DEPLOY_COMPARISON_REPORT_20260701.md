# Simulation and Deployment Dry-Run Comparison

Analysis date: 2026-07-01

## Input files

- `grallator_sim_policy_input_log_20260701_065137.csv`
- `grallator_sim_motor_log_20260701_065137.csv`
- Deployment policy SHA-256:
  `330e02f2406e129bb6ed6ec4a6c6bcd5f80d7e7799a2ab087e81eda65d2fae69`

The analysis used `src/compare_sim_deploy_logs.py` and produced:

- `logs/deploy_dry_comparison_grallator_sim_policy_input_log_20260701_065137.csv`

## Executive conclusion

The deployed TorchScript policy and the logged simulation policy are the same.
The 48-value observation formula is also correct. Replaying every exact logged
observation through deployment reproduces the simulation action with a maximum
absolute difference of only `1.907e-6`.

The gait mismatch is therefore not caused by a different neural network, a
34/45/48 observation-size error, or nonzero base-linear-velocity slots.

The largest measured deployment mismatch is the joint target rate limiter. With
the current `0.04 rad/step` limit, deployment modifies `77.7%` of forward joint
targets and `82.2%` of backward joint targets. Deployment also normally changes
from policy mode to hold during neutral commands, while simulation continues to
run the policy at zero command.

## Log integrity and actual command sequence

The policy log contains 552 rows at exactly 20 ms intervals. The motor log
contains 6,618 rows. It has 12 motor rows for steps 1-551 and only six rows for
the final step 552, so the last motor step is truncated.

The recorded command sequence is:

| Segment | Steps | Recorded duration | Command |
|---|---:|---:|---:|
| Initial stop | 1-235 | 4.70 s | `[0.0, 0.0, 0.0]` |
| Forward | 236-323 | 1.76 s | `[+0.8, 0.0, 0.0]` |
| Neutral | 324-403 | 1.60 s | `[0.0, 0.0, 0.0]` |
| Backward | 404-518 | 2.30 s | `[-0.8, 0.0, 0.0]` |
| Final stop | 519-552 | 0.68 s | `[0.0, 0.0, 0.0]` |

The CSV does not contain five seconds of forward and five seconds of backward
motion. This does not invalidate the comparison, but a future frequency/steady
gait comparison should capture the full requested duration.

## Verified policy joint order

The simulation motor log records this exact policy/action order:

```text
00 BL_hip_joint
01 BR_hip_joint
02 FL_hip_joint
03 FR_hip_joint
04 BL_thigh_joint
05 BR_thigh_joint
06 FL_thigh_joint
07 FR_thigh_joint
08 BL_calf_joint
09 BR_calf_joint
10 FL_calf_joint
11 FR_calf_joint
```

The current `config/joint_map.yaml` matches this order exactly. This is now
verified from the simulation log rather than inferred from TorchScript shapes.

## Observation reconstruction

For each complete motor-log step, the observation was independently rebuilt as:

```text
obs[0:3]   = [0, 0, 0]
obs[3:6]   = logged body angular velocity
obs[6:9]   = logged projected gravity
obs[9:12]  = logged command
obs[12:24] = logged q - default_q, in logged joint order
obs[24:36] = logged qd, in logged joint order
obs[36:48] = previous logged raw action
```

Across 551 complete steps:

```text
mean absolute reconstruction error = 0.0
maximum absolute reconstruction error = 0.0
```

The log's measured base-linear-velocity columns become nonzero as the simulated
robot moves, but `obs_000`, `obs_001`, and `obs_002` remain exactly zero for all
552 policy rows. This proves the policy keeps the slots but does not consume the
measured base velocity.

## Exact policy replay

Each logged `obs_000..obs_047` vector was passed directly to the approved
deployment `policy.pt` and compared with `action_00..action_11`.

```text
rows replayed                 = 552
mean absolute action error    = 1.151e-7
root-mean-square action error = 1.640e-7
maximum absolute action error = 1.907e-6
```

This is floating-point-level agreement. The policy artifact, input positions,
and action positions are connected correctly when given the same robot state.

## Target conversion

The motor CSV confirms exactly, with zero error:

```text
q_target_est = default_q + 0.25 * action
```

All logged `default_q` values are zero. Deployment uses the same action scale
and formula.

## Safety limiter comparison

The simulation policy targets were passed through the current deployment
`SafetyMonitor`, preserving target history exactly as the deployment limiter
does.

Overall results:

```text
rate-limited joint samples       = 2,415 / 6,624 (36.5%)
hard position clips              = 4
mean |safe_target-sim_target|    = 0.0776 rad
maximum target difference        = 1.9920 rad
```

By command segment:

| Command | Rate-limited samples | Mean target error | Maximum target error |
|---|---:|---:|---:|
| Stop | 460 / 4,188 (11.0%) | 0.0241 rad | 1.9920 rad |
| Forward | 821 / 1,056 (77.7%) | 0.1568 rad | 0.8878 rad |
| Backward | 1,134 / 1,380 (82.2%) | 0.1793 rad | 1.0217 rad |

The four hard position clips occur on `BL_calf_joint` during startup steps 3-6,
where simulation targets are `+1.54` to `+2.07 rad` while the configured maximum
is `+1.42 rad`.

The simulation target-change distribution is much faster than the configured
deployment rate limit:

| Command | Median | 95th percentile | Maximum |
|---|---:|---:|---:|
| Forward | 0.0473 rad/step | 0.2088 rad/step | 0.4757 rad/step |
| Backward | 0.0564 rad/step | 0.2571 rad/step | 0.4896 rad/step |

At 50 Hz, the current deployment limit is `0.04 rad/step = 2 rad/s`. The logged
95th-percentile policy target change is approximately `10.4 rad/s` forward and
`12.9 rad/s` backward. The position policy is producing dynamic PD setpoints,
not a trajectory that the joint is expected to reach fully within one policy
step.

## Simulation motor tracking

Simulation itself does not instantly track each policy target:

```text
mean |q_target_est-q| = 0.1958 rad
maximum error         = 1.8625 rad
```

This is expected for PD-controlled locomotion. The actuator torque responds to
the moving target and current joint state. Rate-limiting every policy target to
what the joint can physically reach in one step changes the controller the
policy was trained with.

## Closed-loop fake dry replay

The recorded command sequence was also run as a deployment dry loop with fake
upright IMU and ideal target-following joint state.

```text
main-state-machine dry action error during movement:
  mean = 1.0959 action units
  max  = 5.3925 action units

always-policy fake dry action error:
  mean = 0.8948 action units
  max  = 8.8994 action units
```

This disagreement is expected and is not evidence of a policy mismatch. The
simulation action depends on its evolving gyro, projected gravity, joint
positions, joint velocities, and previous raw action. A fake dry robot has none
of the simulated physics. Exact observation replay is the valid deterministic
policy comparison, and that comparison passes.

## Neutral-command state mismatch

The simulation logs show nonzero policy actions throughout zero-command
segments. Simulation keeps the locomotion policy active for standing and
balance.

Normal deployment behavior is different:

- Policy runs while the movement command exceeds the walk threshold.
- Releasing movement enters hold mode.
- Hold mode resets previous raw action to zero.
- The next movement starts the recurrent observation history from zero action.

`--stand-policy-stabilization` makes deployment continue policy inference at
zero command after stand auto-zero. That mode is closer to the simulation
control lifecycle.

## Command magnitude mismatch

The simulation command is exactly `+/-0.8 m/s`. With the current keyboard
defaults:

```text
max_vx = 1.8
speed_scale maximum = 0.40
largest default keyboard vx = 0.72 m/s
```

Several safe-preview commands further cap speed scale to `0.15`, producing only
`0.27 m/s`. A lower command may legitimately produce a different gait from the
logged `0.8 m/s` simulation gait.

## Conclusions

1. Policy file: verified correct.
2. Observation dimension: verified 48.
3. Base-linear-velocity slots: verified exact zeros.
4. Observation field order: verified correct.
5. Previous action: verified as previous raw policy action.
6. Action scale: verified `0.25`.
7. Joint/action order: current YAML matches simulation exactly.
8. Main gait mismatch: most strongly associated with policy-target rate limiting
   and different neutral-command state behavior.
9. Fake dry dynamics cannot reproduce physical simulation actions; exact logged
   observation replay is the appropriate policy-equivalence test.

## Recommended next steps

1. Keep the current 48-slot observation contract and current joint order.
2. Separate pose-transition rate limiting from policy locomotion limiting.
   Sit/stand can remain slow; policy walking targets should not be reshaped by a
   `0.04 rad/step` trajectory limiter if simulation matching is the goal.
3. Keep hard mechanical joint limits active. Review the `BL_calf_joint +1.42`
   maximum against the measured safe mechanism range before changing it.
4. Use stand-policy stabilization when comparing real neutral/starting behavior
   with simulation.
5. Capture another log with a verified five seconds forward, 1-2 seconds
   neutral, and five seconds backward. The current log contains only 1.76 and
   2.30 seconds of movement.
6. Compare a real Jetson telemetry log against the same observation/action fields
   after the rate/state-machine decisions are settled.
