# Simulation/HIL Deployment Audit - 2026-07-08

## Inputs

- Baseline simulation: `sim_logs/sim_logs_sim_data`
  - 1,984 policy rows, 39.68 s, two episode IDs.
- Recorded-real-IMU HIL: `sim_logs/sim_log_real_imu_data`
  - 740 policy rows, 14.80 s, one episode.

These are not a controlled A/B pair. The durations and command coverage differ,
so comparisons use matched command labels and authoritative actor columns.

## Authoritative Columns

- Actor input: `obs_000..obs_047`
- Actor output: `policy_action_00..policy_action_11`
- Joint target: `joint_pos_target`
- Applied simulation torque: `applied_torque_postclip`

The descriptive `command_*` and raw Observation Manager term columns are one
step out of phase with the actor tensor. In both logs, step 1 metadata contains
`yaw=-0.8` while actor observation slots 9:12 correctly contain zero.

## Verified Contracts

- Observation dimension/order: 48 slots, with `obs[0:3]` exactly zero.
- Joint/action order: BL, BR, FL, FR for hip, then thigh, then calf.
- Previous action: exact previous raw actor output on continuous rows.
- Action scale: `q_target = 0.25 * action`, with zero default pose.
- Motor direction round-trip: zero error.
- Torque direction round-trip: zero error.
- Hip/thigh directions: -1. Calf directions: +1.

## Policy Artifact Mismatch

Replaying the exact logged actor tensor through deployed `policy/policy.pt` gives:

| Run | Mean absolute action error | Maximum action error |
|---|---:|---:|
| Baseline | 0.02160 | 0.12381 |
| HIL | 0.02363 | 0.15260 |

The same-step correlation is above 0.9997, but this is not floating-point-level
agreement. The logs were produced with `model_6225.pt`; that exact exported actor
is not present in this repository. Exact sim-to-real action matching requires an
export of that checkpoint and verification of its SHA-256 before deployment.

## IMU Findings

- Baseline actor IMU slots match the previous row's post-step simulation state
  exactly, confirming the documented `obs_t -> action_t -> state_(t+1)` order.
- HIL gravity remains near `[-0.0154, -0.0159, -0.9998]` and gyro remains near
  zero while the simulated body moves.
- Replacing HIL slots 3:9 with the corresponding simulated IMU changes current
  deployed-policy actions by mean 0.315, 95th percentile 1.589, maximum 8.579.
- HIL yaw tracking overshoots to about +1.58/-1.38 rad/s for +/-0.8 commands;
  baseline tracks about +0.83/-0.78 rad/s.

This stationary replay verifies parsing, transport, freshness, and slot mapping.
It cannot validate closed-loop locomotion or horizontal IMU axis signs. A live
or recorded IMU trajectory rigidly coupled to the simulated base is required.

## Deployment Shaping

Previous defaults (`clip=0.8`, smoothing=0.5, rate=0.01 rad/step) changed nearly
every target and produced mean target error about 0.20-0.21 rad.

The revised middle profile (`clip=3.0`, smoothing=0.1, rate=0.10 rad/step) gives:

| Run | Mean target error | Action samples clipped | Rate-adjusted samples |
|---|---:|---:|---:|
| Baseline | 0.1002 rad | 733 / 23,808 | 4,364 |
| HIL | 0.1221 rad | 309 / 8,880 | 1,740 |

Hard position bounds and the estimated/measured torque safety layers remain.
`--policy-sim-match` is an explicit suspended-test option that bypasses action
shaping and policy target slew, but it does not bypass hard joint, encoder,
tilt, or torque safety.

## Torque Identification and Fix

Least-squares fits of simulation `computed_torque_preclip` produce R2 values of
0.983-1.000. Identified official-unit gains are approximately:

- Hips: Kp 80, Kd 4.5
- Rear thighs: Kp 130, Kd 7.2
- Front thighs: Kp 112, Kd 6.2
- Rear calves: Kp 135, Kd 7.6
- Front calves: Kp 113, Kd 6.3

Policy and leveling frames now use official RS04 encoding and these policy gains.
The old implementation encoded configured 20/0.8 through legacy 500/5 ranges,
which RS04 decodes as approximately 200/16, while software estimated torque as
20/0.8. This 10-20x accounting mismatch is fixed. The limiter now also adjusts
velocity target when damping torque alone would exceed the configured 40 Nm.

Simulation frequently applies more than 40 Nm (95th percentiles often 40-85 Nm,
peaks to 130 Nm). Deployment intentionally remains capped at 40 Nm, so physical
torque cannot exactly match these simulation runs without an explicit safety
decision and hardware validation.

## Outputs

- `logs/sim_comparison/baseline_replay_20260708.csv`
- `logs/sim_comparison/hil_replay_20260708.csv`
- Matching `.txt` summaries in the same directory.
- Six directional controller dry runs under
  `logs/sim_comparison/controller_dry/`.

