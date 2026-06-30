# Grallator Sim-to-Real Policy Contract

## Policy

- File: `policy/policy.pt`
- Format: exported TorchScript actor
- SHA-256: `330e02f2406e129bb6ed6ec4a6c6bcd5f80d7e7799a2ab087e81eda65d2fae69`
- Observation dimension: 48
- Action dimension: 12
- Control period: 0.02 s

The controller refuses to start if the configured dimensions, observation
layout, joint order, poses, or action scale violate this contract.

## Observation Layout

| Indices | Values |
| --- | --- |
| 0:3 | `base_lin_vel = [0, 0, 0]` exactly |
| 3:6 | IMU body angular velocity |
| 6:9 | IMU projected gravity in body frame |
| 9:12 | command `[vx, vy, yaw_rate]` |
| 12:24 | `joint_pos - q_default` |
| 24:36 | joint velocity |
| 36:48 | previous policy action |

Real or estimated base linear velocity is never supplied to this policy.
`policy_contract.force_zero_base_linear_velocity` must remain `true`.

## Joint And Action Order

1. `BL_hip_joint`
2. `BR_hip_joint`
3. `FL_hip_joint`
4. `FR_hip_joint`
5. `BL_thigh_joint`
6. `BR_thigh_joint`
7. `FL_thigh_joint`
8. `FR_thigh_joint`
9. `BL_calf_joint`
10. `BR_calf_joint`
11. `FL_calf_joint`
12. `FR_calf_joint`

The joint target conversion is exactly:

```text
q_target = q_default + policy_action_scale * action
```

`policy_action_scale` is `0.25`. The target then passes through joint position
and per-step rate safety limits before MIT command packing.

## Joint Feedback Units

The policy receives joint-space radians, never raw MIT motor radians. Live
feedback follows this path:

```text
MIT position -> subtract joint offset -> apply joint direction -> wrap 2*pi
             -> q_current joint radians -> observation[12:24] - q_default
```

Joint velocity is also transformed by the configured joint direction before it
enters observation indices `24:36`. Raw position, velocity, and torque remain in
the feedback dictionary as diagnostic fields such as `position_raw`; they are
not exposed by the policy joint-state API.

## Command Convention

The shared convention is configured in `config/joystick.yaml`:

- `vx > 0`: forward; `vx < 0`: backward
- `vy > 0`: left; `vy < 0`: right
- `yaw > 0`: left/CCW; `yaw < 0`: right/CW
- Keyboard: `W/S` control `+X/-X`, `A/D` control `+Y/-Y`, and `Q/E`
  control positive/negative yaw. Direction keys may be combined.

Joystick, keyboard, main controller, and checker all read the same lateral
sign field: `command_convention.vy_left_positive`.

## Motion Assists

Pure policy deployment is the default. `gait_assist` and `imu_posture` are
disabled in `config/motion_assist.yaml`. Enabling either prints a warning
because it changes the learned action before safety filtering.

## Safe Validation

Run from the repository root:

```bash
export PYTHONPATH="$PWD/src"
/usr/bin/python3 src/check_policy_limits.py
/usr/bin/python3 src/main_controller.py \
  --dry-run-policy-check \
  --policy-path policy/policy.pt
```

Both commands are hardware-free. The one-shot dry run creates no CAN object and
performs no CAN write.

## Motor-Disabled Loop

This runs the normal policy loop and prints all 12 MIT packets without opening
CAN:

```bash
/usr/bin/python3 src/main_controller.py \
  --motors-disabled \
  --policy-path policy/policy.pt \
  --feedback-source fake \
  --imu-source fake \
  --command-source fixed \
  --vx 0.20 --vy 0.0 --yaw 0.0 \
  --start-control-mode policy \
  --initial-zero-frame stand \
  --policy-steps 100 \
  --policy-obs-check-every 10 \
  --log-every 10 \
  --no-gait-assist \
  --no-imu-stabilization
```

## First Real Walk

Only run this after the policy checker, IMU test, CAN routing check, and all 12
motor feedback checks pass. Secure or suspend the robot for the first run.
Adjust the three exported device paths to match the target computer.

```bash
export FRONT_CAN=/dev/ttyUSB1
export BACK_CAN=/dev/ttyUSB2
export IMU_PORT=/dev/ttyUSB0

/usr/bin/python3 src/main_controller.py \
  --mode mit-signal \
  --can-count 2 \
  --can-ports "$FRONT_CAN" "$BACK_CAN" \
  --feedback-source mit \
  --imu-source xsens \
  --imu-port "$IMU_PORT" \
  --command-source keyboard \
  --base-lin-vel-source zero \
  --start-control-mode idle \
  --startup-action hold \
  --initial-zero-frame crouch \
  --no-auto-zero-on-startup \
  --auto-stand-zero \
  --speed-scale-initial 0.15 \
  --speed-scale-min 0.10 \
  --speed-scale-max 0.30 \
  --policy-action-clip 0.0 \
  --policy-action-smoothing 0.0 \
  --policy-obs-check-every 25 \
  --no-gait-assist \
  --no-imu-stabilization \
  --policy-steps 0 \
  --log-every 25
```

Start in the configured crouch pose, press Space to stand, wait for stand zero
calibration to complete, then hold a movement key. Walking remains blocked if
the IMU is not live, MIT feedback is missing or stale, motor fault bits are
nonzero, projected gravity is unsafe, or the stand zero frame is incomplete.
Missing or stale MIT feedback enters `FEEDBACK HOLD`: policy and pose motion are
cancelled and the last safe target is maintained without disabling the motors.
After feedback recovers, release the movement keys once to re-arm motion.
Measured motor fault bits, impossible encoder angles, and body safety faults
remain emergency-stop conditions.
