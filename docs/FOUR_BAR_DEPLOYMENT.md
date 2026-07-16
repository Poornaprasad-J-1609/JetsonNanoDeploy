# Nonlinear four-bar virtual-knee adapter

This adapter is prepared for the `walk-branch-jun18` deployment architecture.
It keeps the trained 48-observation policy in the original Isaac/URDF calf
coordinate and changes only the motor command and encoder-feedback boundaries.

## Implemented mapping

```text
q_k = f(theta_m)
J   = dq_k / dtheta_m

qdot_k = J * theta_dot_m

tau_k = efficiency * tau_m / J
tau_m = J * tau_k / efficiency

Kp_motor = J^2 * Kp_virtual / efficiency
Kd_motor = J^2 * Kd_virtual / efficiency
```

For joints without a configured transmission, the mapping remains identity.

## Files installed

```text
src/four_bar_transmission.py
src/four_bar_motor_command_layer.py
config/four_bar_transmission.yaml
tests/test_four_bar_transmission.py
```

The installer also makes two small edits:

- `main_controller.py` constructs `FourBarMotorCommandLayer` instead of the
  direct `MotorCommandLayer`.
- `state_estimator.py` converts raw motor position, velocity, and torque into
  virtual calf feedback before policy observations and safety checks use them.

## Calibration coordinate

Every `motor_angle_rad` sample must use:

```text
theta_m = motor_direction * (raw_encoder_position - joint_offset)
```

The shared `calf_common.knee_angle_rad` array is a **positive flexion
magnitude**. The YAML `virtual_sign` maps it into the repository's signed
coordinates:

```text
BL, FL: +1
BR, FR: -1
```

Using one shared profile assumes all four manufactured linkages are equivalent.
For final validation, separate profiles per leg are safer if measured curves
differ.

## Current repository range mismatch

On `walk-branch-jun18`, the calf hard limits reach `+/-1.70 rad`, while the
mechanism range previously supplied was `0..1.56 rad`. The calibration must
cover the full commanded range, or all calf limits and crouch/stand pose values
must be reduced to the measured safe range before enabling the adapter.

The adapter deliberately refuses extrapolation.

## Install

Extract the bundle, then run from any location:

```bash
cd ~/JetsonNanoDeploy
git checkout walk-branch-jun18

python3 /path/to/jetson_four_bar_adapter_bundle/apply_four_bar_adapter.py \
  --repo "$PWD"
```

The supplied configuration remains disabled after installation.

## Calibrate and enable

1. Record motor angle and independently measured knee angle through the full
   safe range.
2. Fill `config/four_bar_transmission.yaml` with strictly monotonic samples.
3. Keep the range on one assembly branch and away from the toggle.
4. Set a conservative `min_abs_jacobian`.
5. Start with measured efficiency set to `1.0`; reduce it after loaded tests.
6. Change `enabled: false` to `enabled: true` only after validation.

Validate before CAN use:

```bash
PYTHONPATH=src python3 -m unittest tests/test_four_bar_transmission.py
python3 -m py_compile \
  src/four_bar_transmission.py \
  src/four_bar_motor_command_layer.py \
  src/state_estimator.py \
  src/main_controller.py
```

## Fail-closed behavior

When enabled, the code rejects:

- missing or non-monotonic calibration samples;
- requests outside the measured range;
- a Jacobian below `min_abs_jacobian`;
- nonlinear commands without fresh motor feedback;
- transformed motor positions outside the MIT range.

By default, policy-only calf overtravel is clipped to the hard mechanical limit.

## Staged hardware validation

```text
motor disabled / manual linkage motion
-> suspended single leg with low gains
-> verify motor-angle/knee-angle direction and round trip
-> verify virtual knee velocity sign
-> verify torque estimate and 120 Nm motor clamp
-> default-pose hold
-> crouch-to-stand
-> commanded stand
-> slow stepping
-> locomotion
```

Log these added fields during validation:

```text
motor_position, motor_velocity, motor_torque
joint_position, joint_velocity, joint_torque
transmission_jacobian, transmission_efficiency
motor_position_des, motor_velocity_des
tau_motor_pd_est, motor_torque_limit_nm, motor_torque_limited
```
