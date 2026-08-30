# Grallator MEVIUS-Style Controller

This worktree isolates the simpler motor-facing controller used during the
earlier MEVIUS-style Grallator tests.

It intentionally keeps Grallator's trained policy contract:

- 48 observations and the validated Grallator joint order
- `q_target = q_policy_reference + 0.25 * raw_action`
- motor signs applied downstream by `motor_directions.yaml`
- Grallator's configured joint gains and physical safety limits
- 50 Hz policy loop and 200 Hz CAN command stream

The MEVIUS-style launch profile removes routine action EMA and per-step delta
limiting after the entry blend. It does not substitute MEVIUS2's policy,
default pose, `0.2` action scale, or symmetry vector.

Standing never starts policy inference automatically. After stand settles,
press an explicit movement key (`W/A/S/D/Q/E`) to request policy takeover.
