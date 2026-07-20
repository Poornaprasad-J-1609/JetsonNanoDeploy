# Patch Report: slcan Keyboard Policy Deployment

Branch: `fix/slcan-keyboard-policy-gait`

Commit message to use: `fix: stabilize slcan keyboard policy deployment`

## Scope

This patch repairs the one-CAN SocketCAN deployment path for the Grallator
Jetson controller while preserving the learned RL policy contract.

The active walking path is still the learned actor, not a scripted gait.
Four-bar compensation remains disabled for walking.

## Modified Files

- `src/can_topology.py`
- `src/joystick_interface.py`
- `src/main_controller.py`
- `scripts/validate_policy_gait.py`
- `tests/conftest.py`
- `tests/test_deployment_contract.py`
- `PATCH_REPORT.md`

## Key Fixes

- Default deployment topology is now one SocketCAN interface:
  `--can-count 1`, `--port slcan0`, `--can-backend socketcan`.
- `--command-source` now defaults to `keyboard`.
- `--control-hz 0` keeps the trained 50 Hz policy rate.
- `--feedback-source auto` resolves to MIT feedback in `mit-signal` mode.
- SocketCAN preflight checks `/sys/class/net/<iface>`, state, MTU,
  tx queue length, and BUS-OFF status before motors are enabled.
- SocketCAN no longer inherits serial-AT frame pacing assumptions.
- Duplicate motor-ID checks use physical CAN interface identity.
- One-CAN mode routes all active joints to logical bus `can0`.
- Four-bar transmission is rejected at startup if enabled, because this
  deployment contract requires a 1:1 motor/joint walking path.
- Keyboard source identity was added so keyboard-only grace/timeout logic
  cannot affect joystick or fixed-command modes.
- Keyboard speed-scale changes print the effective clipped command maximum.
- The policy loop now separates `previous_raw_action` from
  `previous_sent_action`.
- Observation slots `obs[36:48]` receive previous raw actor output.
- Exact-policy mode is enabled by default. During policy entry it blends
  joint targets from the last safe stand target to the actor target; after
  entry the sent action equals the raw actor action unless verified safety
  limits intervene.
- Gait-assist and direct IMU posture overlays are skipped in exact-policy mode.
- Policy clipping and torque-limiter counts are logged per joint.
- CSV run logs include command line, runtime control frequency, action scale,
  policy hash, joint order, CAN topology, backend, and timing diagnostics.
- CSV Git push is opt-in. `--auto-push-log` now defaults to false.
- The control loop uses a monotonic deadline scheduler and logs missed
  deadlines, consecutive overruns, and maximum overrun.
- Stale or missing live motor feedback freezes/falls back to the latest safe
  target instead of sending policy commands from stale state.

## Policy Contract Preserved

- Policy file: `policy/policy.pt`
- SHA256: `965e94c4cebfc45b9ef609d4a677a5ee35961a895700aad478603f43844b3779`
- Observation dimension: 48
- Action dimension: 12
- Action scale: 0.25
- Control period: 0.02 s
- `obs[0:3]`: literal zero base linear velocity
- `obs[3:6]`: IMU/body gyro
- `obs[6:9]`: projected gravity
- `obs[9:12]`: final keyboard velocity command
- `obs[12:24]`: joint position observation
- `obs[24:36]`: joint velocity observation
- `obs[36:48]`: previous raw actor action

## Validation Results

Passed:

```bash
python3 -m compileall -q src scripts tests
```

Passed:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
for path in sorted(Path("config").glob("*.yaml")):
    with path.open("r", encoding="utf-8") as stream:
        yaml.safe_load(stream)
    print("YAML OK:", path)
PY
```

Could not run in this workstation environment:

```bash
python3 -m pytest -q
```

Reason:

```text
/home/poornaprasad/miniconda3/bin/python3: No module named pytest
```

Passed:

```bash
git diff --check
```

Passed no-motor controller dry run:

```bash
python3 src/main_controller.py --mode print --command-source keyboard --feedback-source fake --imu-source fake --control-hz 0 --policy-steps 300 --no-gait-assist --no-imu-stabilization --no-auto-push-log
```

Observed stable 50 Hz idle loop and no CAN traffic in `--mode print`.

## Policy Gait Replay Summary

Forward replay:

```bash
python3 scripts/validate_policy_gait.py --vx 0.35 --duration 8 --entry-seconds 2
```

Result: PASS.

- Steady steps: 277
- Max raw/sent action error after entry: 0
- Policy target clipping: 0.00% for all joints
- Torque limiter: 0.36% for `FL_calf_joint`, 0.00% for all other joints
- Peak-to-peak joint target by leg:
  - FR: 0.2448 rad
  - FL: 0.2714 rad
  - BR: 0.1610 rad
  - BL: 0.2068 rad

Additional dry replays passed:

- Backward: `--vx -0.35`
- Left: `--vx 0.0 --vy 0.25`
- Right: `--vx 0.0 --vy -0.25`

The lateral replay target motion is much smaller than forward/backward. That is
a policy/log characteristic, not a handwritten limiter.

## Simulation Log Replay Check

Existing `logs_sim` policy observation logs were replayed through the current
deployment policy loader.

- `grallator_sim_policy_input_log_20260718_162351.csv`:
  rows 1615, max action error `1.43051e-06`, max logged action `7.767`,
  max command `0.500`
- `policy_io_50hz.csv`:
  rows 1617, max action error `1.43051e-06`, max logged action `7.767`
- `grallator_sim_policy_input_log_20260713_124106.csv`:
  rows 456, max action error `1.43051e-06`, max logged action `5.736`,
  max command `0.800`

This confirms the deployed policy loader, observation slots, and action output
match the simulation logs to floating-point tolerance.

Simulation motor torque envelope from available logs:

- Max applied torque: `130.000 Nm`
- Max computed preclip torque: `203.422 Nm`

The no-hardware replay does not measure physical torque; it only checks whether
the deployment torque limiter would rewrite targets.

## Remaining Hardware Assumptions

- `slcan0` is already created and attached by `slcand` or `slcan_attach`.
- `slcan0` is not BUS-OFF and has correct CAN bitrate configured externally.
- Run before motor control:

```bash
sudo ip link set slcan0 txqueuelen 32
```

- All 12 RS04 motor IDs are unique on the single physical CAN bus.
- Xsens IMU is on its own serial device and is not the CAN adapter.
- `config/four_bar_transmission.yaml` stays `enabled: false`.
- Hardware stand zero, motor directions, encoder offsets, and mechanical
  limits must already be physically verified.

## Exact Dry-Run Command

```bash
cd ~/JetsonNanoDeploy
export PYTHONPATH="$PWD/src"

python3 src/main_controller.py \
  --mode print \
  --command-source keyboard \
  --feedback-source fake \
  --imu-source fake \
  --control-hz 0 \
  --policy-steps 300 \
  --no-gait-assist \
  --no-imu-stabilization \
  --no-auto-push-log
```

## Exact Suspended-Hardware Command

```bash
cd ~/JetsonNanoDeploy
export PYTHONPATH="$PWD/src"
export CAN0=slcan0
export IMU_PORT=/dev/ttyUSB0
sudo ip link set "$CAN0" txqueuelen 32

python3 src/main_controller.py \
  --mode mit-signal \
  --can-backend socketcan \
  --can-count 1 \
  --port "$CAN0" \
  --command-source keyboard \
  --feedback-source mit \
  --imu-source xsens \
  --imu-port "$IMU_PORT" \
  --control-hz 0 \
  --start-control-mode idle \
  --startup-action hold \
  --initial-zero-frame stand \
  --no-gait-assist \
  --no-imu-stabilization \
  --no-stand-policy-stabilization \
  --no-auto-push-log \
  --joint-debug
```

Expected sequence:

1. Secure or suspend the robot.
2. Start the controller.
3. Confirm SocketCAN preflight and 12/12 fresh MIT feedback.
4. Confirm IMU projected gravity is near `[0, 0, -1]`.
5. Press Space to stand.
6. Wait for walking to arm.
7. Hold `w` to enter learned forward gait.
8. Use arrow-up slowly if more command speed is needed.
9. Release `w` to return smoothly to stand.
10. Press `x` for emergency stop.

## Full Unified Diff

Before the commit:

```bash
git diff -- src/can_topology.py src/joystick_interface.py src/main_controller.py scripts/validate_policy_gait.py tests/conftest.py tests/test_deployment_contract.py PATCH_REPORT.md
```

After the commit:

```bash
git show --format=fuller --patch HEAD
```

Do not claim real gait is proven until the suspended hardware test and then
ground walking validation have both been completed.

## Timing Scheduler Repair Addendum

Commit message: `fix: distinguish timing backlog from sustained control overruns`

Additional files changed:

- `src/timing_scheduler.py`
- `tests/test_timing_scheduler.py`
- `src/motor_command_layer.py`

Fix:

- Replaced the old lateness-based consecutive watchdog with a deadline
  scheduler that separates current-cycle workload from accumulated deadline
  backlog.
- New timing metrics:
  - `loop_period_ms`
  - `cycle_work_ms`
  - `deadline_lateness_ms`
  - `missed_deadlines_total`
  - `consecutive_work_overruns`
  - `scheduler_resync_count`
  - `max_cycle_work_ms`
  - `max_lateness_ms`
- A one-time slow transition cycle can now resynchronize the scheduler without
  creating a false 25-cycle timing fault.
- Sustained workload overruns still trip the timing fault after the configured
  consecutive count.
- Added CLI options:
  - `--deadline-tolerance-ms`
  - `--deadline-resync-ms`
  - `--timing-fault-consecutive`
- Added separate pose torque override:
  - `--pose-pd-torque-limit`
- `--policy-pd-torque-limit` remains policy walking only.
- `--pose-pd-torque-limit` covers the pose command phases used by sit, stand,
  and hold without changing the YAML defaults unless explicitly supplied.

Timing validation:

- Direct deterministic scheduler tests passed for:
  - one-time 80 ms transition delay followed by 100 normal 5 ms cycles
  - sustained 25 ms workload overload
  - minor wakeup lateness below the work budget
  - explicit mode-transition resync
  - true repeated workload overrun
- 30-second fake-device controller timing run passed at 50 Hz with no false
  timing fault.
- Forward policy gait replay still passes with raw/sent action error after
  entry equal to 0.

`pytest` still could not run in this workstation environment because the active
Python does not have the `pytest` package installed.

## SocketCAN Feedback Pipeline Addendum

Commit message: `fix: pipeline socketcan feedback within policy deadline`

Fix:

- The runtime loop now drains queued CAN feedback nonblocking at the beginning
  of each cycle, uses that fresh cached state for the current policy
  observation, sends one 12-motor command set, then performs only a short
  ordinary post-command drain.
- Added `--steady-feedback-budget-ms` with default `1.5` and validation range
  `0.0..5.0`. This does not replace `--feedback-timeout`; the longer timeout
  remains for startup, calibration, hold capture, and stale/missing feedback
  recovery.
- SocketCAN and serial-AT frame readers can now return early once feedback from
  all expected active motor IDs has arrived.
- MIT feedback state now tracks command send timestamps, feedback timestamps,
  current-cycle feedback, previous-cycle fresh feedback, stale feedback, and
  missing feedback.
- Added timing log fields for `imu_read_ms`, `policy_inference_ms`,
  `command_build_ms`, `can_tx_ms`, `pre_feedback_read_ms`,
  `steady_feedback_read_ms`, `safety_check_ms`, `logging_ms`, and
  `cycle_work_ms`.
- Timing-fault console output now includes the final overloaded cycle
  breakdown.

Preserved:

- Policy file/hash, 48-observation layout, action scale, joint order, motor
  IDs, directions, offsets, Kp/Kd, torque limits, joint limits, four-bar
  disabled state, exact-policy-after-entry behavior, one-CAN SocketCAN
  topology, and the 50 Hz trained control rate.

Validation:

- `python3 -m compileall -q src tests` passed.
- Direct feedback-pipeline and timing test harness passed.
- No-hardware controller dry run passed at 50 Hz.
- `python3 scripts/validate_policy_gait.py --vx 0.35 --duration 8
  --entry-seconds 2` passed with raw/sent action error after entry equal to 0.
- `git diff --check` passed.
- `pytest -q` could not run in this workstation environment because `pytest`
  is not installed.
