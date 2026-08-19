# RS04 Actuator Characterization Bench

This application is an isolated single-actuator tool for RobStride RS04 bench
testing, system identification, and PD gain exploration. It is independent of
the quadruped policy controller.

The actuator loop runs in a dedicated Python thread at a requested 200 Hz. The
Tk GUI runs at about 25 Hz and only consumes immutable snapshots. Linux/Python
is not hard real time, so the application measures and logs actual loop timing,
jitter, lateness, and missed cycles.

## Safety

- Rigidly mount the actuator and clear the full configured travel envelope.
- Start in `--mock` mode.
- Verify motor ID, SocketCAN interface, hard travel limits, and torque limit.
- Keep a physical power disconnect accessible.
- `CONNECT` does not enable torque. `ENABLE MOTOR` first captures feedback and
  holds the measured position to avoid an initial target jump.
- Any configured position, velocity, torque, current, temperature, feedback,
  communication, motor-fault, or timing violation stops the experiment and
  sends the RS04 stop frame.
- Free-decay explicitly disables impedance control and requires an operator
  confirmation. Only use it when the mechanism remains mechanically safe.

## Dependencies

Run the application from the local laptop repository. The Jetson is not used
for this single-actuator bench test.

```bash
cd /home/poornaprasad/Work/WORKSPACE/Issacsim/isaaclab/GRALLATOR_DEPLOY_FIX_WORKTREE
source /home/poornaprasad/Work/Motor_control/.venv/bin/activate
sudo apt install python3-tk
python3 -m pip install -r requirements.txt
```

The existing `/home/poornaprasad/Work/Motor_control/.venv` on this laptop was
verified to contain Tkinter, Matplotlib, python-can, robstride-dynamics,
NumPy, and PyYAML. Activate it for the commands below. The base Conda Python
currently lacks Matplotlib and cannot open the complete GUI.

The key packages are `numpy`, `PyYAML`, `python-can`,
`robstride-dynamics==0.0.1`, and `matplotlib`. Signal filtering and robust
least-squares identification are implemented with NumPy, so SciPy is optional.

## Configuration

Edit [`config.yaml`](config.yaml), or pass a separate YAML with `--config`.
All controller calculations use SI units: radians, rad/s, Nm, A, seconds, and
kg m2.

Important settings include motor ID/interface/bitrate, the fixed 200 Hz motor
rate, gain bounds, position/velocity/torque/current/temperature limits,
feedback timeout, pendulum geometry, filter parameters, and the log directory.

## Mock mode

```bash
cd /home/poornaprasad/Work/WORKSPACE/Issacsim/isaaclab/GRALLATOR_DEPLOY_FIX_WORKTREE
source /home/poornaprasad/Work/Motor_control/.venv/bin/activate
export PYTHONPATH="$PWD"
python3 -m rs04_bench --mock
```

Headless end-to-end validation:

```bash
python3 -m rs04_bench --mock --no-gui --demo-step
```

The mock plant includes inertia, viscous damping, Coulomb friction, command
delay, torque saturation, and a simple thermal state.

## Hardware setup

This reuses the repository's validated SocketCAN implementation and official
`robstride_dynamics` RS04 MIT ranges. Connect the USB-CAN adapter directly to
the laptop. Its setup must already have created a Linux SocketCAN interface
such as `slcan0` or `can0`; verify the actual name before launching:

```bash
ip -br link | grep -E 'can|slcan'
```

At the time this README was updated, this laptop had no CAN/serial adapter
attached, so no interface could be selected automatically. Once the adapter
is attached and `slcan0` exists, use:

```bash
cd /home/poornaprasad/Work/WORKSPACE/Issacsim/isaaclab/GRALLATOR_DEPLOY_FIX_WORKTREE
source /home/poornaprasad/Work/Motor_control/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src"
sudo ip link set slcan0 txqueuelen 32
python3 -m rs04_bench --interface slcan0 --motor-id 1 --bitrate 1000000
```

If the laptop creates `can0`, replace both occurrences of `slcan0` with
`can0`. Do not pass `/dev/ttyUSB0` to `--interface`; this application uses the
repository's SocketCAN transport. Creating an SLCAN interface from a serial
USB-CAN adapter is adapter-specific and must be completed before starting the
bench application.

The direct command is exactly the motor's MIT impedance command:

```text
q_des, qd_des, Kp, Kd, tau_ff
```

No second software PD controller is sent as feedforward torque. The displayed
`tau_commanded_nm` is the diagnostic value
`Kp*(q_des-q) + Kd*(qd_des-qd) + tau_ff`; the motor executes the transmitted
impedance fields internally.

RS04 MIT feedback in this repository exposes position, velocity, torque,
temperature, mode, and fault bits. It does not expose motor current or bus
voltage. Those CSV/GUI fields remain blank and metadata records them as
unavailable. Torque feedback is labeled `rs04_mit_feedback`, independently of
commanded torque. If a future current sensor is added, current-derived torque
is stored in `tau_estimated_nm` only when `motor.torque_constant_nm_per_a` is
configured; the metadata records that calibration and formula.

## GUI workflow

1. Click `CONNECT`. The motor remains disabled.
2. Verify live position and limits.
3. Click `ENABLE MOTOR`; the measured position becomes the initial target.
4. Edit Kp/Kd and use `Update Kp`, `Update Kd`, or `Update Both`.
5. Use the target field, `+`/`-`, or left/right arrow keys for manual movement.
   The angle gauge is also an input: click or drag its orange target pointer.
   Set `Speed [rad/s]` to command a constant-speed ramp toward that target.
   Positive/clockwise dial motion commands increasing motor angle; negative/
   anticlockwise motion commands decreasing motor angle.
6. `HOLD` captures the current measured angle as a constant target.
7. Use the Experiments tab for step, linear/log chirp, and explicit free decay.
8. `DISABLE MOTOR` performs a controlled stop. `EMERGENCY STOP` immediately
   stops the experiment and sends the motor stop command.

## Step test

Specify initial position, step amplitude, pre-step hold and post-step duration.
The exact target discontinuity and all samples are timestamped at 200 Hz. On
completion the application computes steady-state error, 10-90% rise time,
peak/overshoot, 2% settling time, oscillation frequency when peaks are usable,
damping ratio only for meaningful underdamped overshoot, RMS error, maxima,
and command-response delay when cross-correlation is credible.

## Chirp test

The linear and logarithmic chirps use analytical phase and desired velocity.
Keep amplitude small enough that the complete center +/- amplitude envelope is
inside configured travel limits. Chirps generally provide better excitation
for plant identification than one step.

## Pendulum / free decay

Configure point mass, COM radius, lever mass/COM radius, known lever inertia,
gravity, and angle convention in YAML. `downward_vertical` uses
`tau_g = m*g*r*sin(q)`; `horizontal` uses `tau_g = m*g*r*cos(q)`.

The known load inertia is `J_lever + m*r^2`. Total fitted inertia is not mass:
it is the rotational coefficient in `J*qdd`.

## Logging

Every active experiment creates:

```text
logs/rs04_bench/rs04_step_YYYY-MM-DD_HH-MM-SS.csv
logs/rs04_bench/rs04_step_YYYY-MM-DD_HH-MM-SS_metadata.json
```

CSV rows contain wall/monotonic/experiment time, actual timing and frequency,
mode, desired/measured position and velocity, errors, gains, feedforward,
commanded/measured/estimated torque as separate fields, current/voltage/
temperature when available, enable state, safety/event markers, experiment ID,
feedback age, fault bits, and missed-cycle information. A bounded background
writer owns disk I/O; queue overflow is a safety fault rather than silent loss.

Metadata stores motor/interface/rate, experiment parameters, pendulum data,
safety limits, signal availability, notes, and Git commit.

## Offline analysis

```bash
python3 -m rs04_bench --analyze logs/rs04_bench/rs04_chirp_....csv
```

The identifier fits:

```text
tau - tau_g(q) = J*qdd + b*qd + tau_c*sign(qd)
```

Encoder position is resampled at its median sample interval and processed by a
local polynomial (Savitzky-Golay-style) least-squares filter. Velocity and
acceleration are analytical derivatives of that local polynomial; position is
not naively differentiated twice. A Huber iteratively reweighted regression
estimates `J`, `b`, and Coulomb friction and reports RMSE, R2, standard errors,
sample count, torque source and warnings.

The estimate is rejected when there are too few moving samples, velocity
excitation is insufficient, inertia is non-positive, damping is negative, or
fit quality is poor. If only commanded torque exists, output explicitly warns
that saturation and delay can bias the fit.

Given valid `J` and `b`, the application reports the model-based starting
estimate:

```text
Kd = 2*zeta*sqrt(J*Kp) - b
```

This is not a perfect gain. Validate it with a small step because delays,
saturation, compliance, nonlinear friction and discrete-time effects remain.

## Gain sweep

`experiments/gain_sweep.py` provides a sequential safe sweep API. It returns
to neutral, waits for measured settling, executes identical small steps,
analyzes each CSV, penalizes RMS error, settling, overshoot, current and torque
saturation, and returns several candidate gains. Any safety event, timeout or
excessive saturation aborts the entire sweep and disables the actuator. Review
the candidates; the lowest numerical cost is not declared uniquely safe.
