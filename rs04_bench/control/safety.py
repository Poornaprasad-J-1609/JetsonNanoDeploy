from __future__ import annotations

import math


class SafetyMonitor:
    def __init__(self, config):
        self.config = config

    def check(self, state, command, feedback_age_s):
        if not self.config.min_position_rad <= command.q_des <= self.config.max_position_rad:
            return f"desired position exceeds safety limits: {command.q_des:+.4f} rad"
        if abs(command.qd_des) > self.config.max_velocity_rad_s:
            return f"desired velocity exceeds safety limit: {command.qd_des:+.3f} rad/s"
        if state is None:
            if feedback_age_s > min(self.config.feedback_timeout_s, self.config.communication_timeout_s):
                return f"motor feedback timeout ({1000.0 * feedback_age_s:.1f} ms)"
            return None
        values = (state.position, state.velocity)
        if not all(math.isfinite(float(v)) for v in values):
            return "non-finite position or velocity feedback"
        if not self.config.min_position_rad <= state.position <= self.config.max_position_rad:
            return f"position limit violated: {state.position:+.4f} rad"
        if abs(state.velocity) > self.config.max_velocity_rad_s:
            return f"velocity limit violated: {state.velocity:+.3f} rad/s"
        if state.torque_measured is not None and abs(state.torque_measured) > self.config.max_torque_nm:
            return f"measured torque limit violated: {state.torque_measured:+.2f} Nm"
        if state.current is not None and abs(state.current) > self.config.max_current_a:
            return f"current limit violated: {state.current:+.2f} A"
        if state.temperature is not None and state.temperature > self.config.max_temperature_c:
            return f"temperature limit violated: {state.temperature:.1f} C"
        if int(state.fault_bits):
            return f"motor fault bits 0x{int(state.fault_bits):02X}"
        estimated = command.kp * (command.q_des - state.position) + command.kd * (
            command.qd_des - state.velocity
        ) + command.tau_ff
        if abs(estimated) > self.config.max_torque_nm:
            return f"commanded impedance torque exceeds limit: {estimated:+.2f} Nm"
        return None
