#!/usr/bin/env python3
"""Deadline accounting for the 50 Hz deployment control loop."""

import math
import time
from dataclasses import dataclass


@dataclass
class TimingSnapshot:
    cycle_work_s: float = 0.0
    deadline_lateness_s: float = 0.0
    total_missed_deadlines: int = 0
    missed_deadlines_this_cycle: int = 0
    consecutive_work_overruns: int = 0
    maximum_work_s: float = 0.0
    maximum_lateness_s: float = 0.0
    scheduler_resync_count: int = 0
    resynchronized: bool = False
    timing_fault: bool = False
    work_overrun: bool = False


class DeadlineScheduler:
    """Absolute-deadline scheduler with separate backlog and workload counters."""

    def __init__(
        self,
        dt_s,
        deadline_tolerance_s=None,
        deadline_resync_s=None,
        timing_fault_consecutive=25,
        clock=time.monotonic,
        sleep=time.sleep,
        warning_callback=None,
    ):
        self.dt_s = float(dt_s)
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and > 0")
        default_tolerance = max(0.001, 0.05 * self.dt_s)
        default_resync = max(0.050, 2.0 * self.dt_s)
        self.deadline_tolerance_s = (
            default_tolerance
            if deadline_tolerance_s is None
            else float(deadline_tolerance_s)
        )
        self.deadline_resync_s = (
            default_resync
            if deadline_resync_s is None
            else float(deadline_resync_s)
        )
        self.timing_fault_consecutive = int(timing_fault_consecutive)
        if not math.isfinite(self.deadline_tolerance_s) or self.deadline_tolerance_s < 0.0:
            raise ValueError("deadline_tolerance_s must be finite and >= 0")
        if not math.isfinite(self.deadline_resync_s) or self.deadline_resync_s <= 0.0:
            raise ValueError("deadline_resync_s must be finite and > 0")
        if self.timing_fault_consecutive <= 0:
            raise ValueError("timing_fault_consecutive must be > 0")
        self.clock = clock
        self.sleep = sleep
        self.warning_callback = warning_callback
        self.next_deadline = float(self.clock()) + self.dt_s
        self.total_missed_deadlines = 0
        self.consecutive_work_overruns = 0
        self.maximum_work_s = 0.0
        self.maximum_lateness_s = 0.0
        self.scheduler_resync_count = 0
        self._pending_resync_reason = None
        self.last_snapshot = TimingSnapshot()

    def request_resync(self, reason):
        self._pending_resync_reason = str(reason or "one-time blocking operation")

    def _warn(self, message):
        if self.warning_callback is not None:
            self.warning_callback(message)

    def finish_cycle(self, cycle_start_s, work_end_s=None, do_sleep=True):
        work_end_s = float(self.clock()) if work_end_s is None else float(work_end_s)
        cycle_start_s = float(cycle_start_s)
        cycle_work_s = max(0.0, work_end_s - cycle_start_s)
        lateness_s = work_end_s - self.next_deadline
        positive_lateness_s = max(0.0, lateness_s)
        self.maximum_work_s = max(self.maximum_work_s, cycle_work_s)
        self.maximum_lateness_s = max(self.maximum_lateness_s, positive_lateness_s)

        work_overrun = cycle_work_s > self.dt_s + self.deadline_tolerance_s
        if work_overrun:
            self.consecutive_work_overruns += 1
        else:
            self.consecutive_work_overruns = 0

        missed = 0
        resynchronized = False
        pending_reason = self._pending_resync_reason
        self._pending_resync_reason = None

        if lateness_s > 0.0:
            missed = max(1, int(math.floor(lateness_s / self.dt_s)) + 1)
            self.total_missed_deadlines += missed

        should_resync = bool(
            positive_lateness_s >= self.deadline_resync_s
            or (
                pending_reason is not None
                and (positive_lateness_s > 0.0 or work_overrun)
            )
        )
        if should_resync:
            self.scheduler_resync_count += 1
            self.next_deadline = work_end_s + self.dt_s
            resynchronized = True
            # A single slow transition cycle must not poison the watchdog. If
            # multiple cycles have already exceeded the work budget, keep that
            # sustained-overload evidence so the timing safety can still trip.
            if self.consecutive_work_overruns <= 1:
                self.consecutive_work_overruns = 0
            if positive_lateness_s > 0.0 or pending_reason is not None:
                reason = pending_reason or "one-time lateness"
                self._warn(
                    "[TIMING] "
                    f"{reason}; lateness {1000.0 * positive_lateness_s:.1f} ms; "
                    f"missed {missed} deadlines; scheduler resynchronized"
                )
        elif lateness_s > 0.0:
            self.next_deadline += self.dt_s
        else:
            sleep_s = -lateness_s
            if sleep_s > 0.0 and do_sleep:
                self.sleep(sleep_s)
            self.next_deadline += self.dt_s

        timing_fault = self.consecutive_work_overruns >= self.timing_fault_consecutive
        self.last_snapshot = TimingSnapshot(
            cycle_work_s=cycle_work_s,
            deadline_lateness_s=positive_lateness_s,
            total_missed_deadlines=self.total_missed_deadlines,
            missed_deadlines_this_cycle=missed,
            consecutive_work_overruns=self.consecutive_work_overruns,
            maximum_work_s=self.maximum_work_s,
            maximum_lateness_s=self.maximum_lateness_s,
            scheduler_resync_count=self.scheduler_resync_count,
            resynchronized=resynchronized,
            timing_fault=timing_fault,
            work_overrun=work_overrun,
        )
        return self.last_snapshot
