from timing_scheduler import DeadlineScheduler, timing_qualification_passed


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleep_calls = []

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += float(seconds)

    def sleep(self, seconds):
        seconds = max(0.0, float(seconds))
        self.sleep_calls.append(seconds)
        self.advance(seconds)


def run_work_cycles(scheduler, clock, work_times):
    snapshots = []
    for work_s in work_times:
        cycle_start = clock()
        clock.advance(work_s)
        snapshots.append(scheduler.finish_cycle(cycle_start))
    return snapshots


def test_one_time_transition_delay_resynchronizes_without_fault():
    clock = FakeClock()
    warnings = []
    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=clock.sleep,
        warning_callback=warnings.append,
    )

    snapshots = run_work_cycles(scheduler, clock, [0.080] + [0.005] * 100)

    assert snapshots[0].resynchronized is True
    assert snapshots[0].total_missed_deadlines > 0
    assert snapshots[-1].consecutive_work_overruns == 0
    assert not any(snapshot.timing_fault for snapshot in snapshots)
    assert len(warnings) == 1


def test_sustained_work_overload_faults_at_threshold():
    clock = FakeClock()
    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=clock.sleep,
        warning_callback=lambda _message: None,
    )

    snapshots = run_work_cycles(scheduler, clock, [0.025] * 30)

    fault_indices = [
        index for index, snapshot in enumerate(snapshots)
        if snapshot.timing_fault
    ]
    assert fault_indices
    assert fault_indices[0] >= 24
    assert snapshots[fault_indices[0]].consecutive_work_overruns >= 25


def test_minor_wakeup_lateness_does_not_count_as_work_overrun():
    clock = FakeClock()

    def jitter_sleep(seconds):
        clock.sleep(seconds)
        clock.advance(0.0008)

    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=jitter_sleep,
    )

    snapshots = run_work_cycles(scheduler, clock, [0.018] * 100)

    assert not any(snapshot.timing_fault for snapshot in snapshots)
    assert snapshots[-1].consecutive_work_overruns == 0


def test_mode_transition_resync_request_continues_without_fault():
    clock = FakeClock()
    warnings = []
    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=clock.sleep,
        warning_callback=warnings.append,
    )

    scheduler.request_resync("mode transition initialized")
    snapshots = run_work_cycles(scheduler, clock, [0.060] + [0.006] * 20)

    assert snapshots[0].resynchronized is True
    assert snapshots[-1].consecutive_work_overruns == 0
    assert not any(snapshot.timing_fault for snapshot in snapshots)
    assert len(warnings) == 1


def test_real_workload_overrun_is_still_a_timing_fault():
    clock = FakeClock()
    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=clock.sleep,
    )

    snapshots = run_work_cycles(scheduler, clock, [0.0225] * 26)

    assert snapshots[-1].timing_fault
    assert snapshots[-1].consecutive_work_overruns >= 25


def test_policy_cycle_breakdown_fits_twenty_millisecond_budget():
    clock = FakeClock()
    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=clock.sleep,
    )

    cycle_start = clock()
    clock.advance(0.003)  # policy inference
    clock.advance(0.001)  # CAN transmit
    clock.advance(0.002)  # steady feedback drain
    snapshot = scheduler.finish_cycle(cycle_start)

    assert abs(snapshot.cycle_work_s - 0.006) < 1e-12
    assert snapshot.consecutive_work_overruns == 0
    assert not snapshot.timing_fault


def test_repeated_policy_budget_cycles_do_not_trip_watchdog():
    clock = FakeClock()
    scheduler = DeadlineScheduler(
        dt_s=0.02,
        deadline_tolerance_s=0.001,
        deadline_resync_s=0.050,
        timing_fault_consecutive=25,
        clock=clock,
        sleep=clock.sleep,
    )

    snapshots = run_work_cycles(scheduler, clock, [0.018] * 200)

    assert snapshots[-1].consecutive_work_overruns == 0
    assert not any(snapshot.timing_fault for snapshot in snapshots)


def test_timing_qualification_allows_sparse_resync_misses():
    assert timing_qualification_passed(8, 786, consecutive_work_overruns=0)
    assert not timing_qualification_passed(20, 786, consecutive_work_overruns=0)
    assert not timing_qualification_passed(0, 786, consecutive_work_overruns=1)
