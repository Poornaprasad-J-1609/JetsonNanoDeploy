#!/usr/bin/env python3
"""Atomic two-rate CAN command retransmission for the hardware controller."""

import math
import threading
import time


class CanCommandStreamer:
    """Retransmit only the newest command snapshot at a fixed CAN rate.

    The policy producer replaces one immutable snapshot at 50 Hz. The sender
    never queues historical targets, so transport delay cannot replay stale
    gait states after a newer policy target is available.
    """

    def __init__(
        self,
        send_callback,
        receive_callback=None,
        command_dt_s=0.005,
        stale_timeout_s=0.080,
        fault_consecutive_overruns=3,
        transport_label="CAN",
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.send_callback = send_callback
        self.receive_callback = receive_callback
        self.command_dt_s = float(command_dt_s)
        self.stale_timeout_s = float(stale_timeout_s)
        self.fault_consecutive_overruns = int(fault_consecutive_overruns)
        self.transport_label = str(transport_label).strip() or "CAN"
        if not math.isfinite(self.command_dt_s) or self.command_dt_s <= 0.0:
            raise ValueError("command_dt_s must be finite and > 0")
        if not math.isfinite(self.stale_timeout_s) or self.stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be finite and > 0")
        if self.stale_timeout_s < 2.0 * self.command_dt_s:
            raise ValueError("stale_timeout_s must be at least two CAN command periods")
        if self.fault_consecutive_overruns <= 0:
            raise ValueError("fault_consecutive_overruns must be > 0")

        self.clock = clock
        self.sleep = sleep
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._io_idle = threading.Event()
        self._io_idle.set()
        self._thread = None
        self._commands = ()
        self._generation = 0
        self._published_at = None
        self._fault_reason = None
        self._last_batch_duration_s = 0.0
        self._maximum_batch_duration_s = 0.0
        self._last_send_timestamp = None
        self._send_count = 0
        self._missed_deadlines = 0
        self._consecutive_overruns = 0
        self._stale_target_events = 0
        self._last_scheduler_lateness_s = 0.0
        self._maximum_scheduler_lateness_s = 0.0
        self._received_items = []
        self._receive_count = 0
        self._last_receive_duration_s = 0.0
        self._maximum_receive_duration_s = 0.0

    @staticmethod
    def _freeze_commands(commands):
        return tuple(dict(command) for command in commands)

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="grallator-can-200hz",
                daemon=True,
            )
            self._thread.start()

    def submit(self, commands, timestamp=None):
        frozen = self._freeze_commands(commands)
        now = self.clock() if timestamp is None else float(timestamp)
        with self._lock:
            self._commands = frozen
            self._generation += 1
            self._published_at = now if frozen else None
        self._wake.set()
        return self._generation

    def clear(self):
        generation = self.submit(())
        self._io_idle.wait(timeout=max(0.020, 2.0 * self.command_dt_s))
        return generation

    def stop(self, timeout=1.0):
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        self._thread = None
        self.clear()

    @property
    def fault_reason(self):
        with self._lock:
            return self._fault_reason

    @property
    def last_send_timestamp(self):
        with self._lock:
            return self._last_send_timestamp

    @property
    def has_active_commands(self):
        with self._lock:
            return bool(self._commands) and self._fault_reason is None

    def drain_received(self):
        """Return feedback collected by the CAN owner thread exactly once."""
        with self._lock:
            items = self._received_items
            self._received_items = []
        return items

    def _set_fault(self, reason):
        with self._lock:
            if self._fault_reason is None:
                self._fault_reason = str(reason)
            self._commands = ()
            self._published_at = None

    def _snapshot(self):
        with self._lock:
            return self._commands, self._published_at, self._fault_reason

    def _run(self):
        next_deadline = self.clock()
        while not self._stop.is_set():
            commands, published_at, fault = self._snapshot()
            if fault is not None:
                self._wake.wait(timeout=self.command_dt_s)
                self._wake.clear()
                continue
            if not commands:
                self._wake.wait(timeout=self.command_dt_s)
                self._wake.clear()
                next_deadline = self.clock()
                continue

            now = self.clock()
            if now < next_deadline:
                self.sleep(next_deadline - now)
                if self._stop.is_set():
                    break

            woke_at = self.clock()
            scheduler_lateness = max(0.0, woke_at - next_deadline)
            with self._lock:
                self._last_scheduler_lateness_s = scheduler_lateness
                self._maximum_scheduler_lateness_s = max(
                    self._maximum_scheduler_lateness_s,
                    scheduler_lateness,
                )

            # A producer update may arrive while this thread is sleeping. Read
            # the atomic slot again immediately before transmission so a
            # superseded policy target is never sent from a local stale copy.
            commands, published_at, fault = self._snapshot()
            if fault is not None:
                continue
            if not commands:
                next_deadline = self.clock()
                continue
            now = self.clock()
            if published_at is None or now - published_at > self.stale_timeout_s:
                with self._lock:
                    self._stale_target_events += 1
                age = 0.0 if published_at is None else now - float(published_at)
                self._set_fault(
                    "CAN command target became stale: "
                    f"age={age:.3f}s limit={self.stale_timeout_s:.3f}s"
                )
                continue

            send_started = self.clock()
            self._io_idle.clear()
            try:
                self.send_callback(commands)
            except Exception as exc:
                self._set_fault(f"CAN command send failed: {exc}")
                self._io_idle.set()
                continue
            send_finished = self.clock()
            duration = max(0.0, send_finished - send_started)
            with self._lock:
                self._last_batch_duration_s = duration
                self._maximum_batch_duration_s = max(
                    self._maximum_batch_duration_s,
                    duration,
                )
                self._last_send_timestamp = send_finished
                self._send_count += 1
                if duration > self.command_dt_s:
                    self._missed_deadlines += max(
                        1,
                        int(math.ceil(duration / self.command_dt_s)) - 1,
                    )
                    self._consecutive_overruns += 1
                else:
                    self._consecutive_overruns = 0
                consecutive = self._consecutive_overruns
            if consecutive >= self.fault_consecutive_overruns:
                self._set_fault(
                    f"{self.transport_label} TRANSPORT DEADLINE FAILED: "
                    f"CAN batch took {1000.0 * duration:.2f} ms with a "
                    f"{1000.0 * self.command_dt_s:.2f} ms deadline for "
                    f"{consecutive} consecutive batches"
                )
                self._io_idle.set()
                continue

            if self.receive_callback is not None:
                receive_started = self.clock()
                try:
                    received = list(self.receive_callback() or ())
                except Exception as exc:
                    self._set_fault(f"CAN feedback receive failed: {exc}")
                    self._io_idle.set()
                    continue
                receive_finished = self.clock()
                receive_duration = max(0.0, receive_finished - receive_started)
                with self._lock:
                    if received:
                        self._received_items.extend(received)
                        # Four 200 Hz batches contain 48 replies per 50 Hz
                        # policy cycle. Keep ample diagnostic history while
                        # bounding memory if the producer is temporarily late.
                        if len(self._received_items) > 1024:
                            del self._received_items[:-1024]
                        self._receive_count += len(received)
                    self._last_receive_duration_s = receive_duration
                    self._maximum_receive_duration_s = max(
                        self._maximum_receive_duration_s,
                        receive_duration,
                    )
            self._io_idle.set()

            next_deadline += self.command_dt_s
            if send_finished > next_deadline:
                missed = int((send_finished - next_deadline) / self.command_dt_s) + 1
                next_deadline += missed * self.command_dt_s

    def telemetry(self):
        with self._lock:
            published_age = (
                None
                if self._published_at is None
                else max(0.0, self.clock() - self._published_at)
            )
            return {
                "can_command_dt_ms": 1000.0 * self.command_dt_s,
                "can_command_hz": 1.0 / self.command_dt_s,
                "can_command_generation": int(self._generation),
                "can_command_send_count": int(self._send_count),
                "can_command_last_batch_ms": 1000.0 * self._last_batch_duration_s,
                "can_command_max_batch_ms": 1000.0 * self._maximum_batch_duration_s,
                "can_command_missed_deadlines": int(self._missed_deadlines),
                "can_command_consecutive_overruns": int(self._consecutive_overruns),
                "can_command_scheduler_lateness_ms": (
                    1000.0 * self._last_scheduler_lateness_s
                ),
                "can_command_max_scheduler_lateness_ms": (
                    1000.0 * self._maximum_scheduler_lateness_s
                ),
                "can_command_stale_events": int(self._stale_target_events),
                "can_feedback_receive_count": int(self._receive_count),
                "can_feedback_last_drain_ms": (
                    1000.0 * self._last_receive_duration_s
                ),
                "can_feedback_max_drain_ms": (
                    1000.0 * self._maximum_receive_duration_s
                ),
                "can_command_target_age_ms": (
                    "" if published_age is None else 1000.0 * published_age
                ),
                "can_command_fault": "" if self._fault_reason is None else self._fault_reason,
            }
