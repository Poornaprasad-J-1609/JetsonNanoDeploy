#!/usr/bin/env python3
import errno
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CanFrame:
    can_id: int
    data: bytes
    timestamp: float


def serial_encoded_ext_id(can_id: int) -> bytes:
    """
    AT-command USB-CAN extended-frame encoding.
    Intended for CH340 / RobStride-style AT USB-CAN adapter.

    This is only the serial packet layer.
    """
    serial_id = ((can_id & 0x1FFFFFFF) << 3) | 0x04
    return serial_id.to_bytes(4, "big")


def make_at_packet(can_id: int, data: bytes = b"") -> bytes:
    if len(data) > 8:
        raise ValueError("CAN data length must be <= 8 bytes")
    return b"AT" + serial_encoded_ext_id(can_id) + bytes([len(data)]) + data + b"\r\n"


class ATUsbCan:
    requires_frame_gap = True

    def __init__(self, port="/dev/ttyUSB0", baud=921600, timeout=0.02):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.rx_buffer = bytearray()

    def open(self):
        import serial

        self.ser = serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        time.sleep(0.1)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def send_raw(self, can_id: int, data: bytes = b""):
        if self.ser is None:
            raise RuntimeError("Serial port is not open")

        pkt = make_at_packet(can_id, data)
        self.ser.write(pkt)
        self.ser.flush()
        return pkt

    def send_raw_batch(self, frames):
        """Write multiple complete AT packets with one USB serial flush."""
        if self.ser is None:
            raise RuntimeError("Serial port is not open")

        packets = [make_at_packet(can_id, data) for can_id, data in frames]
        if not packets:
            return []
        payload = b"".join(packets)
        written = self.ser.write(payload)
        self.ser.flush()
        if written != len(payload):
            raise IOError(
                f"Short USB-CAN batch write on {self.port}: "
                f"wrote {written}/{len(payload)} bytes"
            )
        return packets

    def send_raw_sequence(self, frames, frame_gap_s=0.0):
        """Write separately paced AT packets, then flush the serial port once."""
        if self.ser is None:
            raise RuntimeError("Serial port is not open")

        packets = [make_at_packet(can_id, data) for can_id, data in frames]
        for index, packet in enumerate(packets):
            written = self.ser.write(packet)
            if written != len(packet):
                raise IOError(
                    f"Short USB-CAN packet write on {self.port}: "
                    f"wrote {written}/{len(packet)} bytes"
                )
            if frame_gap_s > 0.0 and index + 1 < len(packets):
                time.sleep(float(frame_gap_s))
        if packets:
            self.ser.flush()
        return packets

    def _pop_frames_from_rx_buffer(self):
        frames = []

        while True:
            start = self.rx_buffer.find(b"AT")
            if start < 0:
                if self.rx_buffer.endswith(b"A"):
                    self.rx_buffer[:] = b"A"
                else:
                    self.rx_buffer.clear()
                break

            if start > 0:
                del self.rx_buffer[:start]

            header_len = 7  # "AT" + 4-byte serial ID + 1-byte DLC
            if len(self.rx_buffer) < header_len:
                break

            dlc = self.rx_buffer[6]
            if dlc > 8:
                del self.rx_buffer[:2]
                continue

            packet_len = header_len + dlc + 2
            if len(self.rx_buffer) < packet_len:
                break

            if self.rx_buffer[packet_len - 2:packet_len] != b"\r\n":
                del self.rx_buffer[:2]
                continue

            serial_id = int.from_bytes(self.rx_buffer[2:6], "big")
            can_id = (serial_id >> 3) & 0x1FFFFFFF
            data = bytes(self.rx_buffer[7:7 + dlc])
            del self.rx_buffer[:packet_len]
            frames.append(CanFrame(can_id=can_id, data=data, timestamp=time.monotonic()))

        return frames

    @staticmethod
    def _feedback_motor_id(can_id, feedback_comm_types):
        if not feedback_comm_types:
            return None
        comm_type = (int(can_id) >> 24) & 0x1F
        if comm_type not in feedback_comm_types:
            return None
        return ((int(can_id) >> 8) & 0xFFFF) & 0xFF

    def read_available_frames(
        self,
        timeout=0.0,
        max_frames=256,
        expected_motor_ids=None,
        feedback_comm_types=None,
    ):
        """
        Read AT-encoded CAN frames currently available from the adapter.

        Expected packet format matches make_at_packet():
            b"AT" + 4-byte encoded extended ID + 1-byte DLC + data + b"\\r\\n"
        """
        if self.ser is None:
            raise RuntimeError("Serial port is not open")

        frames = []
        expected_motor_ids = {
            int(motor_id) & 0xFF
            for motor_id in (expected_motor_ids or [])
        }
        feedback_comm_types = {
            int(comm_type) & 0x1F
            for comm_type in (feedback_comm_types or [])
        }
        received_expected = set()
        deadline = time.monotonic() + float(timeout)

        while len(frames) < max_frames:
            waiting = getattr(self.ser, "in_waiting", 0)
            should_wait = timeout > 0.0 and time.monotonic() < deadline
            read_size = waiting if waiting > 0 else (512 if should_wait else 0)

            if read_size <= 0:
                break

            chunk = self.ser.read(read_size)
            if chunk:
                self.rx_buffer.extend(chunk)
                new_frames = self._pop_frames_from_rx_buffer()
                frames.extend(new_frames)
                if expected_motor_ids:
                    for frame in new_frames:
                        motor_id = self._feedback_motor_id(
                            frame.can_id,
                            feedback_comm_types,
                        )
                        if motor_id in expected_motor_ids:
                            received_expected.add(motor_id)
                    if expected_motor_ids.issubset(received_expected):
                        break
            elif not should_wait:
                break

        return frames[:max_frames]

    def send_signal_frame(self, motor_id: int):
        """
        Harmless signal test frame.

        This only tests that the USB-CAN serial adapter transmits bytes.
        Motors do not need to be connected.

        This is NOT a RobStride position command.
        """
        can_id = int(motor_id) & 0x1FFFFFFF
        data = b""
        return self.send_raw(can_id, data)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SocketCan:
    """RobStride official SDK transport over a Linux SocketCAN channel."""

    # The 1 ms frame gap belongs to the old serial-AT packet adapter. SocketCAN
    # already serializes frames at the configured CAN bitrate; retaining that
    # userspace gap spreads one 12-motor target over most of a 50 Hz cycle.
    requires_frame_gap = False

    def __init__(
        self,
        channel="can0",
        bitrate=1_000_000,
        timeout=0.02,
        tx_retry_count=4,
        tx_retry_delay=0.0005,
    ):
        self.channel = str(channel)
        self.bitrate = int(bitrate)
        self.timeout = float(timeout)
        self.tx_retry_count = int(max(0, tx_retry_count))
        self.tx_retry_delay = float(max(0.0, tx_retry_delay))
        self.driver = None
        self.bus = None
        self.tx_queue_len = None
        self.last_sequence_duration_s = 0.0
        self.max_sequence_duration_s = 0.0
        self._last_tx_stall_warning_s = -float("inf")

    def open(self):
        from robstride_dynamics import RobstrideBus

        tx_queue_path = Path("/sys/class/net") / self.channel / "tx_queue_len"
        try:
            tx_queue_len = int(tx_queue_path.read_text().strip())
        except (OSError, ValueError):
            tx_queue_len = None
        self.tx_queue_len = tx_queue_len
        if tx_queue_len is not None and tx_queue_len < 16:
            print(
                f"WARNING: {self.channel} txqueuelen={tx_queue_len} can reject "
                "a complete 12-motor command set. Before motor control run: "
                f"sudo ip link set {self.channel} txqueuelen 32"
            )
        elif tx_queue_len is not None and tx_queue_len > 64:
            print(
                f"WARNING: {self.channel} txqueuelen={tx_queue_len} permits stale "
                "motor targets to accumulate and can cause delayed, stepped motion. "
                f"Before motor control run: sudo ip link set {self.channel} "
                "txqueuelen 32"
            )

        self.driver = RobstrideBus(
            channel=self.channel,
            motors={},
            bitrate=self.bitrate,
        )
        self.driver.connect(handshake=False)
        self.bus = self.driver.channel_handler
        return self

    def close(self):
        if self.driver is not None:
            self.driver.disconnect(disable_torque=False)
        elif self.bus is not None:
            self.bus.shutdown()
        self.driver = None
        self.bus = None

    @staticmethod
    def _is_tx_queue_full(exc):
        code = getattr(exc, "error_code", None)
        return code == errno.ENOBUFS or "No buffer space available" in str(exc)

    def send_raw(self, can_id: int, data: bytes = b""):
        if self.bus is None:
            raise RuntimeError("SocketCAN channel is not open")
        if len(data) > 8:
            raise ValueError("CAN data length must be <= 8 bytes")

        import can

        message = can.Message(
            arbitration_id=int(can_id),
            data=bytes(data),
            is_extended_id=True,
        )
        for attempt in range(self.tx_retry_count + 1):
            try:
                # RobstrideBus.transmit() uses a zero-timeout python-can send,
                # which intermittently raises ENOBUFS under a 12-motor burst.
                # Use its connected official SocketCAN channel with a bounded
                # wait so the current command set is submitted or fails now;
                # it is never stored in an unbounded Python retry queue.
                self.bus.send(message, timeout=self.timeout)
                return int(can_id), bytes(data)
            except Exception as exc:
                if not self._is_tx_queue_full(exc) or attempt >= self.tx_retry_count:
                    raise
                time.sleep(self.tx_retry_delay)

    def send_raw_batch(self, frames):
        return [self.send_raw(can_id, data) for can_id, data in frames]

    def send_raw_sequence(self, frames, frame_gap_s=0.0):
        frames = list(frames)
        sent = []
        started = time.monotonic()
        for index, (can_id, data) in enumerate(frames):
            sent.append(self.send_raw(can_id, data))
            if frame_gap_s > 0.0 and index + 1 < len(frames):
                deadline = started + (index + 1) * float(frame_gap_s)
                time.sleep(max(0.0, deadline - time.monotonic()))
        self.last_sequence_duration_s = time.monotonic() - started
        self.max_sequence_duration_s = max(
            self.max_sequence_duration_s,
            self.last_sequence_duration_s,
        )
        expected_s = max(0.002, len(frames) * 0.00025)
        now = time.monotonic()
        if (
            self.last_sequence_duration_s > max(0.050, 4.0 * expected_s)
            and now - self._last_tx_stall_warning_s >= 1.0
        ):
            self._last_tx_stall_warning_s = now
            print(
                f"WARNING: {self.channel} SocketCAN command submission took "
                f"{1000.0 * self.last_sequence_duration_s:.1f} ms; check "
                "txqueuelen, bitrate, adapter load, and CAN termination."
            )
        return sent

    @staticmethod
    def _feedback_motor_id(can_id, feedback_comm_types):
        if not feedback_comm_types:
            return None
        comm_type = (int(can_id) >> 24) & 0x1F
        if comm_type not in feedback_comm_types:
            return None
        return ((int(can_id) >> 8) & 0xFFFF) & 0xFF

    def read_available_frames(
        self,
        timeout=0.0,
        max_frames=256,
        expected_motor_ids=None,
        feedback_comm_types=None,
    ):
        if self.bus is None:
            raise RuntimeError("SocketCAN channel is not open")

        frames = []
        expected_motor_ids = {
            int(motor_id) & 0xFF
            for motor_id in (expected_motor_ids or [])
        }
        feedback_comm_types = {
            int(comm_type) & 0x1F
            for comm_type in (feedback_comm_types or [])
        }
        received_expected = set()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while len(frames) < int(max_frames):
            remaining = max(0.0, deadline - time.monotonic())
            wait = remaining if timeout > 0.0 else 0.0
            message = self.bus.recv(timeout=wait)
            if message is None:
                break
            if not message.is_extended_id:
                continue
            frames.append(CanFrame(
                can_id=int(message.arbitration_id),
                data=bytes(message.data),
                timestamp=time.monotonic(),
            ))
            if expected_motor_ids:
                motor_id = self._feedback_motor_id(
                    message.arbitration_id,
                    feedback_comm_types,
                )
                if motor_id in expected_motor_ids:
                    received_expected.add(motor_id)
                if expected_motor_ids.issubset(received_expected):
                    break
            if timeout > 0.0 and time.monotonic() >= deadline:
                break
        return frames

    def send_signal_frame(self, motor_id: int):
        return self.send_raw(int(motor_id) & 0x1FFFFFFF, b"")

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    print("AT packet example:")
    print(make_at_packet(0x01, b"").hex())
