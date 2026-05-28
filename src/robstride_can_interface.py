#!/usr/bin/env python3
import time
from dataclasses import dataclass


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

    def read_available_frames(self, timeout=0.0, max_frames=256):
        """
        Read AT-encoded CAN frames currently available from the adapter.

        Expected packet format matches make_at_packet():
            b"AT" + 4-byte encoded extended ID + 1-byte DLC + data + b"\\r\\n"
        """
        if self.ser is None:
            raise RuntimeError("Serial port is not open")

        frames = []
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
                frames.extend(self._pop_frames_from_rx_buffer())
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


if __name__ == "__main__":
    print("AT packet example:")
    print(make_at_packet(0x01, b"").hex())
