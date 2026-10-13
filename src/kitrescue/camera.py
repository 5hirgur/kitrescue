"""Offline ESP32 JPEG capture over USB serial."""

import argparse
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import serial


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    sequence: int
    received_at_ns: int
    capture_ms: float

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.jpeg)
        return path

    def as_image(self):
        """Decode into an OpenCV BGR image when OpenCV is installed."""
        import cv2
        import numpy as np

        image = cv2.imdecode(
            np.frombuffer(self.jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError("JPEG decoding failed.")
        return image


class Camera:
    def __init__(self, port, baudrate=921600, timeout=10):
        self.timeout = timeout
        self.serial = serial.Serial(
            port,
            baudrate,
            timeout=0.1,
            write_timeout=2,
            exclusive=True,
        )

        try:
            # Opening serial can reset the ESP32.
            time.sleep(3)
            self.serial.reset_input_buffer()

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.serial.write(b"PING\n")
                response = self.serial.readline().decode(
                    "ascii", errors="replace"
                ).strip()

                if response.startswith("KRCAM/1 READY"):
                    self.status = response
                    break
                if response.startswith("KRCAM/1 ERROR"):
                    raise RuntimeError(response)
                time.sleep(0.2)
            else:
                raise TimeoutError("Camera did not answer PING.")
        except Exception:
            self.close()
            raise

    def _read_exact(self, length, deadline):
        data = bytearray()
        while len(data) < length:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Received {len(data)} of {length} expected bytes."
                )
            data.extend(self.serial.read(length - len(data)))
        return bytes(data)

    def capture(self):
        """Request a frame after scene motion has settled."""
        self.serial.reset_input_buffer()
        start = time.monotonic()
        deadline = start + self.timeout

        self.serial.write(b"CAPTURE\n")
        self.serial.flush()

        # Skip any startup/debug text preceding the frame.
        window = bytearray()
        while bytes(window) != b"KRC1":
            window.extend(self._read_exact(1, deadline))
            if len(window) > 4:
                del window[0]

        sequence, length, checksum = struct.unpack(
            "<III", self._read_exact(12, deadline)
        )

        if length == 0:
            raise RuntimeError("ESP32 failed to capture an image.")
        if length > 1_000_000:
            raise RuntimeError(f"Invalid JPEG length: {length}")

        jpeg = self._read_exact(length, deadline)

        if zlib.crc32(jpeg) != checksum:
            raise RuntimeError("Image CRC failed; retry capture.")
        if not jpeg.startswith(b"\xff\xd8"):
            raise RuntimeError("Received data is not a JPEG.")

        return Frame(
            jpeg=jpeg,
            sequence=sequence,
            received_at_ns=time.time_ns(),
            capture_ms=(time.monotonic() - start) * 1000,
        )

    def close(self):
        self.serial.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--output", default="data/camera/test.jpg")
    args = parser.parse_args()

    with Camera(args.port, args.baud) as camera:
        print(camera.status)
        frame = camera.capture()
        path = frame.save(args.output)
        print(
            f"Saved {path}: sequence={frame.sequence}, "
            f"bytes={len(frame.jpeg)}, time={frame.capture_ms:.0f} ms"
        )


if __name__ == "__main__":
    main()