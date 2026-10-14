"""Separate IK replies without editing the vendor SDK."""

import queue
import struct
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SDK_PATH = (
    ROOT / "vendor/NexArm/src/driver/"
    "ros_robot_controller/ros_robot_controller"
)
sys.path.insert(0, str(SDK_PATH))

from ros_robot_controller_sdk import (
    Board as VendorBoard,
    ESP32Cmd,
)


class Board(VendorBoard):
    ik_timeout = 2.0

    def __init__(self, *args, **kwargs):
        # Initialise before the vendor starts its receiver thread.
        self._ik_replies = queue.Queue(maxsize=64)
        self._recent_rx = deque(maxlen=12)
        super().__init__(*args, **kwargs)

    def dispatch_packet(self, pid, cmd, data):
        self._recent_rx.append((
            time.monotonic(),
            int(pid),
            int(cmd),
            len(data),
        ))

        if (
            pid in (0xFF, 0x5A)
            and cmd == ESP32Cmd.IKINE_RESULT_GET
        ):
            try:
                self._ik_replies.put_nowait(bytes(data))
            except queue.Full:
                pass

            # Keep IK results out of the shared state queue.
            return

        # Preserve vendor handling for all other messages.
        super().dispatch_packet(pid, cmd, data)

    def set_arm_coords(
        self,
        x, y, z, pitch,
        roll=0,
        claw=0,
        time_ms=1000,
        calc_only=False,
    ):
        if not calc_only:
            return super().set_arm_coords(
                x, y, z, pitch, roll, claw, time_ms,
                calc_only=False,
            )

        if not self.enable_recv:
            raise RuntimeError(
                "Enable reception before calculating IK"
            )

        with self.request_lock:
            while True:
                try:
                    self._ik_replies.get_nowait()
                except queue.Empty:
                    break

            payload = struct.pack(
                "<hhhhhhh",
                int(pitch * 10),
                int(x),
                int(y),
                int(z),
                int(roll),
                int(claw),
                int(time_ms),
            )

            start = time.monotonic()
            self.buf_write(
                0xFF,
                ESP32Cmd.IKINE_RESULT_GET,
                payload,
            )

            try:
                data = self._ik_replies.get(
                    timeout=self.ik_timeout
                )
            except queue.Empty:
                observed = [
                    {
                        "id": pid,
                        "cmd": cmd,
                        "bytes": length,
                    }
                    for timestamp, pid, cmd, length
                    in list(self._recent_rx)
                    if timestamp >= start
                ]

                raise RuntimeError(
                    "No command-7 IK reply received. "
                    f"Packets seen while waiting: {observed}. "
                    "No movement requested."
                ) from None

            if len(data) < 24:
                raise RuntimeError(
                    f"Command-7 IK reply has {len(data)} bytes; "
                    "expected at least 24. "
                    "No movement requested."
                )

            return self._decode_pose_servos(data)