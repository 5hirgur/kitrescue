"""Cartesian NexArm control using the unmodified vendor SDK."""

import argparse
import sys
import time
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[2]
SDK_PATH = (
    ROOT / "vendor/NexArm/src/driver/"
    "ros_robot_controller/ros_robot_controller"
)
sys.path.insert(0, str(SDK_PATH))

from ros_robot_controller_sdk import Board

DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"


def connect(port):
    board = Board(device=port, baudrate=1_000_000)

    # Additional startup wait proved necessary on our current hardware.
    time.sleep(8)
    board.enable_reception(True)

    for _ in range(5):
        state = board.get_full_state()
        if state is not None:
            return board, state
        time.sleep(0.5)

    raise RuntimeError("NexArm did not return its state.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--x", required=True, type=int, help="Millimetres")
    parser.add_argument("--y", required=True, type=int, help="Millimetres")
    parser.add_argument("--z", required=True, type=int, help="Millimetres")
    parser.add_argument("--pitch", type=float, help="Degrees; default: current")
    parser.add_argument("--roll", type=int, help="Default: current")
    parser.add_argument("--duration-ms", type=int, default=2000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not 500 <= args.duration_ms <= 10000:
        parser.error("--duration-ms must be between 500 and 10000")

    board, before = connect(args.port)

    target = {
        "x": args.x,
        "y": args.y,
        "z": args.z,
        "pitch": before["pitch"] if args.pitch is None else args.pitch,
        "roll": before["roll"] if args.roll is None else args.roll,
        "claw": before["claw"],
        "time_ms": args.duration_ms,
    }

    print("Current state:")
    pprint(before)
    print("\nRequested target:")
    pprint(target)

    solution = board.set_arm_coords(**target, calc_only=True)
    print("\nIK solution:")
    pprint(solution)

    if not solution or len(solution.get("servos", [])) != 6:
        raise RuntimeError("No complete IK solution; movement cancelled.")

    print("\nServo changes:")
    pprint([
        proposed - current
        for proposed, current in zip(solution["servos"], before["servos"])
    ])

    if not args.execute:
        print("\nCalculation only. Add --execute to request movement.")
        return

    answer = input("\nClear the movement area. Type MOVE to execute: ")
    if answer != "MOVE":
        print("Cancelled.")
        return

    board.set_arm_coords(**target, calc_only=False)
    time.sleep(args.duration_ms / 1000 + 1)

    print("\nReported state after movement:")
    pprint(board.get_full_state())

    print("\nTCP from measured servo positions, if supported:")
    pprint(board.get_real_tcp_pose())


if __name__ == "__main__":
    main()