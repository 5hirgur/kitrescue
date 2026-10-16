"""Reusable NexArm Cartesian control and interactive grasp setup. No ROS required."""

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[2]
SDK_PATH = ROOT / "vendor/NexArm/src/driver/ros_robot_controller/ros_robot_controller"
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
FIELDS = ("x", "y", "z", "pitch", "roll", "claw")


class NexArm:
    """An agent can later call plan(), execute() and nudge() on one connection."""

    def __init__(self, port=DEFAULT_PORT, board=None):
        if board is None:
            sys.path.insert(0, str(SDK_PATH))
            from nexarm_transport import Board
            board = Board(device=port, baudrate=1_000_000)
            time.sleep(8)
        self.board = board
        self.board.enable_reception(True)
        self.claw_target = None
        self.last_target = None
        self.state()

    def state(self):
        for _ in range(5):
            result = self.board.get_full_state()
            if result is not None:
                if not all(k in result for k in (*FIELDS, "servos")):
                    raise RuntimeError("Incomplete robot state.")
                if len(result["servos"]) != 6:
                    raise RuntimeError("Expected six servo values.")
                return result
            time.sleep(0.3)
        raise RuntimeError("NexArm did not return its state.")

    def show_state(self):
        print("Reported state:")
        pprint(self.state())
        method = getattr(self.board, "get_real_tcp_pose", None)
        if method is not None:
            print("TCP calculated from measured joints, if supported:")
            pprint(method())

    def solve_ik(self, target, attempts=4):
        """Reject mismatched replies; every request remains calculation-only."""
        tolerances = {
            "x": 1.0, "y": 1.0, "z": 1.0,
            "pitch": 0.11, "roll": 1.0, "claw": 1.0,
        }
        last_reply = None
        for attempt in range(1, attempts + 1):
            reply = self.board.set_arm_coords(**target, calc_only=True)
            last_reply = reply
            if isinstance(reply, dict):
                servos = reply.get("servos")
                if isinstance(servos, (list, tuple)) and len(servos) == 6:
                    try:
                        matches = all(
                            math.isfinite(float(reply[key]))
                            and abs(float(reply[key]) - target[key]) <= tolerance
                            for key, tolerance in tolerances.items()
                        )
                    except (KeyError, TypeError, ValueError, OverflowError):
                        matches = False
                    if matches:
                        return reply
            print(f"IK reply {attempt}/{attempts} did not match the target; ignored.")
            if attempt < attempts:
                time.sleep(0.2)
        raise RuntimeError(
            f"No matching IK reply after {attempts} calculation-only requests. "
            f"Target: {target}. Last reply: {last_reply}. No movement requested."
        )

    def plan(self, values, relative=False, duration_ms=2000):
        if not 500 <= duration_ms <= 10000:
            raise ValueError("duration_ms must be 500–10000.")
        if not values or set(values) - set(FIELDS):
            raise ValueError("Use x, y, z, pitch, roll and/or claw.")
        before = self.state()
        target = {k: before[k] for k in FIELDS}
        # Preserve an intentional grip command during later arm movements.
        if self.claw_target is not None:
            target["claw"] = self.claw_target
        for key, value in values.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Coordinates and angles must be finite.")
            target[key] = target[key] + value if relative else value
        for key in FIELDS:
            if key == "pitch":
                target[key] = round(float(target[key]), 1)
            else:
                if not float(target[key]).is_integer():
                    raise ValueError(f"{key} must use whole SDK units.")
                target[key] = int(target[key])
            encoded = target[key] * 10 if key == "pitch" else target[key]
            if not -32768 <= encoded <= 32767:
                raise ValueError(f"{key} exceeds the serial field range.")
        target["time_ms"] = int(duration_ms)
        solution = self.solve_ik(target)
        if not solution or len(solution.get("servos", [])) != 6:
            raise RuntimeError("No complete IK solution; movement cancelled.")
        # Reject firmware adjustments to the requested Cartesian pose.
        for key in ("x", "y", "z", "pitch"):
            if key in solution and abs(solution[key] - target[key]) > 2:
                raise RuntimeError(
                    f"IK adjusted {key}: requested {target[key]}, returned {solution[key]}."
                )
        return {"before": before, "target": target, "solution": solution}


    def execute(self, plan, monitor=None):
        current = self.state()

        for key in ("x", "y", "z", "pitch", "roll"):
            if abs(current[key] - plan["before"][key]) > 3:
                raise RuntimeError(
                    "Arm pose changed after planning; calculate again."
                )

        target = plan["target"]

        if monitor is not None:
            monitor(current)

        self.board.set_arm_coords(**target, calc_only=False)
        self.claw_target = target["claw"]
        self.last_target = dict(target)

        deadline = (
            time.monotonic()
            + target["time_ms"] / 1000
            + 0.7
        )

        if monitor is None:
            time.sleep(max(0, deadline - time.monotonic()))
            return self.state()

        # Prevents further waypoints, not a hardware emergency stop.
        state = current

        while time.monotonic() < deadline:
            time.sleep(min(
                .1, max(0, deadline - time.monotonic())
            ))
            state = self.state()
            monitor(state)

        return state

    def nudge(self, values, duration_ms=2000):
        return self.plan(values, relative=True, duration_ms=duration_ms)

    def save_pose(self, name, path=ROOT / "data/taught_poses.json"):
        path = Path(path)
        saved = json.loads(path.read_text()) if path.exists() else {"poses": {}}
        saved["poses"][name] = {
            "recorded_at_ns": time.time_ns(),
            "reported_state": self.state(),
            "last_requested_target": self.last_target,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"Saved {name} in {path}")

    def close(self):
        self.board.enable_reception(False)
        port = getattr(self.board, "port", None)
        if port is not None:
            port.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def display_plan(plan):
    print("Before:")
    pprint(plan["before"])
    print("Requested target:")
    pprint(plan["target"])
    print("IK solution:")
    pprint(plan["solution"])
    print("Servo differences:")
    pprint([b-a for a, b in zip(
        plan["before"]["servos"], plan["solution"]["servos"]
    )])


def confirm_and_execute(arm, plan):
    display_plan(plan)
    if input("Hands clear. Type MOVE to execute: ").strip() != "MOVE":
        print("Cancelled.")
        return
    print("Reported state after movement:")
    pprint(arm.execute(plan))


def assignments(tokens):
    values = {}
    for token in tokens:
        key, value = token.split("=", 1)
        if key not in FIELDS or key in values:
            raise ValueError(f"Invalid or duplicate field: {key}")
        values[key] = float(value)
    return values


def jog(arm, duration_ms):
    print("Commands:")
    print("  state                       read robot state")
    print("  preview pitch=-90 roll=0    calculate only; no movement")
    print("  nudge claw=5                add 5 SDK gripper units")
    print("  nudge pitch=-10             decrease pitch by 10 degrees")
    print("  nudge z=-5                  lower reported Z by 5 mm")
    print("  move pitch=-90 roll=0       absolute target; asks for MOVE")
    print("  save NAME                   record the current reported pose")
    print("  quit")
    arm.show_state()
    while True:
        try:
            tokens = shlex.split(input("nexarm> "))
            if not tokens:
                continue
            command = tokens[0]
            if command in ("quit", "exit"):
                return
            if command == "state":
                arm.show_state()
            elif command == "save" and len(tokens) == 2:
                arm.save_pose(tokens[1])
            elif command in ("preview", "move", "nudge"):
                values = assignments(tokens[1:])
                plan = arm.plan(values, relative=command == "nudge",
                                duration_ms=duration_ms)
                if command == "preview":
                    display_plan(plan)
                else:
                    confirm_and_execute(arm, plan)
            else:
                print("Unknown command. Use state, preview, move, nudge, save or quit.")
        except (ValueError, RuntimeError) as exc:
            print(f"Stopped: {exc}")
        except (EOFError, KeyboardInterrupt):
            print("\nLeaving jog mode.")
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--jog", action="store_true")
    parser.add_argument("--state", action="store_true")
    parser.add_argument("--duration-ms", type=int, default=2000)
    parser.add_argument("--execute", action="store_true")
    for key in FIELDS:
        parser.add_argument("--" + key, type=float)
        parser.add_argument("--d" + key, type=float)
    args = parser.parse_args()
    if not 500 <= args.duration_ms <= 10000:
        parser.error("--duration-ms must be 500–10000")
    absolute = {k: getattr(args, k) for k in FIELDS if getattr(args, k) is not None}
    relative = {k: getattr(args, "d"+k) for k in FIELDS if getattr(args, "d"+k) is not None}
    if absolute and relative:
        parser.error("Use absolute flags or delta flags, not both.")
    if (args.jog or args.state) and (absolute or relative):
        parser.error("Use --jog/--state without coordinate flags.")
    with NexArm(args.port) as arm:
        if args.jog:
            jog(arm, args.duration_ms)
        elif args.state or not (absolute or relative):
            arm.show_state()
        else:
            plan = arm.plan(absolute or relative, relative=bool(relative),
                            duration_ms=args.duration_ms)
            if args.execute:
                confirm_and_execute(arm, plan)
            else:
                display_plan(plan)
                print("Calculation only. Add --execute to request movement.")


if __name__ == "__main__":
    main()