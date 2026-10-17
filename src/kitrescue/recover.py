"""Guarded empty-gripper vertical recovery, then normal parking."""
import argparse
import math
import subprocess
import sys
from pathlib import Path

from ik import NexArm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()

with NexArm() as arm:
    origin = arm.state()
    print("Recovery start:", origin)

    if abs(origin["claw"] + 80) > 5:
        raise RuntimeError("Recovery requires an empty, open gripper.")
    if not -98 <= origin["pitch"] <= 12:
        raise RuntimeError("Pitch outside recovery range; no movement.")
    if not -90 <= origin["roll"] <= 90:
        raise RuntimeError("Wrist outside recovery range; no movement.")
    if origin["z"] < 5:
        raise RuntimeError("Below software floor; no movement.")

    if args.execute:
        input(
            "Gripper EMPTY, vertical path clear, power switch accessible. "
            "Enter once to lift and park: "
        )

    for index in range(8):
        before = arm.state()
        if before["z"] >= 100:
            break

        target = {
            "x": before["x"],
            "y": before["y"],
            "z": before["z"] + 20,
            "pitch": origin["pitch"],
            "roll": origin["roll"],
            "claw": -80,
        }
        plan = arm.plan(target, duration_ms=5000)
        solution = plan["solution"]

        for key in ("x", "y", "z", "pitch", "roll"):
            if abs(solution[key] - target[key]) > 1:
                raise RuntimeError("IK changed recovery target; stopped.")

        print("Vertical recovery target:", target)
        if not args.execute:
            print("Calculation only. Add --execute to lift and park.")
            raise SystemExit(0)

        def monitor(state):
            if not all(math.isfinite(float(state[k]))
                       for k in ("x", "y", "z", "pitch", "roll")):
                raise RuntimeError("Invalid feedback.")
            if state["z"] < before["z"] - 3:
                raise RuntimeError("Recovery moved downward; stopped.")
            if any(abs(state[k] - origin[k]) > 12 for k in ("x", "y")):
                raise RuntimeError("Recovery drifted sideways; stopped.")
            if abs(state["pitch"] - origin["pitch"]) > 6:
                raise RuntimeError("Recovery pitch deviation; stopped.")
            if abs(state["roll"] - origin["roll"]) > 5:
                raise RuntimeError("Recovery wrist deviation; stopped.")

        arm.execute_joints(plan, monitor=monitor)
        after = arm.state()
        monitor(after)
        rise = after["z"] - before["z"]
        print("Measured:", after, "Rise:", rise)

        if rise < 3:
            raise RuntimeError(
                "Insufficient upward progress. No retry or park sent."
            )
    else:
        raise RuntimeError("Recovery step limit reached; no park sent.")

    if not args.execute:
        print("Already raised. Add --execute to park.")
        raise SystemExit(0)

    if arm.state()["z"] < 100:
        raise RuntimeError("Measured clearance not reached; no park sent.")

# Close recovery's serial connection before opening the normal controller.
script = Path(__file__).with_name("manipulation.py")
subprocess.run(
    [sys.executable, str(script),
     "park", "--execute", "--one-confirm", "--fast"],
    check=True,
)
