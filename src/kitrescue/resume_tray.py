"""Finish a released tray transfer without repeating the pick."""
import copy
import json
import math
from pathlib import Path

import manipulation as m


def main():
    folder = Path("data/manipulation")
    plan = json.loads((folder / "plan.json").read_text())
    cal = json.loads(Path("config/robot_camera.json").read_text())

    if (
        plan.get("destination_kind") != "tray_floor"
        or len(plan.get("transfers", [])) != 1
    ):
        raise RuntimeError("Expected the saved single-cube tray transfer.")

    steps = plan["transfers"][0]["steps"]
    index = next(i for i, s in enumerate(steps)
                 if s["action"] == "retract")
    remaining = steps[index:]

    if [s["action"] for s in remaining] != [
        "retract", "return_to_park", "camera_clear"
    ]:
        raise RuntimeError("Unexpected remaining route.")

    release = next(s["pose"] for s in steps
                   if s["action"] == "release")

    if hasattr(m, "enable_run_logging"):
        m.enable_run_logging()

    with m.NexArm() as arm:
        robot = m.KitRescue(
            arm, None, cal, allow_unvalidated=True
        )
        robot.fast_motion = True
        state = arm.state()

        if (
            abs(state["claw"] + 80) > 5
            or abs(state["pitch"] - cal["reference"]["pitch"]) > 6
            or abs(state["roll"] - remaining[0]["pose"]["roll"]) > 5
            or math.dist(
                [state["x"], state["y"]],
                [release["x"], release["y"]],
            ) > 15
            or not release["z"] - 4 <= state["z"] <= remaining[0]["pose"]["z"] + 16
        ):
            raise RuntimeError(f"Unexpected recovery pose: {state}")

        lift = copy.deepcopy(remaining[0])
        lift["pose"].update(
            x=state["x"], y=state["y"],
            roll=state["roll"], claw=-80,
        )
        lift["minimum_z"] = None

        need_lift = state["z"] < lift["pose"]["z"]
        route = ([lift] if need_lift else []) + remaining[1:]

        for step in route:
            arm.plan(step["pose"], duration_ms=robot.duration)

        print("Current:", state)
        print("Lift vertically at current XY, then return to camera_clear.")
        input("Orange released, gripper empty, hands/routes clear; Enter: ")

        if need_lift:
            robot.move(lift)

        raised = arm.state()
        minimum = remaining[1]["minimum_z"]
        if raised["z"] < minimum:
            raise RuntimeError(
                f"Measured clearance insufficient: {raised['z']} < {minimum}"
            )

        for step in remaining[1:]:
            robot.move(step)

        robot.require_park()
        print("Retraction and camera_clear completed.")


if __name__ == "__main__":
    main()
