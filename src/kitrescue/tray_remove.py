"""Fresh vision-located tray removal."""
import argparse
import copy
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

import manipulation as m
from tray import detect_tray
from smooth_motion import Motion as SharedMotion


class TrayVision(m.Perception):
    def snapshot(self, preview=False):
        with self.lock:
            if self.paused or (
                preview and getattr(self, "_refreshing", False)
            ):
                if self.cached is None:
                    raise RuntimeError("Waiting for a planning image")
                return self.cached

            frame = self.camera.capture()
            image = m.prepare_image(frame, self.config)
            tray, overlay = detect_tray(image)
            self.tray_observation = tray
            records = []

            if tray["status"] == "detected":
                # Extend the search region only. Do not modify calibration.
                search = copy.deepcopy(self.cal)
                search["mapping"]["pixel_hull"] = tray["corners_px"]
                records, overlay = m.detect_top_faces(
                    image, self.profile, search
                )
                cv2.polylines(
                    overlay,
                    [np.rint(tray["corners_px"]).astype(np.int32)],
                    True, (0, 255, 0), 3,
                )

            hull = np.rint(
                self.cal["mapping"]["pixel_hull"]
            ).astype(np.int32)
            cv2.polylines(
                overlay, [hull], True, (255, 255, 0), 2
            )
            cv2.putText(
                overlay,
                f"Tray search green | approximate XY | tops "
                f"{self.source_top_height_mm:g} mm",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                .7, (0, 255, 255), 2,
            )
            self.cached = records, overlay, frame.received_at_ns
            return self.cached


class RemovalMotion(SharedMotion):
    def segment(
        self, target, label,
        maintenance=False, minimum=None, precise=False,
    ):
        self._removal_label = label
        return super().segment(
            target, label,
            maintenance=maintenance,
            minimum=minimum,
            precise=precise,
        )

    def travel_tolerance(self, key, before, target):
        label = getattr(self, "_removal_label", "")
        departing = (
            label.startswith("lift:")
            and target["claw"] == m.CLOSED
        )
        retracting = (
            label.startswith("retract:")
            and target["claw"] == m.OPEN
            and abs(before["claw"] - m.OPEN) <= 5
        )
        if (
            key == "z"
            and getattr(self.r, "fast_motion", False)
            and (departing or retracting)
        ):
            return 16
        return super().travel_tolerance(key, before, target)


class TrayRemoval(m.KitRescue):
    def controller(self):
        if not hasattr(self, "_motion_controller"):
            self._motion_controller = RemovalMotion(self)
        return self._motion_controller

    def xy(self, values):
        point = np.asarray(values, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Need finite XY")

        try:
            return super().xy(point)
        except ValueError:
            if not self.allow_extrapolation:
                raise ValueError(
                    "Tray source needs --allow-extrapolation"
                )

            tray = getattr(self.vision, "tray_observation", {})
            if tray.get("status") != "detected":
                raise ValueError("No clear tray")

            matrix = np.float64(
                self.cal["mapping"]["homography_pixel_to_command_xy"]
            )
            polygon = cv2.perspectiveTransform(
                np.float64(tray["corners_px"]).reshape(-1, 1, 2),
                matrix,
            ).reshape(-1, 2).astype(np.float32)

            rounded = np.rint(point).astype(int)
            for p in (point, rounded.astype(float)):
                if cv2.pointPolygonTest(
                    polygon, tuple(p), True
                ) < 30:
                    raise ValueError(
                        "Keep the source well away from tray walls"
                    )
            return rounded.tolist()

    def steps(self, source, destination, layer):
        if layer != 0:
            raise ValueError("One tray cube to table only")

        route = super().steps(source, destination, layer)
        for step in route:
            if step["action"] in ("grasp_height", "close"):
                step["pose"]["z"] = int(math.ceil(
                    self.base["z"] + self.floor_mm
                ))

            if step["minimum_z"] is not None:
                step["minimum_z"] += self.floor_mm
                step["pose"]["z"] = int(math.ceil(
                    step["pose"]["z"] + self.floor_mm
                ))
                if step["pose"]["z"] > self.park["z"]:
                    raise ValueError(
                        "Required travel exceeds camera_clear"
                    )
        return route


def finish_pending_retract(robot):
    """Use a saved route only to finish empty retraction, never to grasp."""
    if robot.controller().is_parked():
        return

    saved = json.loads(
        Path("data/manipulation/plan.json").read_text()
    )
    if (
        saved.get("destination_kind") != "tray_floor"
        or len(saved.get("transfers", [])) != 1
    ):
        raise RuntimeError("Reach camera_clear first")

    steps = saved["transfers"][0]["steps"]
    start = next(
        i for i, s in enumerate(steps)
        if s["action"] == "retract"
    )
    remaining = copy.deepcopy(steps[start:])

    if [s["action"] for s in remaining] != [
        "retract", "return_to_park", "camera_clear"
    ]:
        raise RuntimeError("Unexpected pending route")

    state = robot.arm.state()
    release = next(
        s["pose"] for s in steps if s["action"] == "release"
    )

    if (
        abs(state["claw"] + 80) > 5
        or abs(state["pitch"] - robot.base["pitch"]) > 6
        or abs(state["roll"] - remaining[0]["pose"]["roll"]) > 5
        or math.dist(
            [state["x"], state["y"]],
            [release["x"], release["y"]],
        ) > 15
        or not release["z"] - 4 <= state["z"] <= remaining[0]["pose"]["z"] + 16
    ):
        raise RuntimeError(
            "Unexpected start; reach camera_clear first"
        )

    lift = remaining[0]
    lift["pose"].update(
        x=state["x"], y=state["y"], roll=state["roll"]
    )
    lift["minimum_z"] = None
    route = (
        [lift] if state["z"] < lift["pose"]["z"] else []
    ) + remaining[1:]

    robot.preflight(route)
    with robot.vision.lock:
        robot.vision.paused = True
    try:
        for step in route:
            robot.move(step)
    finally:
        with robot.vision.lock:
            robot.vision.paused = False

    robot.require_park()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("colour", nargs="?", default="orange")
    parser.add_argument("--to", nargs=2, type=float, required=True)
    parser.add_argument("--floor-mm", type=float, default=8)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.floor_mm) or not 0 <= args.floor_mm < 20:
        parser.error("Floor height must be 0..<20 mm")

    config = m.load_config("config/camera.json")
    cal = json.loads(Path("config/robot_camera.json").read_text())
    profile = json.loads(Path("config/cube_colours.json").read_text())

    if config != cal["camera_config"]:
        parser.error("Camera settings differ from calibration")

    m.PAGE = re.sub(
        r"view\.naturalWidth\s*\*\s*Number\(zoom\.value\)",
        "view.parentElement.clientWidth * Number(zoom.value)",
        m.PAGE,
    )

    if hasattr(m, "enable_run_logging"):
        m.enable_run_logging()

    with m.open_camera(config) as camera:
        vision = TrayVision(camera, config, cal, profile)
        vision.source_top_height_mm = 30 + args.floor_mm
        vision.preview_full_frame = True
        viewer = m.WorkspaceView(vision, port=8766, zoom=1)

        try:
            with m.NexArm() as arm:
                robot = TrayRemoval(
                    arm, vision, cal,
                    allow_unvalidated=args.allow_unvalidated,
                )
                robot.fast_motion = args.fast
                robot.floor_mm = args.floor_mm
                robot.allow_extrapolation = args.allow_extrapolation

                # Table destination must remain inside calibration.
                destination = m.KitRescue.xy(robot, args.to)

                if args.execute:
                    input(
                        "Gripper EMPTY, table destination/routes clear; "
                        "Enter once to finish any pending retract, "
                        "freshly observe and remove: "
                    )
                    finish_pending_retract(robot)

                robot.require_park()

                with vision.lock:
                    vision._refreshing = True
                try:
                    report = robot.plan_task({
                        "operation": "transfer",
                        "object": args.colour,
                        "destination_xy": destination,
                    })
                finally:
                    with vision.lock:
                        vision._refreshing = False

                if report["observed"][args.colour]["spread_mm"] > 4.2:
                    raise ValueError(
                        "Source spread exceeds 4.2 mapped mm"
                    )

                report.update(
                    source_kind="fresh_vision_in_tray",
                    source_top_height_mm=30 + args.floor_mm,
                    xy_height_correction_applied=False,
                )
                robot.last_plan = copy.deepcopy(report)
                m.write_json(robot.log_dir / "plan.json", report)

                print(json.dumps({
                    "fresh_source": report["observed"][args.colour],
                    "destination_xy": destination,
                    "grasp_z": robot.base["z"] + args.floor_mm,
                    "release_z": robot.base["z"],
                }, indent=2))

                if args.execute:
                    result = robot.execute_plan(report)
                    print(
                        result["status"]
                        + "; inspect placement physically"
                    )
                else:
                    print("Planned only; no movement sent")

        except (RuntimeError, ValueError) as exc:
            print("Task stopped:", exc, flush=True)
            input("No further waypoints. Enter to close preview: ")
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
