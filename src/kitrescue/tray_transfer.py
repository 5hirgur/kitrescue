"""One table cube into a detected tray using the working controller."""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from . import manipulation as m
    from .tray import detect_tray
except ImportError:
    import manipulation as m
    from tray import detect_tray


def enable_tray_preview(vision):
    def observe(image):
        vision.tray_observation = detect_tray(image)[0]

    def draw(image):
        observation = getattr(vision, "tray_observation", {})
        status = observation.get("status", "waiting")
        colour = (0, 255, 0) if status == "detected" else (0, 180, 255)

        if "corners_px" in observation:
            points = np.rint(observation["corners_px"]).astype(np.int32)
            cv2.polylines(image, [points], True, colour, 3)
            centre = tuple(np.rint(observation["center_px"]).astype(int))
            cv2.drawMarker(
                image, centre, colour, cv2.MARKER_CROSS, 25, 2
            )

        cv2.putText(
            image, "Tray: " + status + " | XY estimate unvalidated",
            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, .8, colour, 2,
        )
        return image

    vision.frame_observer = observe
    vision.preview_overlay = draw
    vision.preview_full_frame = True

class TrayTransfer(m.KitRescue):
    def __init__(
        self,
        *args,
        floor_mm=8,
        allow_extrapolation=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not math.isfinite(floor_mm) or not 0 <= floor_mm < 20:
            raise ValueError(
                "Floor elevation must be 0..<20 mm above the table"
            )

        self.floor_mm = float(floor_mm)
        self.allow_extrapolation = allow_extrapolation
        self.tray_destination = None

    def xy(self, values):
        point = np.asarray(values, dtype=float)

        # Only the selected tray destination can bypass the usual hull.
        if (
            self.tray_destination is not None
            and point.shape == (2,)
            and np.isfinite(point).all()
            and np.linalg.norm(point - self.tray_destination) < 0.01
        ):
            rounded = np.rint(point).astype(int)
            distances = [
                cv2.pointPolygonTest(self.hull, tuple(p), True)
                for p in (point, rounded.astype(float))
            ]

            if min(distances) < 0 and not self.allow_extrapolation:
                raise ValueError(
                    "Tray centre is outside calibration; "
                    "require --allow-extrapolation"
                )

            # Bound this initial supervised trial.
            if min(distances) < 0:
                print(
                    f"Extrapolated tray XY: {rounded.tolist()}; "
                    f"{-min(distances):.1f} mm beyond calibration",
                    flush=True,
                )

            return rounded.tolist()

        return super().xy(values)

    def steps(self, source, destination, layer):
        if layer != 0:
            raise ValueError(
                "This wrapper supports one table-to-tray transfer only"
            )

        route = super().steps(source, destination, layer)

        for step in route:
            # Source grasp stays at the original table height.
            if step["action"] in ("place_height", "release"):
                step["pose"]["z"] = int(
                    math.ceil(self.base["z"] + self.floor_mm)
                )

            # Raise travel targets and their measured-clearance requirements.
            if step["minimum_z"] is not None:
                step["minimum_z"] += self.floor_mm
                step["pose"]["z"] = int(
                    math.ceil(step["pose"]["z"] + self.floor_mm)
                )

                if step["pose"]["z"] > self.park["z"]:
                    raise ValueError(
                        "Required clearance exceeds camera_clear"
                    )

        return route

    def plan_to_tray(self, colour="orange"):
        self.last_plan = None
        self.tray_destination = None
        self.require_park()

        observations = []
        output = Path("data/tray")
        output.mkdir(parents=True, exist_ok=True)

        # Observe only from camera_clear, before picking.
        for _ in range(7):
            with self.vision.lock:
                frame = self.vision.camera.capture()
                image = m.prepare_image(frame, self.vision.config)

            cv2.imwrite(str(output / "transfer_input.png"), image)
            tray, overlay = detect_tray(image)
            print("Tray frame:", tray, flush=True)

            if not cv2.imwrite(
                str(output / "current_overlay.png"), overlay
            ):
                raise RuntimeError("Cannot save tray preview")

            if tray["status"] != "detected":
                continue

            observations.append(tray)

        centres = np.asarray([t["center_px"] for t in observations], dtype=float)
        diagnostic = {
            "valid_frames": len(observations),
            "required_frames": 5,
            "centres_px": centres.tolist(),
        }
        if len(observations) < 5:
            m.write_json(output / "stability.json", diagnostic)
            raise ValueError("Fewer than 5 clear tray frames; see data/tray/stability.json")

        matrix = np.float64(self.cal["mapping"]["homography_pixel_to_command_xy"])
        mapped_centres = cv2.perspectiveTransform(
            centres.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)

        median = np.median(mapped_centres, axis=0)
        distances = np.linalg.norm(mapped_centres - median, axis=1)
        keep = np.flatnonzero(distances <= 2.1)
        spread = float(np.max(np.linalg.norm(
            mapped_centres[:, None] - mapped_centres[None, :], axis=2
        )))

        diagnostic.update(
            centres_approx_xy_mm=mapped_centres.tolist(),
            distances_from_median_mm=distances.tolist(),
            inlier_indices=keep.tolist(),
            all_frame_spread_mm=spread,
            maximum_consensus_spread_mm=4.2,
            last_frame_status=tray["status"],
        )
        m.write_json(output / "stability.json", diagnostic)
        print(
            f"Tray consensus: {len(keep)}/7 frames; "
            f"all-frame spread {spread:.2f} mapped mm",
            flush=True,
        )

        if len(keep) < 5:
            raise ValueError(
                "Need 5 agreeing tray centres including the latest frame; "
                "see data/tray/stability.json"
            )

        chosen = int(keep[np.argmin(distances[keep])])
        tray = observations[chosen]
        centre = np.asarray(tray["center_px"], dtype=float)

        overlay = image.copy()
        cv2.polylines(
            overlay,
            [np.rint(tray["corners_px"]).astype(np.int32)],
            True, (0, 255, 0), 2,
        )
        cv2.drawMarker(
            overlay, tuple(np.rint(centre).astype(int)),
            (0, 255, 0), cv2.MARKER_CROSS, 20, 2,
        )
        if not cv2.imwrite(str(output / "current_overlay.png"), overlay):
            raise RuntimeError("Cannot save selected tray preview")
        matrix = np.float64(
            self.cal["mapping"]["homography_pixel_to_command_xy"]
        )

        points = np.float64(
            [centre.tolist()] + tray["corners_px"]
        ).reshape(-1, 1, 2)

        mapped = cv2.perspectiveTransform(
            points, matrix
        ).reshape(-1, 2)

        self.tray_destination = np.rint(mapped[0]).astype(float)
        destination = self.xy(self.tray_destination)

        # Approximate outer-outline clearance, not full collision checking.
        outline = np.float32(mapped[1:])
        clearance = cv2.pointPolygonTest(
            outline,
            tuple(map(float, destination)),
            True,
        )
        if clearance < 45:
            raise ValueError(
                "Estimated centre has less than 45 mm clearance "
                "from outer outline"
            )

        report = super().plan_task({
            "operation": "transfer",
            "object": colour,
            "destination_xy": destination,
        })

        for transfer in report["transfers"]:
            transfer["release_z"] = int(
                math.ceil(self.base["z"] + self.floor_mm)
            )

        report.update(
            destination_kind="tray_floor",
            floor_elevation_mm=self.floor_mm,
            tray_observation=tray,
            xy_height_correction_applied=False,
            destination_metric_pose_verified=False,
        )

        self.last_plan = copy.deepcopy(report)
        m.write_json(self.log_dir / "plan.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("colour", nargs="?", default="orange")
    parser.add_argument("--floor-mm", type=float, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--preview-zoom", type=float, default=1)
    parser.add_argument("--port", default=m.DEFAULT_PORT)
    parser.add_argument(
        "--camera-config", default="config/camera.json"
    )
    parser.add_argument(
        "--calibration", default="config/robot_camera.json"
    )
    parser.add_argument(
        "--colour-profile", default="config/cube_colours.json"
    )
    args = parser.parse_args()

    if not .5 <= args.preview_zoom <= 4:
        parser.error("Preview zoom must be 0.5..4")

    config = m.load_config(args.camera_config)
    cal = json.loads(Path(args.calibration).read_text())
    profile = json.loads(Path(args.colour_profile).read_text())

    if not cal.get("mapping") or config != cal["camera_config"]:
        parser.error(
            "Completed calibration and matching camera "
            "configuration required"
        )

    if hasattr(m, "enable_run_logging"):
        m.enable_run_logging()

    with m.open_camera(config) as camera:
        vision = m.Perception(camera, config, cal, profile)
        enable_tray_preview(vision)
        vision.preview_full_frame = True
        import re
        m.PAGE = re.sub(
            r"view\.naturalWidth\s*\*\s*Number\(zoom\.value\)",
            "view.parentElement.clientWidth * Number(zoom.value)",
            m.PAGE,
        )
        viewer = m.WorkspaceView(
            vision, port=8766, zoom=args.preview_zoom
        )

        try:
            with m.NexArm(args.port) as arm:
                robot = TrayTransfer(
                    arm,
                    vision,
                    cal,
                    allow_unvalidated=args.allow_unvalidated,
                    floor_mm=args.floor_mm,
                    allow_extrapolation=args.allow_extrapolation,
                )
                robot.fast_motion = args.fast

                if args.execute:
                    input(
                        "Orange alone on table; gripper EMPTY and "
                        "raised, routes clear. Unvalidated tray XY; "
                        "Enter once to PARK, PLAN and EXECUTE: "
                    )
                    robot.controller().park()

                report = robot.plan_to_tray(args.colour)

                print(json.dumps({
                    "object": args.colour,
                    "destination_xy": report["destination_xy"],
                    "release_z": report["transfers"][0]["release_z"],
                    "xy_height_correction_applied": False,
                }, indent=2))

                if args.execute:
                    result = robot.execute_plan(report)
                    print(
                        result["status"]
                        + "; inspect cube placement physically"
                    )
                else:
                    print(
                        "Planned only. Add --execute to move; "
                        "start at camera_clear."
                    )

        except (ValueError, RuntimeError) as exc:
            print("Task stopped:", exc, flush=True)
            input(
                "No further waypoints will be sent. "
                "Enter to close preview: "
            )
        finally:
            viewer.close()


if __name__ == "__main__":
    main()