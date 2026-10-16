"""KitRescue: top-face guided transfers and stacks. Vendor SDK unchanged.
NO ROTATION FILE"""

import argparse
import copy
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

try:
    from .camera import load_config, open_camera, prepare_image
    from .top_faces import detect_top_faces, project
    from .ik import NexArm, DEFAULT_PORT, ROOT
except ImportError:
    from camera import load_config, open_camera, prepare_image
    from top_faces import detect_top_faces, project
    from ik import NexArm, DEFAULT_PORT, ROOT


VERSION = "KitRescue manipulation topfaces-1"

EDGE, OPEN, CLOSED = 30, -80, -15
FLOOR, SEPARATION, STABILITY = 5, 45, 15.5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Perception:
    def __init__(self, camera, config, calibration, profile):
        self.camera = camera
        self.config = config
        self.cal = calibration
        self.profile = profile

        self.lock = threading.Lock()
        self.paused = False
        self.cached = None
        self.destination = None

    def snapshot(self):
        # One camera connection, shared by the task and browser.
        with self.lock:
            if self.paused:
                if self.cached is None:
                    raise RuntimeError("Preview paused during robot movement")
                return self.cached

            frame = self.camera.capture()
            image = prepare_image(frame, self.config)
            records, overlay = detect_top_faces(
                image, self.profile, self.cal
            )

            self.cached = (
                records,
                overlay,
                frame.received_at_ns,
            )
            return self.cached

    def observe(self, names):
        """Use the best top-face proposal, including ambiguous detections."""
        samples = {name: [] for name in names}
        ambiguous_frames = {name: 0 for name in names}

        for _ in range(5):
            records, overlay, stamp = self.snapshot()

            for name in names:
                matches = [
                    r for r in records
                    if r["colour"] == name
                ]

                if len(matches) != 1:
                    found = [r["status"] for r in matches]
                    raise ValueError(
                        f"{name}: expected one cube candidate, "
                        f"got {found}. Check preview."
                    )

                candidate = matches[0]
                status = candidate["status"]

                if status == "ok":
                    xy = candidate.get("xy_mm")

                elif status == "ambiguous":
                    xy = candidate.get("proposal", {}).get("xy_mm")
                    ambiguous_frames[name] += 1

                else:
                    xy = None

                point = np.asarray(xy, dtype=float)

                if point.shape != (2,) or not np.isfinite(point).all():
                    raise ValueError(
                        f"{name}: no usable top-face centre. "
                        "Check preview."
                    )

                samples[name].append(point.tolist())

        result = {}

        for name, values in samples.items():
            points = np.asarray(values, dtype=float)

            spread = float(
                np.linalg.norm(
                    points[:, None] - points[None, :],
                    axis=2,
                ).max()
            )

            if spread > STABILITY:
                raise ValueError(
                    f"{name}: top-centre variation {spread:.2f} mm "
                    f"exceeds {STABILITY} mm"
                )

            result[name] = {
                "xy": np.median(points, axis=0).tolist(),
                "spread_mm": spread,
                "received_at_ns": stamp,
                "ambiguous_frames": ambiguous_frames[name],
                "observation_count": 5,
                "geometry_kind": "best_top_face_proposal",
            }

        write_json(
            ROOT / "data/manipulation/observations.json",
            result,
        )

        cv2.imwrite(
            str(ROOT / "data/manipulation/observation.png"),
            overlay,
        )

        return result, records


PAGE = '''<!doctype html>
<meta charset="utf-8">
<title>KitRescue</title>
<style>
body {
    background:#182029;
    color:#eee;
    font:16px system-ui;
    margin:20px;
}
.window { overflow:auto; max-height:80vh; }
img { max-width:none; }
pre { white-space:pre-wrap; }
</style>
<h2>KitRescue workspace</h2>
<p>
Green: accepted top face. Orange: best ambiguous top-face proposal, also used for picking.
Cyan: calibrated patch. Magenta: destination.
Preview freezes during motion.
</p>
<label>
Display zoom
<input id="zoom" type="range"
       min="0.5" max="4" step="0.25" value="ZOOM_VALUE">
<span id="value"></span>×
</label>
<div class="window"><img id="view"></div>
<pre id="status"></pre>
<script>
const view = document.getElementById('view');
const zoom = document.getElementById('zoom');
const status = document.getElementById('status');
const value = document.getElementById('value');

function resize() {
    value.textContent = zoom.value;
    if (view.naturalWidth) {
        view.style.width =
            (view.naturalWidth * Number(zoom.value)) + 'px';
    }
}
zoom.oninput = resize;

function next() {
    view.src = '/frame.jpg?t=' + Date.now();
}

view.onload = async () => {
    resize();
    try {
        status.textContent = JSON.stringify(
            await (await fetch('/status')).json(),
            null, 2
        );
    } catch (e) {
        status.textContent = String(e);
    }
    setTimeout(next, 700);
};

view.onerror = () => {
    status.textContent = 'Check terminal';
    setTimeout(next, 1500);
};
next();
</script>'''


class WorkspaceView:
    def __init__(self, vision, port=8766, zoom=1):
        owner = self
        self.latest = {}
        self.vision = vision

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    route = urlsplit(self.path).path

                    if route == "/":
                        body = PAGE.replace(
                            "ZOOM_VALUE", str(zoom)
                        ).encode()
                        kind = "text/html"

                    elif route == "/frame.jpg":
                        records, image, stamp = vision.snapshot()
                        image = image.copy()

                        hull = np.float32(
                            vision.cal["mapping"]["pixel_hull"]
                        )

                        if vision.destination is not None:
                            matrix = np.float64(
                                vision.cal["mapping"][
                                    "homography_pixel_to_command_xy"
                                ]
                            )
                            inverse = np.linalg.inv(matrix)
                            point = tuple(
                                np.rint(
                                    project(
                                        [vision.destination],
                                        inverse,
                                    )[0]
                                ).astype(int)
                            )

                            cv2.drawMarker(
                                image,
                                point,
                                (255, 0, 255),
                                cv2.MARKER_CROSS,
                                25,
                                2,
                            )

                        lo = np.maximum(
                            np.floor(hull.min(axis=0)).astype(int) - 80,
                            [0, 0],
                        )
                        hi = np.minimum(
                            np.ceil(hull.max(axis=0)).astype(int) + 81,
                            [image.shape[1], image.shape[0]],
                        )

                        ok, encoded = cv2.imencode(
                            ".jpg",
                            image[lo[1]:hi[1], lo[0]:hi[0]],
                        )
                        if not ok:
                            raise RuntimeError(
                                "Preview encoding failed"
                            )

                        owner.latest = {
                            "objects": records,
                            "received_at_ns": stamp,
                            "paused": vision.paused,
                            "destination_xy": vision.destination,
                        }

                        body = encoded.tobytes()
                        kind = "image/jpeg"

                    elif route == "/status":
                        body = json.dumps(owner.latest).encode()
                        kind = "application/json"

                    else:
                        self.send_error(404)
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header(
                        "Content-Length", str(len(body))
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)

                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    self.send_error(500, str(exc))

        self.server = ThreadingHTTPServer(
            ("127.0.0.1", port),
            Handler,
        )
        threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        ).start()

        actual_port = self.server.server_port
        print(
            f"Forward VSCode port {actual_port}; "
            f"open http://localhost:{actual_port}"
        )

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class KitRescue:
    """Plan JSON tasks, then execute explicitly.

    All source cubes must initially rest on the calibrated table.
    """

    def __init__(
        self,
        arm,
        vision,
        calibration,
        duration_ms=5000,
        allow_unvalidated=False,
    ):
        self.arm = arm
        self.vision = vision
        self.cal = calibration
        self.base = calibration["reference"]
        self.park = calibration["park"]
        self.duration = duration_ms
        self.log_dir = ROOT / "data/manipulation"

        if (
            not calibration["mapping"]["accepted"]
            and not allow_unvalidated
        ):
            raise ValueError(
                "Calibration failed its threshold; "
                "supervised trials need --allow-unvalidated"
            )

        if not 500 <= duration_ms <= 10000:
            raise ValueError("Duration must be 500..10000 ms")

        if (
            abs(self.base["pitch"] + 90) > 1
            or self.base["claw"] != OPEN
        ):
            raise ValueError(
                "Reference must be the taught top-down, "
                "claw=-80 pose"
            )

        fit = [
            s["commanded_xy_mm"]
            for s in calibration["samples"]
            if s["role"] == "fit"
        ]
        self.hull = cv2.convexHull(np.float32(fit))
        self.last_plan = None

    def xy(self, values):
        point = np.asarray(values, dtype=float)

        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("XY requires two finite numbers")

        rounded = np.rint(point).astype(int)

        for p in (point, rounded.astype(float)):
            if cv2.pointPolygonTest(
                self.hull, tuple(p), False
            ) < 0:
                raise ValueError(
                    f"XY {point.tolist()} is outside "
                    "the calibrated patch"
                )

        return rounded.tolist()

    def require_park(self):
        state = self.arm.state()

        if any(
            abs(state[k] - self.park[k]) > 10
            for k in ("x", "y", "z")
        ):
            raise ValueError(
                "Start at saved camera_clear, "
                "with the gripper empty"
            )

        if any(
            abs(state[k] - self.base[k]) > 5
            for k in ("pitch", "roll", "claw")
        ):
            raise ValueError(
                "Start with the taught top-down "
                "orientation and open claw"
            )

    def pose(self, xy, z, claw):
        result = {
            k: self.base[k]
            for k in ("x", "y", "z", "pitch", "roll", "claw")
        }
        result.update(
            x=int(xy[0]),
            y=int(xy[1]),
            z=int(math.ceil(z)),
            claw=claw,
        )

        if result["z"] < FLOOR:
            raise ValueError(
                "Requested Z is below the software floor"
            )

        return result

    def steps(self, source, destination, layer):
        release = self.base["z"] + layer * EDGE

        minimum = (
            max(self.base["z"], release)
            + EDGE
            + 20
        )
        travel = max(
            int(self.cal.get("travel_z", 84)),
            minimum + 20,
        )

        if travel > self.park["z"]:
            raise ValueError(
                "Required stack clearance exceeds camera_clear"
            )

        park_xy = [self.park["x"], self.park["y"]]

        # label, XY, Z, claw command, minimum measured travel Z
        route = [
            (
                "travel_at_park",
                park_xy, travel, OPEN, minimum,
            ),
            (
                "approach_source",
                source, travel, OPEN, minimum,
            ),
            (
                "grasp_height",
                source, self.base["z"], OPEN, None,
            ),
            (
                "close",
                source, self.base["z"], CLOSED, None,
            ),
            (
                "lift",
                source, travel, CLOSED, minimum,
            ),
            (
                "carry",
                destination, travel, CLOSED, minimum,
            ),
            (
                "place_height",
                destination, release, CLOSED, None,
            ),
            (
                "release",
                destination, release, OPEN, None,
            ),
            (
                "retract",
                destination, travel, OPEN, minimum,
            ),
            (
                "return_to_park",
                park_xy, travel, OPEN, minimum,
            ),
            (
                "camera_clear",
                park_xy, self.park["z"], OPEN, None,
            ),
        ]

        return [
            {
                "action": label,
                "pose": self.pose(xy, z, claw),
                "minimum_z": low,
            }
            for label, xy, z, claw, low in route
        ]

    def plan_task(self, task):
        """JSON: operation, order/object, optional destination_xy."""
        self.last_plan = None
        self.require_park()

        operation = task.get("operation")

        if operation == "stack":
            names = task.get("order", [])
            if not isinstance(names, list) or not 2 <= len(names) <= 3:
                raise ValueError(
                    "Stack order needs two or three names, "
                    "bottom first"
                )

        elif operation == "transfer":
            names = [task.get("object")]

        else:
            raise ValueError(
                "Operation must be stack or transfer"
            )

        if any(
            not isinstance(n, str)
            or n not in self.vision.profile["parts"]
            for n in names
        ):
            raise ValueError("Use learned cube colour names")

        if len(set(names)) != len(names):
            raise ValueError(
                "Each colour can appear only once"
            )

        if any(
            abs(
                float(
                    self.vision.profile["parts"][n].get(
                        "edge_mm", EDGE
                    )
                ) - EDGE
            ) > .1
            for n in names
        ):
            raise ValueError(
                "This grasp setup requires 30 mm cubes"
            )

        observed, records = self.vision.observe(names)
        positions = {
            n: self.xy(o["xy"])
            for n, o in observed.items()
        }

        existing_base = (
            operation == "stack"
            and task.get("destination_xy") is None
        )

        if not existing_base and task.get("destination_xy") is None:
            raise ValueError(
                "Transfer requires destination_xy"
            )

        destination = (
            positions[names[0]]
            if existing_base
            else self.xy(task["destination_xy"])
        )
        self.vision.destination = destination

        for i, name in enumerate(names):
            for other in names[i + 1:]:
                if math.dist(
                    positions[name], positions[other]
                ) < SEPARATION:
                    raise ValueError(
                        f"Separate {name} and {other} centres "
                        f"by at least {SEPARATION} mm"
                    )

        # Simple descent-area guard, not geometric collision checking.
        for record in records:
            if (
                existing_base
                and record["colour"] == names[0]
            ):
                continue

            location = record.get(
                "proposal", {}
            ).get("xy_mm")

            if location is None:
                matrix = np.float64(
                    self.cal["mapping"][
                        "homography_pixel_to_command_xy"
                    ]
                )
                location = project(
                    [record["silhouette_center_px"]],
                    matrix,
                )[0]

            if math.dist(location, destination) < SEPARATION:
                raise ValueError(
                    "Destination is too close to "
                    "another detected cube"
                )

        transfers = []

        for layer, name in enumerate(names):
            if existing_base and layer == 0:
                continue

            route = self.steps(
                positions[name], destination, layer
            )
            transfers.append({
                "object": name,
                "source_xy": positions[name],
                "layer": layer,
                "release_z": self.base["z"] + layer * EDGE,
                "steps": route,
            })

        # ALL layers checked before any movement.
        # Plans are refreshed during execution.
        for transfer in transfers:
            for step in transfer["steps"]:
                self.arm.plan(
                    step["pose"],
                    duration_ms=self.duration,
                )

        report = {
            "task": copy.deepcopy(task),
            "destination_xy": destination,
            "base_left_in_place": (
                names[0] if existing_base else None
            ),
            "observed": observed,
            "transfers": transfers,
            "status": "planned_only",
            "physical_success_verified": False,
        }

        self.last_plan = copy.deepcopy(report)
        write_json(self.log_dir / "plan.json", report)
        return report

    @staticmethod
    def endpoint_ok(state, pose, minimum_z):
        if state["z"] < FLOOR:
            return False

        if minimum_z is None:
            z_ok = abs(state["z"] - pose["z"]) <= 10
        else:
            z_ok = (
                minimum_z <= state["z"] <= pose["z"] + 10
            )

        return (
            z_ok
            and all(
                abs(state[k] - pose[k]) <= 10
                for k in ("x", "y")
            )
            and all(
                abs(state[k] - pose[k]) <= 5
                for k in ("pitch", "roll")
            )
        )

    def move(self, step):
        pose = step["pose"]
        minimum = step["minimum_z"]
        before = self.arm.state()

        lateral = math.dist(
            [before["x"], before["y"]],
            [pose["x"], pose["y"]],
        ) > 10

        if (
            before["z"] < FLOOR
            or (
                minimum is not None
                and lateral
                and before["z"] < minimum
            )
        ):
            raise RuntimeError(
                "Lateral move blocked: "
                "insufficient reported clearance"
            )

        plan = self.arm.plan(
            pose,
            duration_ms=self.duration,
        )
        state = self.arm.execute(plan)

        readings = [state]
        deadline = time.monotonic() + 3
        passed = False

        while time.monotonic() < deadline:
            if state["z"] < FLOOR:
                break

            time.sleep(.25)
            state = self.arm.state()
            readings.append(state)
            recent = readings[-3:]

            quiet = len(recent) == 3 and all(
                max(s[k] for s in recent)
                - min(s[k] for s in recent)
                <= (
                    1 if k in ("x", "y", "z") else .4
                )
                for k in ("x", "y", "z", "pitch", "roll")
            )

            if quiet and self.endpoint_ok(
                state, pose, minimum
            ):
                passed = True
                break

        record = {
            "action": step["action"],
            "requested": pose,
            "minimum_z": minimum,
            "before": before,
            "after": state,
            "passed": passed,
            "feedback": readings,
        }

        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / "moves.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

        if not passed:
            write_json(
                self.log_dir / "last_motion_failure.json",
                record,
            )
            raise RuntimeError(
                f"Following error at {step['action']}: "
                f"{state}. No next waypoint sent."
            )

        print(
            f"  {step['action']}: "
            f"XYZ={[state[k] for k in ('x', 'y', 'z')]}",
            flush=True,
        )

    def execute_plan(self, report, step_mode=False):
        """Execute this instance's last plan once, after authorization."""
        if self.last_plan is None or report != self.last_plan:
            raise ValueError(
                "Plan is missing or changed; "
                "call plan_task again"
            )

        self.last_plan = None
        self.require_park()

        # Recheck the base and all sources before starting.
        current, _ = self.vision.observe(
            list(report["observed"])
        )

        for name, original in report["observed"].items():
            if math.dist(
                current[name]["xy"],
                original["xy"],
            ) > STABILITY:
                raise RuntimeError(
                    f"{name} moved after planning; replan"
                )

        for transfer in report["transfers"]:
            self.require_park()
            name = transfer["object"]

            # Only locate the next unpicked cube on the table.
            # Do not map raised stack tops through the table-plane mapping.
            current, _ = self.vision.observe([name])

            if math.dist(
                current[name]["xy"],
                report["observed"][name]["xy"],
            ) > STABILITY:
                raise RuntimeError(
                    f"{name} moved before pickup; replan"
                )

            print(
                f"Moving {name} "
                f"to layer {transfer['layer'] + 1}",
                flush=True,
            )

            with self.vision.lock:
                self.vision.paused = True

            try:
                for step in transfer["steps"]:
                    if step_mode:
                        input(
                            f"{step['action']}. "
                            "Check route/held cube; "
                            "hands clear, Enter: "
                        )

                    self.move(step)

            finally:
                with self.vision.lock:
                    self.vision.paused = False

        result = dict(
            report,
            status="motion_completed",
            physical_success_verified=False,
        )
        write_json(self.log_dir / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--version",
        action="version",
        version=VERSION,
    )
    parser.add_argument(
        "operation",
        choices=["view", "stack", "transfer"],
    )
    parser.add_argument(
        "cubes",
        nargs="*",
        help="Bottom-to-top order, or one cube for transfer",
    )
    parser.add_argument(
        "--to",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Enter before each movement",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--preview-zoom",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8766,
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--camera-config",
        default=str(ROOT / "config/camera.json"),
    )
    parser.add_argument(
        "--calibration",
        default=str(ROOT / "config/robot_camera.json"),
    )
    parser.add_argument(
        "--colour-profile",
        default=str(ROOT / "config/cube_colours.json"),
    )

    args = parser.parse_args()

    if not .5 <= args.preview_zoom <= 4:
        parser.error("Preview zoom must be 0.5..4")

    if args.operation == "view" and (
        args.execute or args.cubes or args.to
    ):
        parser.error(
            "view takes no cubes, destination or --execute"
        )

    if args.operation == "transfer" and (
        len(args.cubes) != 1 or args.to is None
    ):
        parser.error("Use transfer COLOUR --to X Y")

    if args.operation == "stack" and not 2 <= len(args.cubes) <= 3:
        parser.error(
            "Use stack BOTTOM TOP "
            "or stack BOTTOM MIDDLE TOP"
        )

    config = load_config(args.camera_config)
    cal = json.loads(Path(args.calibration).read_text())
    profile = json.loads(
        Path(args.colour_profile).read_text()
    )

    if not cal.get("mapping") or config != cal["camera_config"]:
        parser.error(
            "Completed calibration and matching "
            "camera configuration required"
        )

    print(
        "Keep cube sides aligned as during calibration; "
        "wrist roll stays taught."
    )
    print(
        "Endpoint/clearance checks only. Clear all routes; "
        "no automatic grasp verification."
    )

    with open_camera(config) as camera:
        vision = Perception(camera, config, cal, profile)
        viewer = WorkspaceView(
            vision,
            args.web_port,
            args.preview_zoom,
        )

        try:
            if args.operation == "view":
                input("Camera only. Enter to close: ")
                return

            input(
                "Position cubes using preview; clear hands, "
                "then Enter to PLAN (no movement): "
            )

            task = {
                "operation": args.operation,
                "destination_xy": args.to,
            }

            if args.operation == "stack":
                task["order"] = args.cubes
            else:
                task["object"] = args.cubes[0]

            with NexArm(args.port) as arm:
                robot = KitRescue(
                    arm,
                    vision,
                    cal,
                    args.duration_ms,
                    args.allow_unvalidated,
                )

                report = robot.plan_task(task)

                print(json.dumps(
                    {
                        k: v for k, v in report.items()
                        if k != "transfers"
                    },
                    indent=2,
                ))

                for transfer in report["transfers"]:
                    print(
                        f"{transfer['object']}: "
                        f"layer {transfer['layer'] + 1}, "
                        f"release Z={transfer['release_z']}"
                    )

                if args.execute:
                    input(
                        "All layers passed endpoint IK. "
                        "Inspect route; hands clear, "
                        "Enter to EXECUTE: "
                    )

                    result = robot.execute_plan(
                        report,
                        args.step,
                    )
                    print(
                        result["status"]
                        + "; inspect grasp/stack physically."
                    )

                else:
                    print(
                        "Planned only: no robot movement. "
                        "Add --execute to run."
                    )

            input("Preview active. Enter to close: ")

        finally:
            viewer.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(
            "\nStopped. No automatic recovery movement sent."
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit("Task stopped: " + str(exc))