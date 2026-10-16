"""KitRescue: top-face guided transfers and stacks. Vendor SDK unchanged."""

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


VERSION = "KitRescue manipulation topfaces-6 absolute-waypoints"

EDGE, OPEN, CLOSED = 30, -80, -15
FLOOR, SEPARATION, STABILITY = 5, 45, 14


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
        """Require five fresh agreeing detections; tolerate brief dropouts."""
        names = list(names)
        samples = {name: [] for name in names}
        reasons = {name: "not observed" for name in names}
        seen_stamps = set()
        result = None

        for attempt in range(12):
            records, overlay, stamp = self.snapshot()

            if stamp in seen_stamps:
                continue
            seen_stamps.add(stamp)

            for name in names:
                matches = [
                    r for r in records if r["colour"] == name
                ]
                usable = []

                for candidate in matches:
                    if candidate.get("status") not in ("ok", "ambiguous"):
                        continue

                    proposal = candidate.get("proposal", {})
                    xy = candidate.get("xy_mm")
                    if xy is None:
                        xy = proposal.get("xy_mm")

                    try:
                        point = np.asarray(xy, dtype=float)
                        angle = float(
                            proposal.get("angle_deg", float("nan"))
                        )
                        valid = (
                            point.shape == (2,)
                            and np.isfinite(point).all()
                            and math.isfinite(angle)
                        )
                    except (TypeError, ValueError):
                        valid = False

                    if valid:
                        usable.append((candidate, point, angle))

                if len(usable) != 1:
                    reasons[name] = (
                        "; ".join(
                            str(r.get("reason") or r.get("status"))
                            for r in matches
                        )
                        if matches else
                        "No colour candidate inside the calibrated patch"
                    )
                    if len(usable) > 1:
                        reasons[name] = (
                            "Multiple usable same-colour cubes; "
                            "separate/remove extras"
                        )
                    continue

                candidate, point, angle = usable[0]

                samples[name].append((
                    point.tolist(),
                    angle,
                    stamp,
                    candidate.get("status") == "ambiguous",
                ))
                samples[name] = samples[name][-5:]
                reasons[name] = (
                    f"{len(samples[name])}/5 usable observations"
                )

            result = {}

            for name in names:
                entries = samples[name]

                if len(entries) < 5:
                    result = None
                    break

                if stamp - entries[-1][2] > 3_000_000_000:
                    reasons[name] = (
                        "Latest usable detection is older than 3 seconds"
                    )
                    result = None
                    break

                points = np.asarray(
                    [entry[0] for entry in entries], dtype=float
                )
                spread = float(
                    np.linalg.norm(
                        points[:, None] - points[None, :],
                        axis=2,
                    ).max()
                )

                angles = np.asarray([entry[1] for entry in entries])
                unwrapped = (
                    angles[0] + (angles - angles[0] + 45) % 90 - 45
                )

                if spread > STABILITY:
                    reasons[name] = (
                        f"Top-centre variation {spread:.2f} mm "
                        f"exceeds {STABILITY} mm"
                    )
                    result = None
                    break

                if np.ptp(unwrapped) > 10:
                    reasons[name] = (
                        "Top-face angle varies by more than 10 degrees"
                    )
                    result = None
                    break

                result[name] = {
                    "xy": np.median(points, axis=0).tolist(),
                    "spread_mm": spread,
                    "received_at_ns": entries[-1][2],
                    "angle_deg": float(np.median(unwrapped) % 90),
                    "ambiguous_frames": sum(
                        int(entry[3]) for entry in entries
                    ),
                    "observation_count": 5,
                    "geometry_kind": "best_top_face_proposal",
                }

            if result is not None:
                break

        if result is None:
            detail = "; ".join(
                f"{name}: {reasons[name]}" for name in names
            )
            write_json(
                ROOT / "data/manipulation/last_observation_failure.json",
                {
                    "reasons": reasons,
                    "objects": records,
                    "usable_counts": {
                        name: len(values)
                        for name, values in samples.items()
                    },
                },
            )
            raise ValueError(
                "No consistent observation after 12 frames. " + detail
            )

        write_json(
            ROOT / "data/manipulation/observations.json", result
        )
        cv2.imwrite(
            str(ROOT / "data/manipulation/observation.png"), overlay
        )
        return result, records

    def track(self, name):
        """Image-only consistency check for a raised cube."""
        points = []

        for _ in range(5):
            records, _, stamp = self.snapshot()
            matches = [r for r in records if r["colour"] == name]

            if len(matches) != 1:
                raise ValueError(
                    f"Cannot see one {name} stack-top region"
                )

            points.append(matches[0]["silhouette_center_px"])

        points = np.asarray(points, dtype=float)
        spread = float(
            np.linalg.norm(
                points[:, None] - points[None, :], axis=2
            ).max()
        )

        if not np.isfinite(points).all() or spread > 12:
            raise ValueError(
                f"{name}: unstable stack image region"
            )

        return {
            "pixel_xy": np.median(points, axis=0).tolist(),
            "received_at_ns": stamp,
        }


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
        self.controller().require_park()

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

    def _plan_positions(self, task):
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








    def controller(self):
        if not hasattr(self, "_motion_controller"):
            try:
                from .motion import Motion
            except ImportError:
                from motion import Motion
            self._motion_controller = Motion(self)
        return self._motion_controller

    def move(self, step):
        return self.controller().move(step)


    @staticmethod
    def angle_difference(a, b):
        return (float(a) - float(b) + 45) % 90 - 45

    @staticmethod
    def wrist_roll(xy, face_angle, near=0):
        # face_angle is in calibrated robot XY, not image pixels.
        bearing = math.degrees(math.atan2(xy[1], xy[0]))
        raw = float(face_angle) - bearing

        choices = [
            raw + 90 * k
            for k in range(-4, 5)
            if -90 <= raw + 90 * k <= 90
        ]

        if not choices:
            raise ValueError(
                "No wrist orientation within the application range"
            )

        return int(round(
            min(choices, key=lambda value: abs(value - near))
        ))


    def oriented_steps(self, transfer, report, source):
        source_xy = self.xy(source["xy"])
        destination = self.xy(report["destination_xy"])
        route = self.steps(
            source_xy, destination, transfer["layer"]
        )

        # Keep the chosen symmetry branch across camera refreshes.
        pick_roll = self.wrist_roll(
            source_xy,
            source["angle_deg"],
            transfer.get("pick_roll", self.base["roll"]),
        )
        place_roll = self.wrist_roll(
            destination,
            report["destination_angle_deg"],
            transfer.get("place_roll", pick_roll),
        )

        for key, new in (
            ("pick_roll", pick_roll),
            ("place_roll", place_roll),
        ):
            if key in transfer and abs(new - transfer[key]) > 15:
                raise RuntimeError(
                    "Wrist branch would change abruptly; "
                    "replan from camera_clear"
                )

        transfer["pick_roll"] = pick_roll
        transfer["place_roll"] = place_roll

        for step in route:
            if step["action"] in (
                "approach_source", "grasp_height", "close", "lift"
            ):
                step["pose"]["roll"] = pick_roll

            elif step["action"] in (
                "carry", "place_height", "release", "retract"
            ):
                step["pose"]["roll"] = place_roll

        return route

    def preflight(self, route):
        for step in route:
            self.arm.plan(
                step["pose"], duration_ms=self.duration
            )

    def plan_task(self, task):
        report = self._plan_positions(task)
        self.last_plan = None

        bottom = (
            task["order"][0]
            if task["operation"] == "stack"
            else task["object"]
        )
        report["destination_angle_deg"] = (
            report["observed"][bottom]["angle_deg"]
        )

        for transfer in report["transfers"]:
            source = report["observed"][transfer["object"]]
            transfer["steps"] = self.oriented_steps(
                transfer, report, source
            )
            self.preflight(transfer["steps"])

        self.last_plan = copy.deepcopy(report)
        write_json(self.log_dir / "plan.json", report)
        return report

    def validate_refresh(self, name, current, original):
        if math.dist(current["xy"], original["xy"]) > 4.20:
            raise RuntimeError(
                f"{name} moved beyond 4.20 mm; replan"
            )

        if abs(self.angle_difference(
            current["angle_deg"], original["angle_deg"]
        )) > 10:
            raise RuntimeError(
                f"{name} rotated after planning; replan"
            )

        self.xy(current["xy"])

    def fresh_observation(self, name, original):
        current, _ = self.vision.observe([name])
        current = current[name]
        self.validate_refresh(name, current, original)
        return current

    def camera_read(self, function):
        # Called only after the arm has settled at a waypoint.
        with self.vision.lock:
            self.vision.paused = False

        try:
            return function()
        finally:
            with self.vision.lock:
                self.vision.paused = True

    def inspection_park(self, route, claw):
        # Travel at clearance, then raise at camera_clear XY.
        for step in route[-2:]:
            step = copy.deepcopy(step)
            step["action"] = "inspection_" + step["action"]
            step["pose"]["claw"] = claw
            self.move(step)

    def check_stack_image(self, name, baseline):
        current = self.vision.track(name)

        if math.dist(
            current["pixel_xy"], baseline["pixel_xy"]
        ) > 12:
            raise RuntimeError(
                "Stack image shifted by more than 12 pixels; "
                "inspect and replan"
            )

        return current

    def require_empty(self, destination, records, held_name):
        for record in records:
            if record["colour"] == held_name:
                continue

            location = record.get("proposal", {}).get("xy_mm")

            if location is None:
                matrix = np.float64(
                    self.cal["mapping"][
                        "homography_pixel_to_command_xy"
                    ]
                )
                location = project(
                    [record["silhouette_center_px"]], matrix
                )[0]

            if math.dist(location, destination) < SEPARATION:
                raise RuntimeError(
                    "Destination is no longer clear"
                )

    def execute_plan(self, report, step_mode=False):
        """Observe from park only; freeze each transfer through release."""
        if self.last_plan is None or report != self.last_plan:
            raise ValueError(
                "Plan missing or changed; call plan_task again"
            )
        self.last_plan = None
        self.require_park()
        work = copy.deepcopy(report)
        executed = []
        top_name = work["base_left_in_place"]
        top_image = None

        for index, transfer in enumerate(work["transfers"]):
            self.require_park()
            name = transfer["object"]

            # Initial observations were already captured at park.
            source = (
                work["observed"][name]
                if index == 0
                else self.fresh_observation(
                    name, work["observed"][name]
                )
            )
            if top_image is not None:
                self.check_stack_image(top_name, top_image)

            route = self.oriented_steps(transfer, work, source)
            self.preflight(route)
            print(
                f"Moving {name} to layer {transfer['layer'] + 1}; "
                "targets frozen",
                flush=True,
            )

            with self.vision.lock:
                self.vision.paused = True
            try:
                for step in route:
                    if step_mode:
                        input(
                            f"{step['action']}; hands clear, Enter: "
                        )
                    self.move(step)
            finally:
                with self.vision.lock:
                    self.vision.paused = False

            executed.append(dict(
                transfer,
                steps=route,
                source_observation=source,
            ))
            top_name = name

            if index + 1 < len(work["transfers"]):
                # Empty gripper has returned to park.
                # Raised stack tops are checked in image pixels only.
                self.require_park()
                top_image = self.vision.track(top_name)

        result = dict(
            work,
            transfers=executed,
            status="motion_completed",
            physical_success_verified=False,
            camera_policy="park_only",
        )
        write_json(self.log_dir / "result.json", result)
        return result



def task_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument(
        "operation", choices=["view", "stack", "transfer"]
    )
    parser.add_argument(
        "cubes", nargs="*",
        help="Bottom-to-top order, or one cube for transfer"
    )
    parser.add_argument("--to", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--step", action="store_true",
        help="Enter before each movement"
    )
    parser.add_argument(
        "--one-confirm", action="store_true",
        help="Only one Enter before task execution"
    )
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument("--preview-zoom", type=float, default=1)
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--camera-config",
        default=str(ROOT / "config/camera.json")
    )
    parser.add_argument(
        "--calibration",
        default=str(ROOT / "config/robot_camera.json")
    )
    parser.add_argument(
        "--colour-profile",
        default=str(ROOT / "config/cube_colours.json")
    )
    args = parser.parse_args()

    if args.one_confirm and (
        args.step or not args.execute or args.operation == "view"
    ):
        parser.error(
            "--one-confirm requires --execute "
            "and cannot be combined with --step or view"
        )

    if not .5 <= args.preview_zoom <= 4:
        parser.error("Preview zoom must be 0.5..4")

    if args.operation == "view" and (
        args.execute or args.cubes or args.to
    ):
        parser.error("view takes no cubes, destination or --execute")

    if args.operation == "transfer" and (
        len(args.cubes) != 1 or args.to is None
    ):
        parser.error("Use transfer COLOUR --to X Y")

    if args.operation == "stack" and not 2 <= len(args.cubes) <= 3:
        parser.error("Use stack BOTTOM TOP or stack BOTTOM MIDDLE TOP")

    config = load_config(args.camera_config)
    cal = json.loads(Path(args.calibration).read_text())
    profile = json.loads(Path(args.colour_profile).read_text())

    if not cal.get("mapping") or config != cal["camera_config"]:
        parser.error(
            "Completed calibration and matching camera configuration required"
        )

    print("Top-face wrist alignment enabled; pitch remains top-down.")
    print(
        "Endpoint/clearance checks only. Clear all routes; "
        "no automatic grasp verification."
    )

    with open_camera(config) as camera:
        vision = Perception(camera, config, cal, profile)
        viewer = WorkspaceView(
            vision, args.web_port, args.preview_zoom
        )

        try:
            if args.operation == "view":
                input("Camera only. Enter to close: ")
                return

            if not args.one_confirm:
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
                    arm, vision, cal,
                    args.duration_ms, args.allow_unvalidated
                )
                if args.execute:
                    input(
                        "Gripper EMPTY, raised above travel Z, routes clear. "
                        "Enter to PARK, observe and run this task: "
                    )
                    robot.controller().park()
                while True:
                    try:
                        report = robot.plan_task(task)
                        break
                    except (ValueError, RuntimeError) as exc:
                        print("Planning stopped: " + str(exc), flush=True)
                        print(
                            "Planning paused; no new motion sent. Adjust cubes in preview.",
                            flush=True,
                        )
                        input("Enter to retry PLAN, or Ctrl+C to quit: ")

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
                    if not args.one_confirm:
                        input(
                            "All layers passed endpoint IK. "
                            "Hands clear, Enter to EXECUTE: "
                        )
                    result = robot.execute_plan(report, args.step)
                    print(
                        result["status"]
                        + "; inspect grasp/stack physically."
                    )
                else:
                    print(
                        "Planned only: no robot movement. "
                        "Add --execute to run."
                    )

            if not args.one_confirm:
                input("Preview active. Enter to close: ")

        except (ValueError, RuntimeError) as exc:
            print("Task stopped: " + str(exc), flush=True)
            print(
                "Preview remains open. No further robot commands will be sent.",
                flush=True,
            )
            input("Inspect the scene; Enter to close, or Ctrl+C: ")
        finally:
            viewer.close()




def controller_limits(arm):
    """Read only. Firmware pitch limits use tenths of a degree."""
    raw = arm.board.get_coord_limits()
    keys = [
        axis + end
        for axis in "xyzprc"
        for end in ("min", "max")
    ]

    if not isinstance(raw, dict) or not all(k in raw for k in keys):
        raise RuntimeError(
            "Cannot read controller limits; no movement sent"
        )

    limits = {k: float(raw[k]) for k in keys}
    for key in ("pmin", "pmax"):
        limits[key] /= 10.0

    if not all(math.isfinite(v) for v in limits.values()):
        raise RuntimeError("Invalid controller limits")

    if any(
        limits[axis + "min"] > limits[axis + "max"]
        for axis in "xyzprc"
    ):
        raise RuntimeError("Reversed controller limits")

    return limits


def maintenance_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation", choices=("park", "test-motion")
    )
    parser.add_argument(
        "axis", nargs="?", choices=("z", "roll")
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--calibration",
        default=str(ROOT / "config/robot_camera.json"),
    )
    parser.add_argument(
        "--duration-ms", type=int, default=5000
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--one-confirm", action="store_true")
    args = parser.parse_args(argv)

    if (args.operation == "test-motion") != (args.axis is not None):
        parser.error(
            "Use park, test-motion z or test-motion roll"
        )

    cal = json.loads(Path(args.calibration).read_text())
    with NexArm(args.port) as arm:
        robot = KitRescue(
            arm, None, cal, args.duration_ms,
            allow_unvalidated=True,
        )
        motion = robot.controller()
        print("Current state:", json.dumps(arm.state()))
        print(
            "Commanded camera_clear:",
            json.dumps(motion.park_pose()),
        )

        if not args.execute:
            print(
                "Park endpoint IK:",
                arm.plan(
                    motion.park_pose(), duration_ms=5000
                )["solution"],
            )
            print(
                "Calculation only. Add --execute; "
                "test-motion parks before testing."
            )
            return

        input(
            "Empty gripper, already raised, routes clear; "
            "Enter to PARK and run: "
        )
        motion.park()

        if args.operation == "test-motion":
            motion.test(args.axis)

        print("Completed. Inspect the physical pose.")



def enable_run_logging():
    """Record console output, host writes and returned robot states."""
    import atexit
    import sys
    from functools import wraps

    path = ROOT / "data/manipulation/latest_run.log"
    if getattr(NexArm, "_kitrescue_trace_ready", False):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    sink = path.open("w", buffering=1)
    lock = threading.RLock()
    stdout, stderr = sys.stdout, sys.stderr
    counter = [0]

    def record(kind, **data):
        with lock:
            counter[0] += 1
            sink.write(json.dumps(dict(
                event=counter[0],
                host_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(),
                kind=kind,
                **data
            ), default=str) + "\n")

    class Tee:
        def __init__(self, original):
            self.original = original

        def write(self, text):
            with lock:
                result = self.original.write(text)
                sink.write(text)
                return result

        def flush(self):
            with lock:
                self.original.flush()
                sink.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

    sys.stdout, sys.stderr = Tee(stdout), Tee(stderr)

    def finish():
        sys.stdout, sys.stderr = stdout, stderr
        sink.close()

    atexit.register(finish)

    old_init = NexArm.__init__

    @wraps(old_init)
    def traced_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        port = self.board.port
        old_write = port.write

        def traced_write(data):
            raw = bytes(data)
            record("TX_ATTEMPT", hex=raw.hex(" "))
            try:
                count = old_write(data)
            except BaseException as exc:
                record("TX_ERROR", error=repr(exc))
                raise
            record(
                "TX_RETURN",
                bytes_written=count,
                expected=len(raw),
            )
            return count

        port.write = traced_write
        record(
            "CONNECTED",
            device=str(getattr(port, "port", "unknown")),
        )

    NexArm.__init__ = traced_init

    old_state = NexArm.state

    @wraps(old_state)
    def traced_state(self, *args, **kwargs):
        result = old_state(self, *args, **kwargs)
        record("STATE", state=result)
        return result

    NexArm.state = traced_state

    def wrap_method(name):
        original = getattr(NexArm, name, None)
        if original is None:
            return

        @wraps(original)
        def traced(self, *args, **kwargs):
            record(
                "CALL",
                method=name,
                args=args,
                kwargs={
                    k: v for k, v in kwargs.items()
                    if k != "monitor"
                },
            )
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                record("ERROR", method=name, error=repr(exc))
                raise
            record("RETURN", method=name, result=result)
            return result

        setattr(NexArm, name, traced)

    for name in ("plan", "execute", "execute_joints"):
        wrap_method(name)

    NexArm._kitrescue_trace_ready = True
    record(
        "RUN_START",
        argv=sys.argv,
        note="Host timestamps. TX_RETURN is not a device acknowledgement.",
    )
    print("Run log:", path, flush=True)
    return path


def main():
    enable_run_logging()
    import sys

    if (
        len(sys.argv) > 1
        and sys.argv[1] in ("park", "test-motion")
    ):
        maintenance_main(sys.argv[1:])
    else:
        task_main()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(
            "\nStopped. No automatic recovery movement sent."
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit("Task stopped: " + str(exc))