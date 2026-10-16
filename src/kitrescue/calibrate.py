"""Guided planar calibration. Browser preview; robot commands stay in terminal."""

import argparse
from contextlib import ExitStack
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from camera import load_config, open_camera, prepare_image, camera_metadata
from ik import NexArm, DEFAULT_PORT, ROOT
from markers import detect_markers


MIN_Z_MM = 5
CUBE_EDGE_MM = 30
TRAVEL_MARGIN_MM = 20
VERSION = "KitRescue calibration workspace-5"


def check_z(pose, label):
    z = float(pose["z"])
    if not np.isfinite(z) or z < MIN_Z_MM:
        raise RuntimeError(
            f"{label}: Z={z} violates the software floor Z >= {MIN_Z_MM}. "
            "No further commands will be sent. This is not a physical collision check."
        )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


PREVIEW_PAGE = '''<!doctype html><meta charset="utf-8">
<title>KitRescue calibration</title>
<style>
body{font:16px system-ui;background:#182029;color:#eee;margin:16px}
button,input{font:inherit}#viewport{overflow:auto;height:76vh;border:1px solid #52616e}
#view{display:block;max-width:none;height:auto}label{display:block;margin:12px 0}
</style>
<h2>Calibration preview</h2>
<p>Use the terminal to move and record. Top marker: ID __MARKER_ID__.
Zoom changes the display only; recordings and marker coordinates use native pixels.</p>
<label>Display zoom <input id="zoom" type="number" min="1" max="6" step="0.25"
value="__ZOOM__"> &times; <button id="reset">Reset</button></label>
<p>Scroll horizontally or vertically to inspect the enlarged image.</p>
<div id="viewport"><img id="view" alt="Waiting for camera"></div>
<script>
const view=document.getElementById('view'), zoom=document.getElementById('zoom'),
      viewport=document.getElementById('viewport');
function applyZoom(){
  const value=Number(zoom.value);
  if(!Number.isFinite(value)||value<1||value>6)return;
  const oldWidth=view.getBoundingClientRect().width||viewport.clientWidth;
  const cx=(viewport.scrollLeft+viewport.clientWidth/2)/oldWidth;
  const oldHeight=view.getBoundingClientRect().height||viewport.clientHeight;
  const cy=(viewport.scrollTop+viewport.clientHeight/2)/oldHeight;
  view.style.width=(100*value)+'%';
  viewport.scrollLeft=cx*view.getBoundingClientRect().width-viewport.clientWidth/2;
  viewport.scrollTop=cy*view.getBoundingClientRect().height-viewport.clientHeight/2;
}
zoom.onchange=applyZoom;
document.getElementById('reset').onclick=()=>{zoom.value='1';applyZoom();
  viewport.scrollLeft=0;viewport.scrollTop=0;};
function next(){view.src='/frame.jpg?t='+Date.now()}
view.onload=()=>setTimeout(next,300);
view.onerror=()=>setTimeout(next,1000);
applyZoom();next();
</script>'''


def preview_server(camera, config, port, zoom=1.0, marker_id=5):
    if not np.isfinite(zoom) or not 1 <= zoom <= 6:
        raise ValueError("Preview zoom must be between 1 and 6")

    page = PREVIEW_PAGE.replace("__ZOOM__", str(float(zoom))).replace(
        "__MARKER_ID__", str(int(marker_id))
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                if self.path == "/":
                    body, kind = page, "text/html; charset=utf-8"

                elif self.path.startswith("/frame.jpg"):
                    frame = camera.capture()
                    image = prepare_image(frame, config)
                    _, overlay = detect_markers(image)
                    ok, encoded = cv2.imencode(".jpg", overlay)

                    if not ok:
                        raise RuntimeError("Preview encoding failed")

                    body, kind = encoded.tobytes(), "image/jpeg"

                else:
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            except (BrokenPipeError, ConnectionResetError):
                pass

            except Exception as exc:
                self.send_error(500, str(exc))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def check_endpoints(arm, base, park, targets, travel_z, duration):
    """Query all endpoints without moving. These checks do not test route clearance."""
    initial = arm.state()
    check_z(initial, "Current reported pose")

    initial_high = {
        **base,
        "x": initial["x"],
        "y": initial["y"],
        "z": travel_z,
    }

    endpoints = [
        {**base, "x": x, "y": y, "z": z}
        for _, x, y in targets
        for z in (base["z"], travel_z)
    ] + [
        park,
        {**park, "z": travel_z},
        initial_high,
    ]

    failures = []
    checked = set()

    for endpoint in endpoints:
        key = tuple(
            endpoint[k]
            for k in ("x", "y", "z", "pitch", "roll", "claw")
        )

        if key in checked:
            continue

        checked.add(key)
        check_z(endpoint, "Preflight endpoint")

        try:
            arm.plan(endpoint, duration_ms=duration)

        except RuntimeError as exc:
            failures.append({
                "target": endpoint,
                "reason": str(exc),
            })

            print(
                f"REJECTED: X={endpoint['x']} "
                f"Y={endpoint['y']} Z={endpoint['z']}: {exc}",
                flush=True,
            )

        else:
            print(
                f"IK OK: X={endpoint['x']} "
                f"Y={endpoint['y']} Z={endpoint['z']}",
                flush=True,
            )

    if failures:
        print(json.dumps({"rejected_endpoints": failures}, indent=2))
        raise RuntimeError(
            f"{len(failures)} of {len(checked)} IK endpoints were rejected. "
            "No movement requested; adjust the ranges or inspect the IK replies."
        )

    print(
        f"All {len(checked)} endpoint IK calculations passed. "
        "No movement requested."
    )
    return initial_high


def endpoint_errors(state, target):
    return {
        key: round(float(state[key]) - float(target[key]), 3)
        for key in ("x", "y", "z", "pitch", "roll")
    }


def endpoint_matches(state, target, minimum_z=None):
    errors = endpoint_errors(state, target)

    if minimum_z is None:
        z_ok = abs(errors["z"]) <= 10
    else:
        z_ok = (
            float(minimum_z)
            <= float(state["z"])
            <= float(target["z"]) + 10
        )

    return (
        z_ok
        and all(abs(errors[key]) <= 10 for key in ("x", "y"))
        and all(abs(errors[key]) <= 5 for key in ("pitch", "roll"))
    )


def settled_feedback(arm, first, target, timeout=3.0, minimum_z=None):
    """Wait for three quiet readings, without sending another movement command."""
    readings = [first]
    check_z(first, "Reported pose after movement")

    deadline = time.monotonic() + timeout
    quiet = False

    while time.monotonic() < deadline:
        time.sleep(0.25)

        state = arm.state()
        check_z(state, "Reported pose while settling")
        readings.append(state)

        if len(readings) >= 3:
            recent = readings[-3:]

            quiet = all(
                max(float(s[key]) for s in recent)
                - min(float(s[key]) for s in recent)
                <= (1 if key in ("x", "y", "z") else 0.4)
                for key in ("x", "y", "z", "pitch", "roll")
            )

            if quiet and endpoint_matches(state, target, minimum_z):
                return state, readings, True

    return readings[-1], readings, quiet


def move(arm, target, duration, minimum_z=None):
    """Execute a full pose and verify either endpoint accuracy or travel clearance."""
    check_z(target, "Requested target")

    if minimum_z is not None:
        if (
            not np.isfinite(minimum_z)
            or not MIN_Z_MM <= minimum_z <= target["z"]
        ):
            raise ValueError("Invalid minimum reported travel Z")

    current = arm.state()
    check_z(current, "Current reported pose")

    if (
        minimum_z is not None
        and current["z"] < minimum_z
        and any(
            abs(current[key] - target[key]) > 10
            for key in ("x", "y")
        )
    ):
        raise RuntimeError(
            f"Lateral travel blocked: reported Z={current['z']} "
            f"is below required clearance Z={minimum_z}. "
            "Raise at current XY first."
        )

    plan = arm.plan(target, duration_ms=duration)

    if "z" in plan["solution"]:
        check_z(plan["solution"], "IK endpoint")

    first = arm.execute(plan)

    state, readings, settled = settled_feedback(
        arm,
        first,
        target,
        minimum_z=minimum_z,
    )

    if not settled or not endpoint_matches(state, target, minimum_z):
        expected = plan["solution"].get("servos", [])
        measured = state.get("servos", [])

        servo_errors = None
        if len(expected) == len(measured) == 6:
            servo_errors = [
                int(actual) - int(requested)
                for actual, requested in zip(measured, expected)
            ]

        diagnostics = {
            "requested": target,
            "reported": state,
            "settled": settled,
            "minimum_z": minimum_z,
            "errors": endpoint_errors(state, target),
            "ik_solution": plan["solution"],
            "servo_error_ticks_measured_minus_ik": servo_errors,
            "feedback": readings,
        }

        path = ROOT / "data/calibration/last_motion_failure.json"
        write_json(path, diagnostics)

        raise RuntimeError(
            "Movement endpoint failed after settling. "
            f"Requested XYZ={[target[k] for k in ('x', 'y', 'z')]}; "
            f"reported XYZ={[state[k] for k in ('x', 'y', 'z')]}; "
            f"errors={diagnostics['errors']}; "
            f"minimum travel Z={minimum_z}; "
            f"servo errors={servo_errors}. "
            f"Details: {path}. No subsequent waypoint was sent."
        )

    if minimum_z is not None:
        print(
            f"Travel clearance: reported Z={state['z']} >= {minimum_z} mm "
            f"(commanded {target['z']})",
            flush=True,
        )

    return state


def vertical_move(arm, target, duration, minimum_z=None):
    """Use one native motion, without imposing intermediate stopping poses."""
    current = arm.state()
    check_z(current, "Current pose before vertical leg")

    if (
        abs(current["x"] - target["x"]) > 10
        or abs(current["y"] - target["y"]) > 10
    ):
        raise RuntimeError(
            "Vertical leg would require more than 10 mm XY correction; stopped"
        )

    print(
        f"Vertical endpoint: X={target['x']} "
        f"Y={target['y']} Z={target['z']}",
        flush=True,
    )

    return move(
        arm,
        target,
        duration,
        minimum_z=minimum_z,
    )


def capture_marker(camera, config, marker_id):
    centres = []
    last = None

    for _ in range(5):
        frame = camera.capture()
        image = prepare_image(frame, config)
        observations, overlay = detect_markers(image)

        matches = [
            marker
            for marker in observations
            if marker["id"] == marker_id
        ]

        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one ID {marker_id}")

        centres.append(matches[0]["center_px"])
        last = (frame, image, overlay, matches[0])

    points = np.asarray(centres, dtype=float)

    if np.ptp(points, axis=0).max() > 2:
        raise RuntimeError(
            "Marker moved more than 2 pixels; leave cube still"
        )

    return points.mean(axis=0).tolist(), last


def automatic_capture(camera, config, marker_id, timeout=12.0):
    """Automatically record a stable marker after parking; bounded on failure."""
    deadline = time.monotonic() + timeout
    last_error = "No capture attempted"

    while time.monotonic() < deadline:
        try:
            return capture_marker(camera, config, marker_id)

        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(0.2)

    raise RuntimeError(
        f"Automatic capture could not find a stable ID {marker_id}: "
        f"{last_error}"
    )


def fit_mapping(samples, max_error):
    training = [
        sample for sample in samples
        if sample["role"] == "fit"
    ]
    tests = [
        sample for sample in samples
        if sample["role"] == "test"
    ]

    if len(training) < 4 or not tests:
        raise ValueError(
            "Need at least four calibration points and one independent check"
        )

    pixels = np.float64([
        sample["pixel_xy"]
        for sample in training
    ])

    robot = np.float64([
        sample["commanded_xy_mm"]
        for sample in training
    ])

    if np.linalg.matrix_rank(pixels - pixels.mean(axis=0)) < 2:
        raise RuntimeError(
            "Calibration points do not span an image area"
        )

    homography, _ = cv2.findHomography(
        pixels,
        robot,
        method=0,
    )

    if homography is None or not np.isfinite(homography).all():
        raise RuntimeError("Cannot fit homography")

    for sample in samples:
        predicted = cv2.perspectiveTransform(
            np.float64([[sample["pixel_xy"]]]),
            homography,
        )[0, 0]

        sample["predicted_xy_mm"] = predicted.tolist()
        sample["error_mm"] = float(
            np.linalg.norm(
                predicted - sample["commanded_xy_mm"]
            )
        )

    maximum = max(sample["error_mm"] for sample in tests)

    return {
        "homography_pixel_to_command_xy": homography.tolist(),
        "pixel_hull": (
            cv2.convexHull(pixels.astype(np.float32))
            .reshape(-1, 2)
            .tolist()
        ),
        "fit_rms_mm": float(
            np.sqrt(
                np.mean([
                    sample["error_mm"] ** 2
                    for sample in training
                ])
            )
        ),
        "held_out_max_error_mm": maximum,
        "acceptance_threshold_mm": max_error,
        "accepted": maximum <= max_error,
        "fit_sample_count": len(training),
        "validation_sample_count": len(tests),
        "validation_scope": (
            "Single check point; accuracy elsewhere is unverified"
            if len(tests) == 1
            else "Held-out checks at sampled locations"
        ),
    }


def pixel_to_robot(pixel_xy, calibration):
    """Reusable by later picking tools; reject extrapolation."""
    mapping = calibration["mapping"]

    if not mapping or not mapping["accepted"]:
        raise ValueError(
            "Calibration failed its held-out checks"
        )

    hull = np.float32(mapping["pixel_hull"])
    point = tuple(float(value) for value in pixel_xy)

    if cv2.pointPolygonTest(hull, point, False) < 0:
        raise ValueError(
            "Point lies outside the calibrated area"
        )

    xy = cv2.perspectiveTransform(
        np.float64([[point]]),
        np.float64(mapping["homography_pixel_to_command_xy"]),
    )[0, 0]

    return float(xy[0]), float(xy[1])


def resume_session(args, base, park, config, targets):
    path = Path(args.output)

    if not args.resume:
        if path.exists():
            answer = input(
                "Replace existing robot_camera.json? Type REPLACE: "
            )

            if answer != "REPLACE":
                raise SystemExit(
                    "Cancelled; saved samples unchanged"
                )

        return None

    if not path.exists():
        raise ValueError(
            "No saved calibration to resume"
        )

    saved = json.loads(path.read_text())

    checks = {
        "reference": base,
        "park": park,
        "camera_config": config,
        "marker_id": args.marker_id,
    }

    for key, value in checks.items():
        if saved.get(key) != value:
            raise ValueError(
                f"Cannot resume: {key} differs from saved session"
            )

    samples = saved.get("samples", [])

    if len(samples) >= len(targets):
        raise ValueError(
            "All samples already collected; do not overwrite them"
        )

    for index, sample in enumerate(samples, 1):
        role, x, y = targets[index - 1]

        if (
            sample.get("index") != index
            or sample.get("role") != role
            or sample.get("commanded_xy_mm") != [x, y]
        ):
            raise ValueError(
                "Saved samples do not match this target sequence"
            )

    if saved.get("travel_z") != args.travel_z:
        previous = saved.get("travel_z")

        saved.setdefault("route_changes", []).append({
            "recorded_at_ns": time.time_ns(),
            "after_sample": len(samples),
            "previous_travel_z": previous,
            "travel_z": args.travel_z,
        })

        saved["travel_z"] = args.travel_z

        print(
            f"Travel height changed {previous} -> {args.travel_z} mm. "
            "Saved alignment poses and image points retained; "
            "new route IK checked."
        )

    saved["mapping"] = None

    print(
        f"Resuming at sample {len(samples) + 1}/{len(targets)}"
    )
    print(
        "Resume assumes camera, robot, support surface "
        "and lighting are unchanged."
    )

    return saved


def quick_targets(x_range, y_range, saved_samples=()):
    """Keep an incomplete session, then add missing corners and a centre check."""
    xmin, xmax = x_range
    ymin, ymax = y_range
    targets = []

    for sample in saved_samples:
        role = sample.get("role")
        xy = sample.get("commanded_xy_mm", [])

        if (
            role not in ("fit", "test")
            or len(xy) != 2
            or not all(np.isfinite(value) for value in xy)
            or not xmin <= xy[0] <= xmax
            or not ymin <= xy[1] <= ymax
        ):
            raise ValueError(
                "Saved calibration point is outside the requested "
                "quick-calibration ranges"
            )

        targets.append((role, xy[0], xy[1]))

    corners = (
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    )

    for x, y in corners:
        point = ("fit", x, y)

        if point not in targets:
            targets.append(point)

    centre = (
        "test",
        round((xmin + xmax) / 2),
        round((ymin + ymax) / 2),
    )

    if centre not in targets:
        targets.append(centre)

    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--version",
        action="version",
        version=VERSION,
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--camera-config",
        default=str(ROOT / "config/camera.json"),
    )
    parser.add_argument(
        "--poses",
        default=str(ROOT / "data/taught_poses.json"),
    )
    parser.add_argument(
        "--reference",
        default="grasp_reference",
    )
    parser.add_argument(
        "--park",
        default="camera_clear",
    )
    parser.add_argument(
        "--x-range",
        type=int,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "--y-range",
        type=int,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "--travel-z",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=2500,
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8766,
    )
    parser.add_argument(
        "--preview-zoom",
        type=float,
        default=1.0,
        help="Initial display magnification, 1 to 6; adjustable in browser",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check all IK endpoints only; no camera, motion or file changes",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-error-mm",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "config/robot_camera.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Four corners plus a centre check; "
            "retain existing points with --resume"
        ),
    )

    args = parser.parse_args()

    if not 500 <= args.duration_ms <= 10000:
        parser.error(
            "--duration-ms must be between 500 and 10000"
        )

    if (
        not np.isfinite(args.preview_zoom)
        or not 1 <= args.preview_zoom <= 6
    ):
        parser.error(
            "--preview-zoom must be between 1 and 6"
        )

    if (
        not np.isfinite(args.max_error_mm)
        or args.max_error_mm <= 0
    ):
        parser.error(
            "--max-error-mm must be positive and finite"
        )

    poses = json.loads(
        Path(args.poses).read_text()
    )["poses"]

    for name in (args.reference, args.park):
        if name not in poses:
            parser.error(
                f"Saved pose '{name}' is missing from {args.poses}. "
                f"Available poses: {', '.join(poses) or '(none)'}. "
                f"Teach it in ik.py --jog using 'save {name}' first. "
                "No movement made."
            )

    reference = poses[args.reference]
    source = reference.get("last_requested_target")

    if not source:
        raise ValueError(
            "Reference must include its commanded target"
        )

    base = {
        key: source[key]
        for key in ("x", "y", "z", "pitch", "roll", "claw")
    }

    check_z(base, "Saved grasp reference")

    minimum_travel_z = (
        base["z"]
        + CUBE_EDGE_MM
        + TRAVEL_MARGIN_MM
    )

    if args.travel_z < minimum_travel_z:
        raise ValueError(
            f"Travel Z must be at least {minimum_travel_z} mm "
            "(grasp Z + 30 mm cube + 20 mm clearance)"
        )

    park_record = poses[args.park]
    park_source = (
        park_record.get("last_requested_target")
        or park_record["reported_state"]
    )

    park = {
        **base,
        **{
            key: park_source[key]
            for key in ("x", "y", "z")
        },
    }

    if park["z"] < args.travel_z:
        raise ValueError(
            "Teach camera_clear at or above travel Z"
        )

    xmin, xmax = args.x_range
    ymin, ymax = args.y_range

    if xmax - xmin < 20 or ymax - ymin < 20:
        raise ValueError(
            "Each calibration range must span at least 20 mm"
        )

    # Four X values, three Y values; snake order reduces travel.
    xs = [
        int(round(value))
        for value in np.linspace(xmin, xmax, 4)
    ]
    ys = [
        int(round(value))
        for value in np.linspace(ymin, ymax, 3)
    ]

    targets = [
        ("fit", x, y)
        for row, y in enumerate(ys)
        for x in (xs if row % 2 == 0 else xs[::-1])
    ]

    for fx, fy in (
        (0.2, 0.25),
        (0.8, 0.25),
        (0.2, 0.75),
        (0.8, 0.75),
    ):
        targets.append((
            "test",
            round(xmin + fx * (xmax - xmin)),
            round(ymin + fy * (ymax - ymin)),
        ))

    if args.quick:
        previous_samples = []

        if args.resume:
            previous = json.loads(
                Path(args.output).read_text()
            )

            if previous.get("mapping") is not None:
                raise ValueError(
                    "This calibration is already complete; "
                    "quick resume requires an incomplete session"
                )

            previous_samples = previous.get("samples", [])

        targets = quick_targets(
            args.x_range,
            args.y_range,
            previous_samples,
        )

    config = load_config(args.camera_config)

    if config["backend"] != "webcam":
        raise ValueError(
            "This preview routine requires the Logitech webcam backend"
        )

    print("Grasp target:", base)
    print(
        "Camera-clear target (same pitch/roll/open claw):",
        park,
    )
    print(
        "Travel Z:",
        args.travel_z,
        "Targets:",
        targets,
    )
    print(
        f"Travel moves require settled reported Z >= {minimum_travel_z} mm; "
        "grasp and camera-clear poses retain their endpoint position checks."
    )

    if args.quick:
        print(
            "Quick calibration: corners plus one centre check. "
            "This is a prototype check, not a measurement of error "
            "throughout the workspace."
        )

    print(
        "IK checks endpoints only; this does not establish "
        "collision-free routes."
    )

    if not args.check_only:
        print(
            "Release any held object first. "
            "Remove loose objects from the movement routes."
        )
        input(
            "Press Enter after clearing the workspace: "
        )

    with ExitStack() as resources:
        arm = resources.enter_context(
            NexArm(args.port)
        )

        initial_high = check_endpoints(
            arm,
            base,
            park,
            targets,
            args.travel_z,
            args.duration_ms,
        )

        if args.check_only:
            print(
                "Range check complete. Camera not opened; "
                "saved calibration unchanged."
            )
            return

        resumed_session = resume_session(
            args,
            base,
            park,
            config,
            targets,
        )

        camera = resources.enter_context(
            open_camera(config)
        )

        server = preview_server(
            camera,
            config,
            args.web_port,
            zoom=args.preview_zoom,
            marker_id=args.marker_id,
        )

        print(
            f"Forward VSCode port {args.web_port}; "
            f"open http://localhost:{args.web_port}"
        )

        session = resumed_session or {
            "version": 1,
            "created_at_ns": time.time_ns(),
            "reference": base,
            "cube_edge_mm": CUBE_EDGE_MM,
            "marker_id": args.marker_id,
            "travel_z": args.travel_z,
            "park": park,
            "camera_config": config,
            "coordinate_space": "commanded_robot_xy_mm",
            "scope": "30 mm cubes on the calibrated support plane only",
            "samples": [],
            "mapping": None,
        }

        session["calibration_mode"] = (
            "quick" if args.quick else "grid16"
        )
        session["minimum_reported_travel_z"] = minimum_travel_z

        try:
            input(
                "Remove the calibration cube from all routes. "
                "Hands clear; press Enter to start positioning: "
            )

            if arm.state()["z"] < args.travel_z:
                vertical_move(
                    arm,
                    initial_high,
                    args.duration_ms,
                    minimum_z=minimum_travel_z,
                )
            else:
                move(
                    arm,
                    initial_high,
                    args.duration_ms,
                    minimum_z=minimum_travel_z,
                )

            move(
                arm,
                {**park, "z": args.travel_z},
                args.duration_ms,
                minimum_z=minimum_travel_z,
            )

            move(
                arm,
                park,
                args.duration_ms,
            )

            first_pending = len(session["samples"]) + 1

            for index, (role, x, y) in enumerate(targets, 1):
                if index <= len(session["samples"]):
                    continue

                print(
                    f"\nSample {index}/{len(targets)} [{role}]: "
                    f"X={x}, Y={y}, Z={base['z']}"
                )

                if index != first_pending:
                    input(
                        "Remove the previous cube and clear the route. "
                        "Hands clear; press Enter for the next position: "
                    )

                above = {
                    **park,
                    "z": args.travel_z,
                }
                travel = {
                    **base,
                    "x": x,
                    "y": y,
                    "z": args.travel_z,
                }
                target = {
                    **base,
                    "x": x,
                    "y": y,
                }

                move(
                    arm,
                    above,
                    args.duration_ms,
                    minimum_z=minimum_travel_z,
                )

                move(
                    arm,
                    travel,
                    args.duration_ms,
                    minimum_z=minimum_travel_z,
                )

                vertical_move(
                    arm,
                    target,
                    args.duration_ms,
                )

                print(
                    "With the arm stationary, place cube flat and "
                    "centre it in BOTH XY directions."
                )
                print(
                    f"Top ID {args.marker_id} up; cube sides aligned "
                    "to jaws. Do not move the robot or table."
                )

                input(
                    "Cube aligned and hands clear? Press Enter to "
                    "retract, park and capture automatically: "
                )

                # Record the robot state BEFORE retracting.
                aligned_state = arm.state()

                vertical_move(
                    arm,
                    travel,
                    args.duration_ms,
                    minimum_z=minimum_travel_z,
                )

                move(
                    arm,
                    above,
                    args.duration_ms,
                    minimum_z=minimum_travel_z,
                )

                move(
                    arm,
                    park,
                    args.duration_ms,
                )

                while True:
                    try:
                        print(
                            f"Automatically capturing stable ID "
                            f"{args.marker_id}...",
                            flush=True,
                        )

                        centre, (
                            frame,
                            image,
                            overlay,
                            marker,
                        ) = automatic_capture(
                            camera,
                            config,
                            args.marker_id,
                        )
                        break

                    except RuntimeError as exc:
                        print("Capture rejected:", exc)
                        input(
                            "Arm parked. Keep the cube fixed; clear hands/"
                            "check marker visibility, then Enter to retry: "
                        )

                image_path = (
                    ROOT
                    / "data/calibration"
                    / f"sample_{index:02d}.png"
                )
                image_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(
                        "Could not save calibration image"
                    )

                sample = {
                    "index": index,
                    "role": role,
                    "commanded_xy_mm": [x, y],
                    "pixel_xy": centre,
                    "commanded_pose": target,
                    "aligned_reported_state": aligned_state,
                    "travel_z_mm": args.travel_z,
                    "image": str(image_path),
                    "last_marker": marker,
                    "received_at_ns": frame.received_at_ns,
                }

                session["samples"].append(sample)
                session["camera"] = camera_metadata(
                    camera,
                    config,
                    image,
                )

                write_json(
                    args.output,
                    session,
                )

                print(
                    "Saved:",
                    [x, y],
                    "<->",
                    centre,
                )

            session["mapping"] = fit_mapping(
                session["samples"],
                args.max_error_mm,
            )

            write_json(
                args.output,
                session,
            )

            print(json.dumps(
                session["mapping"],
                indent=2,
            ))

            if session["mapping"]["accepted"]:
                print("PASS")
            else:
                print("FAIL: inspect alignment and samples")

            print("Saved", args.output)

        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nStopped. No automatic recovery movement was sent; "
            "saved samples remain available."
        )

    except (RuntimeError, ValueError) as exc:
        print(f"Calibration stopped: {exc}")
        raise SystemExit(2)