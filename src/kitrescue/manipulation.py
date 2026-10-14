"""Pick/place and stacking primitives. Vendor SDK stays unchanged."""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

try:
    from .camera import load_config, open_camera, prepare_image
    from .markers import detect_markers
    from .ik import NexArm, DEFAULT_PORT, ROOT
except ImportError:
    from camera import load_config, open_camera, prepare_image
    from markers import detect_markers
    from ik import NexArm, DEFAULT_PORT, ROOT


CUBE_MM = 30
VERSION = "KitRescue manipulation workspace-2"


def workspace_overlay(image, calibration, profile=None, destination=None):
    """Draw in native processed-image pixels. Display only; no robot commands."""
    mapping = calibration["mapping"]
    hull = np.float32(mapping["pixel_hull"]).reshape(-1, 2)
    polygon = np.rint(hull).astype(np.int32)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    canvas = (image.astype(np.float32) * 0.45).astype(np.uint8)
    canvas[mask > 0] = image[mask > 0]
    cv2.polylines(canvas, [polygon], True, (255, 255, 0), 2)
    status = dict(calibration_accepted=mapping["accepted"],
                  held_out_max_error_mm=mapping.get("held_out_max_error_mm"),
                  cubes=[], markers=[], notes=[],
                  source_plane="30 mm cube tops on the original table")
    if profile is not None:
        try:
            try:
                from . import cubes
            except ImportError:
                import cubes
            detections = cubes.detect_cubes(image, profile, min_area=900)[0]
            roi = profile.get("roi_xywh")
            if roi:
                x, y, w, h = roi
                cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (0, 255, 255), 1)
                if any(not (x <= px < x+w and y <= py < y+h) for px, py in hull):
                    status["notes"].append("Colour ROI does not cover the entire calibrated patch")
            for detection in detections:
                point = tuple(float(v) for v in detection["bbox_center_px"])
                inside = cv2.pointPolygonTest(hull, point, False) >= 0
                flags = [k for k in ("touches_image_edge", "touches_roi_edge", "overlaps_other_colour")
                         if detection.get(k)]
                colour = (0, 220, 255) if inside and not flags else (0, 0, 255)
                x, y, w, h = detection["bbox_xywh_px"]
                cv2.rectangle(canvas, (x,y), (x+w-1,y+h-1), colour, 2)
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 4, colour, -1)
                label = detection["colour"] + (" ~inside" if inside else " outside")
                cv2.putText(canvas, label, (x, max(18,y-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1)
                xy = cv2.perspectiveTransform(np.float64([[point]]),
                     np.float64(mapping["homography_pixel_to_command_xy"]))[0,0]
                status["cubes"].append(dict(colour=detection["colour"], pixel_xy=list(point),
                       estimated_xy_mm=[round(float(v),1) for v in xy],
                       centre_inside_patch=bool(inside), flags=flags, approximate_grasp=True))
        except Exception as exc:
            status["notes"].append("Colour detection: " + str(exc))
    else:
        status["notes"].append("No colour profile loaded; showing the calibrated patch only")
    markers, _ = detect_markers(image)
    for marker in markers:
        corners = np.rint(marker["corners_px"]).astype(np.int32)
        cv2.polylines(canvas, [corners], True, (80,255,80), 1)
        status["markers"].append(dict(id=marker["id"], pixel_xy=marker["center_px"]))
    if destination is not None:
        inverse = np.linalg.inv(np.float64(mapping["homography_pixel_to_command_xy"]))
        destination = [float(v) for v in destination]
        angles = np.linspace(0, 2*np.pi, 65)
        ring = np.array([[destination[0]+35*np.cos(a), destination[1]+35*np.sin(a)] for a in angles])
        ring_px = cv2.perspectiveTransform(ring.reshape(1,-1,2), inverse)[0]
        cv2.polylines(canvas, [np.rint(ring_px).astype(np.int32)], True, (255,0,255), 1)
        centre = cv2.perspectiveTransform(np.float64([[destination]]), inverse)[0,0]
        p = tuple(np.rint(centre).astype(int))
        cv2.drawMarker(canvas,p,(255,0,255),cv2.MARKER_CROSS,20,2)
        cv2.putText(canvas,"stack base",(p[0]+8,p[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),1)
        status["destination_xy_mm"] = destination
    return canvas, status


class WorkspaceView:
    def __init__(self, camera, config, calibration, profile_path, destination=None, port=8766):
        if config["backend"] != "webcam":
            raise ValueError("Shared live preview requires the webcam backend")
        self.camera, self.config, self.cal = camera, config, calibration
        self.destination = destination
        self.profile = json.loads(Path(profile_path).read_text()) if Path(profile_path).exists() else None
        self.latest = {"status":"Waiting for first preview frame"}
        self.lock = threading.Lock()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                request = urlsplit(self.path)
                try:
                    if request.path == "/":
                        body = WORKSPACE_PAGE.encode()
                        kind = "text/html; charset=utf-8"
                    elif request.path == "/frame.jpg":
                        zoom = parse_qs(request.query).get("zoom", ["1"])[0] == "1"
                        body = owner.frame(zoom)
                        kind = "image/jpeg"
                    elif request.path == "/status":
                        body = json.dumps(owner.latest).encode()
                        kind = "application/json"
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    self.send_error(500, str(exc))
        self.server = ThreadingHTTPServer(("127.0.0.1",port),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        print(f"Browser preview: forward VSCode port {self.server.server_port}; open http://localhost:{self.server.server_port}")

    def frame(self, zoom=True):
        with self.lock:
            frame = self.camera.capture()
            image = prepare_image(frame,self.config)
            expected = self.cal.get("camera",{})
            if [image.shape[1],image.shape[0]] != [expected.get("width"),expected.get("height")]:
                raise ValueError("Preview resolution differs from calibration")
            overlay, status = workspace_overlay(image,self.cal,self.profile,self.destination)
            status["received_at_ns"] = frame.received_at_ns
            if zoom:
                hull = np.asarray(self.cal["mapping"]["pixel_hull"])
                lo = np.maximum(np.floor(hull.min(axis=0)).astype(int)-120,[0,0])
                hi = np.minimum(np.ceil(hull.max(axis=0)).astype(int)+121,[image.shape[1],image.shape[0]])
                overlay = overlay[lo[1]:hi[1],lo[0]:hi[0]]
                status["crop_origin_px"] = lo.tolist()
            ok, encoded = cv2.imencode(".jpg",overlay,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:
                raise RuntimeError("Preview encoding failed")
            self.latest = status
            return encoded.tobytes()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


WORKSPACE_PAGE = '''<!doctype html><meta charset="utf-8">
<title>KitRescue workspace</title>
<style>body{font:16px system-ui;background:#182029;color:#eee;max-width:1200px;margin:20px auto;padding:0 15px}
img{width:100%;height:auto}pre{white-space:pre-wrap}label{display:block;margin:12px 0}</style>
<h2>KitRescue workspace — read-only</h2>
<p>Cyan: measured calibration patch. Put source cube centres inside it.
Magenta cross: stack base. Keep source centres outside its 35 mm separation ring.</p>
<p>Yellow dots are approximate colour-mask centres; red dots are outside the patch or ambiguous.
This is a placement guide, not collision checking. Positions assume 30 mm cubes on the original table.</p>
<label><input id="zoom" type="checkbox" checked> Zoom to calibrated patch</label>
<img id="view" alt="Waiting for camera"><pre id="status">Waiting...</pre>
<script>
const view=document.getElementById('view'),zoom=document.getElementById('zoom'),status=document.getElementById('status');
function next(){view.src='/frame.jpg?zoom='+(zoom.checked?'1':'0')+'&t='+Date.now()}
view.onload=async()=>{try{status.textContent=JSON.stringify(await(await fetch('/status')).json(),null,2)}catch(e){status.textContent=String(e)}setTimeout(next,500)};
view.onerror=()=>{status.textContent='Preview unavailable; check terminal.';setTimeout(next,1500)};
next();
</script>'''


def require_confirmation(message, token):
    if input(f"{message} Type {token}: ").strip() != token:
        raise RuntimeError("Cancelled; no automatic recovery movement will be sent")


class KitRescue:
    def __init__(self, arm, camera, camera_config, calibration, sources,
                 execute=False, allow_unvalidated=False, allow_extrapolation=False,
                 duration_ms=5000, release_z=None, clearance_mm=20,
                 confirm=require_confirmation, log_path=None,
                 colour_profile=None, allow_approximate_grasps=False):
        self.arm, self.camera = arm, camera
        self.config, self.cal = camera_config, calibration
        self.sources = sources
        self.execute = execute
        self.allow_extrapolation = allow_extrapolation
        self.allow_approximate_grasps = allow_approximate_grasps
        self.colour_profile = Path(colour_profile or ROOT / "config/cube_colours.json")
        self.duration = duration_ms
        self.confirm = confirm
        self.base, self.park = calibration["reference"], calibration["park"]
        self.release_z = self.base["z"] if release_z is None else release_z
        self.clearance = clearance_mm
        self.placed = {}
        self.session_id = time.time_ns()
        self.log_path = Path(log_path or ROOT / "data/manipulation/events.jsonl")
        mapping = calibration.get("mapping")
        if not mapping:
            raise ValueError("Calibration has no fitted mapping")
        if not mapping["accepted"] and not allow_unvalidated:
            raise ValueError("Calibration unvalidated. A manual trial requires --allow-unvalidated")
        if camera_config != calibration["camera_config"]:
            raise ValueError("Camera configuration differs from calibration")
        if not 500 <= duration_ms <= 10000 or clearance_mm < 15:
            raise ValueError("Use duration 500..10000 ms and clearance >= 15 mm")
        if not math.isfinite(float(self.release_z)) or self.release_z < 5:
            raise ValueError("Release Z violates the software floor")
        fit = [s["commanded_xy_mm"] for s in calibration["samples"] if s["role"] == "fit"]
        self.robot_hull = cv2.convexHull(np.float32(fit))
        self._require_park()

    def _event(self, kind, **values):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(session_id=self.session_id, recorded_at_ns=time.time_ns(),
                      event=kind, **values)
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def _require_park(self):
        state = self.arm.state()
        if any(abs(state[k] - self.park[k]) > 10 for k in ("x", "y", "z")):
            raise ValueError("Start at saved camera_clear with an empty, open gripper")
        if (abs(state["pitch"] - self.base["pitch"]) > 5
                or abs(state["roll"] - self.base["roll"]) > 5
                or abs(state["claw"] - self.base["claw"]) > 5):
            raise ValueError("Start with the taught top-down orientation and open claw")

    def _check_xy(self, xy):
        point = tuple(float(v) for v in xy)
        if len(point) != 2 or not all(math.isfinite(v) for v in point):
            raise ValueError("XY must contain two finite numbers")
        if not self.allow_extrapolation:
            if cv2.pointPolygonTest(self.robot_hull, point, False) < 0:
                raise ValueError(f"XY {point} is outside the calibrated patch")
        return [int(round(v)) for v in point]

    def _map_pixel(self, pixel):
        pixel = np.float64(pixel)
        mapping = self.cal["mapping"]
        if not self.allow_extrapolation:
            if cv2.pointPolygonTest(np.float32(mapping["pixel_hull"]), tuple(pixel), False) < 0:
                raise ValueError("Object is outside the calibrated image patch")
        xy = cv2.perspectiveTransform(
            pixel.reshape(1, 1, 2),
            np.float64(mapping["homography_pixel_to_command_xy"]),
        )[0, 0]
        return self._check_xy(xy)

    def _observe_colour(self, name, colour):
        if not self.allow_approximate_grasps:
            raise ValueError(
                "Colour detection currently provides silhouette centres, not verified "
                "top-face centres. A staged trial requires --allow-approximate-grasps."
            )
        try:
            from . import cubes
        except ImportError:
            import cubes
        profile = json.loads(self.colour_profile.read_text())
        signature = getattr(cubes, "geometry_signature", None)
        if callable(signature) and profile.get("geometry_signature"):
            if profile["geometry_signature"] != signature(self.config):
                raise ValueError("Colour profile camera geometry differs; resample colours")
        if colour not in profile["parts"]:
            raise ValueError(f"No learned colour profile for {colour}")
        self._require_park()
        centres = []
        for _ in range(5):
            frame = self.camera.capture()
            image = prepare_image(frame, self.config)
            result = cubes.detect_cubes(image, profile, min_area=900)
            observations, overlay = result[0], result[2]
            debug_path = self.log_path.parent / "latest_colours.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), overlay)
            matches = [o for o in observations if o["colour"] == colour]
            if len(matches) != 1:
                raise ValueError(f"Expected one {colour} candidate, found {len(matches)}; inspect {debug_path}")
            candidate = matches[0]
            if any(candidate.get(k, False) for k in (
                "touches_image_edge", "touches_roi_edge", "overlaps_other_colour"
            )):
                raise ValueError(f"Ambiguous/clipped {colour} candidate; inspect {debug_path}")
            centres.append(candidate["bbox_center_px"])
        points = np.float64(centres)
        if np.ptp(points, axis=0).max() > 2:
            raise ValueError("Colour candidate moved while observing")
        pixel = points.mean(axis=0)
        return dict(name=name, colour=colour, xy=self._map_pixel(pixel), method="colour",
                    pixel_xy=pixel.tolist(), received_at_ns=frame.received_at_ns,
                    approximate_grasp=True, geometry_kind="silhouette_centre")

    def observe(self, name):
        """Source cubes must be on the original calibration support plane."""
        if name not in self.sources:
            raise ValueError(f"Unknown object: {name}")
        if name in self.placed:
            raise ValueError(f"{name} already placed in this session")
        specification = self.sources[name]
        kind, value = specification.split(":", 1)
        if kind in ("colour", "color"):
            result = self._observe_colour(name, value)
        elif kind in ("xy", "taught"):
            if kind == "taught":
                poses = json.loads((ROOT / "data/taught_poses.json").read_text())["poses"]
                pose = poses[value].get("last_requested_target")
                if not pose:
                    raise ValueError("Taught source must include a commanded pose")
                values = [pose["x"], pose["y"]]
            else:
                values = [float(v) for v in value.split(",")]
            xy = self._check_xy(values)
            result = dict(name=name, xy=xy, method="taught_xy")
        elif kind == "marker":
            self._require_park()
            marker_id = int(value)
            centres = []
            for _ in range(5):
                frame = self.camera.capture()
                image = prepare_image(frame, self.config)
                expected = self.cal.get("camera", {})
                if [image.shape[1], image.shape[0]] != [expected.get("width"), expected.get("height")]:
                    raise ValueError("Camera resolution differs from calibration")
                observations, _ = detect_markers(image)
                found = [m for m in observations if m["id"] == marker_id]
                if len(found) != 1:
                    raise ValueError(f"Expected exactly one top marker ID {marker_id}")
                centres.append(found[0]["center_px"])
            points = np.float64(centres)
            if np.ptp(points, axis=0).max() > 2:
                raise ValueError("Cube/camera moved while observing")
            pixel = points.mean(axis=0)
            result = dict(name=name, xy=self._map_pixel(pixel), method="marker",
                          marker_id=marker_id, pixel_xy=pixel.tolist(),
                          received_at_ns=frame.received_at_ns)
        else:
            raise ValueError("Use colour:NAME, marker:ID, taught:POSE_NAME or xy:X,Y")
        self._event("observation", observation=result)
        return result

    def get_scene(self):
        return {"sources": [self.observe(n) for n in self.sources if n not in self.placed],
                "operator_confirmed_placements": self.placed}

    def _pose(self, xy, z, claw):
        pose = dict(self.base, x=int(round(xy[0])), y=int(round(xy[1])),
                    z=int(round(z)), claw=int(claw))
        if pose["z"] < 5:
            raise ValueError("Requested pose violates Z >= 5")
        # No time_ms in this dictionary: NexArm.plan receives duration separately.
        return {k: pose[k] for k in ("x", "y", "z", "pitch", "roll", "claw")}

    def plan_pick(self, source_xy, safe_z):
        return [
            ("lower_at_park", self._pose([self.park["x"], self.park["y"]], safe_z, self.base["claw"])),
            ("approach_source", self._pose(source_xy, safe_z, self.base["claw"])),
            ("lower_to_grasp", self._pose(source_xy, self.base["z"], self.base["claw"])),
            ("close", self._pose(source_xy, self.base["z"], -15)),
            ("lift", self._pose(source_xy, safe_z, -15)),
        ]

    def plan_place(self, destination_xy, release_z, safe_z):
        return [
            ("carry", self._pose(destination_xy, safe_z, -15)),
            ("lower_to_place", self._pose(destination_xy, release_z, -15)),
            ("release", self._pose(destination_xy, release_z, self.base["claw"])),
            ("retract", self._pose(destination_xy, safe_z, self.base["claw"])),
            ("return_to_park_xy", self._pose([self.park["x"], self.park["y"]], safe_z, self.base["claw"])),
            ("camera_clear", self._pose([self.park["x"], self.park["y"]], self.park["z"], self.base["claw"])),
        ]

    def pick_place(self, name, destination_xy, support_height_mm=0, layer=0):
        """One transaction; LLM tools should supply object/destination names via a registry."""
        if not math.isfinite(float(support_height_mm)) or support_height_mm < 0:
            raise ValueError("Support height must be nonnegative")
        if layer not in (0, 1, 2):
            raise ValueError("This initial implementation supports at most three layers")
        destination = self._check_xy(destination_xy)
        self._require_park()
        source = self.observe(name)
        # Avoid trying to place onto the source cube's own footprint.
        if math.dist(source["xy"], destination) < CUBE_MM + 5:
            raise ValueError("Source and destination need at least 35 mm centre separation")
        release_z = self.release_z + support_height_mm + layer * CUBE_MM
        # Include the new stack height, so the open gripper can retract above it.
        highest_top = max(CUBE_MM, support_height_mm + (layer + 1) * CUBE_MM)
        safe_z = int(math.ceil(max(self.base["z"], self.release_z) + highest_top + self.clearance))
        if safe_z > self.park["z"]:
            raise ValueError("Required clearance exceeds the taught camera-clear height")
        steps = self.plan_pick(source["xy"], safe_z) + self.plan_place(destination, release_z, safe_z)
        # Every endpoint checked before the first movement of this transaction.
        for label, pose in steps:
            self.arm.plan(pose, duration_ms=self.duration)
        report = dict(object=name, source=source, destination_xy=destination,
                      release_z=release_z, safe_z=safe_z,
                      steps=[dict(action=label, pose=pose) for label, pose in steps])
        print(json.dumps(report, indent=2))
        if not self.execute:
            return dict(status="planned_only", **report)
        self.confirm("Check the cube orientation, destination and complete route.", "START")
        for label, pose in steps:
            self.confirm(f"{label}: {pose}. Hands clear.", "MOVE")
            before = self.arm.state()
            if before["z"] < 5:
                raise RuntimeError("Reported Z below software floor; stopped")
            plan = self.arm.plan(pose, duration_ms=self.duration)
            after = self.arm.execute(plan)
            self._event("movement", action=label, requested=pose,
                        before=before, after=after)
            if after["z"] < 5 or any(abs(after[k] - pose[k]) > 10 for k in ("x", "y", "z")):
                raise RuntimeError(f"Following error at {label}: {after}")
            if abs(after["pitch"] - pose["pitch"]) > 5 or abs(after["roll"] - pose["roll"]) > 5:
                raise RuntimeError(f"Orientation error at {label}: {after}")
            if label == "lift":
                self.confirm("Confirm the cube is actually held. Cancel if it is not.", "HELD")
        self.confirm("Inspect the destination. Confirm the cube was placed correctly.", "PLACED")
        self.placed[name] = dict(xy=destination, layer=layer, support_height_mm=support_height_mm)
        self._event("placement_confirmed", object=name, placement=self.placed[name])
        return dict(status="operator_confirmed", **report)

    def stack(self, bottom_to_top, destination_xy, support_height_mm=0):
        if not 1 <= len(bottom_to_top) <= 3 or len(set(bottom_to_top)) != len(bottom_to_top):
            raise ValueError("Supply one to three distinct objects, bottom first")
        if set(bottom_to_top) - set(self.sources):
            raise ValueError("Stack order contains unknown objects")
        # Inspect all remaining sources before stacking. Keep the destination empty.
        scene = self.get_scene()
        destination = self._check_xy(destination_xy)
        if any(math.dist(o["xy"], destination) < CUBE_MM + 5 for o in scene["sources"]):
            raise ValueError("The stack destination overlaps a source cube")
        # Check the entire planned stack before starting the first pickup.
        positions = {o["name"]: o["xy"] for o in scene["sources"]}
        if not math.isfinite(float(support_height_mm)) or support_height_mm < 0:
            raise ValueError("Support height must be nonnegative")
        for layer, name in enumerate(bottom_to_top):
            release_z = self.release_z + support_height_mm + layer * CUBE_MM
            highest = max(CUBE_MM, support_height_mm + (layer + 1) * CUBE_MM)
            safe_z = int(math.ceil(max(self.base["z"], self.release_z) + highest + self.clearance))
            if safe_z > self.park["z"]:
                raise ValueError("Stack clearance exceeds the taught parking height")
            for _, pose in self.plan_pick(positions[name], safe_z) + self.plan_place(destination, release_z, safe_z):
                self.arm.plan(pose, duration_ms=self.duration)
        results = []
        for layer, name in enumerate(bottom_to_top):
            results.append(self.pick_place(name, destination, support_height_mm, layer))
        return dict(operation="stack", bottom_to_top=bottom_to_top, results=results)

    def stack_on(self, top, base, support_height_mm=0, destination_callback=None):
        """Leave the existing base cube in place; move only the top cube."""
        if top == base or top not in self.sources or base not in self.sources:
            raise ValueError("Choose two distinct known objects for top and base")
        if support_height_mm != 0:
            raise ValueError("stack-on currently requires both cubes on the original table")
        base_observation = self.observe(base)
        destination = base_observation["xy"]
        if destination_callback:
            destination_callback(destination)
        # Base cube is 30 mm high. Source cube still rests on the original table.
        result = self.pick_place(top, destination, support_height_mm, layer=1)
        return dict(operation="stack_on", top=top, base=base,
                    base_observation=base_observation, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("operation", choices=["view", "transfer", "stack", "stack-on"])
    parser.add_argument("--source", action="append", default=[],
                        help="NAME=colour:NAME, NAME=marker:ID, NAME=taught:POSE or NAME=xy:X,Y")
    parser.add_argument("--order", nargs="+", help="Stack order, bottom first")
    parser.add_argument("--top", help="Object to pick for stack-on")
    parser.add_argument("--base", help="Existing base cube to leave in place")
    parser.add_argument("--to", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--support-height", type=float, default=0,
                        help="Tray floor elevation above the calibrated table, mm")
    parser.add_argument("--release-z", type=float, help="Taught release Z on the original table")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--camera-config", default=str(ROOT / "config/camera.json"))
    parser.add_argument("--calibration", default=str(ROOT / "config/robot_camera.json"))
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--colour-profile", default=str(ROOT / "config/cube_colours.json"))
    parser.add_argument("--allow-approximate-grasps", action="store_true")
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    sources = {}
    for item in args.source:
        name, specification = item.split("=", 1)
        if not name or name in sources:
            parser.error("Object names must be nonempty and unique")
        kind, value = specification.split(":", 1)
        if kind not in ("xy", "marker", "taught", "colour", "color"):
            parser.error("Sources must use colour:NAME, xy:X,Y, taught:POSE or marker:ID")
        sources[name] = specification
    if args.operation == "transfer" and len(sources) != 1:
        parser.error("transfer requires exactly one source")
    if args.operation == "stack" and not args.order:
        parser.error("stack requires --order, bottom first")
    if args.operation in ("transfer", "stack") and (not sources or args.to is None):
        parser.error("transfer/stack require --source and --to")
    if args.operation == "stack-on":
        if not args.top or not args.base or args.top == args.base:
            parser.error("stack-on requires distinct --top and --base names")
        if args.top not in sources or args.base not in sources:
            parser.error("Both --top and --base must be declared with --source")
        if args.to is not None:
            parser.error("stack-on gets its destination from --base; omit --to")
    if args.operation == "view" and (args.execute or args.no_browser):
        parser.error("view is camera-only and requires the browser preview")
    if args.to is not None and not all(math.isfinite(v) for v in args.to):
        parser.error("Destination coordinates must be finite")
    calibration = json.loads(Path(args.calibration).read_text())
    config = load_config(args.camera_config)
    if config != calibration["camera_config"]:
        parser.error("Camera configuration differs from calibration")
    if not calibration.get("mapping"):
        parser.error("No completed calibration mapping found")
    print("Endpoint IK checks only; routes are not geometrically collision checked.")
    print("Fixed taught roll: source cubes must be aligned as during calibration.")
    print("Without --execute, observations and IK calculations only; no movement.")
    with open_camera(config) as camera:
        viewer = None
        try:
            if not args.no_browser:
                viewer = WorkspaceView(camera, config, calibration, args.colour_profile,
                                       args.to, args.web_port)
            if args.operation == "view":
                input("Camera-only preview active. Press Enter or Ctrl+C to close: ")
                return
            if viewer:
                input("Use the browser to position cubes. Hands clear; Enter to calculate the task: ")
            exit_code = 0
            try:
                with NexArm(args.port) as arm:
                    robot = KitRescue(arm, camera, config, calibration, sources,
                                     execute=args.execute, allow_unvalidated=args.allow_unvalidated,
                                     allow_extrapolation=args.allow_extrapolation,
                                     duration_ms=args.duration_ms, release_z=args.release_z,
                                     colour_profile=args.colour_profile,
                                     allow_approximate_grasps=args.allow_approximate_grasps)
                    if args.operation == "transfer":
                        result = robot.pick_place(next(iter(sources)), args.to, args.support_height)
                    elif args.operation == "stack":
                        result = robot.stack(args.order, args.to, args.support_height)
                    else:
                        callback = (lambda xy: setattr(viewer, "destination", xy)) if viewer else None
                        result = robot.stack_on(args.top, args.base, args.support_height,
                                                destination_callback=callback)
                    print(json.dumps(result, indent=2))
            except (ValueError, RuntimeError) as exc:
                print("Task stopped:", exc)
                exit_code = 2
            if viewer:
                input("Preview remains active. Press Enter to close: ")
            if exit_code:
                raise SystemExit(exit_code)
        finally:
            if viewer:
                viewer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. No automatic recovery movement was sent.")