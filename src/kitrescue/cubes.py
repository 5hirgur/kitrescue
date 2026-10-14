"""Sample colors and inspect cube candidates. No robot motion."""

import argparse
import copy
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


def rectangle(image, rect):
    x, y, w, h = map(int, rect)
    height, width = image.shape[:2]
    if (
        x < 0 or y < 0 or w <= 0 or h <= 0
        or x + w > width or y + h > height
    ):
        raise ValueError(f"Rectangle {rect} exceeds {width}x{height} image")
    return image[y:y+h, x:x+w]


def pixels(image, rectangles):
    return np.concatenate([
        rectangle(image, rect).reshape(-1, 3)
        for rect in rectangles
    ])


def image_metadata(path):
    sidecar = Path(path).with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text()).get("camera")
    return None


def geometry_signature(camera):
    if camera is None:
        return None
    controls = camera.get("controls", {})
    return {
        **{
            key: camera.get(key)
            for key in ("backend", "device", "width", "height", "mirror")
        },
        "focus": controls.get("focus_absolute"),
        "zoom": controls.get("zoom_absolute"),
    }


def default_samples():
    return {
        "roi_xywh": None,
        "parts": {
            "black": {
                "edge_mm": 30,
                "mode": "achromatic",
                "samples_xywh": [],
            },
            "grey": {
                "edge_mm": 30,
                "mode": "achromatic",
                "samples_xywh": [],
            },
            "orange": {
                "edge_mm": 30,
                "mode": "chromatic",
                "samples_xywh": [],
            },
        },
    }


def hue_statistics(values):
    angle = values[:, 0] * (2 * np.pi / 180)
    hue = float(
        np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
        * 180 / (2 * np.pi) % 180
    )
    errors = np.abs((values[:, 0] - hue + 90) % 180 - 90)
    return hue, errors


def learn_profile(image, samples, camera=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    roi = samples.get("roi_xywh")
    if roi is not None:
        rectangle(image, roi)

    parts = {}
    if not samples.get("parts"):
        raise ValueError("No named parts in the samples file.")

    for name, part in samples["parts"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid part name: {name}")

        patches = part.get("samples_xywh", [])
        if not patches:
            raise ValueError(f"Select patches for {name}.")

        values = pixels(hsv, patches).astype(float)
        saturation = values[:, 1]
        brightness = values[:, 2]

        if np.median(brightness) < 12:
            raise ValueError(
                f"{name} samples are too dark. Choose a visible surface."
            )

        mode = part.get("mode", "achromatic")
        edge = float(part.get("edge_mm", 30))
        if edge <= 0:
            raise ValueError(f"Invalid edge length for {name}")

        learned = {
            "mode": mode,
            "edge_mm": edge,
            "samples_xywh": patches,
            "sample_bgr": np.median(
                pixels(image, patches), axis=0
            ).tolist(),
        }

        if mode == "chromatic":
            hue, errors = hue_statistics(values)
            if (
                np.percentile(errors, 95) > 25
                or np.median(saturation) < 25
            ):
                raise ValueError(
                    f"{name} patches contain inconsistent colors. "
                    "Avoid edges, glare and background."
                )

            learned.update(
                hue=hue,
                hue_tolerance=float(max(
                    5, min(25, np.percentile(errors, 95) + 5)
                )),
                s_min=float(max(
                    25, np.percentile(saturation, 5) - 20
                )),
                v_min=float(max(
                    8, np.percentile(brightness, 5) * 0.6
                )),
            )

        elif mode == "achromatic":
            # Learn the recorded color cast instead of assuming that
            # physical black and grey always have low image saturation.
            learned.update(
                s_max=float(min(
                    255, max(25, np.percentile(saturation, 95) + 15)
                )),
                v_min=float(max(
                    0, np.percentile(brightness, 5) - 20
                )),
                v_max=float(min(
                    255, np.percentile(brightness, 95) + 20
                )),
            )

            if np.median(saturation) >= 45:
                hue, errors = hue_statistics(values)
                if np.percentile(errors, 95) <= 20:
                    learned.update(
                        cast_hue=hue,
                        cast_hue_tolerance=25.0,
                    )
        else:
            raise ValueError(f"Unknown mode for {name}: {mode}")

        parts[name] = learned

    return {
        "version": 2,
        "image_wh": [image.shape[1], image.shape[0]],
        "camera": camera,
        "geometry_signature": geometry_signature(camera),
        "roi_xywh": roi,
        "parts": parts,
    }


def validate_camera(profile, camera):
    warnings = []
    previous = profile.get("camera")

    if previous is not None and camera is not None:
        if geometry_signature(camera) != profile.get("geometry_signature"):
            raise ValueError(
                "Camera, resolution, mirror, focus or zoom changed. "
                "Recheck geometry and resample colors."
            )
        if previous.get("controls") != camera.get("controls"):
            warnings.append(
                "Camera lighting controls changed; validate or resample."
            )
    else:
        warnings.append("Camera metadata is missing.")

    controls = (camera or previous or {}).get("controls", {})
    if (
        controls.get("white_balance_automatic") == 1
        or controls.get("auto_exposure") == 3
    ):
        warnings.append("Exposure or white balance is automatic.")

    return warnings


def marker_exclusion(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    corners, ids, _ = detector.detectMarkers(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    if ids is not None:
        for corner in corners:
            points = corner.reshape(4, 2)
            centre = points.mean(axis=0)
            points = (points - centre) * 1.12 + centre
            cv2.fillConvexPoly(
                mask, np.rint(points).astype(np.int32), 255
            )
    return mask


def candidates(mask, name, min_area):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = mask.shape
    results = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1)
        extent = area / max(w * h, 1)

        if not (
            0.4 <= w / max(h, 1) <= 2.5
            and solidity >= 0.75
            and extent >= 0.45
        ):
            continue

        results.append({
            "colour": name,
            "bbox_xywh_px": [x, y, w, h],
            "bbox_center_px": [x + (w-1)/2, y + (h-1)/2],
            "contour_px": contour.reshape(-1, 2).tolist(),
            "area_px": float(area),
            "solidity": float(solidity),
            "touches_image_edge": (
                x == 0 or y == 0
                or x+w == width or y+h == height
            ),
            "geometry_kind": "silhouette_candidate_not_grasp_pose",
        })

    return results


def detect_cubes(image, profile, min_area):
    if profile.get("version") != 2:
        raise ValueError("Learn a new color profile.")

    if [image.shape[1], image.shape[0]] != profile["image_wh"]:
        raise ValueError("Image resolution differs from the color profile.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    allowed = np.full(image.shape[:2], 255, np.uint8)
    roi = profile.get("roi_xywh")

    if roi is not None:
        rectangle(image, roi)
        allowed[:] = 0
        rx, ry, rw, rh = roi
        allowed[ry:ry+rh, rx:rx+rw] = 255

    allowed[marker_exclusion(image) > 0] = 0
    masks = {}
    detections = []
    kernel = np.ones((3, 3), np.uint8)

    for name, part in profile["parts"].items():
        if part["mode"] == "chromatic":
            distance = np.abs((h - part["hue"] + 90) % 180 - 90)
            selected = (
                (distance <= part["hue_tolerance"])
                & (s >= part["s_min"])
                & (v >= part["v_min"])
            )
        else:
            selected = (
                (s <= part["s_max"])
                & (v >= part["v_min"])
                & (v <= part["v_max"])
            )
            if "cast_hue" in part:
                distance = np.abs(
                    (h - part["cast_hue"] + 90) % 180 - 90
                )
                selected &= (
                    (s < 25)
                    | (distance <= part["cast_hue_tolerance"])
                )

        mask = cv2.bitwise_and(
            selected.astype(np.uint8) * 255, allowed
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.bitwise_and(mask, allowed)
        masks[name] = mask

        for item in candidates(mask, name, min_area):
            item["edge_mm"] = part["edge_mm"]
            item["touches_roi_edge"] = False
            if roi is not None:
                x, y, w, hh = item["bbox_xywh_px"]
                item["touches_roi_edge"] = (
                    x <= rx or y <= ry
                    or x+w >= rx+rw or y+hh >= ry+rh
                )
            detections.append(item)

    for item in detections:
        x, y, w, hh = item["bbox_xywh_px"]
        item["overlaps_other_colour"] = False
        for other in detections:
            if other["colour"] == item["colour"]:
                continue
            bx, by, bw, bh = other["bbox_xywh_px"]
            overlap = (
                max(0, min(x+w, bx+bw) - max(x, bx))
                * max(0, min(y+hh, by+bh) - max(y, by))
            )
            if overlap / max(min(w*hh, bw*bh), 1) > 0.15:
                item["overlaps_other_colour"] = True

    overlay = image.copy()
    if roi is not None:
        cv2.rectangle(
            overlay, (rx, ry), (rx+rw-1, ry+rh-1), (0, 255, 255), 1
        )

    for item in detections:
        x, y, w, hh = item["bbox_xywh_px"]
        ambiguous = item["overlaps_other_colour"]
        colour = (0, 0, 255) if ambiguous else (0, 255, 0)
        cv2.rectangle(
            overlay, (x, y), (x+w-1, y+hh-1), colour, 2
        )
        label = item["colour"] + (
            " ambiguous" if ambiguous else " candidate"
        )
        cv2.putText(
            overlay, label, (x, max(15, y-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1,
        )

    return detections, masks, overlay


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Cannot save {path}")


def sample_colors(image, image_path, samples_path, port):
    path = Path(samples_path)
    samples = (
        json.loads(path.read_text())
        if path.exists() else default_samples()
    )
    if not samples.get("parts"):
        samples = default_samples()

    samples = copy.deepcopy(samples)
    for part in samples["parts"].values():
        part["samples_xywh"] = []

    samples.update(
        roi_xywh=None,
        image_wh=[image.shape[1], image.shape[0]],
        source_image=str(image_path),
        source_sha256=hashlib.sha256(
            Path(image_path).read_bytes()
        ).hexdigest(),
    )

    saved = threading.Event()
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode the sampling image.")
    png = encoded.tobytes()

    page = r'''<!doctype html>
<meta charset="utf-8">
<title>KitRescue color samples</title>
<style>
body{font:16px system-ui;max-width:1400px;margin:20px auto;padding:0 15px;background:#182029;color:#eee}
button,select{font:inherit;margin:5px;padding:8px}
canvas{max-width:100%;height:auto;cursor:crosshair;touch-action:none}
#status{white-space:pre-wrap}
</style>
<h2>KitRescue color sampling</h2>
<p>Draw the workspace first. Then select each color and draw 2–3 small
patches inside its top face. Avoid edges, tray pixels and reflections.</p>
<select id="kind"><option value="__workspace">Workspace</option></select>
<button id="undo">Undo selected</button>
<button id="save">Save and continue</button>
<span id="coordinates"></span>
<p id="status">Loading…</p>
<canvas id="canvas"></canvas>
<script>
let data=__DATA__, active=null;
const sel=document.getElementById('kind');
const canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d');
const status=document.getElementById('status');
const photo=new Image();

for(const name of Object.keys(data.parts)){
  let o=document.createElement('option');
  o.value=name;o.textContent=name;sel.appendChild(o);
}
function pos(e){
  let r=canvas.getBoundingClientRect();
  return [
    Math.max(0,Math.min(canvas.width-1,
      Math.floor((e.clientX-r.left)*canvas.width/r.width))),
    Math.max(0,Math.min(canvas.height-1,
      Math.floor((e.clientY-r.top)*canvas.height/r.height)))
  ];
}
function rect(a,b){
  return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),
    Math.abs(a[0]-b[0])+1,Math.abs(a[1]-b[1])+1];
}
function box(r,label,color){
  if(!r)return;
  ctx.strokeStyle=color;ctx.fillStyle=color;
  ctx.lineWidth=Math.max(2,canvas.width/700);
  ctx.strokeRect(...r);
  ctx.font=Math.max(16,canvas.width/80)+'px sans-serif';
  ctx.fillText(label,r[0],Math.max(20,r[1]-4));
}
function draw(){
  ctx.drawImage(photo,0,0);
  box(data.roi_xywh,'Workspace','#ffdb4d');
  for(let [name,part] of Object.entries(data.parts))
    for(let r of part.samples_xywh)box(r,name,'#ff74e3');
  if(active)box(rect(active.start,active.end),'New','#8aff83');
}
function summary(){
  status.textContent=Object.entries(data.parts)
    .map(([n,p])=>n+': '+p.samples_xywh.length+' patches').join(' | ');
}
photo.onload=()=>{
  canvas.width=photo.naturalWidth;
  canvas.height=photo.naturalHeight;
  draw();summary();
};
photo.src='/image.png';
canvas.onpointerdown=e=>{
  canvas.setPointerCapture(e.pointerId);
  let p=pos(e);active={start:p,end:p};draw();
};
canvas.onpointermove=e=>{
  let p=pos(e);
  document.getElementById('coordinates').textContent='Pixel '+p.join(', ');
  if(active){active.end=p;draw();}
};
canvas.onpointerup=e=>{
  if(!active)return;
  let r=rect(active.start,pos(e));active=null;
  if(r[2]>=3&&r[3]>=3){
    if(sel.value==='__workspace')data.roi_xywh=r;
    else data.parts[sel.value].samples_xywh.push(r);
  }
  draw();summary();
};
canvas.onpointercancel=()=>{active=null;draw();};
document.getElementById('undo').onclick=()=>{
  if(sel.value==='__workspace')data.roi_xywh=null;
  else data.parts[sel.value].samples_xywh.pop();
  draw();summary();
};
document.getElementById('save').onclick=async()=>{
  try{
    let response=await fetch('/save',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(data)
    });
    status.textContent=await response.text();
  }catch(error){status.textContent=String(error);}
};
</script>'''.replace(
        "__DATA__", json.dumps(samples).replace("<", "\\u003c")
    )

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload, content_type="text/plain"):
            if isinstance(payload, str):
                payload = payload.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif route == "/image.png":
                self.reply(200, png, "image/png")
            else:
                self.reply(404, "Not found")

        def do_POST(self):
            if self.path != "/save":
                self.reply(404, "Not found")
                return
            try:
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected JSON")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    raise ValueError("Invalid request size")

                received = json.loads(self.rfile.read(length))
                validated = copy.deepcopy(samples)
                roi = received.get("roi_xywh")
                if roi is None:
                    raise ValueError("Draw the workspace first.")

                rectangle(image, roi)
                roi = list(map(int, roi))
                validated["roi_xywh"] = roi
                rx, ry, rw, rh = roi

                for name, part in validated["parts"].items():
                    patches = received["parts"][name]["samples_xywh"]
                    if not patches:
                        raise ValueError(
                            f"Draw at least one patch for {name}."
                        )
                    for patch in patches:
                        rectangle(image, patch)
                        x, y, w, h = map(int, patch)
                        if not (
                            rx <= x and ry <= y
                            and x+w <= rx+rw and y+h <= ry+rh
                        ):
                            raise ValueError(
                                f"{name} patch is outside the workspace."
                            )
                    part["samples_xywh"] = [
                        list(map(int, patch)) for patch in patches
                    ]

                learn_profile(image, validated, image_metadata(image_path))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(validated, indent=2) + "\n")
                saved.set()
                self.reply(
                    200,
                    "Saved. Return to the terminal; "
                    "processing continues automatically.",
                )
                threading.Thread(
                    target=self.server.shutdown, daemon=True
                ).start()

            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, f"Not saved: {exc}")

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{port}", flush=True)
        print(
            f"In VSCode Remote SSH, forward port {port}. "
            "Draw samples and click Save and continue.",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

    if not saved.is_set():
        raise ValueError("Sampling cancelled; no new profile created.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", action="version",
        version="KitRescue cubes: automatic browser sampling",
    )
    parser.add_argument(
        "mode", choices=("sample", "learn", "detect", "check")
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--samples", default="config/colour_samples.json")
    parser.add_argument("--profile", default="config/cube_colours.json")
    parser.add_argument("--output-dir", default="data/cube_outputs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--min-area", type=float, default=900)
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("Invalid port.")
    if args.min_area <= 0:
        parser.error("--min-area must be positive.")

    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Cannot read {args.image}")

    if args.mode == "sample":
        sample_colors(image, args.image, args.samples, args.port)
        return

    camera = image_metadata(args.image)
    profile_path = Path(args.profile)

    if args.mode == "learn":
        samples_path = Path(args.samples)
        samples = (
            json.loads(samples_path.read_text())
            if samples_path.exists() else {}
        )
        image_hash = hashlib.sha256(
            Path(args.image).read_bytes()
        ).hexdigest()

        missing = (
            not samples.get("parts")
            or any(
                not part.get("samples_xywh")
                for part in samples.get("parts", {}).values()
            )
            or samples.get("source_sha256") != image_hash
        )

        if args.resample or missing:
            sample_colors(image, args.image, args.samples, args.port)
            samples = json.loads(samples_path.read_text())

        profile = learn_profile(image, samples, camera)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")

        overlay = image.copy()
        for name, part in profile["parts"].items():
            for x, y, w, h in part["samples_xywh"]:
                cv2.rectangle(
                    overlay, (x, y), (x+w-1, y+h-1), (0, 255, 255), 1
                )
                cv2.putText(
                    overlay, name, (x, max(15, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )
        write_image(profile_path.with_suffix(".samples.png"), overlay)
        print(f"Saved color profile: {profile_path}")
    else:
        profile = json.loads(profile_path.read_text())

    warnings = validate_camera(profile, camera)
    detections, masks, overlay = detect_cubes(
        image, profile, args.min_area
    )
    counts = {
        name: sum(item["colour"] == name for item in detections)
        for name in profile["parts"]
    }

    expected = {}
    for entry in args.expect:
        try:
            name, value = entry.split("=", 1)
            count = int(value)
            if (
                count < 0 or name in expected
                or name not in profile["parts"]
            ):
                raise ValueError
            expected[name] = count
        except ValueError:
            parser.error("Use --expect black=1 with valid part names.")

    passed = None
    if args.mode == "check":
        if not expected:
            parser.error("check requires --expect NAME=COUNT.")
        expected = {
            name: expected.get(name, 0) for name in profile["parts"]
        }
        passed = (
            counts == expected
            and not warnings
            and not any(
                item["overlaps_other_colour"]
                or item["touches_image_edge"]
                or item["touches_roi_edge"]
                for item in detections
            )
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    write_image(output / f"{stem}_overlay.png", overlay)
    for name, mask in masks.items():
        write_image(output / f"{stem}_{name}.png", mask)

    report = {
        "image": args.image,
        "profile": str(profile_path),
        "warnings": warnings,
        "counts": counts,
        "expected": expected,
        "color_check_passed": passed,
        "check_scope": "Color candidates only; not robot alignment.",
        "candidates": detections,
    }
    (output / f"{stem}_cubes.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps({
        "counts": counts,
        "warnings": warnings,
        "color_check_passed": passed,
    }, indent=2))
    print(f"Inspect: {output / f'{stem}_overlay.png'}")

    if args.mode == "check" and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}")