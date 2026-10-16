"""Find cube top faces using colour, edges and the existing calibration.

Diagnostic only: this file never imports or commands the robot.
"""

import argparse
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    from .cubes import detect_cubes
except ImportError:
    from cubes import detect_cubes


def project(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return cv2.perspectiveTransform(
        points.reshape(1, -1, 2), matrix
    )[0]


def draw_face(image, record):
    if "proposal" not in record:
        return

    proposal = record["proposal"]
    colour = (
        (0, 255, 0)
        if record["status"] == "ok"
        else (0, 165, 255)
    )

    corners = np.rint(proposal["corners_px"]).astype(np.int32)
    centre = tuple(np.rint(proposal["center_px"]).astype(int))

    cv2.polylines(image, [corners], True, colour, 2)
    cv2.drawMarker(
        image, centre, colour, cv2.MARKER_CROSS, 15, 2
    )

    cv2.putText(
        image,
        record["colour"] + ": " + record["status"],
        tuple(corners.min(axis=0) - [0, 7]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colour,
        1,
    )


def fit_top_face(image, colour_mask, detection, homography, hull):
    """Rectify a local crop, then match a square to observed line edges."""
    name = detection["colour"]

    record = {
        "colour": name,
        "status": "no_face",
        "reason": "",
        "silhouette_center_px": detection["bbox_center_px"],
    }

    if detection.get("touches_image_edge"):
        record["reason"] = "Object is clipped by the image boundary"
        return record

    side = float(detection.get("edge_mm", 30))
    x, y, width, height = detection["bbox_xywh_px"]

    boundary = project(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        homography,
    )

    lo = boundary.min(axis=0) - 4
    hi = boundary.max(axis=0) + 4

    if (
        not np.isfinite(boundary).all()
        or np.any(hi - lo > side * 4)
    ):
        record["reason"] = (
            "Colour region is too large for an isolated cube"
        )
        return record

    # Rectification resolution, in pixels per mapped millimetre.
    scale = 4.0

    transform = np.array([
        [scale, 0, -lo[0] * scale],
        [0, scale, -lo[1] * scale],
        [0, 0, 1.0],
    ])

    warp = transform @ homography
    wh = np.ceil((hi - lo) * scale).astype(int)

    if np.any(wh < 10):
        record["reason"] = "Candidate is too small"
        return record

    patch = cv2.warpPerspective(image, warp, tuple(wh))
    mask = cv2.warpPerspective(
        colour_mask,
        warp,
        tuple(wh),
        flags=cv2.INTER_NEAREST,
    )

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Enhance grayscale contrast for geometry only.
    # The colour profile and original colour image are unchanged.
    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    ).apply(gray)

    edges = (
        cv2.Canny(
            cv2.GaussianBlur(gray, (3, 3), 0), 15, 45
        )
        | cv2.Canny(
            cv2.GaussianBlur(enhanced, (3, 3), 0), 25, 75
        )
    )

    # Black faces can have useful but weak intensity boundaries.
    if np.any(mask) and np.median(gray[mask > 0]) < 60:
        edges |= cv2.Canny(
            cv2.GaussianBlur(gray, (3, 3), 0), 5, 15
        )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        30,
        minLineLength=int(side * scale * 0.45),
        maxLineGap=int(scale * 3),
    )

    if lines is None:
        record["reason"] = "Not enough straight edges"
        return record

    lines = lines.reshape(-1, 2, 2).astype(float) / scale
    lengths = np.linalg.norm(
        lines[:, 1] - lines[:, 0], axis=1
    )

    # Prefer longer lines and keep the search bounded.
    lines = lines[np.argsort(-lengths)][:60]

    angles = []

    for a, b in lines:
        vector = b - a
        angle = np.arctan2(
            vector[1], vector[0]
        ) % (np.pi / 2)

        similar = any(
            abs(
                (angle - old + np.pi / 4)
                % (np.pi / 2)
                - np.pi / 4
            ) < np.deg2rad(3)
            for old in angles
        )

        if not similar:
            angles.append(angle)

        if len(angles) >= 12:
            break

    distance = cv2.distanceTransform(
        255 - edges,
        cv2.DIST_L2,
        3,
    ) / scale

    proposals = []
    fractions = np.linspace(0.08, 0.92, 25)

    interior = np.array([
        [a, b]
        for a in [0.2, 0.35, 0.5, 0.65, 0.8]
        for b in [0.2, 0.35, 0.5, 0.65, 0.8]
    ])

    for angle in angles:
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])

        uv = lines @ rotation
        vectors = uv[:, 1] - uv[:, 0]
        length = np.linalg.norm(vectors, axis=1)

        u_edges = uv[
            np.abs(vectors[:, 0]) < length * 0.12
        ].mean(axis=1)[:, 0]

        v_edges = uv[
            np.abs(vectors[:, 1]) < length * 0.12
        ].mean(axis=1)[:, 1]

        if not len(u_edges) or not len(v_edges):
            continue

        # Allow modest edge noise and approximate command-space scale.
        for edge in side + np.array([-4.0, -2.0, 0.0, 2.0, 4.0]):
            if edge <= 0:
                continue

            us = np.unique(
                np.round(
                    np.r_[u_edges, u_edges - edge] * 2
                ) / 2
            )
            vs = np.unique(
                np.round(
                    np.r_[v_edges, v_edges - edge] * 2
                ) / 2
            )

            origins = (
                np.array(np.meshgrid(us, vs))
                .reshape(2, -1)
                .T
            )

            corners = (
                origins[:, None, :]
                + [
                    [0, 0],
                    [edge, 0],
                    [edge, edge],
                    [0, edge],
                ]
            ) @ rotation.T

            valid = (
                (corners >= 0)
                & (corners * scale < wh - 1)
            ).all(axis=(1, 2))

            corners = corners[valid]

            if not len(corners):
                continue

            centres = corners.mean(axis=1)
            cp = np.clip(
                np.rint(centres * scale).astype(int),
                [0, 0],
                wh - 1,
            )

            corners = corners[
                mask[cp[:, 1], cp[:, 0]] > 0
            ]

            if not len(corners):
                continue

            # Sample points along every proposed square edge.
            points = (
                corners[:, :, None, :]
                * (1 - fractions[None, None, :, None])
                + np.roll(corners, -1, axis=1)[:, :, None, :]
                * fractions[None, None, :, None]
            )

            px = np.clip(
                np.rint(points * scale).astype(int),
                [0, 0],
                wh - 1,
            )

            support = (
                distance[
                    px[:, :, :, 1],
                    px[:, :, :, 0],
                ] <= 0.75
            ).mean(axis=2)

            # Require evidence on all four sides.
            valid = (
                (np.sort(support, axis=1)[:, 1] >= 0.65)
                & (support.min(axis=1) >= 0.45)
                & (support.mean(axis=1) >= 0.8)
            )

            for quad, evidence in zip(
                corners[valid], support[valid]
            ):
                inside = (
                    quad[0]
                    + interior[:, 0, None] * (quad[1] - quad[0])
                    + interior[:, 1, None] * (quad[3] - quad[0])
                )

                ip = np.clip(
                    np.rint(inside * scale).astype(int),
                    [0, 0],
                    wh - 1,
                )

                coverage = float(
                    (
                        mask[ip[:, 1], ip[:, 0]] > 0
                    ).mean()
                )

                if coverage < 0.8:
                    continue

                centre_xy = quad.mean(axis=0) + lo
                centre_px = project(
                    [centre_xy],
                    np.linalg.inv(homography),
                )[0]

                if cv2.pointPolygonTest(
                    hull,
                    tuple(centre_px),
                    False,
                ) < 0:
                    continue

                score = float(
                    evidence.mean()
                    + 0.12 * coverage
                    - 0.15 * abs(edge - side) / side
                )

                proposals.append({
                    "score": score,
                    "xy_mm": centre_xy.tolist(),
                    "center_px": centre_px.tolist(),
                    "corners_px": project(
                        quad * scale,
                        np.linalg.inv(warp),
                    ).tolist(),
                    "side_mm": float(edge),
                    "edge_support": evidence.tolist(),
                    "colour_coverage": coverage,
                    "angle_deg": float(
                        np.degrees(angle) % 90
                    ),
                })

    if not proposals:
        record["reason"] = (
            "No square has enough edge and colour evidence"
        )
        return record

    proposals.sort(key=lambda item: -item["score"])
    best = proposals[0]

    # Reject distinct centres that fit nearly equally well.
    alternatives = [
        proposal
        for proposal in proposals[1:]
        if np.linalg.norm(
            np.array(proposal["xy_mm"]) - best["xy_mm"]
        ) > 3.0
    ]

    gap = (
        best["score"] - alternatives[0]["score"]
        if alternatives
        else None
    )

    record.update({
        "proposal": best,
        "score_gap": gap,
        "geometry_kind": "edge_supported_top_square",
        "orientation_note": (
            "Plane angle only; not a verified wrist-roll command"
        ),
    })

    if gap is not None and gap < 0.08:
        record.update({
            "status": "ambiguous",
            "reason": (
                "Different top-face centres fit almost equally well"
            ),
        })
    else:
        record.update({
            "status": "ok",
            "reason": "",
            "top_center_px": best["center_px"],
            "xy_mm": best["xy_mm"],
        })

    return record


def detect_top_faces(image, profile, calibration):
    """Return face hypotheses and an overlay. No silhouette-centre fallback."""
    expected = calibration.get("camera", {})
    size = [image.shape[1], image.shape[0]]

    if expected.get("width") and size != [
        expected["width"], expected["height"]
    ]:
        raise ValueError(
            "Image resolution differs from robot-camera calibration"
        )

    mapping = calibration["mapping"]

    matrix = np.asarray(
        mapping["homography_pixel_to_command_xy"],
        dtype=float,
    )
    hull = np.float32(
        mapping["pixel_hull"]
    ).reshape(-1, 2)

    active_profile = copy.deepcopy(profile)

    # An old manual ROI must not clip the cube.
    # Accepted face centres must still lie inside the calibrated hull.
    active_profile["roi_xywh"] = None

    detected = detect_cubes(
        image, active_profile, min_area=900
    )
    objects, masks = detected[0], detected[1]

    overlay = image.copy()
    cv2.polylines(
        overlay,
        [np.rint(hull).astype(np.int32)],
        True,
        (255, 255, 0),
        2,
    )

    records = []

    for candidate in objects:
        x, y, w, h = candidate["bbox_xywh_px"]

        bounds = np.float32([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ])

        if cv2.intersectConvexConvex(
            bounds,
            cv2.convexHull(hull),
        )[0] <= 0:
            continue

        record = fit_top_face(
            image,
            masks[candidate["colour"]],
            candidate,
            matrix,
            hull,
        )
        records.append(record)

        old = tuple(
            np.rint(
                candidate["bbox_center_px"]
            ).astype(int)
        )

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 180, 180),
            1,
        )
        cv2.circle(
            overlay, old, 3, (0, 180, 180), -1
        )

        draw_face(overlay, record)

    return records, overlay


PAGE = '''<!doctype html><meta charset="utf-8">
<title>KitRescue top faces</title>
<style>
body{background:#182029;color:#eee;font:16px system-ui;max-width:1100px;margin:20px auto}
img{width:100%;height:auto}pre{white-space:pre-wrap}
</style>
<h2>Top-face test — camera only</h2>
<p>Green: accepted top-face hypothesis. Orange: ambiguous hypothesis.
Small yellow dot: old silhouette centre. Cross: proposed top-face centre.</p>
<p>Check the outline follows the four top edges. Robot motion is disabled in this tool.</p>
<img id="view"><pre id="status"></pre>
<script>
const image=document.getElementById('view'),status=document.getElementById('status');
function next(){image.src='/frame.jpg?t='+Date.now()}
image.onload=async()=>{
  try{status.textContent=JSON.stringify(await(await fetch('/status')).json(),null,2)}
  catch(e){status.textContent=String(e)}
  setTimeout(next,500)
};
image.onerror=()=>{
  status.textContent='Check the terminal for camera errors';
  setTimeout(next,1500)
};
next();
</script>'''


def save_result(records, overlay, calibration, output):
    output.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(
        str(output / "overlay.png"), overlay
    ):
        raise RuntimeError("Cannot write overlay")

    result = {
        "calibration_accepted": calibration["mapping"]["accepted"],
        "diagnostic_only": True,
        "source_plane": (
            "30 mm cube tops on the original table"
        ),
        "objects": records,
    }

    (output / "centres.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    return result


def live_view(profile, calibration, config_path, port, output):
    try:
        from .camera import (
            load_config,
            open_camera,
            prepare_image,
        )
    except ImportError:
        from camera import (
            load_config,
            open_camera,
            prepare_image,
        )

    config = load_config(config_path)

    if config != calibration["camera_config"]:
        raise ValueError(
            "Camera configuration differs from calibration"
        )

    latest = {}
    lock = threading.Lock()

    with open_camera(config) as camera:

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    if self.path == "/":
                        body = PAGE.encode()
                        kind = "text/html; charset=utf-8"

                    elif self.path.startswith("/frame.jpg"):
                        with lock:
                            raw = prepare_image(
                                camera.capture(), config
                            )

                            records, overlay = detect_top_faces(
                                raw, profile, calibration
                            )

                            latest.clear()
                            latest.update(
                                save_result(
                                    records,
                                    overlay,
                                    calibration,
                                    output,
                                )
                            )

                            if not cv2.imwrite(
                                str(output / "input.png"), raw
                            ):
                                raise RuntimeError(
                                    "Cannot save input frame"
                                )

                            # Display crop only; saved coordinates stay native.
                            hull = np.float32(
                                calibration["mapping"]["pixel_hull"]
                            )

                            lo = np.maximum(
                                np.floor(
                                    hull.min(axis=0)
                                ).astype(int) - 80,
                                [0, 0],
                            )
                            hi = np.minimum(
                                np.ceil(
                                    hull.max(axis=0)
                                ).astype(int) + 81,
                                [raw.shape[1], raw.shape[0]],
                            )

                            ok, encoded = cv2.imencode(
                                ".jpg",
                                overlay[
                                    lo[1]:hi[1],
                                    lo[0]:hi[0],
                                ],
                            )

                            if not ok:
                                raise RuntimeError(
                                    "Cannot encode preview"
                                )

                            body = encoded.tobytes()
                            kind = "image/jpeg"

                    elif self.path == "/status":
                        with lock:
                            body = json.dumps(latest).encode()
                            kind = "application/json"

                    else:
                        self.send_error(404)
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header(
                        "Content-Length", str(len(body))
                    )
                    self.send_header(
                        "Cache-Control", "no-store"
                    )
                    self.end_headers()
                    self.wfile.write(body)

                except (
                    BrokenPipeError,
                    ConnectionResetError,
                ):
                    pass

                except Exception as exc:
                    print(
                        "Preview error:", exc, flush=True
                    )
                    self.send_error(500, str(exc))

        server = ThreadingHTTPServer(
            ("127.0.0.1", port), Handler
        )
        threading.Thread(
            target=server.serve_forever,
            daemon=True,
        ).start()

        print(
            f"Forward VSCode port {port}; "
            f"open http://localhost:{port}"
        )

        try:
            input(
                "Camera-only test. Press Enter to stop: "
            )
        finally:
            server.shutdown()
            server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )
    source.add_argument("--image")
    source.add_argument(
        "--live", action="store_true"
    )

    parser.add_argument(
        "--profile",
        default="config/cube_colours.json",
    )
    parser.add_argument(
        "--calibration",
        default="config/robot_camera.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/top_faces",
    )
    parser.add_argument(
        "--camera-config",
        default="config/camera.json",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8766,
    )

    args = parser.parse_args()

    profile = json.loads(
        Path(args.profile).read_text()
    )
    calibration = json.loads(
        Path(args.calibration).read_text()
    )
    output = Path(args.output_dir)

    if args.live:
        live_view(
            profile,
            calibration,
            args.camera_config,
            args.web_port,
            output,
        )
        return

    image = cv2.imread(args.image)

    if image is None:
        raise SystemExit(
            "Cannot read image: " + args.image
        )

    records, overlay = detect_top_faces(
        image, profile, calibration
    )

    result = save_result(
        records, overlay, calibration, output
    )

    print(json.dumps(result, indent=2))
    print("Overlay:", output / "overlay.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped camera-only test.")