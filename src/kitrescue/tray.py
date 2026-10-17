"""Markerless tray outline detection. Vision only: never moves the robot."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def ordered_corners(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(
        points[:, 1] - centre[1],
        points[:, 0] - centre[0],
    )
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)



def _refine_tray_quad(points, contour):
    """Fit boundary lines while ignoring short contour branches."""
    cloud = contour.reshape(-1, 2).astype(np.float32)
    lines = []

    for a, b in zip(points, np.roll(points, -1, axis=0)):
        vector = b - a
        length = np.linalg.norm(vector)
        if length < 1:
            return points

        unit = vector / length
        relative = cloud - a
        along = relative @ unit
        distance = np.abs(
            relative[:, 0] * unit[1] - relative[:, 1] * unit[0]
        )
        selected = cloud[
            (along > length * .08)
            & (along < length * .92)
            & (distance < 8)
        ]
        if len(selected) < 12:
            return points

        vx, vy, x, y = cv2.fitLine(
            selected, cv2.DIST_WELSCH, 0, .01, .01
        ).flatten()
        normal = np.array([-vy, vx])
        lines.append((normal, float(normal @ np.array([x, y]))))

    result = []
    for i in range(4):
        n1, d1 = lines[i - 1]
        n2, d2 = lines[i]
        matrix = np.stack([n1, n2])
        if abs(np.linalg.det(matrix)) < .15:
            return points
        result.append(np.linalg.solve(matrix, [d1, d2]))

    result = np.float32(result)
    if (
        not cv2.isContourConvex(result)
        or np.linalg.norm(result - points, axis=1).max() > 15
    ):
        return points
    return result


def detect_tray(image, outer_mm=(180, 140), height_mm=20):
    """Return an outline observation and overlay in input-image pixels.

    Corner zero is an image convention, not a persistent physical corner.
    This function does not calculate robot coordinates.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale)
    h, w = small.shape[:2]

    gray = cv2.GaussianBlur(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 120), (179, 100, 255))
    candidates = []

    for low, high in ((10, 30), (20, 60), (30, 90), (50, 150), (80, 180)):
        edges = cv2.Canny(gray, low, high)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        distance = cv2.distanceTransform(
            255 - edges,
            cv2.DIST_L2,
            3,
        )

        # Add closed white-region outlines when Canny contours fragment.
        region = cv2.inRange(hsv, (0, 0, 155), (179, 50, 255))
        region = cv2.morphologyEx(
            region, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        region_contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = list(contours) + list(region_contours)

        for contour in contours:
            area = cv2.contourArea(contour)
            if not 0.02 * h * w < area < 0.55 * h * w:
                continue

            polygon = cv2.approxPolyDP(cv2.convexHull(contour), 0.02 * cv2.arcLength(cv2.convexHull(contour), True), True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue

            points = _refine_tray_quad(ordered_corners(polygon), contour)

            # Require the whole outline to be visible.
            if (
                points[:, 0].min() < 3
                or points[:, 1].min() < 3
                or points[:, 0].max() > w - 4
                or points[:, 1].max() > h - 4
            ):
                continue

            sides = np.linalg.norm(
                np.roll(points, -1, axis=0) - points,
                axis=1,
            )
            if sides.min() < 25 or sides.max() / sides.min() > 4:
                continue

            interior = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(
                interior,
                points.astype(np.int32),
                255,
            )
            interior = cv2.erode(
                interior,
                np.ones((7, 7), np.uint8),
            )
            coverage = float(np.mean(white[interior > 0] > 0))
            if coverage < 0.65:
                continue

            # Check that all four proposed sides have nearby image edges.
            support = []
            for a, b in zip(points, np.roll(points, -1, axis=0)):
                pixels = np.rint(
                    a
                    + np.linspace(0.08, 0.92, 80)[:, None]
                    * (b - a)
                ).astype(int)
                support.append(
                    float(
                        np.mean(
                            distance[pixels[:, 1], pixels[:, 0]] <= 3
                        )
                    )
                )

            if min(support) < 0.55:
                continue

            candidates.append({
                "points": points,
                "white_coverage": coverage,
                "edge_support": support,
                "score": coverage + float(np.mean(support)),
            })

    # Merge repeated detections of the same rectangle.
    unique = []
    for item in sorted(
        candidates,
        key=lambda candidate: candidate["score"],
        reverse=True,
    ):
        duplicate = False

        for other in unique:
            intersection, _ = cv2.intersectConvexConvex(
                item["points"],
                other["points"],
            )
            area_a = cv2.contourArea(item["points"])
            area_b = cv2.contourArea(other["points"])
            overlap = intersection / max(
                1,
                area_a + area_b - intersection,
            )
            containment = intersection / max(1, min(area_a, area_b))
            area_ratio = min(area_a, area_b) / max(1, max(area_a, area_b))
            centre_gap = np.linalg.norm(
                item["points"].mean(axis=0)
                - other["points"].mean(axis=0)
            )
            nested = (
                containment > 0.95
                and area_ratio > 0.60
                and centre_gap < 0.05 * np.sqrt(max(area_a, area_b))
                and abs(item["score"] - other["score"]) < 0.15
            )
            if overlap > 0.8 or nested:
                if nested and area_a > area_b:
                    other.update(item)
                duplicate = True
                break

        if not duplicate:
            unique.append(item)

    overlay = image.copy()
    result = {
        "status": "not_found",
        "image_size": [width, height],
        "outer_dimensions_mm": list(outer_mm),
        "height_mm": height_mm,
        "robot_xy_mm": None,
        "metric_pose_verified": False,
    }

    if not unique:
        return result, overlay

    best = unique[0]
    points = best["points"] / scale

    # Perspective-correct centre: intersection of rectangle diagonals.
    square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    transform = cv2.getPerspectiveTransform(
        square,
        points.astype(np.float32),
    )
    centre = cv2.perspectiveTransform(
        np.float32([[[0.5, 0.5]]]),
        transform,
    )[0, 0]

    ambiguous = (
        len(unique) > 1
        and best["score"] - unique[1]["score"] < 0.15
    )

    result.update(
        status="ambiguous" if ambiguous else "detected",
        corners_px=points.tolist(),
        center_px=centre.tolist(),
        edge_support=best["edge_support"],
        white_coverage=best["white_coverage"],
        candidate_count=len(unique),
        geometry_kind="outer_outline",
    )

    colour = (0, 180, 255) if ambiguous else (0, 255, 0)
    cv2.polylines(
        overlay,
        [np.rint(points).astype(np.int32)],
        True,
        colour,
        2,
    )
    cv2.drawMarker(
        overlay,
        tuple(np.rint(centre).astype(int)),
        colour,
        cv2.MARKER_CROSS,
        16,
        2,
    )

    for index, point in enumerate(points):
        cv2.putText(
            overlay,
            str(index),
            tuple(np.rint(point).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )

    return result, overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        help="Already processed image; no additional mirroring.",
    )
    source.add_argument(
        "--camera-config",
        help="Capture using the existing camera module.",
    )
    parser.add_argument(
        "--outer-mm",
        nargs=2,
        type=float,
        default=[180, 140],
    )
    parser.add_argument("--height-mm", type=float, default=20)
    parser.add_argument("--calibration")
    parser.add_argument("--output-dir", default="data/tray")
    args = parser.parse_args()

    if min(args.outer_mm) <= 0 or args.height_mm <= 0:
        parser.error("Dimensions must be positive millimetres.")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Cannot read {args.image}")
    else:
        try:
            from .camera import load_config, open_camera, prepare_image
        except ImportError:
            from camera import load_config, open_camera, prepare_image

        config = load_config(args.camera_config)
        with open_camera(config) as camera:
            frame = camera.capture()
            image = prepare_image(frame, config)

    result, overlay = detect_tray(
        image,
        args.outer_mm,
        args.height_mm,
    )

    if args.calibration:
        calibration = json.loads(
            Path(args.calibration).read_text()
        )
        settings = calibration.get(
            "camera",
            calibration.get("camera_config", {}),
        )
        expected = [
            settings.get("width"),
            settings.get("height"),
        ]
        if expected != [image.shape[1], image.shape[0]]:
            raise ValueError(
                "Image resolution differs from calibration; "
                "use an original camera frame."
            )

        hull = np.float32(
            calibration["mapping"]["pixel_hull"]
        )
        cv2.polylines(
            overlay,
            [np.rint(hull).astype(np.int32)],
            True,
            (0, 200, 255),
            2,
        )
        cv2.putText(
            overlay,
            "Yellow: 30 mm reference-plane coverage",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

        if "center_px" in result:
            result["center_in_reference_image_hull"] = bool(
                cv2.pointPolygonTest(
                    hull,
                    tuple(result["center_px"]),
                    False,
                ) >= 0
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name, pixels in (
        ("current_processed.png", image),
        ("current_overlay.png", overlay),
    ):
        if not cv2.imwrite(str(output / name), pixels):
            raise RuntimeError(f"Cannot save {name}")

    (output / "current.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    print(f"Overlay: {output / 'current_overlay.png'}")


if __name__ == "__main__":
    main()