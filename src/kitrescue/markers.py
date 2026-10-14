"""Capture and inspect ArUco markers without controlling the robot."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .camera import load_config, open_camera, prepare_image, camera_metadata
except ImportError:
    from camera import load_config, open_camera, prepare_image, camera_metadata


def draw_box(image, points, colour, label=None):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    x, y, w, h = cv2.boundingRect(points)
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), colour, 1)
    if label:
        cv2.putText(image, label, (x, max(15, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return [x, y, w, h]


def detect_markers(image, dictionary_name="DICT_4X4_50", debug_candidates=False):
    """Return observations in input-image pixels and a debug overlay."""
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    overlay = image.copy()
    observations = []
    if debug_candidates:
        for candidate in rejected:
            draw_box(overlay, candidate, (255, 0, 255))
    if ids is None:
        return observations, overlay
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        points = marker_corners.reshape(4, 2).astype(np.float32)
        transform = cv2.getPerspectiveTransform(square, points)
        centre = cv2.perspectiveTransform(
            np.array([[[0.5, 0.5]]], dtype=np.float32), transform)[0, 0]
        bbox = draw_box(overlay, points, (0, 255, 0), label=f"ID {int(marker_id)}")
        observations.append({"id": int(marker_id), "corners_px": points.tolist(),
                             "center_px": centre.tolist(), "bbox_xywh_px": bbox})
        cv2.circle(overlay, tuple(np.rint(centre).astype(int)), 4, (0, 0, 255), -1)
    return observations, overlay


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="Camera JSON; default config/camera.json")
    source.add_argument("--port", help="Legacy ESP32 port")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--output-dir", default="data/markers")
    parser.add_argument("--mirror", action="store_true", default=None,
                        help="Legacy override; normally set mirror in camera JSON")
    parser.add_argument("--debug-candidates", action="store_true")
    args = parser.parse_args()
    config = {"backend": "esp32", "port": args.port, "mirror": False} if args.port else load_config(
        args.config or "config/camera.json")
    if args.mirror is not None:
        config["mirror"] = args.mirror
    with open_camera(config) as camera:
        print(camera.status)
        frame = camera.capture()
        image = prepare_image(frame, config)
        source_metadata = camera_metadata(camera, config, image)
    observations, overlay = detect_markers(image, args.dictionary, args.debug_candidates)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = str(frame.received_at_ns)
    raw_path = frame.save(output / f"{stem}_raw.jpg")
    processed_path = output / f"{stem}_processed.png"
    overlay_path = output / f"{stem}_markers.png"
    save_image(processed_path, image)
    save_image(overlay_path, overlay)
    metadata = {"dictionary": args.dictionary, "width": image.shape[1],
                "height": image.shape[0], "sequence": frame.sequence,
                "received_at_ns": frame.received_at_ns,
                "timestamp_kind": "host_receive_time_not_sensor_exposure",
                "capture_ms": frame.capture_ms,
                "image_mirrored": config.get("mirror", False),
                "coordinate_image": processed_path.name,
                "camera": source_metadata, "markers": observations}
    metadata_path = output / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    # Sidecar next to the coordinate image is also read by cubes.py.
    processed_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Raw image: {raw_path}\nCoordinate image: {processed_path}")
    print(f"Debug overlay: {overlay_path}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()