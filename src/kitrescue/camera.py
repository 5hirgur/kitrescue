"""ESP32 serial and Linux webcam capture. No robot commands."""

import argparse
import copy
import json
import struct
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import serial


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    sequence: int
    received_at_ns: int
    capture_ms: float

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.jpeg)
        return path

    def as_image(self):
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("JPEG decoding failed.")
        return image


class Camera:
    """Original ESP32 interface; Camera(port, baudrate) remains supported."""

    def __init__(self, port, baudrate=921600, timeout=10):
        self.timeout = timeout
        self.serial = serial.Serial(port, baudrate, timeout=0.1,
                                    write_timeout=2, exclusive=True)
        try:
            time.sleep(3)
            self.serial.reset_input_buffer()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.serial.write(b"PING\n")
                response = self.serial.readline().decode("ascii", errors="replace").strip()
                if response.startswith("KRCAM/1 READY"):
                    self.status = response
                    break
                if response.startswith("KRCAM/1 ERROR"):
                    raise RuntimeError(response)
                time.sleep(0.2)
            else:
                raise TimeoutError("Camera did not answer PING.")
        except Exception:
            self.close()
            raise

    def _read_exact(self, length, deadline):
        data = bytearray()
        while len(data) < length:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Received {len(data)} of {length} expected bytes.")
            data.extend(self.serial.read(length - len(data)))
        return bytes(data)

    def capture(self):
        self.serial.reset_input_buffer()
        start = time.monotonic()
        deadline = start + self.timeout
        self.serial.write(b"CAPTURE\n")
        self.serial.flush()
        window = bytearray()
        while bytes(window) != b"KRC1":
            window.extend(self._read_exact(1, deadline))
            if len(window) > 4:
                del window[0]
        sequence, length, checksum = struct.unpack("<III", self._read_exact(12, deadline))
        if length == 0:
            raise RuntimeError("ESP32 failed to capture an image.")
        if length > 1_000_000:
            raise RuntimeError(f"Invalid JPEG length: {length}")
        jpeg = self._read_exact(length, deadline)
        if zlib.crc32(jpeg) != checksum:
            raise RuntimeError("Image CRC failed; retry capture.")
        if not jpeg.startswith(b"\xff\xd8"):
            raise RuntimeError("Received data is not a JPEG.")
        return Frame(jpeg, sequence, time.time_ns(), (time.monotonic() - start) * 1000)

    def close(self):
        self.serial.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def v4l(device, *args):
    try:
        result = subprocess.run(["v4l2-ctl", "--device", str(device), *args],
                                text=True, capture_output=True, timeout=10, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Install v4l-utils: sudo apt install v4l-utils") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError((exc.stderr + exc.stdout).strip()) from exc
    return result.stdout


def get_controls(device, names):
    if not names:
        return {}
    result = {}
    for line in v4l(device, "--get-ctrl=" + ",".join(names)).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = int(value.strip().split()[0])
    missing = set(names) - result.keys()
    if missing:
        raise RuntimeError(f"Camera did not report controls: {sorted(missing)}")
    return result


def set_controls(device, controls):
    """Apply saved controls; continue only after stable exact readback."""
    import time

    if not controls:
        return {}

    switches = {
        k: v for k, v in controls.items()
        if k in (
            "auto_exposure",
            "white_balance_automatic",
            "focus_automatic_continuous",
        )
    }
    values = {k: v for k, v in controls.items() if k not in switches}
    actual = {}
    last_error = None

    for attempt in range(8):
        try:
            for group in (switches, values):
                if group:
                    v4l(device, "--set-ctrl=" + ",".join(
                        f"{k}={int(v)}" for k, v in group.items()
                    ))

            # Reapply exposure/gain after other controls have settled.
            time.sleep(0.25)
            for key in ("exposure_time_absolute", "gain"):
                if key in controls:
                    v4l(device, f"--set-ctrl={key}={int(controls[key])}")

            time.sleep(0.25)
            actual = get_controls(device, list(controls))
            if actual == controls:
                time.sleep(0.25)
                actual = get_controls(device, list(controls))
                if actual == controls:
                    return actual
        except RuntimeError as exc:
            last_error = str(exc)

        print(f"Camera controls settling: attempt {attempt + 1}/8",
              flush=True)

    raise RuntimeError(
        f"Camera controls did not settle: requested={controls}, "
        f"actual={actual}; last_error={last_error}"
    )



def _webcam_device_index(device):
    import os
    import re

    path = Path(device).resolve(strict=True)

    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError(f"No camera access: {path}")

    match = re.fullmatch(r"/dev/video(\d+)", str(path))
    if not match:
        raise ValueError(
            f"Expected a V4L2 camera, got {path}"
        )

    return int(match.group(1))


class WebcamCamera:
    """Continuously drain video frames so idle time does not accumulate a backlog."""

    def __init__(self, config):
        import cv2
        self.config = copy.deepcopy(config)
        self.device = config["device"]
        self.timeout = float(config.get("timeout_seconds", 5))
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._sequence = 0
        self._thread = None
        self._cap = cv2.VideoCapture(_webcam_device_index(self.device), cv2.CAP_V4L2)
        try:
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open {self.device}; check connection, video group and other apps.")
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
            self._cap.set(cv2.CAP_PROP_FPS, config.get("fps", 30))
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; driver may ignore it.
            self.controls = {}
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            self.capture()
            self.controls = set_controls(self.device, config.get("controls", {}))
            deadline = time.monotonic() + float(config.get("warmup_seconds", 2))
            while True:
                frame = self.capture()
                if time.monotonic() >= deadline:
                    break
            image = frame.as_image()
            actual_wh = [image.shape[1], image.shape[0]]
            if actual_wh != [config["width"], config["height"]]:
                raise RuntimeError(f"Resolution differs: requested {config['width']}x{config['height']}, got {actual_wh}")
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            requested = config.get("controls", {})
            self.controls = get_controls(self.device, list(requested))
            if self.controls != requested:
                print("Startup control changes:", json.dumps({
                    "requested": requested, "actual": self.controls
                }))
                set_controls(self.device, requested)
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    self.capture()
                self.controls = get_controls(self.device, list(requested))
                if self.controls != requested:
                    raise RuntimeError("Controls remain different: " + json.dumps({
                        "requested": requested, "actual": self.controls
                    }))
            self.status = f"C920/V4L2 READY {actual_wh[0]} {actual_wh[1]} fps={self.fps:g}"
        except Exception:
            self.close()
            raise

    def _read_loop(self):
        try:
            while not self._stop.is_set():
                ok, image = self._cap.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames.")
                with self._condition:
                    self._sequence += 1
                    self._latest = (self._sequence, time.monotonic(), time.time_ns(), image)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def capture(self):
        import cv2
        start = time.monotonic()
        with self._condition:
            # Wait past two new deliveries; this also skips a read already in progress.
            target_sequence = self._sequence + 2
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._stop.is_set()
                or (self._latest is not None and self._latest[0] >= target_sequence),
                timeout=self.timeout,
            )
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            if not ready or self._stop.is_set():
                raise TimeoutError("No fresh webcam frame arrived.")
            sequence, _, received_at_ns, image = self._latest
            image = image.copy()
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Cannot encode webcam frame.")
        return Frame(jpeg.tobytes(), sequence, received_at_ns,
                     (time.monotonic() - start) * 1000)

    def freeze_lighting(self):
        """Read settled automatic values, switch to manual, and return saved controls."""
        values = get_controls(self.device, ["exposure_time_absolute", "gain",
                                           "white_balance_temperature"])
        controls = {**self.config.get("controls", {}), **values,
                    "auto_exposure": 1, "white_balance_automatic": 0}
        self.controls = set_controls(self.device, controls)
        self.config["controls"] = dict(self.controls)
        return copy.deepcopy(self.config)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        # Do not release while another thread is inside read().
        if self._thread is None or not self._thread.is_alive():
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("backend") not in ("webcam", "esp32"):
        raise ValueError("Camera backend must be webcam or esp32.")
    if not isinstance(config.get("mirror", False), bool):
        raise ValueError("mirror must be true or false.")
    return config


def open_camera(config):
    if config["backend"] == "webcam":
        return WebcamCamera(config)
    camera = Camera(config["port"], config.get("baudrate", 921600),
                    config.get("timeout_seconds", 10))
    camera.config = copy.deepcopy(config)
    return camera


def prepare_image(frame, config):
    import cv2
    image = frame.as_image()
    return cv2.flip(image, 1) if config.get("mirror", False) else image


def camera_metadata(camera, config, image):
    return {"backend": config["backend"],
            "device": config.get("device", config.get("port")),
            "width": image.shape[1], "height": image.shape[0],
            "mirror": config.get("mirror", False),
            "controls": dict(getattr(camera, "controls", {})),
            "reported_fps": getattr(camera, "fps", None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("capture", "setup"), default="capture")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--port", help="Legacy ESP32 serial mode")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--output", default="data/camera/test.jpg")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {
        "backend": "esp32", "port": args.port, "baudrate": args.baud, "mirror": False}
    if args.mode == "setup":
        if config["backend"] != "webcam" or not args.config:
            parser.error("setup requires a webcam --config")
        config = copy.deepcopy(config)
        config["controls"] = {k: v for k, v in config.get("controls", {}).items()
                              if k not in ("exposure_time_absolute", "gain", "white_balance_temperature")}
        config["controls"].update(auto_exposure=3, white_balance_automatic=1)
    with open_camera(config) as camera:
        print(camera.status)
        if args.mode == "setup":
            config = camera.freeze_lighting()
            # Consume several deliveries after changing controls.
            for _ in range(5):
                camera.capture()
        frame = camera.capture()
        image = prepare_image(frame, config)
        metadata = camera_metadata(camera, config, image)
        metadata.update(sequence=frame.sequence, received_at_ns=frame.received_at_ns,
                        capture_ms=frame.capture_ms,
                        timestamp_kind="host_receive_time_not_sensor_exposure")
        raw_path = frame.save(args.output)
        import cv2
        processed = raw_path.with_name(raw_path.stem + "_processed.png")
        if not cv2.imwrite(str(processed), image):
            raise RuntimeError(f"Cannot save {processed}")
        processed.with_suffix(".json").write_text(json.dumps({"camera": metadata}, indent=2))
        if args.mode == "setup":
            config_path = Path(args.config)
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            print(f"Saved manual exposure/white balance in {config_path}")
        print(json.dumps(metadata, indent=2))
        print(f"Raw: {raw_path}\nProcessed: {processed}")


if __name__ == "__main__":
    main()