"""Guided planar calibration. Browser preview; robot commands stay in terminal."""
import argparse
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

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def preview_server(camera, config, port):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                if self.path == '/':
                    body = b'<!doctype html>\n<title>KitRescue calibration</title>\n<h2>Calibration preview - no robot controls</h2>\n<p>Use the terminal for movement and recording.\nTop marker must be ID 5.</p>\n<img id="view" style="max-width:100%;height:auto">\n<script>\nconst view=document.getElementById(\'view\');\nfunction next(){view.src=\'/frame.jpg?t=\'+Date.now()}\nview.onload=()=>setTimeout(next,250);\nview.onerror=()=>setTimeout(next,1000);\nnext();\n</script>'
                    kind = 'text/html'
                elif self.path.startswith('/frame.jpg'):
                    frame = camera.capture()
                    image = prepare_image(frame, config)
                    _, overlay = detect_markers(image)
                    ok, encoded = cv2.imencode('.jpg', overlay)
                    if not ok:
                        raise RuntimeError('Preview encoding failed')
                    body = encoded.tobytes()
                    kind = 'image/jpeg'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.send_error(500, str(exc))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
MIN_Z_MM = 5

def check_z(pose, label):
    z = float(pose['z'])
    if not np.isfinite(z) or z < MIN_Z_MM:
        raise RuntimeError(f'{label}: Z={z} violates Z >= {MIN_Z_MM}. Calibration stopped.')

def move(arm, target, duration):
    check_z(target, 'Requested target')
    check_z(arm.state(), 'Current reported pose')
    plan = arm.plan(target, duration_ms=duration)
    if 'z' in plan['solution']:
        check_z(plan['solution'], 'IK endpoint')
    state = arm.execute(plan)
    check_z(state, 'Reported pose after movement')
    for key in ('x', 'y', 'z'):
        if abs(state[key] - target[key]) > 10:
            raise RuntimeError(f'Unexpected following error: {key}, {state}')
    if abs(state['pitch'] - target['pitch']) > 5:
        raise RuntimeError(f'Unexpected pitch difference: {state}')
    return state

def capture_marker(camera, config, marker_id):
    centres = []
    last = None
    for _ in range(5):
        frame = camera.capture()
        image = prepare_image(frame, config)
        observations, overlay = detect_markers(image)
        matches = [marker for marker in observations if marker['id'] == marker_id]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one ID {marker_id}')
        centres.append(matches[0]['center_px'])
        last = (frame, image, overlay, matches[0])
    points = np.asarray(centres, dtype=float)
    if np.ptp(points, axis=0).max() > 2:
        raise RuntimeError('Marker moved more than 2 pixels; leave cube still')
    return (points.mean(axis=0).tolist(), last)

def fit_mapping(samples, max_error):
    training = [sample for sample in samples if sample['role'] == 'fit']
    tests = [sample for sample in samples if sample['role'] == 'test']
    pixels = np.float64([sample['pixel_xy'] for sample in training])
    robot = np.float64([sample['commanded_xy_mm'] for sample in training])
    if np.linalg.matrix_rank(pixels - pixels.mean(axis=0)) < 2:
        raise RuntimeError('Calibration points do not span an image area')
    homography, _ = cv2.findHomography(pixels, robot, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise RuntimeError('Cannot fit homography')
    for sample in samples:
        predicted = cv2.perspectiveTransform(np.float64([[sample['pixel_xy']]]), homography)[0, 0]
        sample['predicted_xy_mm'] = predicted.tolist()
        sample['error_mm'] = float(np.linalg.norm(predicted - sample['commanded_xy_mm']))
    maximum = max((sample['error_mm'] for sample in tests))
    return {'homography_pixel_to_command_xy': homography.tolist(), 'pixel_hull': cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).tolist(), 'fit_rms_mm': float(np.sqrt(np.mean([sample['error_mm'] ** 2 for sample in training]))), 'held_out_max_error_mm': maximum, 'acceptance_threshold_mm': max_error, 'accepted': maximum <= max_error}

def pixel_to_robot(pixel_xy, calibration):
    """Reusable by picking tools; reject extrapolation."""
    mapping = calibration['mapping']
    if not mapping or not mapping['accepted']:
        raise ValueError('Calibration failed its held-out checks')
    hull = np.float32(mapping['pixel_hull'])
    point = tuple((float(value) for value in pixel_xy))
    if cv2.pointPolygonTest(hull, point, False) < 0:
        raise ValueError('Point lies outside the calibrated area')
    xy = cv2.perspectiveTransform(np.float64([[point]]), np.float64(mapping['homography_pixel_to_command_xy']))[0, 0]
    return (float(xy[0]), float(xy[1]))

def resume_session(args, base, park, config, targets):
    path = Path(args.output)
    if not args.resume:
        if path.exists():
            if input('Replace existing calibration? Type REPLACE: ') != 'REPLACE':
                raise SystemExit('Cancelled; saved samples unchanged')
        return None
    if not path.exists():
        raise ValueError('No saved calibration to resume')
    saved = json.loads(path.read_text())
    checks = dict(reference=base, park=park, camera_config=config, travel_z=args.travel_z, marker_id=args.marker_id)
    for key, value in checks.items():
        if saved.get(key) != value:
            raise ValueError(f'Cannot resume: {key} changed')
    samples = saved.get('samples', [])
    if len(samples) >= len(targets):
        raise ValueError('All samples already collected')
    for index, sample in enumerate(samples, 1):
        role, x, y = targets[index - 1]
        if sample.get('index') != index or sample.get('role') != role or sample.get('commanded_xy_mm') != [x, y]:
            raise ValueError('Saved samples do not match this target sequence')
    saved['mapping'] = None
    print(f'Resuming at sample {len(samples) + 1}/{len(targets)}')
    print('Camera, robot, surface and lighting must remain unchanged.')
    return saved

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default=DEFAULT_PORT)
    parser.add_argument('--camera-config', default=str(ROOT / 'config/camera.json'))
    parser.add_argument('--poses', default=str(ROOT / 'data/taught_poses.json'))
    parser.add_argument('--reference', default='grasp_reference')
    parser.add_argument('--park', default='camera_clear')
    parser.add_argument('--x-range', type=int, nargs=2, required=True)
    parser.add_argument('--y-range', type=int, nargs=2, required=True)
    parser.add_argument('--travel-z', type=int, required=True)
    parser.add_argument('--duration-ms', type=int, default=2500)
    parser.add_argument('--web-port', type=int, default=8766)
    parser.add_argument('--marker-id', type=int, default=5)
    parser.add_argument('--max-error-mm', type=float, default=3)
    parser.add_argument('--output', default=str(ROOT / 'config/robot_camera.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    poses = json.loads(Path(args.poses).read_text())['poses']
    reference = poses[args.reference]
    source = reference.get('last_requested_target')
    if not source:
        raise ValueError('Reference must include its commanded target')
    fields = ('x', 'y', 'z', 'pitch', 'roll', 'claw')
    base = {key: source[key] for key in fields}
    if args.travel_z < base['z'] + 40:
        raise ValueError('Travel Z must be at least 40 mm above grasp Z')
    park_record = poses[args.park]
    park_source = park_record.get('last_requested_target') or park_record['reported_state']
    park = {**base, **{key: park_source[key] for key in ('x', 'y', 'z')}}
    if park['z'] < args.travel_z:
        raise ValueError('Teach camera_clear at or above travel Z')
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    if xmax - xmin < 20 or ymax - ymin < 20:
        raise ValueError('Each calibration range must span at least 20 mm')
    xs = [int(round(value)) for value in np.linspace(xmin, xmax, 4)]
    ys = [int(round(value)) for value in np.linspace(ymin, ymax, 3)]
    targets = [('fit', x, y) for row, y in enumerate(ys) for x in (xs if row % 2 == 0 else xs[::-1])]
    for fx, fy in ((0.2, 0.25), (0.8, 0.25), (0.2, 0.75), (0.8, 0.75)):
        targets.append(('test', round(xmin + fx * (xmax - xmin)), round(ymin + fy * (ymax - ymin))))
    config = load_config(args.camera_config)
    if config['backend'] != 'webcam':
        raise ValueError('This routine requires the webcam backend')
    resumed_session = resume_session(args, base, park, config, targets)
    print('Grasp target:', base)
    print('Camera-clear target:', park)
    print('Travel Z:', args.travel_z)
    print('Targets:', targets)
    print('Release any held object first. Clear loose objects from the movement routes.')
    print('IK checks endpoints only. Inspect travel routes for clearance.')
    if input('Type READY after clearing the workspace: ') != 'READY':
        return
    with NexArm(args.port) as arm, open_camera(config) as camera:
        for _, x, y in targets:
            heights = dict.fromkeys((base['z'], args.travel_z, park['z']))
            for z in (base['z'], args.travel_z):
                endpoint = {**base, 'x': x, 'y': y, 'z': z}
                print('Checking endpoint:', endpoint, flush=True)
                try:
                    arm.plan(endpoint, duration_ms=args.duration_ms)
                except RuntimeError as exc:
                    raise RuntimeError(f'Preflight failed for {endpoint}: {exc}') from exc
        arm.plan(park, duration_ms=args.duration_ms)
        arm.plan({**park, 'z': args.travel_z}, duration_ms=args.duration_ms)
        initial = arm.state()
        initial_high = {**base, 'x': initial['x'], 'y': initial['y'], 'z': args.travel_z}
        arm.plan(initial_high, duration_ms=args.duration_ms)
        print('All endpoint IK calculations passed. No movements made yet.')
        server = preview_server(camera, config, args.web_port)
        print(f'Forward VSCode port {args.web_port}; open http://localhost:{args.web_port}')
        session = resumed_session or {'version': 1, 'created_at_ns': time.time_ns(), 'reference': base, 'cube_edge_mm': 30, 'marker_id': args.marker_id, 'travel_z': args.travel_z, 'park': park, 'camera_config': config, 'coordinate_space': 'commanded_robot_xy_mm', 'scope': '30 mm cubes on the calibrated support plane only', 'samples': [], 'mapping': None}
        try:
            if input('Type MOVE to reach travel Z, move to park XY, then raise to camera_clear: ') != 'MOVE':
                return
            move(arm, initial_high, args.duration_ms)
            move(arm, {**park, 'z': args.travel_z}, args.duration_ms)
            move(arm, park, args.duration_ms)
            for index, (role, x, y) in enumerate(targets, 1):
                if index <= len(session['samples']):
                    continue
                print(f'\nSample {index}/16 [{role}]: X={x}, Y={y}, Z={base['z']}')
                print('Remove the cube from the previous position and the approach route.')
                if input('Type MOVE to travel above the target, then descend: ') != 'MOVE':
                    raise RuntimeError('Cancelled; arm remains at its last position')
                above = {**park, 'z': args.travel_z}
                travel = {**base, 'x': x, 'y': y, 'z': args.travel_z}
                target = {**base, 'x': x, 'y': y}
                move(arm, above, args.duration_ms)
                move(arm, travel, args.duration_ms)
                move(arm, target, args.duration_ms)
                print('With the arm stationary, place the cube flat and centre it in BOTH XY directions.')
                print('Top ID 5 up; cube sides aligned to jaws. Do not move the robot or table.')
                if input('Hands clear? Type CLEAR to record alignment and retract: ') != 'CLEAR':
                    raise RuntimeError('Cancelled; arm remains at alignment pose')
                aligned_state = arm.state()
                move(arm, travel, args.duration_ms)
                move(arm, above, args.duration_ms)
                move(arm, park, args.duration_ms)
                while True:
                    input('Check preview: cube untouched, top marker visible, no hands. Enter to capture: ')
                    try:
                        centre, result = capture_marker(camera, config, args.marker_id)
                        frame, image, _, marker = result
                        break
                    except RuntimeError as exc:
                        print('Capture rejected:', exc, 'Retry without moving the cube.')
                image_path = ROOT / 'data/calibration' / f'sample_{index:02d}.png'
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError('Could not save calibration image')
                session['samples'].append({'index': index, 'role': role, 'commanded_xy_mm': [x, y], 'pixel_xy': centre, 'commanded_pose': target, 'aligned_reported_state': aligned_state, 'image': str(image_path), 'last_marker': marker, 'received_at_ns': frame.received_at_ns})
                session['camera'] = camera_metadata(camera, config, image)
                write_json(args.output, session)
                print('Saved:', [x, y], '<->', centre)
            session['mapping'] = fit_mapping(session['samples'], args.max_error_mm)
            write_json(args.output, session)
            print(json.dumps(session['mapping'], indent=2))
            print('PASS' if session['mapping']['accepted'] else 'FAIL: inspect alignment and samples')
            print('Saved', args.output)
        finally:
            server.shutdown()
            server.server_close()
if __name__ == '__main__':
    main()
