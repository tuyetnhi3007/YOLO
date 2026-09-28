from datetime import datetime, timedelta, timezone
import json
import os
import socket
import threading
import time
from urllib import error, request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from settings import load_env_file


load_env_file()


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Bangkok").strip() or "Asia/Bangkok"
DEFAULT_TIMEZONE = timezone(timedelta(hours=7), name="Asia/Bangkok")


def get_app_timezone():
    try:
        return ZoneInfo(APP_TIMEZONE)
    except ZoneInfoNotFoundError:
        return DEFAULT_TIMEZONE

FLIP_CAMERA = True
ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "1") == "1"
MIN_TRACK_HISTORY = int(os.getenv("MIN_TRACK_HISTORY", "2"))
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
INFER_IMG_SIZE = int(os.getenv("INFER_IMG_SIZE", "416"))
PROCESS_EVERY_N_FRAMES = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "1")))
INFERENCE_MAX_FPS = max(0.1, float(os.getenv("INFERENCE_MAX_FPS", "8")))
FALLBACK_DETECT_INTERVAL = max(1, int(os.getenv("FALLBACK_DETECT_INTERVAL", "6")))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
CAMERA_BUFFER_SIZE = max(1, int(os.getenv("CAMERA_BUFFER_SIZE", "1")))
IDLE_SLEEP_SECONDS = float(os.getenv("IDLE_SLEEP_SECONDS", "0.005"))
TRACKER_CONFIG = os.getenv("YOLO_TRACKER_CONFIG", "tracker.yaml")

people_count = 0
last_updated = ""
detections = []
state_lock = threading.Lock()
last_sync_at = 0.0
last_sync_ok = False


def get_public_api_base_url():
    render_api_url = os.getenv("RENDER_API_URL", "").strip().rstrip("/")
    if render_api_url and "ten-app-cua-ban.onrender.com" not in render_api_url:
        return render_api_url
    return ""


def wake_render_service(render_api_url, timeout):
    health_url = f"{render_api_url}/health"
    try:
        with request.urlopen(health_url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (error.URLError, TimeoutError, OSError) as exc:
        print(f"[Render Sync] Wake-up failed for {health_url}: {exc}")
        return False


def sync_state_to_render(count, current_detections, camera_enabled=True, force=False):
    global last_sync_at, last_sync_ok

    render_api_url = os.getenv("RENDER_API_URL", "").strip().rstrip("/")
    if not render_api_url:
        return

    sync_interval = float(os.getenv("RENDER_SYNC_INTERVAL", "1.0"))
    now = time.time()
    if not force and now - last_sync_at < sync_interval:
        return

    payload = json.dumps(
        {
            "people_count": int(count),
            "time": datetime.now(get_app_timezone()).strftime("%Y-%m-%d %H:%M:%S"),
            "source": socket.gethostname(),
            "detections": current_detections,
            "camera_enabled": camera_enabled,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    update_token = os.getenv("UPDATE_TOKEN", "").strip()
    if update_token:
        headers["X-Update-Token"] = update_token

    update_url = f"{render_api_url}/update"
    req = request.Request(update_url, data=payload, headers=headers, method="POST")
    sync_timeout = float(os.getenv("RENDER_SYNC_TIMEOUT", "20"))
    wake_timeout = float(os.getenv("RENDER_WAKE_TIMEOUT", "70"))
    wake_after = float(os.getenv("RENDER_WAKE_AFTER", "600"))

    try:
        if not last_sync_ok or (now - last_sync_at) >= wake_after:
            wake_render_service(render_api_url, timeout=wake_timeout)

        with request.urlopen(req, timeout=sync_timeout) as response:
            if 200 <= response.status < 300:
                last_sync_at = now
                if not last_sync_ok:
                    print(f"[Render Sync] Sync resumed successfully to {update_url}")
                last_sync_ok = True
    except error.HTTPError as exc:
        last_sync_ok = False
        try:
            details = exc.read().decode("utf-8", errors="replace")
        except Exception:
            details = exc.reason
        print(f"[Render Sync] HTTP {exc.code} when syncing to {update_url}: {details}")
    except (error.URLError, TimeoutError, OSError) as exc:
        last_sync_ok = False
        print(f"[Render Sync] Failed to sync to {update_url}: {exc}")

prev_frame = None
track_history = {}


def has_motion(cv2, np, previous_frame, current_frame, box, threshold=1500):
    if previous_frame is None:
        return True

    x1, y1, x2, y2 = box
    prev_crop = previous_frame[y1:y2, x1:x2]
    curr_crop = current_frame[y1:y2, x1:x2]

    if prev_crop.shape != curr_crop.shape or prev_crop.size == 0:
        return True

    diff = cv2.absdiff(prev_crop, curr_crop)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    return np.sum(thresh) > threshold


def is_mirror_pair(box1, box2, frame_width, tolerance=80):
    x1 = (box1[0] + box1[2]) // 2
    x2 = (box2[0] + box2[2]) // 2
    return abs((x1 + x2) - frame_width) < tolerance


def run_camera():
    global people_count, last_updated, prev_frame, track_history, detections

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[Camera] Camera mode disabled because dependencies are missing: {exc}")
        return

    model = YOLO(MODEL_PATH)
    print(
        f"[Camera] Using model={MODEL_PATH}, imgsz={INFER_IMG_SIZE}, "
        f"process_every_n_frames={PROCESS_EVERY_N_FRAMES}, "
        f"inference_max_fps={INFERENCE_MAX_FPS}, tracker={TRACKER_CONFIG}"
    )

    def open_camera():
        candidates = [
            (0, cv2.CAP_DSHOW),
            (0, cv2.CAP_MSMF),
            (1, cv2.CAP_DSHOW),
            (1, cv2.CAP_MSMF),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
            (2, cv2.CAP_ANY),
        ]

        for index, backend in candidates:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

            ret, _ = cap.read()
            if ret:
                print(f"[Camera] Connected with index={index}, backend={backend}")
                return cap

            cap.release()

        return None

    cap = open_camera()
    if cap is None:
        print("[Camera] Cannot open any available camera.")
        with state_lock:
            last_updated = datetime.now(get_app_timezone()).strftime("%Y-%m-%d %H:%M:%S")
        sync_state_to_render(0, [], camera_enabled=False, force=True)
        return

    latest_frame = None
    latest_frame_lock = threading.Lock()
    latest_frame_index = 0
    capture_running = True

    def capture_frames():
        nonlocal latest_frame, latest_frame_index, capture_running

        while capture_running:
            ok, captured_frame = cap.read()
            if not ok:
                print("[Camera] Failed to read frame.")
                capture_running = False
                break

            if FLIP_CAMERA:
                captured_frame = cv2.flip(captured_frame, 1)

            with latest_frame_lock:
                latest_frame = captured_frame
                latest_frame_index += 1

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    latest_result = {"count": 0, "detections": [], "boxes": [], "time": ""}
    latest_result_lock = threading.Lock()
    latest_sync_state = {"count": 0, "detections": []}
    latest_sync_lock = threading.Lock()
    inference_interval = 1.0 / INFERENCE_MAX_FPS

    def process_frame(frame, frame_index):
        global prev_frame

        _, w, _ = frame.shape
        current_time = datetime.now(get_app_timezone()).strftime("%Y-%m-%d %H:%M:%S")
        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            imgsz=INFER_IMG_SIZE,
            tracker=TRACKER_CONFIG,
            verbose=False,
        )
        current_boxes = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0 or box.conf[0] <= 0.4:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if x1 > w * 0.85:
                    continue

                if box.id is not None and not has_motion(cv2, np, prev_frame, frame, (x1, y1, x2, y2)):
                    continue

                if box.id is not None:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    track_history.setdefault(track_id, []).append(center)
                    if len(track_history[track_id]) > 10:
                        track_history[track_id].pop(0)
                    if len(track_history[track_id]) < MIN_TRACK_HISTORY:
                        continue

                current_boxes.append((track_id, x1, y1, x2, y2))

        if not current_boxes and frame_index % FALLBACK_DETECT_INTERVAL == 0:
            detect_results = model(frame, conf=0.4, imgsz=INFER_IMG_SIZE, verbose=False)
            fallback_id = 1
            for result in detect_results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls != 0 or box.conf[0] <= 0.4:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 > w * 0.85:
                        continue

                    current_boxes.append((fallback_id, x1, y1, x2, y2))
                    fallback_id += 1

        removed_ids = set()
        for i in range(len(current_boxes)):
            for j in range(i + 1, len(current_boxes)):
                _, x1a, y1a, x2a, y2a = current_boxes[i]
                id2, x1b, y1b, x2b, y2b = current_boxes[j]
                if is_mirror_pair((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b), w):
                    removed_ids.add(id2)

        current_count = 0
        current_detections = []
        boxes_for_draw = []

        for track_id, x1, y1, x2, y2 in current_boxes:
            if track_id in removed_ids:
                continue

            current_count += 1
            current_detections.append({"id": track_id, "bbox": [x1, y1, x2, y2]})
            boxes_for_draw.append((track_id, x1, y1, x2, y2))

        prev_frame = frame.copy()
        return current_count, current_detections, boxes_for_draw, current_time

    def run_inference():
        global people_count, last_updated, detections

        processed_frame_index = 0
        last_inference_at = 0.0

        while capture_running:
            with latest_frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
                frame_index = latest_frame_index

            if frame is None or frame_index == processed_frame_index:
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            if frame_index % PROCESS_EVERY_N_FRAMES != 0:
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            wait_seconds = inference_interval - (time.time() - last_inference_at)
            if wait_seconds > 0:
                time.sleep(min(wait_seconds, 0.02))
                continue

            processed_frame_index = frame_index
            last_inference_at = time.time()
            current_count, current_detections, boxes_for_draw, processed_time = process_frame(frame, frame_index)

            with latest_result_lock:
                latest_result["count"] = current_count
                latest_result["detections"] = current_detections
                latest_result["boxes"] = boxes_for_draw
                latest_result["time"] = processed_time

            with state_lock:
                people_count = current_count
                detections = current_detections
                last_updated = processed_time

            with latest_sync_lock:
                latest_sync_state["count"] = current_count
                latest_sync_state["detections"] = current_detections

    def sync_to_render_loop():
        while capture_running:
            with latest_sync_lock:
                current_count = latest_sync_state["count"]
                current_detections = list(latest_sync_state["detections"])

            sync_state_to_render(current_count, current_detections, camera_enabled=True)
            time.sleep(IDLE_SLEEP_SECONDS)

    inference_thread = threading.Thread(target=run_inference, daemon=True)
    sync_thread = threading.Thread(target=sync_to_render_loop, daemon=True)
    inference_thread.start()
    sync_thread.start()

    displayed_frame_index = 0

    try:
        while capture_running:
            with latest_frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
                frame_index = latest_frame_index

            if frame is None or frame_index == displayed_frame_index:
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            displayed_frame_index = frame_index

            with latest_result_lock:
                latest_count = latest_result["count"]
                latest_boxes_for_draw = list(latest_result["boxes"])

            for track_id, x1, y1, x2, y2 in latest_boxes_for_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"People: {latest_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture_running = False
        capture_thread.join(timeout=1)
        inference_thread.join(timeout=2)
        sync_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        sync_state_to_render(0, [], camera_enabled=False, force=True)


def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    public_api_url = get_public_api_base_url()
    if public_api_url:
        print(f"[Render Sync] Target API: {public_api_url}/update")
    else:
        print("[Render Sync] RENDER_API_URL is empty. State will not sync to Render.")

    if not ENABLE_CAMERA:
        print("[Camera] ENABLE_CAMERA=0, nothing to run.")
        sync_state_to_render(0, [], camera_enabled=False, force=True)
    else:
        print(f"[Camera] Local IP: {get_local_ip()}")
        run_camera()
