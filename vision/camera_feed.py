"""
vision/camera_feed.py — Live phone camera → YOLOv8 → zone occupancy pipeline.

Pulls an MJPEG stream from IP Webcam (S23 Ultra), runs YOLOv8n person
detection, maps each person to a room zone via perspective homography,
and pushes per-zone occupancy counts to the SRACE backend.

The camera is security-style (angled from a corner).  On first run, a
calibration window opens so you can click the 4 floor corners of the room.
The calibration is saved to config/camera_calib.json for future runs.

Usage:
    # First run — will open calibration window:
    python vision/camera_feed.py --url http://192.168.1.X:8080/video

    # Subsequent runs — loads saved calibration:
    python vision/camera_feed.py --url http://192.168.1.X:8080/video

    # Headless mode (no preview window — for RPi / SSH):
    python vision/camera_feed.py --url http://192.168.1.X:8080/video --headless

    # Custom settings:
    python vision/camera_feed.py --url http://192.168.1.X:8080/video \\
        --interval 2 --conf 0.4 --config config/classroom_real.json

Keys (when preview window is open):
    q — quit
    r — recalibrate (re-click the 4 corners)
    g — toggle zone grid overlay
    s — save a screenshot

Requires:
    pip install ultralytics opencv-python requests
"""

import argparse
import json
import os
import sys
import time
import signal
import threading
from collections import deque
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("✗ OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("✗ Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vision.zone_mapper import HomographyZoneMapper, SimpleZoneMapper


# ══════════════════════════════════════════════════════════════
#  CAMERA PIPELINE
# ══════════════════════════════════════════════════════════════

class CameraPipeline:
    """
    End-to-end pipeline:  MJPEG stream → YOLOv8 → zone occupancy → backend.

    Attributes:
        url: IP Webcam MJPEG stream URL
        model: YOLOv8 model instance
        mapper: HomographyZoneMapper for pixel → zone conversion
        zone_counts: Current per-zone person counts
        running: Whether the pipeline is active
    """

    def __init__(
        self,
        url: str,
        room_config_path: str = None,
        calib_path: str = None,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        push_interval: float = 3.0,
        backend_url: str = "http://localhost:8000",
        headless: bool = False,
        show_grid: bool = True,
        smoothing_window: int = 3,
        skip_calib: bool = False,
        use_aruco: bool = False,
    ):
        self.url = url
        self.confidence = confidence
        self.push_interval = push_interval
        self.backend_url = backend_url
        self.headless = headless
        self.show_grid = show_grid
        self.smoothing_window = smoothing_window
        self.skip_calib = skip_calib
        self.use_aruco = use_aruco

        # Load room config for zone grid dimensions
        self.room_width = 10.8
        self.room_depth = 7.6
        self.n_zone_cols = 4
        self.n_zone_rows = 3

        if room_config_path:
            self._load_room_config(room_config_path)

        self.n_zones = self.n_zone_cols * self.n_zone_rows

        # Detection backend
        if use_aruco:
            # ArUco marker detection (for Lego figurine demo)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.model = None
            print(f"  ✓ ArUco detector ready (DICT_4X4_50, each marker = 1 person)")
        else:
            # YOLO model
            print(f"  Loading YOLOv8 model: {model_name} ...")
            self.model = YOLO(model_name)
            print(f"  ✓ Model loaded")
            self.aruco_detector = None

        # Zone mapper (homography for security-style camera)
        self.mapper = HomographyZoneMapper(
            room_width=self.room_width,
            room_depth=self.room_depth,
            n_zone_cols=self.n_zone_cols,
            n_zone_rows=self.n_zone_rows,
        )

        # Calibration file path
        self.calib_path = calib_path or str(
            PROJECT_ROOT / "config" / "camera_calib.json"
        )

        # State
        self.zone_counts = np.zeros(self.n_zones, dtype=int)
        self.running = False
        self.cap = None
        self.fps = 0.0
        self.total_persons = 0
        self.last_push_time = 0.0
        self.frame_count = 0

        # Temporal smoothing buffer — stores last N frames of zone counts
        self._count_history: deque[np.ndarray] = deque(maxlen=smoothing_window)

        # Stats for /camera_status endpoint
        self.stats = {
            "connected": False,
            "url": url,
            "fps": 0.0,
            "persons_detected": 0,
            "last_update": 0.0,
            "frames_processed": 0,
        }

    def _load_room_config(self, path: str):
        """Load zone grid dimensions from room config JSON."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            room = data["room"]
            zones = data["zones"]
            self.room_width = float(room["width"])
            self.room_depth = float(room["depth"])
            self.n_zone_cols = int(zones["cols"])
            self.n_zone_rows = int(zones["rows"])
            self.n_zones = self.n_zone_cols * self.n_zone_rows
            self.zone_counts = np.zeros(self.n_zones, dtype=int)
            print(f"  ✓ Room config loaded: {room.get('name', 'unnamed')} "
                  f"({self.room_width}×{self.room_depth}m, "
                  f"{self.n_zone_rows}×{self.n_zone_cols} zones)")
        except Exception as e:
            print(f"  ⚠ Failed to load room config: {e}")
            print(f"  → Using defaults: 10.8×7.6m, 3×4 zones")

    def _connect_stream(self) -> bool:
        """Open the MJPEG video stream."""
        print(f"\n  Connecting to stream: {self.url}")
        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            print("  ✗ Failed to open stream!")
            print("  → Check IP Webcam is running on your phone")
            print("  → Verify phone and laptop are on the same WiFi")
            print(f"  → Test in browser: {self.url}")
            return False

        # Read one frame to get dimensions
        ret, frame = self.cap.read()
        if not ret:
            print("  ✗ Connected but failed to grab first frame")
            return False

        h, w = frame.shape[:2]
        print(f"  ✓ Stream connected: {w}×{h} pixels")
        self.stats["connected"] = True
        return True

    def _ensure_calibrated(self, frame: np.ndarray) -> bool:
        """Load saved calibration or run interactive calibration."""
        # Skip calibration mode — just map full frame to room grid linearly
        if self.skip_calib:
            h, w = frame.shape[:2]
            self.mapper = SimpleZoneMapper(
                frame_width=w, frame_height=h,
                room_width=self.room_width, room_depth=self.room_depth,
                n_zone_cols=self.n_zone_cols, n_zone_rows=self.n_zone_rows,
            )
            print("  ✓ Skip-calib mode: full frame mapped to zone grid")
            return True

        # Try loading existing calibration
        if self.mapper.load_calibration(self.calib_path):
            return True

        if self.headless:
            print("  ✗ No calibration found and running in headless mode!")
            print(f"  → Run once without --headless to calibrate")
            print(f"  → Or use --skip-calib for quick testing")
            return False

        # Interactive calibration with LIVE video feed
        print("\n  ═══════════════════════════════════════════")
        print("  CAMERA CALIBRATION (LIVE FEED)")
        print("  Click the 4 corners of the room floor:")
        print("    1. Top-left    2. Top-right")
        print("    3. Bottom-right  4. Bottom-left")
        print("  Press 'r' to reset, 'q' to cancel")
        print("  ═══════════════════════════════════════════\n")

        success = self.mapper.calibrate_interactive(frame, cap=self.cap)
        if success:
            self.mapper.save_calibration(self.calib_path)
        return success

    def _detect_persons(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect persons in a frame using YOLO or ArUco markers.

        Returns list of (foot_x, foot_y, confidence, bbox) for each person.
        """
        if self.use_aruco:
            return self._detect_aruco(frame)
        return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[tuple]:
        """
        Run YOLOv8 person detection on a frame.

        Returns list of (foot_x, foot_y, confidence, bbox) for each person.
        foot_x/foot_y is the bottom-center of the bounding box (where feet
        touch the floor — better for zone mapping than bbox center).
        """
        results = self.model(
            frame,
            classes=[0],  # COCO class 0 = person
            conf=self.confidence,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Foot-point = bottom-center of bounding box
            foot_x = (x1 + x2) / 2
            foot_y = y2  # bottom edge

            detections.append((foot_x, foot_y, conf, (x1, y1, x2, y2)))

        return detections

    def _detect_aruco(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect ArUco markers in a frame. Each marker = 1 person.

        Returns same format as _detect_yolo: (center_x, center_y, 1.0, bbox).
        Uses marker center as the position (Lego figurines stand on the marker).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        detections = []
        if ids is not None:
            for i, marker_corners in enumerate(corners):
                pts = marker_corners[0]  # 4 corner points
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]

                # Bounding box
                x1, y1 = x_coords.min(), y_coords.min()
                x2, y2 = x_coords.max(), y_coords.max()

                # Center of marker = person position
                cx = x_coords.mean()
                cy = y_coords.mean()

                marker_id = int(ids[i][0])
                detections.append((cx, cy, 1.0, (x1, y1, x2, y2, marker_id)))

        return detections

    def _map_to_zones(self, detections: list[tuple]) -> np.ndarray:
        """
        Map person detections to zone indices.
        Returns an array of person counts per zone.
        """
        counts = np.zeros(self.n_zones, dtype=int)

        for foot_x, foot_y, conf, bbox in detections:
            zone_idx = self.mapper.pixel_to_zone(foot_x, foot_y)
            if 0 <= zone_idx < self.n_zones:
                counts[zone_idx] += 1

        return counts

    def _smooth_counts(self, raw_counts: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing (rolling mode) to reduce flicker.
        Uses the most common count per zone over the last N frames.
        """
        self._count_history.append(raw_counts.copy())

        if len(self._count_history) < 2:
            return raw_counts

        # Stack history and take the mode (most common value) per zone
        stacked = np.stack(self._count_history, axis=0)  # (N, n_zones)
        smoothed = np.zeros(self.n_zones, dtype=int)
        for zi in range(self.n_zones):
            values, counts = np.unique(stacked[:, zi], return_counts=True)
            smoothed[zi] = values[np.argmax(counts)]

        return smoothed

    def _push_occupancy(self, zone_counts: np.ndarray):
        """POST zone occupancy to the SRACE FastAPI backend."""
        if not REQUESTS_AVAILABLE:
            return

        try:
            resp = requests.post(
                f"{self.backend_url}/set_occupancy",
                json={"zone_people": zone_counts.tolist()},
                timeout=2.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("total_people", 0)
                occ = data.get("occupied_zones", 0)
                # Compact status line
                print(f"\r  📡 Push OK: {total} people in {occ} zones  ", end="", flush=True)
            else:
                print(f"\r  ⚠ Push failed: HTTP {resp.status_code}  ", end="", flush=True)
        except requests.exceptions.ConnectionError:
            pass  # Backend not running — silent fail
        except Exception as e:
            print(f"\r  ⚠ Push error: {e}  ", end="", flush=True)

    def _draw_overlay(self, frame: np.ndarray, detections: list[tuple]) -> np.ndarray:
        """Draw detection boxes, zone grid, and HUD on the frame."""
        annotated = frame.copy()

        # Draw bounding boxes and foot-points
        for detection in detections:
            foot_x, foot_y, conf = detection[0], detection[1], detection[2]
            bbox_data = detection[3]
            x1, y1, x2, y2 = bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]
            marker_id = bbox_data[4] if len(bbox_data) > 4 else None

            # Bounding box
            color = (255, 128, 0) if self.use_aruco else (0, 255, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            # Label
            label = f"ID:{marker_id}" if marker_id is not None else f"{conf:.0%}"
            cv2.putText(annotated, label,
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Foot-point marker
            cv2.circle(annotated, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)

            # Zone index label at foot
            zone_idx = self.mapper.pixel_to_zone(foot_x, foot_y)
            if zone_idx >= 0:
                cv2.putText(annotated, f"Z{zone_idx}",
                            (int(foot_x) + 8, int(foot_y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Zone grid overlay
        if self.show_grid:
            if hasattr(self.mapper, 'is_calibrated') and self.mapper.is_calibrated:
                annotated = self.mapper.draw_zone_grid_on_frame(annotated, alpha=0.4)
            elif isinstance(self.mapper, SimpleZoneMapper):
                # Draw simple grid lines for skip-calib mode
                h, w = annotated.shape[:2]
                v_lines, h_lines = self.mapper.get_zone_grid_lines()
                for vx in v_lines:
                    cv2.line(annotated, (vx, 0), (vx, h), (0, 255, 255), 1)
                for hy in h_lines:
                    cv2.line(annotated, (0, hy), (w, hy), (0, 255, 255), 1)
                # Zone labels
                for r in range(self.n_zone_rows):
                    for c in range(self.n_zone_cols):
                        cx = int((c + 0.5) * self.mapper.zone_w_px)
                        cy = int((r + 0.5) * self.mapper.zone_h_px)
                        zi = r * self.n_zone_cols + c
                        cv2.putText(annotated, f"Z{zi}", (cx - 10, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # HUD — occupancy count and FPS
        hud_bg = annotated.copy()
        cv2.rectangle(hud_bg, (0, 0), (320, 90), (0, 0, 0), -1)
        annotated = cv2.addWeighted(hud_bg, 0.6, annotated, 0.4, 0)

        cv2.putText(annotated, f"SRACE Vision | {self.fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(annotated, f"Persons: {self.total_persons}  |  Zones: {int((self.zone_counts > 0).sum())}/{self.n_zones}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Per-zone mini status
        zone_str = " ".join(f"{c}" for c in self.zone_counts)
        cv2.putText(annotated, f"[{zone_str}]",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return annotated

    def run(self):
        """
        Main loop: capture → detect → map → push → display.

        Runs until 'q' is pressed or SIGINT is received.
        """
        print("\n" + "▓" * 55)
        print("  SRACE v2 — Vision Pipeline")
        print("  Phone Camera → YOLOv8 → Zone Occupancy → Backend")
        print("▓" * 55)

        # Connect to stream
        if not self._connect_stream():
            return

        # Flush several frames — MJPEG streams often have stale/black
        # frames buffered at connect time
        print("  Flushing stream buffer...")
        first_frame = None
        for _ in range(15):
            ret, frame = self.cap.read()
            if ret:
                first_frame = frame
            time.sleep(0.05)

        if first_frame is None:
            print("  ✗ Failed to grab any frame for calibration")
            return
        print("  ✓ Got live frame")

        # Ensure we have a calibration
        if not self._ensure_calibrated(first_frame):
            print("  ✗ Calibration required. Exiting.")
            return

        # Main loop
        self.running = True
        fps_start = time.time()
        fps_frames = 0

        print("\n  ✓ Pipeline running!")
        if not self.headless:
            print("  Keys: q=quit  r=recalibrate  g=toggle grid  s=screenshot")
        print()

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("\n  ⚠ Frame grab failed — reconnecting...")
                    time.sleep(1)
                    self.cap.release()
                    if not self._connect_stream():
                        break
                    continue

                self.frame_count += 1
                fps_frames += 1

                # FPS calculation (every 10 frames)
                if fps_frames >= 10:
                    elapsed = time.time() - fps_start
                    self.fps = fps_frames / elapsed if elapsed > 0 else 0
                    fps_start = time.time()
                    fps_frames = 0

                # ── Detect persons ──
                detections = self._detect_persons(frame)
                self.total_persons = len(detections)

                # ── Map to zones ──
                raw_counts = self._map_to_zones(detections)
                self.zone_counts = self._smooth_counts(raw_counts)

                # ── Push occupancy at interval ──
                now = time.time()
                if now - self.last_push_time >= self.push_interval:
                    self.last_push_time = now
                    self._push_occupancy(self.zone_counts)
                    self.stats.update({
                        "fps": round(self.fps, 1),
                        "persons_detected": self.total_persons,
                        "last_update": now,
                        "frames_processed": self.frame_count,
                    })

                # ── Display ──
                if not self.headless:
                    annotated = self._draw_overlay(frame, detections)
                    cv2.imshow("SRACE Vision", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n  Shutting down...")
                        break
                    elif key == ord('r'):
                        print("\n  Recalibrating...")
                        self.mapper.calibrate_interactive(frame)
                        self.mapper.save_calibration(self.calib_path)
                    elif key == ord('g'):
                        self.show_grid = not self.show_grid
                    elif key == ord('s'):
                        screenshot_path = str(PROJECT_ROOT / "output" / f"screenshot_{int(time.time())}.png")
                        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                        cv2.imwrite(screenshot_path, annotated)
                        print(f"\n  📸 Screenshot saved: {screenshot_path}")

        except KeyboardInterrupt:
            print("\n  Interrupted — shutting down...")

        finally:
            self.running = False
            self.stats["connected"] = False
            if self.cap:
                self.cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            # Push zero occupancy on shutdown (clear the room)
            self._push_occupancy(np.zeros(self.n_zones, dtype=int))
            print("  ✓ Vision pipeline stopped\n")


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SRACE v2 — Phone camera YOLOv8 occupancy pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vision/camera_feed.py --url http://192.168.1.5:8080/video
  python vision/camera_feed.py --url http://192.168.1.5:8080/video --headless
  python vision/camera_feed.py --url http://192.168.1.5:8080/video --interval 2 --conf 0.4
        """,
    )

    parser.add_argument("--url", type=str, required=True,
                        help="IP Webcam MJPEG stream URL (e.g., http://192.168.1.5:8080/video)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to room config JSON (default: config/classroom_real.json)")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to calibration JSON (default: config/camera_calib.json)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model name (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between occupancy pushes (default: 3.0)")
    parser.add_argument("--backend", type=str, default="http://localhost:8000",
                        help="SRACE backend URL (default: http://localhost:8000)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without preview window (for RPi/SSH)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Don't show zone grid overlay")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Temporal smoothing window size (default: 3 frames)")
    parser.add_argument("--skip-calib", action="store_true",
                        help="Skip calibration — map full frame to zone grid (for quick testing)")
    parser.add_argument("--aruco", action="store_true",
                        help="Use ArUco markers instead of YOLO (for Lego figurine demo)")

    args = parser.parse_args()

    # Default room config path
    config_path = args.config
    if config_path is None:
        default = PROJECT_ROOT / "config" / "classroom_real.json"
        if default.exists():
            config_path = str(default)

    pipeline = CameraPipeline(
        url=args.url,
        room_config_path=config_path,
        calib_path=args.calib,
        model_name=args.model,
        confidence=args.conf,
        push_interval=args.interval,
        backend_url=args.backend,
        headless=args.headless,
        show_grid=not args.no_grid,
        smoothing_window=args.smooth,
        skip_calib=args.skip_calib,
        use_aruco=args.aruco,
    )

    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        pipeline.running = False

    signal.signal(signal.SIGINT, signal_handler)

    pipeline.run()


if __name__ == "__main__":
    main()
