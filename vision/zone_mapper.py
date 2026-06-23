"""
vision/zone_mapper.py — Map pixel coordinates to room zone indices.

Two mapper implementations:
  1. SimpleZoneMapper  — linear mapping for top-down overhead cameras
  2. HomographyZoneMapper — perspective transform for angled (security-style) cameras

The homography mapper uses 4 calibration points: the user clicks the 4 corners
of the room floor in the camera frame, and we compute the perspective warp from
pixel space → room-metre space.  Once calibrated, it maps any (px, py) to a
flat zone index in the room grid.

Usage:
    from vision.zone_mapper import HomographyZoneMapper
    mapper = HomographyZoneMapper(room_width=10.8, room_depth=7.6,
                                  n_zone_cols=4, n_zone_rows=3)
    mapper.calibrate_interactive(frame)   # opens window, click 4 corners
    zone_idx = mapper.pixel_to_zone(px, py)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class SimpleZoneMapper:
    """
    Linear pixel→zone mapper for top-down overhead cameras.

    Assumes the camera frame maps directly to the room rectangle
    with no perspective distortion.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        room_width: float,
        room_depth: float,
        n_zone_cols: int,
        n_zone_rows: int,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.room_width = room_width
        self.room_depth = room_depth
        self.n_zone_cols = n_zone_cols
        self.n_zone_rows = n_zone_rows
        self.n_zones = n_zone_cols * n_zone_rows

        # Pre-compute zone boundaries in pixel space
        self.zone_w_px = frame_width / n_zone_cols
        self.zone_h_px = frame_height / n_zone_rows

    def pixel_to_room(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coords to room coords (metres)."""
        rx = (px / self.frame_width) * self.room_width
        ry = (py / self.frame_height) * self.room_depth
        return rx, ry

    def pixel_to_zone(self, px: float, py: float) -> int:
        """
        Convert pixel coords to flat zone index.
        Returns -1 if outside the room bounds.
        """
        col = int(px / self.zone_w_px)
        row = int(py / self.zone_h_px)

        col = max(0, min(col, self.n_zone_cols - 1))
        row = max(0, min(row, self.n_zone_rows - 1))

        return row * self.n_zone_cols + col

    def get_zone_grid_lines(self) -> tuple[list, list]:
        """Return pixel positions for zone grid lines (for overlay drawing)."""
        v_lines = [int(c * self.zone_w_px) for c in range(1, self.n_zone_cols)]
        h_lines = [int(r * self.zone_h_px) for r in range(1, self.n_zone_rows)]
        return v_lines, h_lines


class HomographyZoneMapper:
    """
    Perspective-corrected pixel→zone mapper for angled (security-style) cameras.

    Uses a homography matrix computed from 4 calibration points — the corners
    of the room floor as seen in the camera frame.

    Calibration order (click these corners in the frame):
        1. Top-left of room floor     → maps to (0, 0)
        2. Top-right of room floor    → maps to (room_width, 0)
        3. Bottom-right of room floor → maps to (room_width, room_depth)
        4. Bottom-left of room floor  → maps to (0, room_depth)
    """

    def __init__(
        self,
        room_width: float,
        room_depth: float,
        n_zone_cols: int,
        n_zone_rows: int,
    ):
        self.room_width = room_width
        self.room_depth = room_depth
        self.n_zone_cols = n_zone_cols
        self.n_zone_rows = n_zone_rows
        self.n_zones = n_zone_cols * n_zone_rows

        self.zone_w = room_width / n_zone_cols
        self.zone_h = room_depth / n_zone_rows

        # Homography matrix (set by calibrate)
        self.H: Optional[np.ndarray] = None

        # Raw calibration points (pixel coords of 4 corners)
        self.calib_points: Optional[np.ndarray] = None

        # Destination points in room-metre space
        self.dst_points = np.array([
            [0, 0],
            [room_width, 0],
            [room_width, room_depth],
            [0, room_depth],
        ], dtype=np.float32)

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    def calibrate_from_points(self, pixel_corners: list[tuple[float, float]]):
        """
        Set the homography from 4 pixel corner coordinates.

        Args:
            pixel_corners: List of 4 (px, py) tuples in order:
                           TL, TR, BR, BL of the room floor.
        """
        if len(pixel_corners) != 4:
            raise ValueError(f"Need exactly 4 calibration points, got {len(pixel_corners)}")

        self.calib_points = np.array(pixel_corners, dtype=np.float32)
        self.H, _ = cv2.findHomography(self.calib_points, self.dst_points)

        if self.H is None:
            raise RuntimeError("Homography computation failed — points may be collinear")

        print(f"✓ Homography calibrated from 4 points")

    def calibrate_interactive(self, frame: np.ndarray, cap=None,
                               window_name: str = "SRACE Calibration"):
        """
        Open an interactive window for the user to click 4 room corners.

        If a cv2.VideoCapture `cap` is provided, the window shows a LIVE
        camera feed so the user can position the camera before clicking.
        Otherwise falls back to the provided static frame.

        Click order:
            1. Top-left corner of room floor
            2. Top-right corner of room floor
            3. Bottom-right corner of room floor
            4. Bottom-left corner of room floor

        Press 'r' to reset, 'q' to cancel.
        Window closes automatically after 4 clicks.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for interactive calibration")

        clicked = []
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
                clicked.append((x, y))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, on_click)

        current_frame = frame.copy()

        while True:
            # Pull live frame if capture is available
            if cap is not None:
                ret, live = cap.read()
                if ret:
                    current_frame = live

            # Build display frame
            display = current_frame.copy()

            # Instructions
            next_label = labels[len(clicked)] if len(clicked) < 4 else "Done!"
            cv2.putText(display, f"Click 4 room floor corners | Next: {next_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display, "Press 'r' to reset, 'q' to cancel",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # Draw already-clicked points
            for i, (cx, cy) in enumerate(clicked):
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(display, f"{i+1}:{labels[i]}",
                            (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(display, clicked[i-1], (cx, cy), (0, 255, 0), 2)
            if len(clicked) == 4:
                cv2.line(display, clicked[3], clicked[0], (0, 255, 0), 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                print("✗ Calibration cancelled")
                return False
            if key == ord('r'):
                clicked.clear()
            if len(clicked) == 4:
                # Show the final polygon for 1 second
                cv2.imshow(window_name, display)
                cv2.waitKey(1000)
                break

        cv2.destroyWindow(window_name)
        self.calibrate_from_points(clicked)
        return True

    def pixel_to_room(self, px: float, py: float) -> tuple[float, float]:
        """
        Transform a pixel coordinate to room coordinates using the homography.

        Returns (room_x, room_y) in metres.
        """
        if self.H is None:
            raise RuntimeError("Not calibrated — call calibrate_from_points() or calibrate_interactive() first")

        pt = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        rx, ry = transformed[0, 0]
        return float(rx), float(ry)

    def pixel_to_zone(self, px: float, py: float) -> int:
        """
        Convert pixel coords to flat zone index via homography.
        Returns -1 if the transformed point is outside the room bounds.
        """
        rx, ry = self.pixel_to_room(px, py)

        # Check bounds
        if rx < 0 or rx > self.room_width or ry < 0 or ry > self.room_depth:
            return -1

        col = int(rx / self.zone_w)
        row = int(ry / self.zone_h)

        col = max(0, min(col, self.n_zone_cols - 1))
        row = max(0, min(row, self.n_zone_rows - 1))

        return row * self.n_zone_cols + col

    def draw_zone_grid_on_frame(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Draw the zone grid overlay on a camera frame using inverse homography.

        Transforms zone boundaries from room-metre space back to pixel space
        and draws them on the frame.
        """
        if self.H is None:
            return frame

        overlay = frame.copy()
        H_inv = np.linalg.inv(self.H)

        # Draw zone grid lines
        # Vertical lines (column boundaries)
        for c in range(self.n_zone_cols + 1):
            x = c * self.zone_w
            top_pt = np.array([[[x, 0]]], dtype=np.float32)
            bot_pt = np.array([[[x, self.room_depth]]], dtype=np.float32)
            top_px = cv2.perspectiveTransform(top_pt, H_inv)[0, 0].astype(int)
            bot_px = cv2.perspectiveTransform(bot_pt, H_inv)[0, 0].astype(int)
            cv2.line(overlay, tuple(top_px), tuple(bot_px), (0, 255, 255), 1)

        # Horizontal lines (row boundaries)
        for r in range(self.n_zone_rows + 1):
            y = r * self.zone_h
            left_pt = np.array([[[0, y]]], dtype=np.float32)
            right_pt = np.array([[[self.room_width, y]]], dtype=np.float32)
            left_px = cv2.perspectiveTransform(left_pt, H_inv)[0, 0].astype(int)
            right_px = cv2.perspectiveTransform(right_pt, H_inv)[0, 0].astype(int)
            cv2.line(overlay, tuple(left_px), tuple(right_px), (0, 255, 255), 1)

        # Draw zone labels at centers
        for r in range(self.n_zone_rows):
            for c in range(self.n_zone_cols):
                cx = (c + 0.5) * self.zone_w
                cy = (r + 0.5) * self.zone_h
                center_pt = np.array([[[cx, cy]]], dtype=np.float32)
                center_px = cv2.perspectiveTransform(center_pt, H_inv)[0, 0].astype(int)
                zone_idx = r * self.n_zone_cols + c
                cv2.putText(overlay, f"Z{zone_idx}",
                            (int(center_px[0]) - 10, int(center_px[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Blend overlay
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return result

    def save_calibration(self, path: str | Path):
        """Save calibration points to a JSON file for reuse."""
        if self.calib_points is None:
            raise RuntimeError("Not calibrated — nothing to save")

        data = {
            "room_width": self.room_width,
            "room_depth": self.room_depth,
            "n_zone_cols": self.n_zone_cols,
            "n_zone_rows": self.n_zone_rows,
            "calibration_points": self.calib_points.tolist(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Calibration saved to {path}")

    def load_calibration(self, path: str | Path) -> bool:
        """Load calibration points from a previously saved JSON file."""
        path = Path(path)
        if not path.exists():
            print(f"⚠ Calibration file not found: {path}")
            return False

        with open(path, "r") as f:
            data = json.load(f)

        points = data["calibration_points"]
        self.calibrate_from_points(points)
        print(f"✓ Calibration loaded from {path}")
        return True
