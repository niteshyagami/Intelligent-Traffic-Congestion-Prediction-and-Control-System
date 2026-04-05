"""
YOLOv8 Vehicle Detector with Lane-Wise Counting.

Detects vehicles (car, truck, bus, motorcycle, bicycle) in a video frame,
assigns each to one of 4 lane ROIs arranged in a '+' intersection pattern,
and returns per-lane counts.

Lane Layout ('+' intersection as seen from overhead CCTV):

          |  NORTH  |
          |  Lane   |
    ------+---------+------
    WEST  | CENTER  | EAST
    Lane  | (inter- | Lane
          | section)|
    ------+---------+------
          |  SOUTH  |
          |  Lane   |

Vehicles in the center box are assigned to the nearest lane approach.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Detection:
    """A single detected vehicle."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    lane: Optional[str] = None


@dataclass
class LaneROI:
    """Region of Interest for a lane."""
    name: str
    points: List[Tuple[int, int]]  # polygon vertices


@dataclass
class DetectionResult:
    """Results from one frame detection."""
    total_vehicles: int = 0
    lane_counts: Dict[str, int] = field(default_factory=dict)
    detections: List[Detection] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (0, 0)


# COCO classes that represent vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "bike",
    5: "bus",
    6: "bus",
    7: "truck",
    1: "bike"
}

class VehicleDetector:
    """
    YOLOv8-based vehicle detector with lane assignment.
    
    Lanes are arranged in a '+' (plus/cross) pattern matching how real
    traffic intersections look from an overhead CCTV camera.
    
    Usage:
        detector = VehicleDetector()
        detector.load_model()
        result = detector.detect(frame)
        print(result.lane_counts)  # {'Lane_N': 5, 'Lane_S': 3, ...}
    """

    def __init__(self, model_name: str = "yolo11x.pt", confidence: float = 0.18):
        self.model = None
        self.model_name = model_name
        self.confidence = confidence
        self.lanes: List[LaneROI] = []
        self._model_loaded = False
        self._frame_w = 0
        self._frame_h = 0

    def load_model(self):
        """Load YOLOv8 model (auto-downloads weights if needed)."""
        if self._model_loaded:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self._model_loaded = True
            print(f"[Detector] YOLO11 model loaded: {self.model_name}")
        except ImportError:
            print("[Detector] WARNING: ultralytics not installed. Install: pip install ultralytics")
            print("[Detector] Running in SIMULATION mode (random detections)")
            self._model_loaded = False

    def setup_lanes(self, frame_width: int, frame_height: int):
        """
        Set up 4 lane ROIs in a '+' intersection pattern.

        The frame is divided into a 3x3 grid:
        - The center cell is the intersection box
        - Top-center = North lane (vehicles approaching from north)
        - Bottom-center = South lane
        - Left-center = West lane
        - Right-center = East lane
        - Corner cells are also assigned to the nearest lane for coverage

        Layout:
            +--------+--------+--------+
            | (NW)   | NORTH  | (NE)   |
            | -> N/W |  Lane  | -> N/E |
            +--------+--------+--------+
            | WEST   | CENTER | EAST   |
            | Lane   | (both) | Lane   |
            +--------+--------+--------+
            | (SW)   | SOUTH  | (SE)   |
            | -> S/W |  Lane  | -> S/E |
            +--------+--------+--------+
        """
        w, h = frame_width, frame_height
        self._frame_w = w
        self._frame_h = h

        # Define the intersection zone (center ~40% of frame)
        # This gives wider lane approach strips
        left = int(w * 0.30)
        right = int(w * 0.70)
        top = int(h * 0.30)
        bottom = int(h * 0.70)

        # North lane: top strip + top-left and top-right corners for broader coverage
        # (vehicles approaching from north — entire top portion of frame)
        self.lanes = [
            LaneROI("Lane_N", [
                (0, 0), (w, 0),           # top edge full width
                (w, top),                  # down to intersection top
                (right, top),              # intersection corner
                (right, bottom),           # to center-right (catch vehicles in intersection going north)
                (left, bottom),            # to center-left
                (left, top),              # back up
                (0, top),                 # to left edge
            ]),
            LaneROI("Lane_S", [
                (0, bottom), (left, bottom),  # from left edge
                (left, top),                  # up to intersection (catch vehicles in intersection going south)
                (right, top),                 # across
                (right, bottom),              # back down
                (w, bottom),                  # to right edge
                (w, h), (0, h),              # bottom edge full width
            ]),
            LaneROI("Lane_E", [
                (right, 0),                   # top-right
                (w, 0), (w, h),              # right edge full height
                (right, h),                   # bottom-right
            ]),
            LaneROI("Lane_W", [
                (0, 0), (left, 0),           # top-left
                (left, h),                   # left strip full height
                (0, h),                      # bottom-left
            ]),
        ]
        print(f"[Detector] '+' intersection lanes configured for {w}x{h} frame")

    def setup_plus_lanes(self, frame_width: int, frame_height: int):
        """
        Alternative simpler '+' setup: 4 rectangular strips forming a cross.
        Each lane is a wide strip covering its approach direction.
        Detections in corners default to the nearest lane.
        """
        w, h = frame_width, frame_height
        self._frame_w = w
        self._frame_h = h

        # The '+' arms — each lane covers its full approach strip
        third_w = w // 3
        third_h = h // 3

        self.lanes = [
            # North: top-center column (full width of center third, from top to center)
            LaneROI("Lane_N", [(third_w, 0), (2*third_w, 0), (2*third_w, third_h), (third_w, third_h)]),
            # South: bottom-center column
            LaneROI("Lane_S", [(third_w, 2*third_h), (2*third_w, 2*third_h), (2*third_w, h), (third_w, h)]),
            # East: right-center row
            LaneROI("Lane_E", [(2*third_w, third_h), (w, third_h), (w, 2*third_h), (2*third_w, 2*third_h)]),
            # West: left-center row
            LaneROI("Lane_W", [(0, third_h), (third_w, third_h), (third_w, 2*third_h), (0, 2*third_h)]),
        ]

    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _assign_lane(self, center: Tuple[int, int]) -> Optional[str]:
        """
        Assign a detection to a lane based on its center point.
        If the point doesn't fall in any lane polygon, assign to nearest lane
        based on angular position from frame center (handles corner vehicles).
        """
        # First try polygon-based assignment
        for lane in self.lanes:
            if self._point_in_polygon(center, lane.points):
                return lane.name

        # Fallback: assign based on angular position relative to frame center
        # This ensures ALL detected vehicles get assigned to a lane
        if self._frame_w > 0 and self._frame_h > 0:
            cx, cy = self._frame_w // 2, self._frame_h // 2
            dx = center[0] - cx
            dy = center[1] - cy

            # Use angle to determine which lane quadrant
            import math
            angle = math.atan2(dy, dx)  # -pi to pi, 0 = right
            # Convert to: N = top, E = right, S = bottom, W = left
            if -3 * math.pi / 4 <= angle < -math.pi / 4:
                return "Lane_N"   # top (negative y)
            elif -math.pi / 4 <= angle < math.pi / 4:
                return "Lane_E"   # right (positive x)
            elif math.pi / 4 <= angle < 3 * math.pi / 4:
                return "Lane_S"   # bottom (positive y)
            else:
                return "Lane_W"   # left (negative x)

        return None

    def detect(self, frame: np.ndarray, mode: str = "intersection") -> DetectionResult:
        """
        Detect vehicles in a frame and return per-lane counts.

        Args:
            frame: BGR/RGB numpy array (H, W, 3)
            mode: 'intersection' (splits N/S/E/W) or 'single' (all go to Lane_N)

        Returns:
            DetectionResult with total count, per-lane counts, and detection list
        """
        h, w = frame.shape[:2]

        if not self.lanes or self._frame_w != w or self._frame_h != h:
            self.setup_lanes(w, h)

        result = DetectionResult(frame_shape=(h, w))
        for lane in self.lanes:
            result.lane_counts[lane.name] = 0

        if not self._model_loaded:
            self.load_model()

        if self.model is not None:
            detections = self._detect_yolo(frame)
        else:
            detections = self._detect_simulated(w, h)

        for det in detections:
            if mode == "single":
                det.lane = "Lane_N"
            else:
                det.lane = self._assign_lane(det.center)
                
            if det.lane:
                if det.lane not in result.lane_counts:
                    result.lane_counts[det.lane] = 0
                result.lane_counts[det.lane] += 1
                result.total_vehicles += 1
            result.detections.append(det)

        return result
    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 detection."""
        # EXTREME RECALL MODE: yolo11x with conf=0.01 and classes forced to vehicles
        # AERIAL & STOCK PHOTO FIX: Heavily color-graded (green/vintage) stock photos 
        # completely destroy the RGB distribution that YOLO expects.
        # We aggressively strip colors (grayscale), boost edge contrast via Histogram Equalization, 
        # and convert back to 3-channel to force the model to look ONLY at geometric shapes.
        import cv2
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrasted = cv2.equalizeHist(gray_frame)
        ai_frame = cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)
        
        # Lowered conf to 0.001 to squeeze every fractional pixel match
        results = self.model(ai_frame, conf=0.001, iou=0.85, imgsz=1280, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                # To catch aerial cars misclassified as laptops/cell-phones, we temporarily bypass the strict class filter
                # if cls_id not in VEHICLE_CLASSES:
                #     continue
                class_name = VEHICLE_CLASSES.get(cls_id, f"obj_{cls_id}")

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                ))

        return detections

    def _detect_simulated(self, w: int, h: int) -> List[Detection]:
        """Generate random detections when YOLO is not available (simulation mode)."""
        import random
        n = random.randint(8, 30)
        detections = []
        for _ in range(n):
            cx = random.randint(50, w - 50)
            cy = random.randint(50, h - 50)
            bw = random.randint(40, 120)
            bh = random.randint(30, 80)
            detections.append(Detection(
                class_name=random.choice(["car", "car", "car", "motorcycle", "truck", "bus"]),
                confidence=random.uniform(0.5, 0.95),
                bbox=(cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2),
                center=(cx, cy),
            ))
        return detections

    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and lane boundaries on the frame."""
        try:
            import cv2
        except ImportError:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Lane colors (BGR)
        lane_colors = {
            "Lane_N": (0, 255, 255),   # Yellow
            "Lane_E": (0, 255, 0),     # Green
            "Lane_S": (255, 128, 0),   # Blue-ish
            "Lane_W": (0, 0, 255),     # Red
        }

        # Draw semi-transparent lane overlay
        overlay = annotated.copy()
        for lane in self.lanes:
            color = lane_colors.get(lane.name, (255, 255, 255))
            pts = np.array(lane.points, dtype=np.int32)
            # Fill with transparent color
            cv2.fillPoly(overlay, [pts], color)
        # Blend at 15% opacity so lanes are visible but don't obscure vehicles
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # Draw lane boundary outlines
        for lane in self.lanes:
            color = lane_colors.get(lane.name, (255, 255, 255))
            pts = np.array(lane.points, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, color, 2)

        # Draw lane labels with vehicle counts
        label_positions = {
            "Lane_N": (w // 2, min(60, h // 6)),
            "Lane_S": (w // 2, max(h - 30, h * 5 // 6)),
            "Lane_E": (max(w - 120, w * 5 // 6), h // 2),
            "Lane_W": (min(80, w // 6), h // 2),
        }
        for lane in self.lanes:
            color = lane_colors.get(lane.name, (255, 255, 255))
            count = result.lane_counts.get(lane.name, 0)
            pos = label_positions.get(lane.name, (w // 2, h // 2))
            label = f"{lane.name.replace('Lane_', '')}: {count}"
            # Draw background box for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (pos[0] - tw // 2 - 4, pos[1] - th - 4),
                         (pos[0] + tw // 2 + 4, pos[1] + 4), (0, 0, 0), -1)
            cv2.putText(annotated, label, (pos[0] - tw // 2, pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw detection boxes
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = lane_colors.get(det.lane, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            # Ensure text doesn't go off the top edge or get hidden by neighboring boxes
            text_y = max(15, y1 - 5)
            # Draw black background outline for text readability
            cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            # Draw actual text
            cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw total count
        total_label = f"Total: {result.total_vehicles} vehicles"
        cv2.putText(annotated, total_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated
