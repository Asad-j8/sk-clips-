"""Face detection, tracking, and crop math using OpenCV YuNet (with Haar Cascade fallback).

Implements:
- YuNet face detection with landmark / bbox extraction
- Haar Cascade fallback if model weights or engine unavailable
- Face ratio rule: Crop width >= 1.1 * face width to ensure no faces are cropped
- IoU-based persistent face tracking across frames
- Camera inset (webcam) detection over screen recordings
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_YUNET_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_HAAR_PATH = Path("/data/user/0/com.termux/files/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")


@dataclass(frozen=True)
class FaceDetection:
    """Detected face bounding box and metadata."""

    face_id: str
    x: float  # Normalized 0..1 or pixel depending on context
    y: float
    w: float
    h: float
    confidence: float
    landmarks: list[tuple[float, float]] | None = None

    @property
    def center_x(self) -> float:
        return self.x + (self.w / 2.0)

    @property
    def center_y(self) -> float:
        return self.y + (self.h / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx, by, bw, bh = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


class FaceTracker:
    """Face detector and persistent multi-frame tracker using OpenCV YuNet."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_YUNET_MODEL
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.yunet_detector = None
        self.haar_detector = None
        self.next_face_idx = 1
        self.prev_tracks: list[FaceDetection] = []

        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize YuNet detector, falling back to Haar cascade if needed."""
        if hasattr(cv2, "FaceDetectorYN") and self.model_path.is_file():
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(self.model_path),
                    "",
                    (320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000,
                )
                logger.info("Initialized OpenCV YuNet face detector from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("YuNet initialization failed: %s. Falling back to Haar Cascade.", e)

        # Fallback to Haar Cascade
        if DEFAULT_HAAR_PATH.is_file():
            try:
                self.haar_detector = cv2.CascadeClassifier(str(DEFAULT_HAAR_PATH))
                logger.info("Initialized OpenCV Haar Cascade face detector from %s", DEFAULT_HAAR_PATH)
            except Exception as e:
                logger.error("Haar cascade initialization failed: %s", e)

    def detect_faces(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect all faces in a single BGR video frame.

        Returns:
            List of FaceDetection objects with pixel coordinates.
        """
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        raw_detections: list[tuple[float, float, float, float, float, list[tuple[float, float]] | None]] = []

        # 1. Try YuNet
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                ret, faces = self.yunet_detector.detect(frame_bgr)
                if ret > 0 and faces is not None:
                    for f in faces:
                        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        conf = float(f[14]) if len(f) > 14 else 1.0
                        # 5 landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner
                        landmarks = []
                        if len(f) >= 14:
                            for l_idx in range(4, 14, 2):
                                landmarks.append((float(f[l_idx]), float(f[l_idx + 1])))
                        raw_detections.append((fx, fy, fw, fh, conf, landmarks))
            except Exception as e:
                logger.warning("YuNet detection error on frame: %s", e)

        # 2. Fallback to Haar cascade if YuNet returned nothing
        if not raw_detections and self.haar_detector is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                boxes = self.haar_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                for (hx, hy, hw, hh) in boxes:
                    raw_detections.append((float(hx), float(hy), float(hw), float(hh), 0.85, None))
            except Exception as e:
                logger.warning("Haar cascade detection error: %s", e)

        # Match with previous tracks using IoU for persistent IDs
        return self._assign_persistent_ids(raw_detections)

    def _assign_persistent_ids(
        self,
        raw_detections: list[tuple[float, float, float, float, float, list[tuple[float, float]] | None]],
    ) -> list[FaceDetection]:
        """Assign persistent IDs (face_1, face_2, ...) using IoU matching."""
        current_faces: list[FaceDetection] = []
        unmatched_prev = list(self.prev_tracks)

        # Sort detections largest area first
        sorted_raw = sorted(raw_detections, key=lambda d: d[2] * d[3], reverse=True)

        for fx, fy, fw, fh, conf, landmarks in sorted_raw:
            best_match: FaceDetection | None = None
            best_iou = 0.0

            for prev in unmatched_prev:
                iou = compute_iou([fx, fy, fw, fh], [prev.x, prev.y, prev.w, prev.h])
                if iou > 0.25 and iou > best_iou:
                    best_iou = iou
                    best_match = prev

            if best_match:
                assigned_id = best_match.face_id
                unmatched_prev.remove(best_match)
            else:
                assigned_id = f"face_{self.next_face_idx}"
                self.next_face_idx += 1

            current_faces.append(
                FaceDetection(
                    face_id=assigned_id,
                    x=fx,
                    y=fy,
                    w=fw,
                    h=fh,
                    confidence=conf,
                    landmarks=landmarks,
                )
            )

        self.prev_tracks = current_faces
        return current_faces

    def detect_camera_inset(
        self, frame_bgr: np.ndarray, source_w: int, source_h: int
    ) -> FaceDetection | None:
        """Find webcam/presenter inset in screencast recordings.

        Usually a smaller face located in one of the 4 screen corners or along the margins.
        """
        faces = self.detect_faces(frame_bgr)
        if not faces:
            return None

        # Look for small face relative to screen (face width between 5% and 35% of frame)
        candidate_insets = [
            f for f in faces if 0.05 * source_w <= f.w <= 0.35 * source_w
        ]
        if candidate_insets:
            # Pick highest confidence or topmost candidate
            return candidate_insets[0]
        return faces[0]


def check_face_crop_rule(crop_w: float, face_w: float) -> bool:
    """Verify face rule: Crop width must be at least 1.1x face width so face is never cropped."""
    if face_w <= 0:
        return True
    return crop_w >= (1.1 * face_w)
