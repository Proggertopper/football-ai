"""YOLO detection interface."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np

from config.settings import Config
from vision.geometry import Rect
from vision.types import Detection

logger = logging.getLogger("football_ai.detector")

# Defaults tuned for 60 FPS targets on mid-range NVIDIA GPUs (e.g. RTX 3060) at 720p capture.
DEFAULT_IMGSZ: int = 320

@runtime_checkable
class DetectorProtocol(Protocol):
    """Any object that can produce detections from a frame."""

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Run inference on *frame* and return detections."""
        ...


class YOLODetector:
    """ultralytics YOLO wrapper with color-based team detection."""

    def __init__(self, cfg: Config) -> None:
        from ultralytics import YOLO
        self._cfg = cfg
        self._cfg_vision = getattr(cfg, 'vision', None)
        model_path = getattr(self._cfg_vision, 'model_path', "models/best.pt")
        self._model = YOLO(model_path)

        self._conf = getattr(self._cfg_vision, 'confidence_threshold', 0.35)
        self._iou = getattr(self._cfg_vision, 'iou_threshold', 0.45)
        self._device = getattr(self._cfg_vision, 'device', "cuda")
        self._input_size = int(getattr(self._cfg_vision, 'input_size', DEFAULT_IMGSZ))
        # Field players / GKs whose raw class id should be split into our vs opp by jersey color
        fp = getattr(self._cfg_vision, 'team_color_split_field_player_ids', None)
        self._split_field: set[int] = set(fp) if fp is not None else {0, 6}
        gk = getattr(self._cfg_vision, 'team_color_split_goalkeeper_ids', None)
        self._split_gk: set[int] = set(gk) if gk is not None else {1}
        self._debug_detections = bool(getattr(self._cfg_vision, 'debug_print_detections', False))
        self._zero_detection_streak = 0
        self._warn_streak = int(getattr(self._cfg_vision, "diagnostics_zero_detection_warn_streak", 30))
        dev = str(self._device).lower()
        self._cuda = dev == "cuda" or dev.startswith("cuda")
        # FP16 inference on GPU (matches TensorRT FP16 engines and speeds up .pt on RTX)
        self._half = bool(getattr(self._cfg_vision, "half_precision", True)) and self._cuda
        if self._cuda and bool(getattr(self._cfg_vision, "cudnn_benchmark", True)):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
            except (ImportError, AttributeError):
                pass
        logger.info(
            "YOLODetector initialized model=%s device=%s imgsz=%s conf=%.2f iou=%.2f half=%s",
            model_path,
            self._device,
            self._input_size,
            self._conf,
            self._iou,
            self._half,
        )

    def _get_team_by_color(self, frame, x1, y1, x2, y2, is_gk):
        """Определяет: наш игрок или чужой (через безопасный getattr)."""
        try:
            # 1. Безопасно достаем цвет из конфига
            tactics_cfg = getattr(self._cfg, 'tactics', None)
            team_colors = getattr(tactics_cfg, 'team_colors', {}) if tactics_cfg is not None else {}
            # Если цвета нет в конфиге, используем синий по умолчанию [B, G, R]
            home_color = getattr(team_colors, 'home', [200, 50, 0])
            our_color = np.array(home_color, dtype=np.uint8)

            h, w = frame.shape[:2]
            # Two ROIs: chest + lower torso — lighting/shadow often shifts one more than the other
            def _mean_bgr(y0f: float, y1f: float) -> np.ndarray | None:
                cy_a, cy_b = int(y1 + (y2 - y1) * y0f), int(y1 + (y2 - y1) * y1f)
                cx_a, cx_b = int(x1 + (x2 - x1) * 0.25), int(x2 - (x2 - x1) * 0.25)
                patch = frame[max(0, cy_a):min(h, cy_b), max(0, cx_a):min(w, cx_b)]
                if patch.size == 0:
                    return None
                return np.mean(patch, axis=(0, 1))

            m1 = _mean_bgr(0.22, 0.52)
            m2 = _mean_bgr(0.38, 0.68)
            if m1 is None and m2 is None:
                return 1 if is_gk else 0
            d1 = float(np.linalg.norm(m1 - our_color)) if m1 is not None else float("inf")
            d2 = float(np.linalg.norm(m2 - our_color)) if m2 is not None else float("inf")
            distance = min(d1, d2)

            base = float(getattr(tactics_cfg, "team_color_distance_threshold", 80) if tactics_cfg else 80)
            soft = float(getattr(tactics_cfg, "team_color_soft_factor", 1.0) if tactics_cfg else 1.0)
            thresh = base * max(1.0, soft)
            is_our_team = distance < thresh

            if is_gk:
                return 1 if is_our_team else 7  # 1: gk_team, 7: gk_opponent
            else:
                return 0 if is_our_team else 6  # 0: player_team, 6: player_opponent
        except Exception as e:
            logger.error(f"Color detection error: {e}")
            return 1 if is_gk else 0

    def detect(self, frame, frame_id):
        if frame is None:
            logger.debug("detect frame=%s skipped frame=None", frame_id)
            return []

        # imgsz=320 + half=True on CUDA are the default path for throughput (see DEFAULT_IMGSZ).
        results = self._model.predict(
            frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            imgsz=self._input_size,
            half=self._half,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())

                if cls_id in self._split_gk:
                    cls_id = self._get_team_by_color(
                        frame, xyxy[0], xyxy[1], xyxy[2], xyxy[3], is_gk=True
                    )
                elif cls_id in self._split_field:
                    cls_id = self._get_team_by_color(
                        frame, xyxy[0], xyxy[1], xyxy[2], xyxy[3], is_gk=False
                    )

                detections.append(Detection(
                    bbox=Rect(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]),
                    class_id=cls_id,
                    confidence=conf,
                    frame_id=frame_id
                ))

        if detections:
            self._zero_detection_streak = 0
            class_counts = dict(Counter(det.class_id for det in detections))
            logger.debug(
                "detect frame=%s detections=%s classes=%s avg_conf=%.3f",
                frame_id,
                len(detections),
                class_counts,
                sum(d.confidence for d in detections) / len(detections),
            )
        else:
            self._zero_detection_streak += 1
            logger.debug("detect frame=%s detections=0", frame_id)
            if self._warn_streak > 0 and self._zero_detection_streak % self._warn_streak == 0:
                logger.warning(
                    "detect frame=%s zero_detection_streak=%s model=%s device=%s imgsz=%s conf=%.2f",
                    frame_id,
                    self._zero_detection_streak,
                    getattr(self._cfg_vision, "model_path", "unknown"),
                    self._device,
                    self._input_size,
                    self._conf,
                )

        if detections and self._debug_detections:
            logger.info(
                "frame=%s raw_detections class_ids=%s conf=%s",
                frame_id,
                [d.class_id for d in detections],
                [round(d.confidence, 2) for d in detections],
            )

        return detections

class MockDetector:
    """Deterministic detector for unit tests and offline evaluation."""

    def __init__(self, frames: dict[int, list[Detection]] | None = None) -> None:
        self._frames: dict[int, list[Detection]] = frames or {}

    @staticmethod
    def detect_static(frame_id: int, frames_data: dict[int, list[Detection]]) -> list[Detection]:
        """Example of a static method if needed."""
        return frames_data.get(frame_id, [])

    def detect(self, _frame: np.ndarray, frame_id: int) -> list[Detection]:
        detections = self._frames.get(frame_id, [])
        logger.debug("mock_detect frame=%s detections=%s", frame_id, len(detections))
        return detections
