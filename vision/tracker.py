"""Multi-object tracker with temporal smoothing.

Uses a simple IoU-based assignment strategy (no Kalman filter dependency).
Tracks are smoothed with exponential moving average and velocities are
estimated from position deltas.

Design choice: we avoid a full Kalman filter to keep the code self-contained
and easy to reason about. For higher accuracy, swap this tracker for a
DeepSORT or ByteTrack implementation behind the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import Config
from vision.geometry import Rect, Vec2
from vision.types import Detection, ObjectClass, TrackedObject

logger = logging.getLogger("football_ai.tracker")


@dataclass
class _TrackState:
    """Internal mutable state for a single track."""

    track_id: int
    obj_class: ObjectClass
    raw_position: Vec2
    smooth_position: Vec2
    velocity: Vec2
    bbox: Rect
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class ObjectTracker:
    """IoU-based multi-object tracker with EMA smoothing.

    Public interface:
        tracker.update(detections, class_map) -> list[TrackedObject]
    """

    def __init__(self, cfg: Config) -> None:
        # Используем getattr, чтобы бот не падал, если в конфиге чего-то нет
        tracker_cfg = getattr(cfg, 'tracker', cfg)
        self._max_age = getattr(tracker_cfg, 'max_age', 5)
        self._min_hits = getattr(tracker_cfg, 'min_hits', 3)
        self._iou_thresh = getattr(tracker_cfg, 'iou_match_threshold', 0.3)
        self._vel_alpha = getattr(tracker_cfg, 'velocity_smoothing_alpha', 0.2)
        self._pos_alpha = getattr(tracker_cfg, 'position_smoothing_alpha', 0.5)
        self._max_disp = getattr(tracker_cfg, 'max_displacement_per_frame', 100.0)
        self._next_id = 1
        self._tracks: list[_TrackState] = []

    def update(
            self, detections: list[Detection], class_map: dict[int, str]
    ) -> list[TrackedObject]:
        """Match detections to existing tracks and return updated TrackedObjects."""
        logger.debug(
            "tracker.update detections=%s existing_tracks=%s",
            len(detections),
            len(self._tracks),
        )
        # 1. Predict positions for existing tracks (linear)
        for t in self._tracks:
            t.raw_position = t.raw_position + t.velocity
            t.age += 1
            t.time_since_update += 1

        # 2. Build IoU cost matrix and greedily match
        matched_det_indices: set[int] = set()
        matched_track_indices: set[int] = set()

        # Sort matches by descending IoU for greedy assignment
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                iou = track.bbox.iou(det.bbox)
                if iou >= self._iou_thresh:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for _iou_val, ti, di in pairs:
            if ti in matched_track_indices or di in matched_det_indices:
                continue
            det = detections[di]
            track = self._tracks[ti]

            det_center = det.center
            displacement = det_center.distance_to(track.raw_position)
            if displacement > self._max_disp:
                continue

            # Update track with matched detection
            old_pos = track.smooth_position
            track.raw_position = det_center
            track.smooth_position = Vec2(
                old_pos.x * (1 - self._pos_alpha) + det_center.x * self._pos_alpha,
                old_pos.y * (1 - self._pos_alpha) + det_center.y * self._pos_alpha,
            )
            new_vel = track.smooth_position - old_pos
            track.velocity = Vec2(
                track.velocity.x * (1 - self._vel_alpha) + new_vel.x * self._vel_alpha,
                track.velocity.y * (1 - self._vel_alpha) + new_vel.y * self._vel_alpha,
            )
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.obj_class = ObjectClass.from_class_id(det.class_id, class_map)
            track.hits += 1
            track.time_since_update = 0
            matched_track_indices.add(ti)
            matched_det_indices.add(di)
            logger.debug(
                "tracker.match track_id=%s class=%s det_idx=%s conf=%.3f",
                track.track_id,
                track.obj_class.value,
                di,
                track.confidence,
            )

        # 3. Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            center = det.center
            new_track = _TrackState(
                track_id=self._next_id,
                obj_class=ObjectClass.from_class_id(det.class_id, class_map),
                raw_position=center,
                smooth_position=center,
                velocity=Vec2.zero(),
                bbox=det.bbox,
                confidence=det.confidence,
                hits=1,
            )
            self._next_id += 1
            self._tracks.append(new_track)
            logger.debug(
                "tracker.new track_id=%s class=%s conf=%.3f",
                new_track.track_id,
                new_track.obj_class.value,
                new_track.confidence,
            )

        # 4. Remove dead tracks
        self._tracks = [t for t in self._tracks if t.time_since_update <= self._max_age]

        # 5. Build output — only confirmed tracks (enough hits)
        result: list[TrackedObject] = []
        for t in self._tracks:
            if t.hits >= self._min_hits:
                result.append(TrackedObject(
                    track_id=t.track_id,
                    obj_class=t.obj_class,
                    position=t.smooth_position,
                    velocity=t.velocity,
                    bbox=t.bbox,
                    confidence=t.confidence,
                    age=t.age,
                    hits=t.hits,
                    time_since_update=t.time_since_update,
                    predicted=t.time_since_update > 0,
                ))
        logger.debug(
            "tracker.result confirmed=%s live_tracks=%s ids=%s",
            len(result),
            len(self._tracks),
            [obj.track_id for obj in result],
        )
        return result

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
