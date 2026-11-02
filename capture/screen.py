from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from config.settings import Config

logger = logging.getLogger("football_ai.capture")


class CaptureProtocol(Protocol):
    """Any object that can produce frames."""

    def grab(self) -> np.ndarray | None: ...

    def release(self) -> None: ...


class ScreenCapture:
    """Live screen capture using mss.

    Captures the configured monitor region at each call to grab().
    """

    def __init__(self, cfg: Config) -> None:
        import mss  # type: ignore[import-untyped]

        self._sct = mss.mss()
        monitor_idx = getattr(cfg.capture, "monitor_index", 1)
        mon = self._sct.monitors[monitor_idx]

        # 1. Получаем регион один раз
        region = getattr(cfg.capture, "region", None)

        # 2. Проверяем наличие всех данных
        has_region = region is not None and all(hasattr(region, a) for a in ["x", "y", "w", "h"])

        if has_region:
            # Используем getattr ПРЯМО ТУТ, чтобы PyCharm не сомневался в типах
            self._region = {
                "left": int(getattr(region, "x")),
                "top": int(getattr(region, "y")),
                "width": int(getattr(region, "w")),
                "height": int(getattr(region, "h")),
            }
        else:
            self._region = {
                "left": int(mon["left"]),
                "top": int(mon["top"]),
                "width": int(getattr(cfg.capture, "width", mon["width"])),
                "height": int(getattr(cfg.capture, "height", mon["height"])),
            }
        logger.info("ScreenCapture initialized monitor_index=%s region=%s", monitor_idx, self._region)

    def grab(self) -> np.ndarray | None:
        """Capture a frame as a BGR numpy array."""
        shot = self._sct.grab(self._region)
        frame = np.array(shot, dtype=np.uint8)
        # mss returns BGRA — drop alpha channel
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        logger.debug("capture.grab backend=mss shape=%s", frame.shape)
        return frame

    def release(self) -> None:
        logger.info("ScreenCapture released")
        self._sct.close()


class DXCamCapture:
    """High-speed DXGI-based capture (Windows + NVIDIA). Often faster than mss for full-screen games.

    Install: ``pip install dxcam``

    Uses asynchronous capture + ``get_latest_frame()`` when ``dxcam_target_fps`` > 0 (recommended for 60 FPS).
    """

    def __init__(self, cfg: Config) -> None:
        import dxcam  # type: ignore[import-untyped]

        cap_cfg = getattr(cfg, "capture", None)
        output_idx = int(getattr(cap_cfg, "dxcam_output_idx", 0))
        target_fps = int(getattr(cap_cfg, "dxcam_target_fps", 60))

        region = getattr(cap_cfg, "region", None)
        has_region = region is not None and all(hasattr(region, a) for a in ("x", "y", "w", "h"))
        rgn: tuple[int, int, int, int] | None
        if has_region:
            x, y = int(getattr(region, "x")), int(getattr(region, "y"))
            w, h = int(getattr(region, "w")), int(getattr(region, "h"))
            rgn = (x, y, x + w, y + h)
        else:
            # Match mss-style crop: primary monitor + optional width/height from config
            import mss  # type: ignore[import-untyped]

            sct = mss.mss()
            monitor_idx = int(getattr(cap_cfg, "monitor_index", 1))
            mon = sct.monitors[monitor_idx]
            left, top = int(mon["left"]), int(mon["top"])
            w = int(getattr(cap_cfg, "width", mon["width"]))
            h = int(getattr(cap_cfg, "height", mon["height"]))
            sct.close()
            rgn = (left, top, left + w, top + h)

        self._cam = dxcam.create(output_idx=output_idx, region=rgn, output_color="BGR")
        self._async = target_fps > 0
        self._target_fps = max(target_fps, 1) if self._async else 0
        if self._async:
            self._cam.start(target_fps=self._target_fps)
            logger.info(
                "DXCam started output_idx=%s monitor_index=%s region=%s target_fps=%s",
                output_idx,
                int(getattr(cap_cfg, "monitor_index", 1)),
                rgn,
                self._target_fps,
            )
        else:
            logger.info(
                "DXCam synchronous grab output_idx=%s monitor_index=%s region=%s",
                output_idx,
                int(getattr(cap_cfg, "monitor_index", 1)),
                rgn,
            )

    def grab(self) -> np.ndarray | None:
        if self._cam is None:
            logger.warning("capture.grab backend=dxcam camera_unavailable")
            return None
        if self._async:
            frame = self._cam.get_latest_frame()
            if frame is None:
                frame = self._cam.grab()
            if frame is not None:
                logger.debug("capture.grab backend=dxcam async shape=%s", frame.shape)
            else:
                logger.debug("capture.grab backend=dxcam async frame=None")
            return frame
        out = self._cam.grab()
        if out is not None:
            logger.debug("capture.grab backend=dxcam sync shape=%s", out.shape)
        else:
            logger.debug("capture.grab backend=dxcam sync frame=None")
        return out

    def release(self) -> None:
        if self._cam is None:
            return
        if self._async:
            try:
                self._cam.stop()
            except (AttributeError, RuntimeError):
                pass
        logger.info("DXCamCapture released")
        self._cam = None


def create_capture(cfg: Config) -> CaptureProtocol:
    """Factory: ``capture.backend`` is ``mss`` (default) or ``dxcam``."""
    cap_cfg = getattr(cfg, "capture", None)
    backend = str(getattr(cap_cfg, "backend", "mss")).lower().strip()
    logger.info("create_capture backend=%s", backend)
    if backend == "dxcam":
        try:
            return DXCamCapture(cfg)
        except ImportError as e:
            logger.error("dxcam not installed: pip install dxcam — falling back to mss (%s)", e)
            return ScreenCapture(cfg)
    return ScreenCapture(cfg)


class VideoFileCapture:
    """Capture frames from a video file (for offline replay testing)."""

    def __init__(self, video_path: str | Path) -> None:
        import cv2
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

    def grab(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()


class StaticFrameCapture:
    """Returns pre-loadead frames in sequence (for unit tests)."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._idx = 0

    def grab(self) -> np.ndarray | None:
        if self._idx >= len(self._frames):
            return None
        frame = self._frames[self._idx]
        self._idx += 1
        return frame

    def release(self) -> None:
        pass
