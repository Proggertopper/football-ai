"""Football AI Research Framework — Main entry point.

Orchestrates: Capture → Detect → Track → Estimate → Tactize → Decide → Execute → Observe
Controls: F1 = activate bot, F2 = pause bot, Q in overlay window = quit

After F1, the bot stays active until F2. Gamepad execution is gated only by that toggle,
not by temporary misses on the ball or other vision gaps (no auto-pause on sparse frames).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path


import numpy as np

try:
    import keyboard
except ImportError:
    keyboard = None  # type: ignore[assignment]

try:
    import cv2
    _CV2 = True
except ImportError:
    cv2 = None
    _CV2 = False

from config.settings import Config, load_config
from control.fc26_mechanics import FC26Mechanics
from control.gamepad import GamepadInterface
from control.controller import ActionController
from decision.engine import DecisionEngine
from observe.logger import EventLogger
from observe.metrics import PerformanceMetrics
from observe.overlay import DebugOverlay
from observe.replay import ReplayRecorder
from observe.telemetry import TelemetryPipeline
from state.estimator import StateEstimator
from tactics.modes import TacticalFSM
from vision.tracker import ObjectTracker
from vision.types import TrackedObject

logger = logging.getLogger("football_ai")


class VisionDebugDumper:
    """Persist sample frames when detection quality collapses."""

    def __init__(self, cfg: Config) -> None:
        obs_cfg = getattr(cfg, "observe", None)
        self._enabled = bool(getattr(obs_cfg, "dump_zero_detection_frames", False))
        self._every_n = max(1, int(getattr(obs_cfg, "dump_zero_detection_every_n", 30)))
        self._sparse_threshold = max(0, int(getattr(obs_cfg, "dump_sparse_detection_threshold", 1)))
        self._sparse_every_n = max(1, int(getattr(obs_cfg, "dump_sparse_detection_every_n", 15)))
        self._out_dir = Path(getattr(obs_cfg, "dump_zero_detection_dir", "logs/vision_debug"))
        self._zero_streak = 0
        self._sparse_streak = 0
        if self._enabled:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "VisionDebugDumper enabled dir=%s zero_every_n=%s sparse_threshold=%s sparse_every_n=%s",
                self._out_dir,
                self._every_n,
                self._sparse_threshold,
                self._sparse_every_n,
            )

    def on_frame(self, frame: np.ndarray | None, frame_id: int, detection_count: int) -> None:
        if not self._enabled:
            return
        if detection_count == 0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0

        if detection_count <= self._sparse_threshold:
            self._sparse_streak += 1
        else:
            self._sparse_streak = 0

        if frame is None:
            return
        if self._zero_streak > 0 and self._zero_streak % self._every_n == 0:
            out_path = _save_frame_image(frame, self._out_dir / f"frame_{frame_id:06d}_zero_{self._zero_streak:04d}.jpg")
            logger.warning(
                "vision_debug saved zero-detection frame=%s streak=%s path=%s",
                frame_id,
                self._zero_streak,
                out_path,
            )
        if self._sparse_streak > 0 and self._sparse_streak % self._sparse_every_n == 0:
            out_path = _save_frame_image(frame, self._out_dir / f"frame_{frame_id:06d}_sparse_{self._sparse_streak:04d}_det_{detection_count}.jpg")
            logger.warning(
                "vision_debug saved sparse-detection frame=%s streak=%s detections=%s path=%s",
                frame_id,
                self._sparse_streak,
                detection_count,
                out_path,
            )


def _save_frame_image(frame: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _CV2 and cv2 is not None and out_path.suffix.lower() != ".npy":
        cv2.imwrite(str(out_path), frame)
        return out_path
    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, frame)
    return npy_path


def _probe_gamepad(duration_sec: float) -> None:
    pad = GamepadInterface()
    logger.info("Gamepad probe started for %.1f seconds", duration_sec)
    try:
        pad.update()
        time.sleep(max(duration_sec, 0.0))
    finally:
        pad.reset()
        pad.update()
        logger.info("Gamepad probe finished")


class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._running = False
        self._frame_id = 0
        self.bot_active = False
        self._bot_activation_frame_id: int | None = None

        self._tracker = ObjectTracker(cfg)
        self._estimator = StateEstimator(cfg)
        self._tactics = TacticalFSM(cfg)
        self._engine = DecisionEngine(cfg)
        self._fc26 = FC26Mechanics(cfg)

        self._gamepad = GamepadInterface()
        self._controller = ActionController(cfg, self._gamepad)

        self._metrics = PerformanceMetrics(window_size=60)
        self._event_logger = EventLogger(cfg)
        self._overlay = DebugOverlay(cfg)
        self._replay = ReplayRecorder(cfg, session_id=str(int(time.time())))
        self._telemetry = TelemetryPipeline(cfg, self._event_logger, self._metrics)
        self._vision_debug = VisionDebugDumper(cfg)

    def run_live(self) -> None:
        from capture.screen import create_capture
        from vision.detector import YOLODetector

        capture = create_capture(self._cfg)
        detector = YOLODetector(self._cfg)
        class_map = self._build_class_map()

        self._running = True
        self.bot_active = False
        self._bot_activation_frame_id = None

        logger.info(">>> SYSTEM INITIALIZED")
        for i in range(5, 0, -1):
            logger.info(f">>> Starting in {i}s... (switch to game window)")
            time.sleep(1)

        logger.info(">>> SYSTEM RUNNING — F1=START, F2=PAUSE")

        obs_cfg = getattr(self._cfg, 'observe', None)
        overlay_on = getattr(obs_cfg, 'overlay_enabled', False) if obs_cfg else False

        if overlay_on and _CV2 and cv2 is not None:
            cv2.namedWindow("Football AI Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Football AI Debug", 960, 540)

        try:
            while self._running:
                # --- Hotkeys ---
                if keyboard is not None:
                    if keyboard.is_pressed("f1") and not self.bot_active:
                        self.bot_active = True
                        self._bot_activation_frame_id = self._frame_id
                        logger.info(">>> BOT ACTIVATED")
                        time.sleep(0.3)
                    if keyboard.is_pressed("f2") and self.bot_active:
                        self.bot_active = False
                        self._bot_activation_frame_id = None
                        self._controller.cancel()
                        self._engine.reset_hysteresis()
                        logger.info(">>> BOT PAUSED")
                        time.sleep(0.3)
                    live_cfg = getattr(self._cfg, "live", None)
                    caps_debug = bool(getattr(live_cfg, "capslock_send_a_debug", False))
                    if caps_debug and keyboard.is_pressed("caps lock"):
                        self._gamepad.press_button("A")
                        self._gamepad.update()
                        logger.info(">>> CAPSLOCK debug: Xbox 'A'")
                        time.sleep(0.1)
                        self._gamepad.release_button("A")
                        self._gamepad.update()
                        time.sleep(0.4)

                self._metrics.total.start()
                self._metrics.tick_frame()
                self._frame_id += 1

                # --- Capture ---
                frame = capture.grab()
                if frame is None:
                    break

                # --- Detect ---
                detections = detector.detect(frame, self._frame_id)
                self._vision_debug.on_frame(frame, self._frame_id, len(detections))
                for det in detections:
                    self._metrics.record_detection_confidence(det.confidence)

                # --- Track ---
                tracked = self._tracker.update(detections, class_map)

                # --- Periodic console log ---
                if self._frame_id % 30 == 0:
                    logger.info(
                        "Frame %s | raw_detections=%s | confirmed_tracks=%s | "
                        "FPS: %.1f | bot_active=%s (F1=start F2=pause)",
                        self._frame_id,
                        len(detections),
                        len(tracked),
                        self._metrics.fps,
                        self.bot_active,
                    )

                # --- Process (estimate → decide → execute) ---
                # execute_gamepad follows F1/F2 only; do not tie to ball/detections.
                self._process_frame(tracked, frame, execute_gamepad=self.bot_active)

                obs_cfg = getattr(self._cfg, 'observe', None)
                overlay_on = getattr(obs_cfg, 'overlay_enabled', False) if obs_cfg else False

                # --- Show overlay ---
                if overlay_on and _CV2 and cv2 is not None:
                    cv2.imshow("Football AI Debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self._metrics.total.stop()

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            if _CV2 and cv2 is not None:
                cv2.destroyAllWindows()
            self._shutdown(capture)

    def _process_frame(
            self,
            tracked: list[TrackedObject],
            frame: np.ndarray | None = None,
            execute_gamepad: bool = True,
    ) -> None:
        """Core per-frame: estimate → tactize → decide → execute."""
        # --- State estimation ---
        current_fps = self._metrics.fps if self._metrics.fps > 0 else 30.0
        state = self._estimator.estimate(tracked, self._frame_id, fps=current_fps)

        state.bot_active = self.bot_active
        if (
            self.bot_active
            and self._bot_activation_frame_id is not None
            and (self._frame_id - self._bot_activation_frame_id) < 100
        ):
            state.set_piece_wait_suppressed = True
        else:
            state.set_piece_wait_suppressed = False

        # --- Tactical FSM ---
        tactical = self._tactics.update(state)

        # --- Decision (single score pass inside decide(); trace via football_ai.decision logger) ---
        result = self._engine.decide(state)
        chosen = result.chosen
        logger.debug(
            "process frame=%s tracked=%s controlled=%s possession=%s chosen=%s score=%.3f risk=%.3f dir=%s execute=%s",
            state.frame_id,
            len(tracked),
            state.controlled_player.track_id if state.controlled_player is not None else None,
            state.possession.value,
            chosen.action_type.value,
            chosen.score,
            chosen.risk,
            chosen.direction,
            execute_gamepad,
        )

        obs_cfg = getattr(self._cfg, "observe", None)
        if obs_cfg and getattr(obs_cfg, "pipeline_print_frames", False):
            logger.info(
                "frame=%s dist_ball=%.1f possession=%s top3=%s chosen=%s",
                state.frame_id,
                state.distance_to_ball if state.distance_to_ball < 1e8 else -1.0,
                state.possession.value,
                [(c.action_type.value, round(c.score, 2)) for c in result.candidates[:3]],
                chosen.action_type.value,
            )

        # --- Execute ---
        if execute_gamepad:
            # Pause / set-piece wait disabled in estimator; never freeze stick from state flags
            self._controller.execute(chosen, freeze_left_stick=False)
            self._controller.tick()
        else:
            self._controller.tick()

        # --- Telemetry & Overlay ---
        self._telemetry.record_decision(chosen.action_type.value)
        self._telemetry.record_mode(tactical.mode.value)
        self._telemetry.tick(self._frame_id)

        if frame is not None:
            self._overlay.render(frame, state, tactical, chosen, self._metrics, tracked)

    def _build_class_map(self) -> dict[int, str]:
        vis_cfg = getattr(self._cfg, 'vision', None)
        raw = getattr(vis_cfg, 'class_map', {}) if vis_cfg else {}
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
        if hasattr(raw, "to_dict"):
            return {int(k): str(v) for k, v in raw.to_dict().items()}
        # Fallback: scan attributes
        result: dict[int, str] = {}
        for i in range(10):
            if hasattr(raw, str(i)):
                result[i] = getattr(raw, str(i))
        return result

    def _shutdown(self, capture=None) -> None:
        self._running = False
        self._controller.cancel()
        self._telemetry.final_flush(self._frame_id)
        self._event_logger.close()
        self._replay.close()
        if capture is not None:
            capture.release()
        logger.info(f"Pipeline stopped after {self._frame_id} frames.")
        logger.info(f"Performance: {self._metrics.summary()}")

    @staticmethod
    def run_replay(video_path: str) -> None:
        """
        Метод для запуска анализа записанного видео вместо живой игры.
        Пока что это просто заглушка, чтобы не было ошибки.
        """
        logger.info(f">>> REPLAY MODE STARTED for: {video_path}")
        # Здесь в будущем будет код, который читает видеофайл кадр за кадром
        pass

    @staticmethod
    def run_snapshot(cfg: Config, delay_sec: float, out_path: str) -> None:
        from capture.screen import create_capture

        capture = create_capture(cfg)
        try:
            logger.info("Snapshot mode: switch to FIFA now, capture in %.1f seconds", delay_sec)
            time.sleep(max(delay_sec, 0.0))
            frame = capture.grab()
            if frame is None:
                raise RuntimeError("Snapshot capture returned no frame")
            saved_path = _save_frame_image(frame, Path(out_path))
            logger.info("Snapshot saved path=%s shape=%s", saved_path, frame.shape)
        finally:
            capture.release()

    def stop(self) -> None:
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Football AI Research Framework — Offline Sandbox"
    )
    parser.add_argument("--mode", choices=["live", "replay", "eval", "snapshot", "gamepad_probe"], default="live")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--capture-backend", choices=["mss", "dxcam"], default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--dump-zero-detections", action="store_true")
    parser.add_argument("--dump-dir", type=str, default=None)
    parser.add_argument("--snapshot-delay", type=float, default=3.0)
    parser.add_argument("--snapshot-out", type=str, default="logs/snapshot.jpg")
    parser.add_argument("--gamepad-probe-seconds", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    overrides: dict[str, dict[str, object]] = {}
    if args.capture_backend is not None:
        overrides.setdefault("capture", {})
        overrides["capture"]["backend"] = args.capture_backend
    if args.model_path is not None or args.conf is not None:
        overrides.setdefault("vision", {})
        if args.model_path is not None:
            overrides["vision"]["model_path"] = args.model_path
        if args.conf is not None:
            overrides["vision"]["confidence_threshold"] = args.conf
    if args.dump_zero_detections or args.dump_dir is not None:
        overrides.setdefault("observe", {})
        if args.dump_zero_detections:
            overrides["observe"]["dump_zero_detection_frames"] = True
        if args.dump_dir is not None:
            overrides["observe"]["dump_zero_detection_dir"] = args.dump_dir

    cfg = load_config(path=args.config, overrides=overrides or None)

    if args.mode == "eval":
        from evaluate.harness import EvaluationHarness

        harness = EvaluationHarness(cfg)
        report = harness.run_all()
        print(report.summary())
        sys.exit(0 if report.failed == 0 else 1)
    elif args.mode == "replay":
        if not args.video:
            parser.error("--video required for replay mode")
        pipeline = Pipeline(cfg)
        signal.signal(signal.SIGINT, lambda *_: pipeline.stop())
        pipeline.run_replay(args.video)
    elif args.mode == "snapshot":
        Pipeline.run_snapshot(cfg, delay_sec=args.snapshot_delay, out_path=args.snapshot_out)
    elif args.mode == "gamepad_probe":
        _probe_gamepad(args.gamepad_probe_seconds)
    else:
        pipeline = Pipeline(cfg)
        signal.signal(signal.SIGINT, lambda *_: pipeline.stop())
        pipeline.run_live()


if __name__ == "__main__":
    main()
