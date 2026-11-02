"""State estimation pipeline.

Transforms tracked objects into a MatchState by computing spatial
relationships, lane analysis, pressure metrics, and possession.
"""

from __future__ import annotations

import logging
import time
import math
from collections import deque

from config.settings import Config
from state.match_state import (
    MatchState,
    PassingLane,
    PossessionState,
    ShotLane,
    Zone,
)
from state.prediction import predict_state
from vision.geometry import Rect, Vec2
from vision.types import ObjectClass, TrackedObject

_log = logging.getLogger("football_ai.state")


class StateEstimator:
    """Builds MatchState from a list of TrackedObjects each frame.

    Maintains a short history buffer for temporal analysis.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        # Safe access to state config
        state_cfg = getattr(cfg, 'state', None)
        self._history: deque[MatchState] = deque(maxlen=getattr(state_cfg, 'history_length', 15))
        self._frame_count: int = 0

        # Cache config values with defaults
        self._possession_dist = getattr(state_cfg, 'possession_distance_threshold', 45.0)
        self._pressure_radius = getattr(state_cfg, 'pressure_radius', 120.0)
        self._pass_lane_w = getattr(state_cfg, 'passing_lane_width', 30.0)
        self._shot_lane_w = getattr(state_cfg, 'shot_lane_width', 25.0)
        self._open_space_r = getattr(state_cfg, 'open_space_radius', 80.0)

        zb = getattr(state_cfg, 'zone_boundaries', None)
        self._zone_def = getattr(zb, 'defensive_third', 0.33)
        self._zone_mid = getattr(zb, 'middle_third', 0.66)
        self._opp_box_nx_min = getattr(state_cfg, 'opponent_box_normalized_x_min', 0.78)

        world_cfg = getattr(cfg, 'world', None)
        self._screen_w = float(getattr(world_cfg, 'screen_width', 1920))
        self._screen_h = float(getattr(world_cfg, 'screen_height', 1080))
        self._crop_top = float(getattr(world_cfg, 'crop_top', 0.0))
        self._crop_bottom = float(getattr(world_cfg, 'crop_bottom', 0.0))
        self._crop_left = float(getattr(world_cfg, 'crop_left', 0.0))
        self._crop_right = float(getattr(world_cfg, 'crop_right', 0.0))

        self._pred_horizon = getattr(state_cfg, 'prediction_horizon_sec', 0.5)
        self._pass_recv_min_opp_dist = float(
            getattr(state_cfg, 'pass_receiver_min_opponent_distance', 72.0)
        )
        self._tackle_carrier_dist_max = float(
            getattr(state_cfg, 'tackle_carrier_distance_normalized_max', 0.06)
        )
        self._ball_ghost_max = int(getattr(state_cfg, "ball_ghost_max_frames", 10))
        self._ball_pause_after = int(getattr(state_cfg, "ball_absent_pause_frames", 30))
        self._ball_resume_conf = float(getattr(state_cfg, "ball_resume_confidence", 0.8))
        self._ball_resume_streak = int(getattr(state_cfg, "ball_resume_streak_frames", 5))
        self._set_piece_speed_max = float(getattr(state_cfg, "set_piece_max_ball_speed", 2.5))
        self._center_circle_r = float(getattr(state_cfg, "set_piece_center_radius_px", 85.0))
        self._boundary_frac = float(getattr(state_cfg, "pass_boundary_margin", 0.10))

        self._last_ball_pos: Vec2 | None = None
        self._last_ball_vel: Vec2 = Vec2.zero()
        self._no_real_ball_frames: int = 0
        self._ball_hi_streak: int = 0
        self._gameplay_frozen: bool = False

    def _filter_tracked_objects(
        self, tracked: list[TrackedObject], raw_track_count: int
    ) -> list[TrackedObject]:
        """Cap player tracks when confirmed_tracks > 22: keep 22 highest-confidence players."""
        player_labels = frozenset({
            "player_team",
            "player_opponent",
            "goalkeeper_team",
            "goalkeeper_opponent",
        })
        players: list[TrackedObject] = []
        rest: list[TrackedObject] = []
        for obj in tracked:
            cls_val = obj.obj_class.value if hasattr(obj.obj_class, "value") else str(obj.obj_class)
            if cls_val in player_labels:
                players.append(obj)
            else:
                rest.append(obj)
        if raw_track_count > 22 and len(players) > 22:
            players.sort(key=lambda o: -o.confidence)
            players = players[:22]
        elif len(players) > 22:
            players.sort(key=lambda o: -o.confidence)
            players = players[:22]
        return rest + players

    @staticmethod
    def _ghost_ball_track(pos: Vec2, vel: Vec2) -> TrackedObject:
        """Linear trajectory hold when the ball is briefly not detected."""
        return TrackedObject(
            track_id=-1,
            obj_class=ObjectClass.BALL,
            position=pos,
            velocity=vel,
            bbox=Rect(0.0, 0.0, 1.0, 1.0),
            confidence=0.2,
            predicted=True,
        )

    def estimate(self, tracked: list[TrackedObject], frame_id: int , fps: float = 30.0) -> MatchState:
        """Build a complete MatchState from current tracked objects."""
        ts = time.monotonic() * 1000
        self._frame_count += 1
        _log.debug("state.estimate frame=%s tracked_in=%s fps=%.2f", frame_id, len(tracked), fps)

        raw_track_count = len(tracked)
        tracked = self._filter_tracked_objects(tracked, raw_track_count)

        # --- Classify objects ---
        det_ball: TrackedObject | None = None
        teammates: list[TrackedObject] = []
        opponents: list[TrackedObject] = []
        gk_team: list[TrackedObject] = []
        gk_opp: list[TrackedObject] = []
        refs: list[TrackedObject] = []

        for obj in tracked:
            # Получаем строковое значение из Enum (например, "player_opponent")
            cls_val = obj.obj_class.value if hasattr(obj.obj_class, 'value') else str(obj.obj_class)

            if cls_val == "ball":
                if det_ball is None or obj.confidence > det_ball.confidence:
                    det_ball = obj
            elif cls_val == "player_team":
                teammates.append(obj)
            elif cls_val == "player_opponent":
                opponents.append(obj)
            elif cls_val == "goalkeeper_team":
                gk_team.append(obj)
            elif cls_val == "goalkeeper_opponent":
                gk_opp.append(obj)
            elif cls_val == "referee":
                refs.append(obj)
            else:
                _log.debug("unmapped detection class label: %s", cls_val)

        # --- Ball: track loss, linear ghost (≤ ball_ghost_max_frames), pause / resume ---
        ball_tracked = det_ball is not None
        if det_ball is not None:
            self._no_real_ball_frames = 0
            self._last_ball_pos = det_ball.position
            self._last_ball_vel = det_ball.velocity
            if det_ball.confidence >= self._ball_resume_conf:
                self._ball_hi_streak += 1
            else:
                self._ball_hi_streak = 0
            if self._ball_hi_streak >= self._ball_resume_streak:
                self._gameplay_frozen = False
            ball = det_ball
            _log.debug(
                "state.ball frame=%s tracked=yes conf=%.3f pos=(%.1f, %.1f) vel=(%.2f, %.2f)",
                frame_id,
                det_ball.confidence,
                det_ball.position.x,
                det_ball.position.y,
                det_ball.velocity.x,
                det_ball.velocity.y,
            )
        else:
            self._no_real_ball_frames += 1
            self._ball_hi_streak = 0
            if self._no_real_ball_frames > self._ball_pause_after:
                self._gameplay_frozen = True
            ball = None
            if (
                self._last_ball_pos is not None
                and self._no_real_ball_frames <= self._ball_ghost_max
            ):
                ghost_pos = self._last_ball_pos + self._last_ball_vel * float(
                    self._no_real_ball_frames
                )
                gx = max(
                    self._crop_left + 2,
                    min(self._screen_w - self._crop_right - 2, ghost_pos.x),
                )
                gy = max(
                    self._crop_top + 2,
                    min(self._screen_h - self._crop_bottom - 2, ghost_pos.y),
                )
                ball = self._ghost_ball_track(Vec2(gx, gy), self._last_ball_vel)
                _log.debug(
                    "state.ball frame=%s ghost=yes absent_frames=%s pos=(%.1f, %.1f)",
                    frame_id,
                    self._no_real_ball_frames,
                    gx,
                    gy,
                )
            else:
                _log.debug(
                    "state.ball frame=%s tracked=no absent_frames=%s frozen=%s",
                    frame_id,
                    self._no_real_ball_frames,
                    self._gameplay_frozen,
                )

        previous = self._history[-1] if self._history else None
        controlled = self._pick_controlled_player(teammates, ball, previous)
        if controlled is not None:
            _log.debug(
                "state.controlled frame=%s track_id=%s pos=(%.1f, %.1f) vel=(%.2f, %.2f)",
                frame_id,
                controlled.track_id,
                controlled.position.x,
                controlled.position.y,
                controlled.velocity.x,
                controlled.velocity.y,
            )
        else:
            _log.debug("state.controlled frame=%s track_id=None teammates=%s", frame_id, len(teammates))

        if not teammates and tracked and frame_id % 90 == 0:
            _log.warning(
                "frame=%s: no player_team — check vision.team_color_split_field_player_ids "
                "and tactics.team_colors.home (BGR). Raw classes may all map to opponent.",
                frame_id,
            )

        eff_w = max(1.0, self._screen_w - self._crop_left - self._crop_right)
        eff_h = max(1.0, self._screen_h - self._crop_top - self._crop_bottom)
        field_diag = math.hypot(eff_w, eff_h)
        screen_center = Vec2(
            self._crop_left + eff_w / 2.0,
            self._crop_top + eff_h / 2.0,
        )

        goal_x = self._screen_w - self._crop_right
        goal_y = self._screen_h / 2.0

        state = MatchState(
            frame_id=frame_id,
            timestamp_ms=ts,
            ball=ball,
            controlled_player=controlled,
            teammates=teammates,
            opponents=opponents,
            goalkeepers_team=gk_team,
            goalkeepers_opponent=gk_opp,
            referees=refs,
            raw_track_count=raw_track_count,
            ball_tracked=ball_tracked,
            goal_point=Vec2(goal_x, goal_y),
            screen_center=screen_center,
            is_paused=self._gameplay_frozen,
            ball_absent_frames=self._no_real_ball_frames,
            ball_high_confidence_streak=self._ball_hi_streak,
        )

        world_cfg = getattr(self._cfg, "world", None)
        state.attack_direction_sign = float(
            getattr(world_cfg, "attack_direction_sign", 1.0) if world_cfg is not None else 1.0
        )
        state.tackle_carrier_distance_threshold = self._tackle_carrier_dist_max

        if ball is not None:
            state.ball_velocity = ball.velocity
            state.ball_speed = ball.speed

        # Set piece / kick-off (skipped when set_piece_max_ball_speed is negative); overridden below
        state.wait_for_input = False
        if (
            self._set_piece_speed_max >= 0.0
            and ball is not None
            and ball_tracked
            and ball.speed <= self._set_piece_speed_max
        ):
            cx = self._crop_left + (self._screen_w - self._crop_left - self._crop_right) / 2.0
            cy = self._crop_top + (self._screen_h - self._crop_top - self._crop_bottom) / 2.0
            if ball.position.distance_to(Vec2(cx, cy)) < self._center_circle_r:
                state.wait_for_input = True
            else:
                nx = self._normalize_x(ball.position.x, self._screen_w, self._crop_left, self._crop_right)
                ny = self._normalize_y(ball.position.y, self._screen_h, self._crop_top, self._crop_bottom)
                if (nx < self._boundary_frac or nx > 1.0 - self._boundary_frac) and (
                    ny < self._boundary_frac or ny > 1.0 - self._boundary_frac
                ):
                    state.wait_for_input = True

        # --- Zones ---
        if ball is not None:
            state.ball_zone = self._classify_zone(ball.position)
        if controlled is not None:
            state.controlled_zone = self._classify_zone(controlled.position)
            nx_ctrl = self._normalize_x(
                controlled.position.x, self._screen_w, self._crop_left, self._crop_right
            )
            state.in_opponent_penalty_area = nx_ctrl >= self._opp_box_nx_min

        # --- Distance to ball ---
        if controlled is not None and ball is not None:
            state.distance_to_ball = controlled.position.distance_to(ball.position)
        elif ball is not None and teammates:
            state.distance_to_ball = min(
                t.position.distance_to(ball.position) for t in teammates
            )

        # --- Tackle range: controlled vs opponent nearest the ball (normalized by field diagonal) ---
        if controlled is not None and ball is not None and opponents:
            carrier = min(
                opponents,
                key=lambda o: o.position.distance_to(ball.position),
            )
            d_pix = controlled.position.distance_to(carrier.position)
            state.dist_ctrl_to_ball_carrier_normalized = d_pix / field_diag
        else:
            state.dist_ctrl_to_ball_carrier_normalized = 1.0

        # --- Possession ---
        state.possession = self._estimate_possession(ball, teammates, opponents)

        # --- Pressure & space ---
        if controlled is not None:
            opp_positions = [o.position for o in opponents]
            state.local_pressure = self._compute_pressure(controlled.position, opp_positions)
            state.nearest_opponent_dist = self._nearest_dist(controlled.position, opp_positions)
            tm_positions = [t.position for t in teammates if t.track_id != controlled.track_id]
            state.nearest_teammate_dist = self._nearest_dist(controlled.position, tm_positions)
            state.dribble_space = self._compute_dribble_space(controlled, opp_positions)
            state.open_space_around = self._compute_open_space(controlled.position, opp_positions)

        # --- Lanes ---
        if controlled is not None:
            opp_pos = [o.position for o in opponents]
            state.passing_lanes = self._analyze_passing_lanes(controlled, teammates, opp_pos)
            state.shot_lanes = self._analyze_shot_lanes(controlled, opp_pos)
            if state.passing_lanes:
                state.best_passing_lane = min(state.passing_lanes, key=lambda l: l.risk_score)
            if state.shot_lanes:
                state.best_shot_lane = max(state.shot_lanes, key=lambda l: l.quality)

        # --- Tactical indicators ---
        state.defensive_danger = self._compute_defensive_danger(state)
        state.counterattack_opportunity = self._compute_counter_opportunity(state)
        state.shot_opportunity = self._compute_shot_opportunity(state)

        # --- Predictions ---
        state.predicted_ball_pos, state.predicted_controlled_pos = predict_state(
            state, self._pred_horizon, fps=fps
        )

        # --- Link history ---
        if self._history:
            state.previous = self._history[-1]
        self._history.append(state)

        # Hard-off ball-loss pause and set-piece wait (config knobs kept for documentation only)
        state.is_paused = False
        state.wait_for_input = False

        _log.debug(
            "state.summary frame=%s team=%s opp=%s ball=%s controlled=%s possession=%s dist_ball=%.1f pressure=%.2f "
            "lanes(pass=%s shot=%s) danger=%.2f counter=%.2f shot=%.2f paused=%s wait=%s",
            frame_id,
            len(state.teammates),
            len(state.opponents),
            state.ball.track_id if state.ball is not None else None,
            state.controlled_player.track_id if state.controlled_player is not None else None,
            state.possession.value,
            state.distance_to_ball if state.distance_to_ball < 1e8 else -1.0,
            state.local_pressure,
            len(state.passing_lanes),
            len(state.shot_lanes),
            state.defensive_danger,
            state.counterattack_opportunity,
            state.shot_opportunity,
            state.is_paused,
            state.wait_for_input,
        )

        return state

    # --- Private helpers ---

    @staticmethod
    def _pick_controlled_player(
        teammates: list[TrackedObject],
        ball: TrackedObject | None,
        previous: MatchState | None,
    ) -> TrackedObject | None:
        if not teammates:
            return None
        if ball is None:
            return max(teammates, key=lambda t: t.confidence)

        ball_pos = ball.position

        if previous is not None and previous.controlled_player is not None:
            prev_id = previous.controlled_player.track_id
            prev_obj = next((t for t in teammates if t.track_id == prev_id), None)
            if prev_obj is not None:
                d_prev = prev_obj.position.distance_to(ball_pos)
                best_other = min(
                    (t.position.distance_to(ball_pos) for t in teammates if t.track_id != prev_id),
                    default=float("inf"),
                )
                if best_other == float("inf") or d_prev <= best_other * 1.2:
                    return prev_obj

        return min(teammates, key=lambda t: t.position.distance_to(ball_pos))

    def _classify_zone(self, pos: Vec2) -> Zone:
        nx = self._normalize_x(pos.x, self._screen_w, self._crop_left, self._crop_right)
        if nx < self._zone_def:
            return Zone.DEFENSIVE_THIRD
        if nx < self._zone_mid:
            return Zone.MIDDLE_THIRD
        return Zone.ATTACKING_THIRD

    @staticmethod
    def _normalize_x(x: float, screen_w: float, crop_left: float, crop_right: float) -> float:
        eff_w = screen_w - crop_left - crop_right
        if eff_w < 1:
            return 0.5
        return max(0.0, min(1.0, (x - crop_left) / eff_w))

    @staticmethod
    def _normalize_y(y: float, screen_h: float, crop_top: float, crop_bottom: float) -> float:
        eff_h = screen_h - crop_top - crop_bottom
        if eff_h < 1:
            return 0.5
        return max(0.0, min(1.0, (y - crop_top) / eff_h))

    def _estimate_possession(
            self,
            ball: TrackedObject | None,
            teammates: list[TrackedObject],
            opponents: list[TrackedObject],
    ) -> PossessionState:
        if ball is None:
            return PossessionState.LOOSE
        bp = ball.position
        team_near = any(t.position.distance_to(bp) < self._possession_dist for t in teammates)
        opp_near = any(o.position.distance_to(bp) < self._possession_dist for o in opponents)
        if team_near and opp_near:
            return PossessionState.CONTESTED
        if team_near:
            return PossessionState.OWN_TEAM
        if opp_near:
            return PossessionState.OPPONENT
        return PossessionState.LOOSE

    def _compute_pressure(self, pos: Vec2, opponents: list[Vec2]) -> float:
        count = sum(1 for o in opponents if pos.distance_to(o) < self._pressure_radius)
        return min(1.0, count / 4.0)

    @staticmethod
    def _nearest_dist(pos: Vec2, others: list[Vec2]) -> float:
        if not others:
            return float("inf")
        return min(pos.distance_to(o) for o in others)

    @staticmethod
    def _compute_dribble_space(player: TrackedObject, opponents: list[Vec2]) -> float:
        """Approximate free space in the player's movement direction."""
        if player is None:
            return 0.0
        direction = player.velocity.normalized()
        if direction.length() < 0.1:
            direction = Vec2(1.0, 0.0)
        check_dist = 200.0
        for d in range(10, int(check_dist), 10):
            probe = player.position + direction * d
            for opp in opponents:
                if probe.distance_to(opp) < 40:
                    return float(d)
        return check_dist


    def _compute_open_space(self, pos: Vec2, opponents: list[Vec2]) -> float:
        nearby = [pos.distance_to(o) for o in opponents if pos.distance_to(o) < self._open_space_r]
        if not nearby:
            return self._open_space_r
        return sum(nearby) / len(nearby)

    def _analyze_passing_lanes(
            self,
            controlled: TrackedObject,
            teammates: list[TrackedObject],
            opp_positions: list[Vec2],
    ) -> list[PassingLane]:
        lanes: list[PassingLane] = []
        from vision.geometry import is_lane_clear
        for tm in teammates:
            if tm.track_id == controlled.track_id:
                continue
            direction = (tm.position - controlled.position).normalized()
            dist = controlled.position.distance_to(tm.position)
            if dist < 20:
                continue
            recv_pressure = False
            min_opp_to_tm = float("inf")
            if opp_positions:
                min_opp_to_tm = min(
                    tm.position.distance_to(o) for o in opp_positions
                )
                if min_opp_to_tm < self._pass_recv_min_opp_dist:
                    recv_pressure = True
            clear = is_lane_clear(
                controlled.position, tm.position, opp_positions, self._pass_lane_w
            )
            risk = 0.0 if clear else 0.5
            risk += min(0.3, dist / 1000.0)
            risk += self._compute_pressure(tm.position, opp_positions) * 0.2
            if recv_pressure:
                risk += 0.18
            tm_nx = self._normalize_x(
                tm.position.x, self._screen_w, self._crop_left, self._crop_right
            )
            tm_ny = self._normalize_y(
                tm.position.y, self._screen_h, self._crop_top, self._crop_bottom
            )
            b_risk = 0.0
            margin = self._boundary_frac
            near_edge = (
                tm_nx < margin
                or tm_nx > 1.0 - margin
                or tm_ny < margin
                or tm_ny > 1.0 - margin
            )
            if near_edge:
                outward = 0.0
                if tm_nx < margin and direction.x < -0.15:
                    outward += 0.45
                if tm_nx > 1.0 - margin and direction.x > 0.15:
                    outward += 0.45
                if tm_ny < margin and direction.y < -0.15:
                    outward += 0.4
                if tm_ny > 1.0 - margin and direction.y > 0.15:
                    outward += 0.4
                b_risk = min(0.85, outward)
                risk += b_risk
            lanes.append(PassingLane(
                target=tm,
                direction=direction,
                distance=dist,
                is_clear=clear,
                obstruction_count=0 if clear else 1,
                risk_score=min(1.0, risk),
                receiver_high_pressure=recv_pressure,
                boundary_risk=b_risk,
            ))
        return lanes

    def _analyze_shot_lanes(
            self, controlled: TrackedObject, opp_positions: list[Vec2]
    ) -> list[ShotLane]:
        goal_x = self._screen_w - self._crop_right
        goal_center_y = self._screen_h / 2
        goal_half_height = 80.0
        lanes: list[ShotLane] = []
        from vision.geometry import is_lane_clear

        for offset in (-goal_half_height, -goal_half_height / 2, 0,
                       goal_half_height / 2, goal_half_height):
            target = Vec2(goal_x, goal_center_y + offset)
            direction = (target - controlled.position).normalized()
            dist = controlled.position.distance_to(target)
            clear = is_lane_clear(
                controlled.position, target, opp_positions, self._shot_lane_w
            )
            center_dir = Vec2(goal_x - controlled.position.x,
                              goal_center_y - controlled.position.y).normalized()
            dot = center_dir.dot(direction)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))

            quality = 0.4 if clear else 0.0
            quality += max(0.0, 1.0 - dist / 800.0) * 0.3
            quality += max(0.0, 1.0 - angle / 45.0) * 0.3

            lanes.append(ShotLane(
                target_point=target,
                direction=direction,
                distance=dist,
                is_clear=clear,
                angle_deg=angle,
                quality=max(0.0, min(1.0, quality)),
            ))
        return lanes

    @staticmethod
    def _compute_defensive_danger(state: MatchState) -> float:
        danger = 0.0
        if state.ball_zone == Zone.DEFENSIVE_THIRD:
            danger += 0.3
        if state.possession == PossessionState.OPPONENT:
            danger += 0.3
        if state.ball is not None:
            if state.ball_velocity.x < -2:
                danger += 0.2
        danger += state.local_pressure * 0.2
        return min(1.0, danger)


    @staticmethod
    def _compute_counter_opportunity(state: MatchState) -> float:
        opp = 0.0
        if state.possession == PossessionState.OWN_TEAM:
            opp += 0.2
        if state.controlled_zone == Zone.MIDDLE_THIRD:
            opp += 0.2
        if state.dribble_space > 150:
            opp += 0.3

        controlled_player = state.controlled_player
        if controlled_player is not None:
            ahead_opps = sum(
                1 for o in state.opponents
                if o.position.x > controlled_player.position.x
            )
            if ahead_opps <= 2:
                opp += 0.3

        return min(1.0, opp)

    @staticmethod
    def _compute_shot_opportunity(state: MatchState) -> float:
        if state.best_shot_lane is None:
            return 0.0
        return state.best_shot_lane.quality
