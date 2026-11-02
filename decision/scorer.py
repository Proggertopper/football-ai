"""Heuristic action scorer."""

from __future__ import annotations

import logging
import math
from typing import Any, Tuple

from config.settings import Config
from decision.actions import ActionSpec, ActionType, check_precondition
from state.match_state import MatchState, PossessionState, Zone
from vision.geometry import Vec2

logger = logging.getLogger("football_ai.scorer")


# (score, risk, expected_value, direction_raw, target_raw, metadata)
ScoreParts = Tuple[float, float, float, Any, Any, dict[str, float]]


class ActionScorer:
    """Produces scored ActionSpecs for all legal actions given a MatchState."""

    def __init__(self, cfg: Config) -> None:
        decision_cfg = getattr(cfg, "decision", None)
        w = getattr(decision_cfg, "weights", None)

        self._w_goal = getattr(w, "goal_threat", 1.5)
        self._w_safety = getattr(w, "safety", 1.2)
        self._w_support = getattr(w, "support", 0.8)
        self._w_turnover = getattr(w, "turnover_risk", -1.0)
        self._w_def_shape = getattr(w, "defensive_shape", 0.7)

        actions_cfg = getattr(cfg, "actions", None)
        self._cooldowns_cfg: dict[str, int] = {}
        self._durations_cfg: dict[str, int] = {}

        cd = getattr(actions_cfg, "cooldowns", {})
        self._cooldowns_cfg = cd.to_dict() if hasattr(cd, "to_dict") else (cd if isinstance(cd, dict) else {})

        dur = getattr(actions_cfg, "durations", {})
        self._durations_cfg = dur.to_dict() if hasattr(dur, "to_dict") else (dur if isinstance(dur, dict) else {})

        self._cooldown_timers: dict[str, int] = {}
        self._last_skipped: list[tuple[str, str]] = []
        self._post_pass_shoot_suppress_remaining: int = 0
        self._post_pass_shoot_suppress_frames: int = int(
            getattr(actions_cfg, "post_pass_shoot_suppress_frames", 18)
        )
        self._tackle_cooldown_frames: int = int(
            self._cooldowns_cfg.get("tackle")
            or self._cooldowns_cfg.get("stand_tackle")
            or 30
        )
        self._rs_flick_mag: float = float(
            getattr(actions_cfg, "rs_switch_flick_magnitude", 1.0)
        )

    _TACKLE_TYPES = frozenset({ActionType.STAND_TACKLE, ActionType.SLIDING_TACKLE})

    _SHOOT_TYPES = frozenset({
        ActionType.SHOOT,
        ActionType.LOW_DRIVEN_SHOT,
        ActionType.FINESSE_SHOT,
        ActionType.TRIVELA_SHOT,
    })
    _PASS_TYPES_FOR_SHOOT_LOCK = frozenset({
        ActionType.PASS,
        ActionType.PRECISION_PASS,
        ActionType.THROUGH_BALL,
    })

    @property
    def last_skipped(self) -> list[tuple[str, str]]:
        """(action.value, reason_code) for actions not scored this frame."""
        return list(self._last_skipped)

    def score_all(self, state: MatchState) -> list[ActionSpec]:
        """Score every legal action and return sorted list (best first)."""
        self._tick_cooldowns()
        self._last_skipped.clear()

        candidates: list[ActionSpec] = []
        for action_type in ActionType:
            if not check_precondition(action_type, state):
                self._last_skipped.append((action_type.value, "precondition_failed"))
                continue
            if self._is_on_cooldown(action_type):
                self._last_skipped.append((action_type.value, "cooldown"))
                continue
            if (
                action_type in self._SHOOT_TYPES
                and self._post_pass_shoot_suppress_remaining > 0
                and not state.in_opponent_penalty_area
            ):
                self._last_skipped.append((action_type.value, "post_pass_shoot_lock"))
                continue

            spec = self._score_action(action_type, state)
            if spec is not None:
                candidates.append(spec)

        candidates.sort(key=lambda s: s.score, reverse=True)
        top = [
            (
                spec.action_type.value,
                round(spec.score, 3),
                round(spec.risk, 3),
                spec.direction,
            )
            for spec in candidates[:5]
        ]
        logger.debug(
            "score_all frame=%s candidates=%s skipped=%s top=%s cooldowns=%s",
            state.frame_id,
            len(candidates),
            len(self._last_skipped),
            top,
            dict(self._cooldown_timers),
        )
        return candidates

    def apply_cooldown(self, action_type: ActionType) -> None:
        """Start cooldown timer after an action is executed."""
        if action_type in self._TACKLE_TYPES:
            self._cooldown_timers["tackle"] = self._tackle_cooldown_frames
        else:
            cd = self._cooldowns_cfg.get(action_type.value, 0)
            if cd > 0:
                self._cooldown_timers[action_type.value] = cd
        if action_type in self._PASS_TYPES_FOR_SHOOT_LOCK:
            self._post_pass_shoot_suppress_remaining = max(
                self._post_pass_shoot_suppress_remaining,
                self._post_pass_shoot_suppress_frames,
            )

    def _tick_cooldowns(self) -> None:
        expired: list[str] = []
        for name, remaining in list(self._cooldown_timers.items()):
            self._cooldown_timers[name] = remaining - 1
            if self._cooldown_timers[name] <= 0:
                expired.append(name)
        for name in expired:
            del self._cooldown_timers[name]
        if self._post_pass_shoot_suppress_remaining > 0:
            self._post_pass_shoot_suppress_remaining -= 1

    def _tackle_cooldown_remaining(self) -> int:
        return int(self._cooldown_timers.get("tackle", 0))

    def _is_on_cooldown(self, action_type: ActionType) -> bool:
        if action_type in self._TACKLE_TYPES:
            return self._tackle_cooldown_remaining() > 0
        return self._cooldown_timers.get(action_type.value, 0) > 0

    def _get_duration(self, action_type: ActionType) -> int:
        if action_type == ActionType.SLIDING_TACKLE:
            return self._durations_cfg.get(
                "sliding_tackle", self._durations_cfg.get("stand_tackle", 10)
            )
        if action_type == ActionType.DRIBBLE_TO_GOAL:
            return self._durations_cfg.get("dribble_to_goal", 8)
        return self._durations_cfg.get(action_type.value, 4)

    @staticmethod
    def _chase_ball_target(state: MatchState) -> Vec2 | None:
        """When hunting the ball, aim at predicted path, not only current position."""
        if state.ball is None or state.controlled_player is None:
            return None
        pred = state.predicted_ball_pos
        if pred.length() >= 1.0:
            return pred
        return state.ball.position

    @staticmethod
    def _goalward_direction(state: MatchState) -> Vec2:
        """Screen-space push toward the ball if visible, else toward opponent goal."""
        ad = state.attack_direction_sign
        gp = state.goal_point
        if gp.x == 0.0 and gp.y == 0.0:
            gp = state.screen_center + Vec2(300.0 * ad, 0.0)
        sc = state.screen_center
        if sc.x == 0.0 and sc.y == 0.0:
            sc = Vec2(640.0, 360.0)

        origin = state.controlled_player.position if state.controlled_player is not None else sc

        if state.ball is not None:
            to_ball = state.ball.position - origin
            if to_ball.length() >= 1e-3:
                return to_ball.normalized()

        to_goal = gp - origin
        if to_goal.length() < 1e-3:
            return Vec2(ad, 0.0)
        return to_goal.normalized()

    def _vec_to_stick(self, v: Vec2) -> tuple[float, float]:
        """Map screen-space vector to left stick [-1, 1] (offline heuristic).

        Short vectors (e.g. normalized intent ~length 1) map to full deflection
        so MOVE does not collapse to near-zero stick output.
        """
        mag = v.length()
        if mag < 1e-6:
            return 0.0, 0.0
        nx, ny = v.x / mag, v.y / mag
        if mag < 1.5:
            return max(-1.0, min(1.0, nx)), max(-1.0, min(1.0, ny))
        scale = min(1.0, mag / 400.0)
        return max(-1.0, min(1.0, nx * scale)), max(-1.0, min(1.0, ny * scale))

    def _rs_switch_direction(self, state: MatchState) -> Vec2 | None:
        """Pro switch: teammate closest to the ball; flick from controlled player toward them."""
        if state.controlled_player is None or state.ball is None:
            return None
        ctrl = state.controlled_player
        bp = state.ball.position
        others = [t for t in state.teammates if t.track_id != ctrl.track_id]
        if state.raw_track_count > 25 and len(others) > 11:
            others = sorted(others, key=lambda t: t.position.distance_to(bp))[:11]
        best_tm_pos: Vec2 | None = None
        best_d = float("inf")
        for tm in others:
            d = tm.position.distance_to(bp)
            if d < best_d:
                best_d = d
                best_tm_pos = tm.position
        if best_tm_pos is None:
            return None
        delta = best_tm_pos - ctrl.position
        if delta.length() < 1e-3:
            return None
        return delta.normalized()

    def _score_action(self, action_type: ActionType, state: MatchState) -> ActionSpec | None:
        """Compute composite score for a single action type."""
        match action_type:
            case ActionType.DRIBBLE_TO_GOAL:
                res = self._score_dribble_to_goal(state)
            case ActionType.MOVE | ActionType.R1_DRIBBLE:
                res = self._score_move(state, action_type)
            case ActionType.SPRINT | ActionType.EXPLOSIVE_BURST:
                res = self._score_sprint(state, action_type)
            case ActionType.PASS | ActionType.PRECISION_PASS:
                res = self._score_pass(state, action_type)
            case ActionType.THROUGH_BALL:
                res = self._score_through_ball(state)
            case ActionType.SHOOT | ActionType.LOW_DRIVEN_SHOT | ActionType.FINESSE_SHOT | ActionType.TRIVELA_SHOT:
                res = self._score_shoot(state, action_type)
            case ActionType.JOCKEY | ActionType.SPEED_JOCKEY:
                s, r, e, m = self._score_jockey(state)
                res = (s, r, e, None, None, m)
            case ActionType.STAND_TACKLE | ActionType.SLIDING_TACKLE:
                s, r, e, m = self._score_tackle(state)
                res = (s, r, e, None, None, m)
            case ActionType.CLEAR:
                s, r, e, m = self._score_clear(state)
                res = (s, r, e, None, None, m)
            case ActionType.CONTAIN:
                s, r, e, m = self._score_contain(state)
                res = (s, r, e, None, None, m)
            case ActionType.SWITCH_PLAYER:
                s, r, e, m = self._score_switch(state, pro_rs=False)
                res = (s, r, e, None, None, m)
            case ActionType.RS_SWITCH:
                s, r, e, m = self._score_switch(state, pro_rs=True)
                rs_dir = self._rs_switch_direction(state)
                if rs_dir is None:
                    return None
                res = (s, r, e, rs_dir, None, m)
            case ActionType.L2_SHIELD:
                s, r, e, m = self._score_shield(state)
                res = (s, r, e, None, None, m)
            case ActionType.CANCEL | ActionType.SUPER_CANCEL:
                res = (0.05, 0.0, 0.05, None, None, {})
            case ActionType.GK_NUDGE:
                res = (0.02, 0.0, 0.02, None, None, {})
            case ActionType.GK_RUSH:
                s, r, e, m = self._score_gk_rush(state)
                res = (s, r, e, None, None, m)
            case _:
                res = (0.0, 0.0, 0.0, None, None, {})

        score, risk, ev, raw_direction, raw_target, metadata = res

        direction_tuple: tuple[float, float] | None = None
        if action_type == ActionType.RS_SWITCH and isinstance(raw_direction, Vec2):
            if raw_direction.length() >= 1e-6:
                n = raw_direction.normalized()
                ad = state.attack_direction_sign
                sm = self._rs_flick_mag
                direction_tuple = (
                    max(-1.0, min(1.0, n.x * ad * sm)),
                    max(-1.0, min(1.0, n.y * sm)),
                )
        elif raw_direction is not None:
            if isinstance(raw_direction, Vec2):
                if raw_direction.length() >= 1e-6:
                    direction_tuple = self._vec_to_stick(raw_direction)
            elif hasattr(raw_direction, "as_tuple"):
                d_tuple = raw_direction.as_tuple()
                if isinstance(d_tuple, tuple) and len(d_tuple) >= 2:
                    vx, vy = float(d_tuple[0]), float(d_tuple[1])
                    v = Vec2(vx, vy)
                    if v.length() >= 1e-6:
                        direction_tuple = self._vec_to_stick(v.normalized())
            elif isinstance(raw_direction, tuple) and len(raw_direction) >= 2:
                vx, vy = float(raw_direction[0]), float(raw_direction[1])
                v = Vec2(vx, vy)
                if v.length() >= 1e-6:
                    direction_tuple = self._vec_to_stick(v.normalized())

        if direction_tuple is not None and action_type != ActionType.RS_SWITCH:
            if action_type in (
                ActionType.MOVE,
                ActionType.R1_DRIBBLE,
                ActionType.SPRINT,
                ActionType.EXPLOSIVE_BURST,
                ActionType.DRIBBLE_TO_GOAL,
                ActionType.JOCKEY,
                ActionType.SPEED_JOCKEY,
            ):
                dx, dy = direction_tuple
                ad = state.attack_direction_sign
                direction_tuple = (
                    max(-1.0, min(1.0, dx * ad)),
                    dy,
                )

        target_tuple: tuple[float, float] | None = None
        if raw_target is not None:
            if hasattr(raw_target, "as_tuple"):
                t = raw_target.as_tuple()
                if isinstance(t, tuple) and len(t) >= 2:
                    target_tuple = (float(t[0]), float(t[1]))
            elif isinstance(raw_target, tuple) and len(raw_target) >= 2:
                target_tuple = (float(raw_target[0]), float(raw_target[1]))

        return ActionSpec(
            action_type=action_type,
            score=float(score),
            risk=float(risk),
            expected_value=float(ev),
            direction=direction_tuple,
            target_position=target_tuple,
            duration_frames=self._get_duration(action_type),
            interruptible=action_type
            not in (
                ActionType.SHOOT,
                ActionType.LOW_DRIVEN_SHOT,
                ActionType.STAND_TACKLE,
                ActionType.SLIDING_TACKLE,
            ),
            metadata=metadata,
        )

    def _score_move(self, state: MatchState, action_type: ActionType) -> ScoreParts:
        if action_type == ActionType.MOVE:
            if state.controlled_player is None:
                return 0.0, 0.0, 0.0, None, None, {}
            direction = self._goalward_direction(state)
            return 1.0, 0.0, 1.0, direction, None, {"forced_move_score": 1.0}

        if state.controlled_player is None:
            return 0.0, 0.0, 0.0, None, None, {}

        score = 0.55
        metadata: dict[str, float] = {}
        direction: Vec2

        if state.possession != PossessionState.OWN_TEAM and state.ball is not None:
            tgt = self._chase_ball_target(state)
            if tgt is None:
                tgt = state.ball.position
            diff = tgt - state.controlled_player.position
            direction = diff.normalized()
            if diff.length() < 1e-3 or direction.length() < 0.01:
                direction = Vec2(1.0, 0.0)
            if state.distance_to_ball < 30:
                direction = Vec2(1.0, 0.0)
            score = 0.72
            metadata["chase_ball"] = 1.0
        else:
            score += min(0.15, state.dribble_space / 200.0 * 0.15)
            direction = state.controlled_player.velocity
            if direction.length() < 0.1:
                direction = Vec2(1.0, 0.0)

        if action_type == ActionType.R1_DRIBBLE:
            if state.nearest_opponent_dist < 100:
                score += 0.12
                metadata["close_control_bonus"] = 0.12

        risk = min(1.0, state.local_pressure * 0.35)
        ev = score - risk
        metadata["pressure_risk"] = risk
        return score, risk, ev, direction, None, metadata

    def _score_sprint(self, state: MatchState, action_type: ActionType) -> ScoreParts:
        if state.controlled_player is None:
            return 0.0, 0.0, 0.0, None, None, {}

        score = 0.18
        metadata: dict[str, float] = {}

        space_bonus = min(0.35, state.dribble_space / 200.0 * 0.35)
        score += space_bonus
        metadata["space_bonus"] = space_bonus

        if state.nearest_opponent_dist > 150:
            score += 0.18
            metadata["clear_ahead"] = 0.18

        if action_type == ActionType.EXPLOSIVE_BURST and state.counterattack_opportunity > 0.5:
            score += 0.12
            metadata["burst_counter_bonus"] = 0.12

        risk = min(1.0, state.local_pressure * 0.45)
        if state.dribble_space < 50:
            risk += 0.25
            risk = min(1.0, risk)

        direction = state.controlled_player.velocity
        if direction.length() < 0.1:
            direction = Vec2(1.0, 0.0)

        if state.possession != PossessionState.OWN_TEAM and state.ball is not None:
            tgt = self._chase_ball_target(state)
            if tgt is not None:
                diff = tgt - state.controlled_player.position
                if diff.length() >= 1e-3:
                    direction = diff.normalized()
                    score += 0.22
                    metadata["sprint_intercept_ball_path"] = 0.22

        ev = score - risk
        return score, risk, ev, direction, None, metadata

    def _score_dribble_to_goal(self, state: MatchState) -> ScoreParts:
        """Always-on attack vector: max score; ball optional (blind goalward if no ball)."""
        direction = self._goalward_direction(state)
        return 1.0, 0.0, 1.0, direction, None, {"forced_dribble_score": 1.0}

    def _score_pass(self, state: MatchState, action_type: ActionType) -> ScoreParts:
        lane = state.best_passing_lane
        if lane is None:
            return 0.1, 0.25, -0.15, None, None, {}

        score = 0.45
        risk = lane.risk_score
        metadata: dict[str, float] = {}

        if lane.risk_score > 0.88:
            return 0.05, 0.62, -0.52, None, None, {"unsafe_pass_lane": 1.0}
        if lane.receiver_high_pressure:
            risk += 0.2
            metadata["receiver_high_pressure"] = 0.2
        if lane.boundary_risk > 0.2:
            risk += lane.boundary_risk * 0.5
            metadata["boundary_risk"] = lane.boundary_risk
        if lane.risk_score < 0.32:
            score += 0.14 * self._w_safety
            metadata["safe_receiver_bonus"] = 0.14 * self._w_safety

        if lane.is_clear:
            score += 0.28 * self._w_safety
            metadata["clear_lane"] = 0.28 * self._w_safety

        dist_factor = max(0.0, 1.0 - lane.distance / 600.0)
        score += dist_factor * 0.18 * self._w_support
        metadata["distance_factor"] = dist_factor * 0.18

        if state.local_pressure > 0.5:
            score += 0.18
            metadata["pressure_escape"] = 0.18

        if action_type == ActionType.PRECISION_PASS:
            score += 0.08
            metadata["precision_bonus"] = 0.08

        ev = score - risk * abs(self._w_turnover)
        target = lane.target.position
        return score, risk, ev, None, target, metadata

    def _score_through_ball(self, state: MatchState) -> ScoreParts:
        lane = state.best_passing_lane
        if lane is None or not lane.is_clear:
            return 0.08, 0.45, -0.37, None, None, {}
        if lane.risk_score > 0.5:
            return 0.07, 0.5, -0.42, None, None, {"risky_through": 1.0}

        score = 0.48
        risk = 0.38
        metadata: dict[str, float] = {}

        if state.controlled_zone == Zone.ATTACKING_THIRD:
            score += 0.18 * self._w_goal
            metadata["final_third_bonus"] = 0.18 * self._w_goal

        if state.counterattack_opportunity > 0.5:
            score += 0.18
            metadata["counter_bonus"] = 0.18

        ev = score - risk * abs(self._w_turnover)
        target = lane.target.position
        return score, risk, ev, None, target, metadata

    def _score_shoot(self, state: MatchState, action_type: ActionType) -> ScoreParts:
        lane = state.best_shot_lane

        if lane is None or lane.distance > 280:
            return 0.0, 0.85, -0.85, None, None, {}

        # ``goal_threat`` in YAML is often <1 for multi-objective tuning; scale so
        # high-quality lanes still outrank transition bursts in obvious finishing situations.
        score = lane.quality * (1.0 + self._w_goal)
        risk = 0.28
        metadata: dict[str, float] = {}

        if not lane.is_clear:
            risk += 0.28
            metadata["blocked_lane"] = 0.28

        dist_penalty = min(0.25, lane.distance / 800.0 * 0.25)
        score -= dist_penalty
        metadata["dist_penalty"] = dist_penalty

        if action_type == ActionType.LOW_DRIVEN_SHOT and lane.distance < 250:
            score += 0.22
            metadata["low_driven_box_bonus"] = 0.22
        elif action_type == ActionType.FINESSE_SHOT:
            if 25 <= lane.angle_deg <= 55:
                score += 0.18
                metadata["finesse_angle_bonus"] = 0.18
        elif action_type == ActionType.TRIVELA_SHOT:
            if 30 <= lane.angle_deg <= 70:
                score += 0.12
                metadata["trivela_angle_bonus"] = 0.12

        ev = score * 0.7 - risk * 0.3
        target = lane.target_point
        return score, min(1.0, risk), ev, None, target, metadata

    @staticmethod
    def _score_jockey(state: MatchState) -> tuple[float, float, float, dict[str, float]]:
        score = 0.0
        metadata: dict[str, float] = {}

        if state.possession in (PossessionState.OPPONENT, PossessionState.CONTESTED):
            score += 0.38
            metadata["defensive_posture"] = 0.38
        if state.defensive_danger > 0.3:
            score += state.defensive_danger * 0.28
            metadata["danger_response"] = state.defensive_danger * 0.28

        risk = 0.12
        ev = score - risk
        return score, risk, ev, metadata

    @staticmethod
    def _score_tackle(state: MatchState) -> tuple[float, float, float, dict[str, float]]:
        score = 0.0
        risk = 0.35
        metadata: dict[str, float] = {}

        if state.distance_to_ball < 100:
            score += 0.65
            metadata["ball_proximity"] = 0.65
        if state.defensive_danger > 0.6:
            score += 0.18
            metadata["emergency_tackle"] = 0.18

        if math.isfinite(state.distance_to_ball):
            risk += (1.0 - min(1.0, state.distance_to_ball / 100.0)) * 0.18
        risk = min(1.0, risk)
        ev = score - risk
        return score, risk, ev, metadata

    @staticmethod
    def _score_clear(state: MatchState) -> tuple[float, float, float, dict[str, float]]:
        score = 0.0
        metadata: dict[str, float] = {}

        if state.defensive_danger > 0.6:
            score += 0.55
            metadata["high_danger_clear"] = 0.55
        if state.ball_zone == Zone.DEFENSIVE_THIRD:
            score += 0.18
            metadata["defensive_zone"] = 0.18

        risk = 0.22
        ev = score - risk
        return score, risk, ev, metadata

    @staticmethod
    def _score_contain(state: MatchState) -> tuple[float, float, float, dict[str, float]]:
        score = 0.0
        metadata: dict[str, float] = {}

        if state.possession == PossessionState.OPPONENT:
            score += 0.32
            metadata["contain_defense"] = 0.32
        score += state.defensive_danger * 0.18
        metadata["danger_factor"] = state.defensive_danger * 0.18

        risk = 0.08
        ev = score - risk
        return score, risk, ev, metadata

    @staticmethod
    def _score_switch(
        state: MatchState, *, pro_rs: bool = False
    ) -> tuple[float, float, float, dict[str, float]]:
        score = 0.12
        metadata: dict[str, float] = {}

        if state.ball is None or state.controlled_player is None:
            return score, 0.08, score - 0.08, metadata

        ctrl_dist = state.distance_to_ball
        for tm in state.teammates:
            if tm.track_id == state.controlled_player.track_id:
                continue
            tm_dist = tm.position.distance_to(state.ball.position)
            if tm_dist < ctrl_dist * 0.55:
                score += 0.55
                metadata["switch_closer_teammate"] = 0.55
                break

        if state.possession in (PossessionState.LOOSE, PossessionState.CONTESTED):
            score += 0.08
            metadata["loose_ball_switch_bias"] = 0.08

        if pro_rs:
            if state.possession == PossessionState.OPPONENT:
                score += 0.32
                metadata["rs_switch_defense"] = 0.32
            elif state.possession in (PossessionState.LOOSE, PossessionState.CONTESTED):
                score += 0.12
                metadata["rs_switch_contest"] = 0.12

        risk = 0.08
        ev = score - risk
        return score, risk, ev, metadata

    @staticmethod
    def _score_shield(state: MatchState) -> tuple[float, float, float, dict[str, float]]:
        score = 0.0
        metadata: dict[str, float] = {}

        if state.nearest_opponent_dist < 60:
            score += 0.38
            metadata["contact_range"] = 0.38
        if state.possession == PossessionState.OWN_TEAM:
            score += 0.12
            metadata["protect_ball"] = 0.12

        risk = 0.06
        ev = score - risk
        return score, risk, ev, metadata

    def _score_gk_rush(self, state: MatchState) -> tuple[float, float, float, dict[str, float]]:
        score = 0.0
        metadata: dict[str, float] = {}
        if state.possession == PossessionState.OPPONENT and state.nearest_opponent_dist_to_gk < 400:
            score += 0.55 * self._w_def_shape
            metadata["gk_rush_opportunity"] = 0.55
        risk = 0.2
        return score, risk, score - risk, metadata

    def reset_cooldowns(self) -> None:
        self._cooldown_timers.clear()
        self._post_pass_shoot_suppress_remaining = 0
