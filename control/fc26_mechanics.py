"""FC 26 specific input mechanics.

Implements the post-patch 1.5.1 meta mechanics.
"""

from __future__ import annotations


from config.settings import Config
from state.match_state import MatchState, PossessionState, Zone
from vision.geometry import Vec2
from vision.types import TrackedObject


class FC26Mechanics:
    """Stateless helper that computes FC26-specific action parameters."""

    def __init__(self, cfg: Config) -> None:
        # Access the 'fc26' section safely
        self._cfg = getattr(cfg, 'fc26', cfg)
        
        # Initialize variables with defaults if _cfg is None or missing attributes
        self._precision_pass_enabled = getattr(self._cfg, 'precision_pass_enabled', True)
        self._low_driven_enabled = getattr(self._cfg, 'low_driven_shot_enabled', True)
        self._r1_dribble_enabled = getattr(self._cfg, 'r1_dribble_enabled', True)
        self._explosive_burst_enabled = getattr(self._cfg, 'explosive_burst_enabled', True)
        self._l2_enforcer_shielding = getattr(self._cfg, 'l2_enforcer_shielding', True)
        self._enforcer_radius = getattr(self._cfg, 'enforcer_contact_radius', 50.0)
        self._speed_jockey_enabled = getattr(self._cfg, 'speed_jockey_enabled', True)
        self._tackle_conf_min = getattr(self._cfg, 'stand_tackle_confidence_min', 0.85)
        self._rs_min_improve = getattr(self._cfg, 'rs_switch_min_improvement', 0.15)
        self._trivela_min = getattr(self._cfg, 'trivela_angle_min', 30.0)
        self._trivela_max = getattr(self._cfg, 'trivela_angle_max', 70.0)
        self._finesse_min = getattr(self._cfg, 'finesse_angle_min', 25.0)
        self._finesse_max = getattr(self._cfg, 'finesse_angle_max', 55.0)
        self._super_cancel_on_stumble = getattr(self._cfg, 'super_cancel_on_stumble', True)
        self._gk_manual_nudge_enabled = getattr(self._cfg, 'gk_manual_nudge_enabled', True)

    # --- Precision Passing ---

    def compute_precision_pass_vector(self, state: MatchState) -> Vec2 | None:
        """Calculate the manual aiming vector for a precision pass (RB + Y)."""
        if not self._precision_pass_enabled:
            return None
            
        lane = state.best_passing_lane
        if lane is None or state.controlled_player is None:
            return None
            
        # Lead the target based on their velocity
        target_pos = lane.target.predict_position(frames_ahead=4)
        direction = (target_pos - state.controlled_player.position).normalized()
        
        print(f"[FC26] Precision Pass Vector calculated: {direction.x:.3f}, {direction.y:.3f}")
        return direction



    # --- Explosive Burst ---

    def should_explosive_burst(self, state: MatchState) -> bool:
        """True if an R1-tap-then-R2 explosive burst is appropriate."""
        if not self._explosive_burst_enabled:
            return False
        if state.controlled_player is None:
            return False
        return (
                state.dribble_space > 120
                and state.nearest_opponent_dist > 80
                and state.counterattack_opportunity > 0.4
        )

    @staticmethod
    def should_controlled_sprint(state: MatchState) -> bool:
        """True if we should use L1 Controlled Sprint for tight maneuvering."""
        if state.controlled_player is None or state.possession != PossessionState.OWN_TEAM:
            return False

        # Если мы в штрафной или рядом, и вокруг враги — включаем точный дриблинг
        return (
                state.ball_zone in [Zone.ATTACKING_THIRD, Zone.BOX]
                and state.nearest_opponent_dist < 100
                and state.dribble_space < 150
        )

    @staticmethod
    def should_gk_rush(state: MatchState) -> bool:
        """True if the goalkeeper should rush out to close the angle."""
        if state.ball is None or state.possession == PossessionState.OWN_TEAM:
            return False

        # Если мяч у врага, он в нашей трети и сближается с воротами
        return (
                state.ball_zone == Zone.DEFENSIVE_THIRD
                and state.nearest_opponent_dist_to_gk < 150  # Если враг близко к вратарю
                and state.counterattack_opportunity > 0.5  # Высокая опасность
        )

    @staticmethod
    def should_power_shot_cancel(state: MatchState) -> bool:
        """Решаем, нужно ли отменить Power Shot."""
        lane = state.best_shot_lane
        # Сначала проверяем, что lane вообще существует (не None)
        if lane is None or state.controlled_player is None:
            return False

        return (
                state.is_charging_shot
                and not lane.is_clear  # Теперь это безопасно
                and state.nearest_opponent_dist < 60
        )

    # --- L2 Enforcer Shielding ---

    def should_l2_shield(self, state: MatchState) -> bool:
        """True if we should hold L2 for enforcer shielding."""
        if not self._l2_enforcer_shielding:
            return False
        if state.controlled_player is None:
            return False
        return (
                state.nearest_opponent_dist < self._enforcer_radius
                and state.possession == PossessionState.OWN_TEAM
        )

    # --- Shot Type Selection ---

    def select_shot_type(self, state: MatchState) -> str:
        """Decide which shot type to use based on position and angle."""
        if state.best_shot_lane is None or state.controlled_player is None:
            return "normal"

        dist = state.best_shot_lane.distance
        angle = state.best_shot_lane.angle_deg

        # Inside box → low driven is meta
        if dist < 250 and self._low_driven_enabled:
            return "low_driven"

        # Finesse angle sweet spot
        if self._finesse_min <= angle <= self._finesse_max:
            return "finesse"

        # trivela angle range
        if self._trivela_min <= angle <= self._trivela_max:
            return "trivela"

        return "normal"

    # --- RS Pro-Switching ---

    def compute_rs_switch_direction(
            self, state: MatchState
    ) -> tuple[Vec2 | None, TrackedObject | None]:
        """Calculate RS flick direction for pro player switching."""
        if state.controlled_player is None or state.ball is None:
            return None, None

        ball_pos = state.ball.position
        ball_vel = state.ball_velocity
        # Predict where the ball will be in ~10 frames
        intercept_point = ball_pos + ball_vel * 10

        current_score = self._interception_score(
            state.controlled_player, intercept_point
        )

        best_target: TrackedObject | None = None
        best_score: float = current_score

        for tm in state.teammates:
            if tm.track_id == state.controlled_player.track_id:
                continue
            score = self._interception_score(tm, intercept_point)
            if score > best_score + self._rs_min_improve:
                best_score = score
                best_target = tm

        if best_target is None:
            return None, None

        # Calculate RS flick direction
        direction = (best_target.position - state.controlled_player.position).normalized()
        return direction, best_target

    @staticmethod
    def _interception_score(player: TrackedObject, intercept_point: Vec2) -> float:
        """Score how well positioned a player is to intercept at a point."""
        dist = player.position.distance_to(intercept_point)
        if dist < 1:
            return 1.0
        # Closer is better (inverse distance, capped)
        dist_score = max(0.0, 1.0 - dist / 500.0)
        # Moving toward the intercept point is better
        to_intercept = (intercept_point - player.position).normalized()
        approach = player.velocity.normalized().dot(to_intercept)
        approach_score = max(0.0, approach) * 0.3
        return dist_score + approach_score

    # --- Stand Tackle Confidence Gate ---

    def can_tackle_confident(self, state: MatchState) -> bool:
        """Only allow stand tackle when ball proximity confidence > threshold."""
        if state.ball is None:
            return False
        return state.ball.confidence >= self._tackle_conf_min

    # --- Super Cancel ---

    def should_super_cancel(self, state: MatchState) -> bool:
        """Detect if super cancel (L2+R2) should trigger for stumble recovery."""
        if not self._super_cancel_on_stumble:
            return False
        if state.controlled_player is None or state.previous is None:
            return False
        prev = state.previous
        if prev.controlled_player is None:
            return False
        cur_speed = state.controlled_player.speed
        prev_speed = prev.controlled_player.speed
        if prev_speed > 3.0 and cur_speed < prev_speed * 0.4:
            if state.nearest_opponent_dist < 80:
                return True
        return False

    # --- GK Nudge ---

    def compute_gk_nudge(self, state: MatchState) -> Vec2 | None:
        """Calculate R3 nudge direction for manual GK positioning."""
        if not self._gk_manual_nudge_enabled:
            return None
        if not state.goalkeepers_team:
            return None

        gk = state.goalkeepers_team[0]
        if state.ball is None:
            return None
        if state.ball_speed < 5.0:
            return None
        if state.ball_zone != Zone.DEFENSIVE_THIRD:
            return None

        predicted = state.predicted_ball_pos
        nudge = (predicted - gk.position).normalized()
        return nudge
