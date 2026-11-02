"""Match state representation.

MatchState is the central data structure that the decision engine consumes.
It is rebuilt every frame by the StateEstimator from tracked objects.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from vision.geometry import Vec2
from vision.types import TrackedObject


class Zone(enum.Enum):
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"
    BOX = "box"


class PossessionState(enum.Enum):
    OWN_TEAM = "own_team"
    OPPONENT = "opponent"
    LOOSE = "loose"
    CONTESTED = "contested"


@dataclass(slots=True)
class PassingLane:
    """A corridor between the controlled player and a teammate."""

    target: TrackedObject
    direction: Vec2
    distance: float
    is_clear: bool
    obstruction_count: int
    risk_score: float  # 0 = safe, 1 = very risky
    # True if an opponent is within pass_receiver_min_opponent_distance of the receiver
    receiver_high_pressure: bool = False
    # Risk from passing toward touchline / goal line when target is near boundary
    boundary_risk: float = 0.0


@dataclass(slots=True)
class ShotLane:
    """Line from controlled player to a goalmouth target point."""

    target_point: Vec2  # point on the goal line
    direction: Vec2
    distance: float
    is_clear: bool
    angle_deg: float  # angle relative to goal center
    quality: float  # 0 = bad, 1 = excellent


@dataclass(slots=True)
class MatchState:
    """Complete snapshot of the match situation for one frame.

    Built by StateEstimator. Consumed by DecisionEngine.
    """

    frame_id: int
    timestamp_ms: float

    # --- Ball ---
    ball: TrackedObject | None = None
    ball_velocity: Vec2 = field(default_factory=Vec2.zero)
    ball_speed: float = 0.0
    # True when a real detector hit (not linear ghost extrapolation)
    ball_tracked: bool = True
    # Goal-side point for dribble / attack direction (screen pixels)
    goal_point: Vec2 = field(default_factory=Vec2.zero)
    # Center of playable (cropped) view — for ball-chase when no teammate tracks
    screen_center: Vec2 = field(default_factory=Vec2.zero)

    # --- Controlled player ---
    controlled_player: TrackedObject | None = None
    is_charging_shot: bool = False

    # --- Players ---
    teammates: list[TrackedObject] = field(default_factory=list)
    opponents: list[TrackedObject] = field(default_factory=list)
    goalkeepers_team: list[TrackedObject] = field(default_factory=list)
    goalkeepers_opponent: list[TrackedObject] = field(default_factory=list)
    referees: list[TrackedObject] = field(default_factory=list)

    # --- Spatial analysis ---
    ball_zone: Zone = Zone.MIDDLE_THIRD
    controlled_zone: Zone = Zone.MIDDLE_THIRD
    # True when the controlled player is deep in the attacking end (toward opponent goal on screen).
    in_opponent_penalty_area: bool = False
    # +1 = attack toward +screen X (goal on the right); -1 if your view attacks left.
    attack_direction_sign: float = 1.0
    possession: PossessionState = PossessionState.LOOSE
    distance_to_ball: float = float("inf")
    # Pixel distance from controlled player to nearest opponent to the ball /  hypot(eff_w, eff_h)
    dist_ctrl_to_ball_carrier_normalized: float = 1.0
    raw_track_count: int = 0
    # Max normalized distance (vs field diagonal) for tackle preconditions; set from config.
    tackle_carrier_distance_threshold: float = 0.06

    # --- Game flow (restarts / set pieces) ---
    is_paused: bool = False  # Long ball loss — freeze movement
    wait_for_input: bool = False  # Kick-off / corner — do not sprint away
    ball_absent_frames: int = 0  # consecutive frames without a real ball detection
    ball_high_confidence_streak: int = 0  # real ball with conf > threshold for resume
    # Live pipeline: F1 active + first N frames after activation — relax set-piece stick freeze / dribble gate
    bot_active: bool = False
    set_piece_wait_suppressed: bool = False

    # --- Pressure & space ---
    local_pressure: float = 0.0  # 0 = no pressure, 1 = surrounded
    nearest_opponent_dist: float = float("inf")
    nearest_teammate_dist: float = float("inf")
    dribble_space: float = 0.0  # pixels of free space ahead
    open_space_around: float = 0.0  # average distance to nearby opponents

    # --- Lanes ---
    passing_lanes: list[PassingLane] = field(default_factory=list)
    shot_lanes: list[ShotLane] = field(default_factory=list)
    best_passing_lane: PassingLane | None = None
    best_shot_lane: ShotLane | None = None

    # --- Tactical indicators ---
    defensive_danger: float = 0.0  # 0 = safe, 1 = extreme danger
    counterattack_opportunity: float = 0.0
    shot_opportunity: float = 0.0

    # --- Predictions ---
    predicted_ball_pos: Vec2 = field(default_factory=Vec2.zero)
    predicted_controlled_pos: Vec2 = field(default_factory=Vec2.zero)

    # --- History ---
    previous: MatchState | None = field(default=None, repr=False)

    def get_all_opponents_positions(self) -> list[Vec2]:
        return [o.position for o in self.opponents]

    def get_all_teammates_positions(self) -> list[Vec2]:
        return [t.position for t in self.teammates]

    @property
    def nearest_opponent_dist_to_gk(self) -> float:
        """Расстояние от ближайшего оппонента до нашего вратаря."""
        if not self.goalkeepers_team or not self.opponents:
            return 999.0
        gk_pos = self.goalkeepers_team[0].position
        return min(p.position.distance_to(gk_pos) for p in self.opponents)