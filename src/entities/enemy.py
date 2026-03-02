from __future__ import annotations

from dataclasses import dataclass, field

from src.ai.agent_control import AimController, ControlArbiter, TargetLock
from src.game.states import INTENT_CHASE, SCRIPT_STATE_PATROL
from src.settings import AGENT_ACTION_INTERVAL_SECONDS, HEALTH_PACK_HEAL, SCRIPT_SIGHT_RANGE_TILES, TILE_SIZE


@dataclass
class Enemy:
    kind: str
    x: float
    y: float
    hp: int
    max_hp: int
    speed: float
    score: int
    radius: int
    size: float
    color: tuple[int, int, int]
    aggro_range: float
    damage: int = 0
    shot_cooldown_max: float = 0.0
    shot_cooldown: float = 0.0
    last_seen_player_cell: tuple[int, int] | None = None
    path: list[tuple[int, int]] = field(default_factory=list)
    path_target_cell: tuple[int, int] | None = None
    repath_timer: float = 0.0
    under_fire_timer: float = 0.0
    angle: float = 0.0
    entity_id: int = 0
    attack_style: str = "ranged"
    melee_range: float = TILE_SIZE * 1.0
    engage_min_range: float = TILE_SIZE * 3.5
    engage_max_range: float = TILE_SIZE * 5.5
    state: str = SCRIPT_STATE_PATROL
    home_cell: tuple[int, int] | None = None
    patrol_points: list[tuple[int, int]] = field(default_factory=list)
    patrol_index: int = 0
    search_target_cell: tuple[int, int] | None = None
    alert_time_left: float = 0.0
    no_los_time: float = 0.0
    strafe_time_left: float = 0.0
    strafe_sign: int = 1
    current_nav_target_cell: tuple[int, int] | None = None
    has_path: bool = False
    next_waypoint: tuple[int, int] | None = None
    current_move_vector: tuple[float, float] = (0.0, 0.0)
    current_target_distance: float = 0.0
    current_has_los: bool = False
    hear_range_tiles: float = 0.0
    sight_range_tiles: float = SCRIPT_SIGHT_RANGE_TILES
    hit_flash_timer: float = 0.0


@dataclass
class RLEnemy(Enemy):
    decision_interval_seconds: float = AGENT_ACTION_INTERVAL_SECONDS
    decision_time_left: float = 0.0
    action_time_left: float = 0.0
    action_locked: bool = False
    emergency_unlock: bool = False
    current_state: int | None = None
    current_obs: tuple[int, int, int, int, int] | None = None
    current_action: int | None = None
    pending_reward: float = 0.0
    last_reward_delta: float = 0.0
    last_has_los: bool = False
    target_lock: TargetLock = field(default_factory=TargetLock)
    arbiter: ControlArbiter = field(default_factory=ControlArbiter)
    aim_controller: AimController = field(default_factory=AimController)
    current_yaw_source: str = "nav"
    current_yaw_rate: float = 0.0
    last_damage_time: float = float("inf")
    current_effective_intent: int = INTENT_CHASE
    current_nav_target_cell: tuple[int, int] | None = None
    has_path: bool = False
    next_waypoint: tuple[int, int] | None = None
    current_move_vector: tuple[float, float] = (0.0, 0.0)
    current_target_distance: float = 0.0
    current_has_los: bool = False


@dataclass
class HealthPackTarget(Enemy):
    heal_amount: int = HEALTH_PACK_HEAL
