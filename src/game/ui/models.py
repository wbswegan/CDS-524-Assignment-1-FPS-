from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UIData:
    score: int
    wave: int
    episode_label: str | None
    hp: int
    kills: int
    ammo_in_mag: int
    magazine_size: int
    is_reloading: bool
    reload_time_left: float
    feedback_messages: list[str] = field(default_factory=list)
    intermission_message: str | None = None
    fps: int = 0


@dataclass
class AgentDebugData:
    mode_label: str
    epsilon: float
    episode_id: int
    episode_return: float
    avg20: float
    reward_total: float
    penalty_total: float
    shoot_ready: bool
    has_los: bool
    is_reloading: bool
    aim_error_degrees: float
    aim_stable_time: float
    goal_type: str
    target_id: str
    target_distance_text: str
    retarget_in_seconds: float
    emergency_switch: bool
    pack_score: float
    enemy_score: float
    pack_distance_text: str
    enemy_distance_text: str
    intent_label: str
    enemy_repulse: bool
    near_wall: bool
    should_move: bool
    bumped: bool
    progress_tiles: float
    stuck_time: float
    recovering: bool
    recovery_side_label: str
    target_type: str
    reason: str
    no_reason_timer: float
    no_reason_cooldown: float
