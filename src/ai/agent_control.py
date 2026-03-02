from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.ai.geometry import normalize_vector
from src.game.states import INTENT_CHASE, TARGET_LOCK_MIN_SECONDS
from src.settings import (
    AGENT_ACTION_INTERVAL_SECONDS,
    AIM_DEADZONE_DEGREES,
    AIM_MAX_DEGREES_PER_SECOND,
    AIM_PROPORTIONAL_GAIN,
    INTENT_HOLD_SECONDS,
    MOVE_FILTER_DEADZONE,
    MOVE_FILTER_RESPONSE,
    PLAYER_AGENT_PATROL_REFRESH_SECONDS,
    PLAYER_AGENT_STRAFE_BURST_SECONDS,
    RETARGET_INTERVAL_SEC,
)

if TYPE_CHECKING:
    from src.game.game import Game
    from src.rl.player_agent import PlayerQLearningAgent


@dataclass
class TargetLock:
    target_key: str | int | None = None
    target_ref: object | None = None
    score: float = float("inf")
    lock_time_left: float = 0.0

    def tick(self, dt: float) -> None:
        self.lock_time_left = max(0.0, self.lock_time_left - dt)

    def clear(self) -> None:
        self.target_key = None
        self.target_ref = None
        self.score = float("inf")
        self.lock_time_left = 0.0

    def assign(self, target_key: str | int, target_ref: object, score: float) -> None:
        self.target_key = target_key
        self.target_ref = target_ref
        self.score = score
        self.lock_time_left = TARGET_LOCK_MIN_SECONDS


class AimController:
    def __init__(
        self,
        proportional_gain: float = AIM_PROPORTIONAL_GAIN,
        deadzone_degrees: float = AIM_DEADZONE_DEGREES,
        max_degrees_per_second: float = AIM_MAX_DEGREES_PER_SECOND,
    ) -> None:
        self.proportional_gain = proportional_gain
        self.deadzone_radians = math.radians(deadzone_degrees)
        self.max_rate_radians = math.radians(max_degrees_per_second)

    def step(self, current_angle: float, target_angle: float, dt: float) -> tuple[float, float, float]:
        from src.ai.geometry import wrap_to_pi

        angle_error = wrap_to_pi(target_angle - current_angle)
        if abs(angle_error) < self.deadzone_radians:
            return current_angle, 0.0, angle_error
        yaw_rate = max(
            -self.max_rate_radians,
            min(self.max_rate_radians, angle_error * self.proportional_gain),
        )
        return (current_angle + yaw_rate * dt) % math.tau, yaw_rate, angle_error


class ControlArbiter:
    def __init__(self) -> None:
        self.filtered_move_x = 0.0
        self.filtered_move_y = 0.0
        self.reset()

    def reset(self) -> None:
        self.desired_move_x = 0.0
        self.desired_move_y = 0.0
        self.move_x = self.filtered_move_x
        self.move_y = self.filtered_move_y
        self.yaw_target: float | None = None
        self.yaw_priority = -1
        self.yaw_source = "nav"

    def suggest_move_toward(
        self,
        actor_x: float,
        actor_y: float,
        target_x: float,
        target_y: float,
        speed: float,
        dt: float,
    ) -> None:
        dir_x, dir_y, distance = normalize_vector(target_x - actor_x, target_y - actor_y)
        if distance <= 0.0001:
            self.desired_move_x = 0.0
            self.desired_move_y = 0.0
            return
        step_distance = min(speed * dt, distance)
        self.desired_move_x = dir_x * step_distance
        self.desired_move_y = dir_y * step_distance

    def finalize_move(self, dt: float) -> tuple[float, float]:
        alpha = min(1.0, MOVE_FILTER_RESPONSE * dt)
        self.filtered_move_x += (self.desired_move_x - self.filtered_move_x) * alpha
        self.filtered_move_y += (self.desired_move_y - self.filtered_move_y) * alpha
        filtered_length = math.hypot(self.filtered_move_x, self.filtered_move_y)
        desired_length = math.hypot(self.desired_move_x, self.desired_move_y)
        if filtered_length < MOVE_FILTER_DEADZONE and desired_length < MOVE_FILTER_DEADZONE:
            self.filtered_move_x = 0.0
            self.filtered_move_y = 0.0
        self.move_x = self.filtered_move_x
        self.move_y = self.filtered_move_y
        return self.move_x, self.move_y

    def clear_filtered_move(self) -> None:
        self.filtered_move_x = 0.0
        self.filtered_move_y = 0.0
        self.move_x = 0.0
        self.move_y = 0.0
        self.desired_move_x = 0.0
        self.desired_move_y = 0.0

    def suggest_yaw(self, source: str, angle: float, priority: int) -> None:
        if priority > self.yaw_priority:
            self.yaw_priority = priority
            self.yaw_target = angle
            self.yaw_source = source


class HumanControl:
    name = "Human"

    def apply(self, game: "Game", dt: float) -> None:
        game._update_mouse_look()
        game.player_bumped = game._update_human_player(dt)


class AgentControl:
    name = "Agent"

    def __init__(self, learner: "PlayerQLearningAgent") -> None:
        self.learner = learner
        self.override_action: int | None = None
        self.last_action: int = INTENT_CHASE
        self.action_interval = AGENT_ACTION_INTERVAL_SECONDS
        self.decision_time_left = 0.0
        self.action_time_left = 0.0
        self.action_locked = False
        self.emergency_unlock = False
        self.path: list[tuple[int, int]] = []
        self.path_target_cell: tuple[int, int] | None = None
        self.repath_timer = 0.0
        self.target_lock = TargetLock()
        self.arbiter = ControlArbiter()
        self.aim_controller = AimController()
        self.current_yaw_source = "nav"
        self.current_yaw_rate = 0.0
        self.current_aim_error = math.pi
        self.current_target_angle = 0.0
        self.aim_stable_time = 0.0
        self.last_seen_target_cell: tuple[int, int] | None = None
        self.current_nav_target_cell: tuple[int, int] | None = None
        self.has_path = False
        self.next_waypoint: tuple[int, int] | None = None
        self.current_move_vector = (0.0, 0.0)
        self.current_effective_intent = INTENT_CHASE
        self.current_target_distance = 0.0
        self.current_has_los = False
        self.shoot_ready = False
        self.shoot_triggered_this_tick = False
        self.blocked_reason = "none"
        self.current_phase = "SEARCH"
        self.last_retarget_time = float("inf")
        self.last_damage_time = float("inf")
        self.last_pack_reward = 0.0
        self.prev_pack_dist_tiles: float | None = None
        self.patrol_target_cell: tuple[int, int] | None = None
        self.patrol_time_left = 0.0
        self.force_patrol_time_left = 0.0
        self.strafe_time_left = 0.0
        self.strafe_sign = 1
        self.no_progress_time = 0.0
        self.move_smooth_x = 0.0
        self.move_smooth_y = 0.0
        self.near_wall = False
        self.recover_time_left = 0.0
        self.recover_sign = 1
        self.replan_after_recover = False
        self.should_move = False
        self.current_progress_tiles = 0.0
        self.current_bumped = False
        self.current_reason = "no_target"
        self.no_reason_timer = 0.0
        self.no_reason_cooldown = 0.0
        self.retarget_timer = 0.0
        self.emergency_retarget_this_tick = False
        self.current_goal_type = "NONE"
        self.current_pack_score = 0.0
        self.current_enemy_score = 0.0
        self.current_pack_dist_tiles: float | None = None
        self.current_enemy_dist_tiles: float | None = None
        self.debug_intent_label = "CHASE"
        self.enemy_repulse_active = False

    def step(self, state: tuple[int, int, int, int, int, int, int, int]) -> int:
        if self.override_action is not None:
            return self.override_action
        return self.learner.select_action(state)

    def set_action(self, action_id: int | None) -> None:
        self.override_action = action_id

    def reset_navigation(self) -> None:
        self.path.clear()
        self.path_target_cell = None
        self.repath_timer = 0.0
        self.target_lock.clear()
        self.arbiter.reset()
        self.arbiter.clear_filtered_move()
        self.current_yaw_source = "nav"
        self.current_yaw_rate = 0.0
        self.current_aim_error = math.pi
        self.current_target_angle = 0.0
        self.aim_stable_time = 0.0
        self.last_seen_target_cell = None
        self.current_nav_target_cell = None
        self.has_path = False
        self.next_waypoint = None
        self.current_move_vector = (0.0, 0.0)
        self.current_effective_intent = INTENT_CHASE
        self.current_target_distance = 0.0
        self.current_has_los = False
        self.shoot_ready = False
        self.shoot_triggered_this_tick = False
        self.blocked_reason = "none"
        self.current_phase = "SEARCH"
        self.last_retarget_time = float("inf")
        self.decision_time_left = 0.0
        self.last_damage_time = float("inf")
        self.last_pack_reward = 0.0
        self.prev_pack_dist_tiles = None
        self.patrol_target_cell = None
        self.patrol_time_left = 0.0
        self.force_patrol_time_left = 0.0
        self.strafe_time_left = 0.0
        self.strafe_sign = 1
        self.no_progress_time = 0.0
        self.move_smooth_x = 0.0
        self.move_smooth_y = 0.0
        self.near_wall = False
        self.recover_time_left = 0.0
        self.recover_sign = 1
        self.replan_after_recover = False
        self.should_move = False
        self.current_progress_tiles = 0.0
        self.current_bumped = False
        self.current_reason = "no_target"
        self.no_reason_timer = 0.0
        self.no_reason_cooldown = 0.0
        self.retarget_timer = 0.0
        self.emergency_retarget_this_tick = False
        self.current_goal_type = "NONE"
        self.current_pack_score = 0.0
        self.current_enemy_score = 0.0
        self.current_pack_dist_tiles = None
        self.current_enemy_dist_tiles = None
        self.debug_intent_label = "CHASE"
        self.enemy_repulse_active = False

    def notify_damage(self) -> None:
        self.emergency_unlock = True
        self.action_time_left = 0.0
        self.decision_time_left = 0.0
        self.action_locked = False
        self.last_damage_time = 0.0

    def mark_target_switch(self, *, emergency: bool) -> None:
        self.retarget_timer = RETARGET_INTERVAL_SEC
        self.last_retarget_time = 0.0
        self.aim_stable_time = 0.0
        self.emergency_retarget_this_tick = emergency

    @property
    def recovery_side_label(self) -> str:
        return "L" if self.recover_sign < 0 else "R"

    def apply(self, game: "Game", dt: float) -> None:
        self.decision_time_left = max(0.0, self.decision_time_left - dt)
        self.action_time_left = max(0.0, self.action_time_left - dt)
        self.last_retarget_time += dt
        self.last_damage_time += dt
        self.patrol_time_left = max(0.0, self.patrol_time_left - dt)
        self.force_patrol_time_left = max(0.0, self.force_patrol_time_left - dt)
        self.strafe_time_left = max(0.0, self.strafe_time_left - dt)
        self.recover_time_left = max(0.0, self.recover_time_left - dt)
        self.no_reason_cooldown = max(0.0, self.no_reason_cooldown - dt)
        self.retarget_timer = max(0.0, self.retarget_timer - dt)
        self.emergency_retarget_this_tick = False
        self.target_lock.tick(dt)
        self.action_locked = self.action_time_left > 0.0 and not self.emergency_unlock

        if self.override_action is not None:
            self.last_action = self.override_action
            self.decision_time_left = self.action_interval
            self.action_time_left = INTENT_HOLD_SECONDS
            self.action_locked = True
        elif self.emergency_unlock:
            self.force_patrol_time_left = 0.0
            self.last_action = self.step(game._build_player_agent_state(self))
            self.decision_time_left = self.action_interval
            self.action_time_left = INTENT_HOLD_SECONDS
            self.action_locked = True
            self.emergency_unlock = False
        elif self.decision_time_left <= 0.0:
            self.decision_time_left = self.action_interval
            if self.action_time_left <= 0.0:
                self.last_action = self.step(game._build_player_agent_state(self))
                self.action_time_left = INTENT_HOLD_SECONDS
                self.action_locked = True
            else:
                self.action_locked = True
        else:
            self.action_locked = True

        game.player_bumped = game._apply_player_intent(self, self.last_action, dt)
