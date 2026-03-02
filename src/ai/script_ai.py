from __future__ import annotations

import math
import random

from src.ai.geometry import cell_center, has_line_of_sight, normalize_vector, project_point, world_to_cell, flank_point, banded_intent, move_with_slide, relative_direction_bin, distance_bin, hp_bin
from src.ai.pathfinding import build_path, reachable_open_cells_within_radius
from src.entities.enemy import Enemy, HealthPackTarget, RLEnemy
from src.game.states import INTENT_CHASE, INTENT_ENGAGE, INTENT_EVADE, INTENT_REPOSITION, SCRIPT_STATE_CHASE, SCRIPT_STATE_ENGAGE, SCRIPT_STATE_PATROL, SCRIPT_STATE_RETURN, SCRIPT_STATE_SEARCH
from src.settings import (
    CRITICAL_HP_RATIO,
    ENEMY_AGGRO_RANGE,
    ENEMY_PATH_RECALC_INTERVAL,
    ENEMY_TYPES,
    INTENT_HOLD_SECONDS,
    PLAYER_AGENT_REWARD_DAMAGE,
    RL_ENEMY_RANGE_MAX,
    RL_ENEMY_RANGE_MIN,
    RL_REWARD_BREAK_LOS,
    RL_REWARD_HIT_PLAYER,
    RL_REWARD_SHAPING,
    RL_REWARD_WALL_BUMP,
    SCRIPT_ALERT_HOLD_SECONDS,
    SCRIPT_HEAR_RANGE_FACTOR,
    SCRIPT_LOST_TARGET_SECONDS,
    SCRIPT_PATROL_RADIUS_MAX_TILES,
    SCRIPT_PATROL_RADIUS_MIN_TILES,
    SCRIPT_PATROL_WAYPOINTS_MAX,
    SCRIPT_PATROL_WAYPOINTS_MIN,
    SCRIPT_SIGHT_RANGE_TILES,
    SFX_VOLUME_PLAYER_HURT,
    TILE_SIZE,
    UNDER_FIRE_WINDOW_SECONDS,
)


class ScriptAIMixin:
    def _script_hear_range_tiles(self) -> float:
        return SCRIPT_HEAR_RANGE_FACTOR * max(len(self.map), len(self.map[0]))

    def _patrol_radius_tiles(self) -> int:
        max_dim = max(len(self.map), len(self.map[0]))
        auto_radius = max_dim // 4
        return max(min(SCRIPT_PATROL_RADIUS_MAX_TILES, auto_radius), min(SCRIPT_PATROL_RADIUS_MIN_TILES, max_dim // 2))

    def _generate_patrol_route(self, start_cell: tuple[int, int], kind: str) -> list[tuple[int, int]]:
        radius_tiles = self._patrol_radius_tiles()
        reachable_cells = reachable_open_cells_within_radius(start_cell, radius_tiles, self.map)
        if not reachable_cells:
            return [start_cell]
        min_points = min(SCRIPT_PATROL_WAYPOINTS_MIN, len(reachable_cells))
        max_points = min(SCRIPT_PATROL_WAYPOINTS_MAX, len(reachable_cells))
        if max_points <= 1:
            return [start_cell]
        seed = start_cell[0] * 73856093 ^ start_cell[1] * 19349663 ^ sum(ord(char) for char in kind) * 83492791
        patrol_rng = random.Random(seed)
        candidates = [cell for cell in reachable_cells if cell != start_cell and math.dist(cell, start_cell) >= 2.0]
        if len(candidates) < min_points:
            candidates = [cell for cell in reachable_cells if cell != start_cell]
        patrol_rng.shuffle(candidates)
        waypoint_count = patrol_rng.randint(min_points, max_points)
        route: list[tuple[int, int]] = []
        for cell in candidates:
            if any(math.dist(cell, other) < 2.0 for other in route):
                continue
            route.append(cell)
            if len(route) >= waypoint_count:
                break
        if not route:
            route = [start_cell]
        return route

    def _configure_script_enemy(self, enemy: Enemy, spawn_cell: tuple[int, int]) -> None:
        definition = ENEMY_TYPES[enemy.kind]
        enemy.home_cell = spawn_cell
        enemy.patrol_points = self._generate_patrol_route(spawn_cell, enemy.kind)
        enemy.patrol_index = 0
        enemy.state = SCRIPT_STATE_PATROL
        enemy.search_target_cell = None
        enemy.alert_time_left = 0.0
        enemy.no_los_time = 0.0
        enemy.path.clear()
        enemy.path_target_cell = None
        enemy.repath_timer = 0.0
        enemy.strafe_time_left = 0.0
        enemy.strafe_sign = 1 if enemy.entity_id % 2 == 0 else -1
        enemy.current_nav_target_cell = enemy.patrol_points[0] if enemy.patrol_points else spawn_cell
        enemy.has_path = False
        enemy.next_waypoint = None
        enemy.current_move_vector = (0.0, 0.0)
        enemy.current_target_distance = 0.0
        enemy.current_has_los = False
        enemy.hear_range_tiles = self._script_hear_range_tiles()
        enemy.sight_range_tiles = float(definition.get("sight_range_tiles", SCRIPT_SIGHT_RANGE_TILES))
        enemy.attack_style = str(definition.get("attack_style", "ranged"))
        enemy.melee_range = TILE_SIZE * float(definition.get("melee_range_tiles", 1.0))
        enemy.engage_min_range = TILE_SIZE * float(definition.get("engage_min_tiles", 3.5 if enemy.attack_style == "ranged" else 0.6))
        enemy.engage_max_range = TILE_SIZE * float(definition.get("engage_max_tiles", 5.5 if enemy.attack_style == "ranged" else 1.4))

    def _move_enemy_by_vector(self, enemy: Enemy, move_x: float, move_y: float) -> bool:
        intended = math.hypot(move_x, move_y)
        next_x, next_y = move_with_slide(enemy.x, enemy.y, move_x, move_y, enemy.radius, self.map)
        actual = math.hypot(next_x - enemy.x, next_y - enemy.y)
        enemy.x, enemy.y = next_x, next_y
        return intended > 0.01 and actual < intended * 0.35

    def _enemy_shoot_player(self, enemy: Enemy) -> bool:
        if enemy.shot_cooldown > 0.0 or enemy.damage <= 0:
            return False
        if not has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map):
            return False
        if math.hypot(self.player.x - enemy.x, self.player.y - enemy.y) > enemy.aggro_range:
            return False
        enemy.shot_cooldown = enemy.shot_cooldown_max
        self.player_hp = max(0, self.player_hp - enemy.damage)
        self.audio.play("player_hurt", SFX_VOLUME_PLAYER_HURT)
        self.player_under_fire_timer = UNDER_FIRE_WINDOW_SECONDS
        self._add_player_agent_reward(PLAYER_AGENT_REWARD_DAMAGE)
        if self.current_control is self.agent_control:
            self.agent_control.notify_damage()
        return True

    def _enemy_melee_attack_player(self, enemy: Enemy) -> bool:
        if enemy.shot_cooldown > 0.0 or enemy.damage <= 0:
            return False
        if math.hypot(self.player.x - enemy.x, self.player.y - enemy.y) > enemy.melee_range:
            return False
        enemy.shot_cooldown = enemy.shot_cooldown_max
        self.player_hp = max(0, self.player_hp - enemy.damage)
        self.audio.play("player_hurt", SFX_VOLUME_PLAYER_HURT)
        self.player_under_fire_timer = UNDER_FIRE_WINDOW_SECONDS
        self._add_player_agent_reward(PLAYER_AGENT_REWARD_DAMAGE)
        if self.current_control is self.agent_control:
            self.agent_control.notify_damage()
        return True
