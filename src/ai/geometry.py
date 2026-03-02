from __future__ import annotations

import math

from src.settings import MAX_DEPTH, PLAYER_RADIUS, TILE_SIZE


def is_wall(world_x: float, world_y: float, level_map: list[str]) -> bool:
    grid_x = int(world_x // TILE_SIZE)
    grid_y = int(world_y // TILE_SIZE)
    if grid_y < 0 or grid_y >= len(level_map):
        return True
    if grid_x < 0 or grid_x >= len(level_map[grid_y]):
        return True
    return level_map[grid_y][grid_x] != "0"


def world_to_cell(world_x: float, world_y: float) -> tuple[int, int]:
    return int(world_x // TILE_SIZE), int(world_y // TILE_SIZE)


def cell_center(cell_x: int, cell_y: int) -> tuple[float, float]:
    return (cell_x + 0.5) * TILE_SIZE, (cell_y + 0.5) * TILE_SIZE


def is_open_cell(cell_x: int, cell_y: int, level_map: list[str]) -> bool:
    return (
        0 <= cell_y < len(level_map)
        and 0 <= cell_x < len(level_map[cell_y])
        and level_map[cell_y][cell_x] == "0"
    )


def can_move_to(world_x: float, world_y: float, radius: float, level_map: list[str]) -> bool:
    return not any(
        is_wall(world_x + offset_x, world_y + offset_y, level_map)
        for offset_x, offset_y in (
            (-radius, -radius),
            (radius, -radius),
            (-radius, radius),
            (radius, radius),
        )
    )


def move_with_slide(
    world_x: float,
    world_y: float,
    dx: float,
    dy: float,
    radius: float,
    level_map: list[str],
) -> tuple[float, float]:
    next_x = world_x + dx
    next_y = world_y + dy
    if can_move_to(next_x, next_y, radius, level_map):
        return next_x, next_y
    moved_x = world_x
    moved_y = world_y
    if can_move_to(world_x + dx, world_y, radius, level_map):
        moved_x = world_x + dx
    if can_move_to(moved_x, world_y + dy, radius, level_map):
        moved_y = world_y + dy
    elif moved_x == world_x and can_move_to(world_x, world_y + dy, radius, level_map):
        moved_y = world_y + dy
    return moved_x, moved_y


def cast_ray_hit(
    origin_x: float,
    origin_y: float,
    ray_angle: float,
    level_map: list[str],
) -> tuple[float, bool, str, float]:
    ray_dir_x = math.cos(ray_angle)
    ray_dir_y = math.sin(ray_angle)
    map_x = int(origin_x // TILE_SIZE)
    map_y = int(origin_y // TILE_SIZE)
    delta_dist_x = abs(TILE_SIZE / ray_dir_x) if abs(ray_dir_x) > 1e-6 else float("inf")
    delta_dist_y = abs(TILE_SIZE / ray_dir_y) if abs(ray_dir_y) > 1e-6 else float("inf")

    if ray_dir_x < 0:
        step_x = -1
        side_dist_x = (origin_x - map_x * TILE_SIZE) / abs(ray_dir_x)
    else:
        step_x = 1
        side_dist_x = ((map_x + 1) * TILE_SIZE - origin_x) / max(abs(ray_dir_x), 1e-6)

    if ray_dir_y < 0:
        step_y = -1
        side_dist_y = (origin_y - map_y * TILE_SIZE) / abs(ray_dir_y)
    else:
        step_y = 1
        side_dist_y = ((map_y + 1) * TILE_SIZE - origin_y) / max(abs(ray_dir_y), 1e-6)

    hit_vertical = False
    while True:
        if side_dist_x < side_dist_y:
            distance = side_dist_x
            side_dist_x += delta_dist_x
            map_x += step_x
            hit_vertical = True
        else:
            distance = side_dist_y
            side_dist_y += delta_dist_y
            map_y += step_y
            hit_vertical = False

        if map_y < 0 or map_y >= len(level_map) or map_x < 0 or map_x >= len(level_map[map_y]):
            return MAX_DEPTH, hit_vertical, "1", 0.0

        cell_value = level_map[map_y][map_x]
        if cell_value != "0":
            hit_x = origin_x + ray_dir_x * distance
            hit_y = origin_y + ray_dir_y * distance
            wall_offset = (hit_y / TILE_SIZE) % 1.0 if hit_vertical else (hit_x / TILE_SIZE) % 1.0
            if (hit_vertical and ray_dir_x > 0.0) or (not hit_vertical and ray_dir_y < 0.0):
                wall_offset = 1.0 - wall_offset
            return min(distance, MAX_DEPTH), hit_vertical, cell_value, wall_offset


def cast_ray(origin_x: float, origin_y: float, ray_angle: float, level_map: list[str]) -> tuple[float, bool]:
    distance, hit_vertical, _wall_type, _wall_offset = cast_ray_hit(
        origin_x,
        origin_y,
        ray_angle,
        level_map,
    )
    return distance, hit_vertical


def has_line_of_sight(
    origin_x: float,
    origin_y: float,
    target_x: float,
    target_y: float,
    level_map: list[str],
) -> bool:
    delta_x = target_x - origin_x
    delta_y = target_y - origin_y
    distance = math.hypot(delta_x, delta_y)
    if distance <= 0.0001:
        return True
    angle = math.atan2(delta_y, delta_x)
    wall_distance, _ = cast_ray(origin_x, origin_y, angle, level_map)
    return wall_distance >= distance - PLAYER_RADIUS


def distance_bin(distance: float) -> int:
    if distance < TILE_SIZE * 2.5:
        return 0
    if distance < TILE_SIZE * 5.0:
        return 1
    return 2


def hp_bin(current_hp: int, max_hp: int) -> int:
    ratio = current_hp / max(max_hp, 1)
    if ratio < 0.34:
        return 0
    if ratio < 0.67:
        return 1
    return 2


def pack_distance_bin(distance: float | None) -> int:
    if distance is None:
        return 3
    if distance <= TILE_SIZE * 4.0:
        return 0
    if distance <= TILE_SIZE * 10.0:
        return 1
    if distance <= TILE_SIZE * 20.0:
        return 2
    return 3


def relative_direction_bin(
    enemy_x: float,
    enemy_y: float,
    player_x: float,
    player_y: float,
    player_angle: float,
) -> int:
    angle_to_enemy = math.atan2(enemy_y - player_y, enemy_x - player_x)
    delta = wrap_to_pi(angle_to_enemy - player_angle)
    if -math.pi / 4 <= delta < math.pi / 4:
        return 0
    if math.pi / 4 <= delta < 3 * math.pi / 4:
        return 1
    if -3 * math.pi / 4 <= delta < -math.pi / 4:
        return 2
    return 3


def normalize_vector(delta_x: float, delta_y: float) -> tuple[float, float, float]:
    distance = math.hypot(delta_x, delta_y)
    if distance <= 0.0001:
        return 0.0, 0.0, 0.0
    return delta_x / distance, delta_y / distance, distance


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def target_angle_and_error(
    actor_x: float,
    actor_y: float,
    actor_yaw: float,
    target_x: float,
    target_y: float,
) -> tuple[float, float]:
    dx = target_x - actor_x
    dy = target_y - actor_y
    target_angle = math.atan2(dy, dx)
    angle_error = wrap_to_pi(target_angle - actor_yaw)
    return target_angle, angle_error


def banded_intent(distance: float, has_los: bool, min_distance: float, max_distance: float) -> int:
    from src.game.states import INTENT_CHASE, INTENT_ENGAGE, INTENT_EVADE

    if not has_los:
        return INTENT_CHASE
    if distance < min_distance:
        return INTENT_EVADE
    if distance > max_distance:
        return INTENT_CHASE
    return INTENT_ENGAGE


def turn_toward_angle(current_angle: float, desired_angle: float, max_step: float) -> float:
    delta = wrap_to_pi(desired_angle - current_angle)
    delta = max(-max_step, min(max_step, delta))
    return (current_angle + delta) % math.tau


def project_point(origin_x: float, origin_y: float, dir_x: float, dir_y: float, distance: float) -> tuple[float, float]:
    return origin_x + dir_x * distance, origin_y + dir_y * distance


def flank_point(
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    side_sign: int,
    distance: float,
) -> tuple[float, float]:
    dir_x, dir_y, _ = normalize_vector(target_x - source_x, target_y - source_y)
    perp_x = -dir_y * side_sign
    perp_y = dir_x * side_sign
    return target_x + perp_x * distance, target_y + perp_y * distance
