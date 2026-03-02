from __future__ import annotations

import math
from collections import deque

from src.ai.geometry import can_move_to, is_open_cell
from src.settings import TILE_SIZE


def build_path(
    start_cell: tuple[int, int],
    target_cell: tuple[int, int],
    level_map: list[str],
) -> list[tuple[int, int]]:
    if start_cell == target_cell:
        return []
    if not is_open_cell(*start_cell, level_map) or not is_open_cell(*target_cell, level_map):
        return []
    queue: deque[tuple[int, int]] = deque([start_cell])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    while queue:
        cell_x, cell_y = queue.popleft()
        if (cell_x, cell_y) == target_cell:
            break
        for next_x, next_y in (
            (cell_x + 1, cell_y),
            (cell_x - 1, cell_y),
            (cell_x, cell_y + 1),
            (cell_x, cell_y - 1),
        ):
            next_cell = (next_x, next_y)
            if next_cell in came_from or not is_open_cell(next_x, next_y, level_map):
                continue
            came_from[next_cell] = (cell_x, cell_y)
            queue.append(next_cell)
    if target_cell not in came_from:
        return []
    path: list[tuple[int, int]] = []
    current = target_cell
    while current != start_cell:
        path.append(current)
        parent = came_from[current]
        if parent is None:
            break
        current = parent
    path.reverse()
    return path


def build_fallback_path_toward_target(
    start_cell: tuple[int, int],
    target_cell: tuple[int, int],
    level_map: list[str],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    if not is_open_cell(*start_cell, level_map):
        return [], None
    queue: deque[tuple[int, int]] = deque([start_cell])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    start_distance = math.dist(start_cell, target_cell)
    best_cell = start_cell
    best_distance = start_distance
    while queue:
        cell_x, cell_y = queue.popleft()
        current_cell = (cell_x, cell_y)
        current_distance = math.dist(current_cell, target_cell)
        if current_distance < best_distance:
            best_distance = current_distance
            best_cell = current_cell
        for next_x, next_y in (
            (cell_x + 1, cell_y),
            (cell_x - 1, cell_y),
            (cell_x, cell_y + 1),
            (cell_x, cell_y - 1),
        ):
            next_cell = (next_x, next_y)
            if next_cell in came_from or not is_open_cell(next_x, next_y, level_map):
                continue
            came_from[next_cell] = current_cell
            queue.append(next_cell)
    if best_cell == start_cell or best_distance >= start_distance:
        return [], None
    path: list[tuple[int, int]] = []
    current = best_cell
    while current != start_cell:
        path.append(current)
        parent = came_from[current]
        if parent is None:
            break
        current = parent
    path.reverse()
    return path, best_cell


def reachable_open_cells_within_radius(
    start_cell: tuple[int, int],
    max_radius_tiles: int,
    level_map: list[str],
) -> list[tuple[int, int]]:
    if not is_open_cell(*start_cell, level_map):
        return []
    queue: deque[tuple[int, int]] = deque([start_cell])
    visited = {start_cell}
    cells: list[tuple[int, int]] = []
    while queue:
        cell_x, cell_y = queue.popleft()
        current_cell = (cell_x, cell_y)
        cells.append(current_cell)
        for next_x, next_y in (
            (cell_x + 1, cell_y),
            (cell_x - 1, cell_y),
            (cell_x, cell_y + 1),
            (cell_x, cell_y - 1),
        ):
            next_cell = (next_x, next_y)
            if next_cell in visited or not is_open_cell(next_x, next_y, level_map):
                continue
            if math.dist(start_cell, next_cell) > max_radius_tiles + 0.5:
                continue
            visited.add(next_cell)
            queue.append(next_cell)
    return cells


def find_free_spawn_position(
    spawn_x: float,
    spawn_y: float,
    radius: float,
    level_map: list[str],
    spawn_index: int,
) -> tuple[float, float]:
    if can_move_to(spawn_x, spawn_y, radius, level_map):
        return spawn_x, spawn_y
    start_cell_x = int(spawn_x // TILE_SIZE)
    start_cell_y = int(spawn_y // TILE_SIZE)
    max_search = max(len(level_map), len(level_map[0]))
    jitter_offsets = (
        (0.0, 0.0),
        (0.16, 0.0),
        (-0.16, 0.0),
        (0.0, 0.16),
        (0.0, -0.16),
        (0.12, 0.12),
        (-0.12, 0.12),
        (0.12, -0.12),
        (-0.12, -0.12),
    )
    for search_radius in range(max_search + 1):
        candidates: list[tuple[float, int, int]] = []
        for cell_y in range(start_cell_y - search_radius, start_cell_y + search_radius + 1):
            if cell_y < 0 or cell_y >= len(level_map):
                continue
            for cell_x in range(start_cell_x - search_radius, start_cell_x + search_radius + 1):
                if cell_x < 0 or cell_x >= len(level_map[cell_y]):
                    continue
                if level_map[cell_y][cell_x] == "1":
                    continue
                center_x = (cell_x + 0.5) * TILE_SIZE
                center_y = (cell_y + 0.5) * TILE_SIZE
                distance_sq = (center_x - spawn_x) ** 2 + (center_y - spawn_y) ** 2
                candidates.append((distance_sq, cell_x, cell_y))
        for _, cell_x, cell_y in sorted(candidates):
            center_x = (cell_x + 0.5) * TILE_SIZE
            center_y = (cell_y + 0.5) * TILE_SIZE
            for offset_index in range(len(jitter_offsets)):
                jitter_x, jitter_y = jitter_offsets[(spawn_index + offset_index) % len(jitter_offsets)]
                candidate_x = center_x + jitter_x * TILE_SIZE
                candidate_y = center_y + jitter_y * TILE_SIZE
                if can_move_to(candidate_x, candidate_y, radius, level_map):
                    return candidate_x, candidate_y
    return spawn_x, spawn_y
