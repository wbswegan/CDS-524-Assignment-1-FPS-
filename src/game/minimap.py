from __future__ import annotations

import math

import pygame

from src.ai.geometry import cell_center
from src.entities.enemy import HealthPackTarget, RLEnemy
from src.game.states import SCRIPT_STATE_COLORS
from src.settings import (
    BG_COLOR,
    FLOOR_COLOR,
    HUD_TEXT_COLOR,
    MINIMAP_MAX_HEIGHT,
    MINIMAP_MAX_WIDTH,
    MINIMAP_PADDING,
    PLAYER_COLOR,
    PLAYER_DIR_COLOR,
    PLAYER_RADIUS,
    RAY_COLOR,
    TILE_SIZE,
    WALL_COLOR,
    WIDTH,
    HEIGHT,
)


class MinimapMixin:
    def _draw_minimap(self) -> None:
        tile_pixels, minimap_scale = self._get_minimap_metrics()
        map_width = len(self.map[0]) * tile_pixels
        map_height = len(self.map) * tile_pixels
        minimap_rect = pygame.Rect(
            MINIMAP_PADDING - 8,
            MINIMAP_PADDING - 8,
            map_width + 16,
            map_height + 16,
        )
        pygame.draw.rect(self.screen, BG_COLOR, minimap_rect, border_radius=8)
        for row_index, row in enumerate(self.map):
            for col_index, cell in enumerate(row):
                tile_rect = pygame.Rect(
                    MINIMAP_PADDING + col_index * tile_pixels,
                    MINIMAP_PADDING + row_index * tile_pixels,
                    tile_pixels,
                    tile_pixels,
                )
                color = WALL_COLOR if cell == "1" else FLOOR_COLOR
                pygame.draw.rect(self.screen, color, tile_rect)
                pygame.draw.rect(self.screen, BG_COLOR, tile_rect, width=1)

        for enemy in self.enemies:
            enemy_x = MINIMAP_PADDING + int(enemy.x * minimap_scale)
            enemy_y = MINIMAP_PADDING + int(enemy.y * minimap_scale)
            enemy_radius = max(3, int(enemy.radius * minimap_scale))
            draw_color = (96, 230, 124) if isinstance(enemy, HealthPackTarget) else enemy.color
            if self.show_script_ai_debug and not isinstance(enemy, RLEnemy):
                state_color = SCRIPT_STATE_COLORS.get(enemy.state, enemy.color)
                hear_radius = max(tile_pixels, int(enemy.hear_range_tiles * tile_pixels))
                sight_radius = max(tile_pixels, int(enemy.sight_range_tiles * tile_pixels))
                pygame.draw.circle(self.screen, state_color, (enemy_x, enemy_y), hear_radius, width=1)
                pygame.draw.circle(self.screen, HUD_TEXT_COLOR, (enemy_x, enemy_y), sight_radius, width=1)
                waypoint_cell = enemy.current_nav_target_cell
                if waypoint_cell is None and enemy.patrol_points:
                    waypoint_cell = enemy.patrol_points[enemy.patrol_index % len(enemy.patrol_points)]
                if waypoint_cell is not None:
                    waypoint_x, waypoint_y = cell_center(*waypoint_cell)
                    waypoint_point = (
                        MINIMAP_PADDING + int(waypoint_x * minimap_scale),
                        MINIMAP_PADDING + int(waypoint_y * minimap_scale),
                    )
                    pygame.draw.line(self.screen, state_color, (enemy_x, enemy_y), waypoint_point, width=1)
                    pygame.draw.circle(self.screen, state_color, waypoint_point, 4, width=1)
            pygame.draw.circle(self.screen, draw_color, (enemy_x, enemy_y), enemy_radius)
            pygame.draw.circle(self.screen, BG_COLOR, (enemy_x, enemy_y), enemy_radius, width=1)
            if self.show_path_debug and enemy.path:
                path_points = [(enemy_x, enemy_y)]
                for cell_x, cell_y in enemy.path:
                    center_x, center_y = cell_center(cell_x, cell_y)
                    path_points.append((MINIMAP_PADDING + int(center_x * minimap_scale), MINIMAP_PADDING + int(center_y * minimap_scale)))
                if len(path_points) > 1:
                    pygame.draw.lines(self.screen, enemy.color, False, path_points, width=2)
                    pygame.draw.circle(self.screen, (255, 255, 255), path_points[1], 4)
            if self.show_script_ai_debug and not isinstance(enemy, RLEnemy):
                state_label = self.font.render(enemy.state[:2], True, SCRIPT_STATE_COLORS.get(enemy.state, HUD_TEXT_COLOR))
                self.screen.blit(state_label, (enemy_x + 6, enemy_y - 10))

        if self.ray_hits:
            ray_step = max(1, len(self.ray_hits) // 80)
            start = (
                MINIMAP_PADDING + int(self.player.x * minimap_scale),
                MINIMAP_PADDING + int(self.player.y * minimap_scale),
            )
            for hit_x, hit_y in self.ray_hits[::ray_step]:
                end = (
                    MINIMAP_PADDING + int(hit_x * minimap_scale),
                    MINIMAP_PADDING + int(hit_y * minimap_scale),
                )
                pygame.draw.line(self.screen, RAY_COLOR, start, end, width=1)

        player_x = MINIMAP_PADDING + int(self.player.x * minimap_scale)
        player_y = MINIMAP_PADDING + int(self.player.y * minimap_scale)
        player_radius = max(4, int(PLAYER_RADIUS * minimap_scale))
        pygame.draw.circle(self.screen, PLAYER_COLOR, (player_x, player_y), player_radius)
        if self.current_control is self.agent_control:
            target = self._get_valid_player_agent_target(self.agent_control)
            if target is not None:
                target_point = (
                    MINIMAP_PADDING + int(target.x * minimap_scale),
                    MINIMAP_PADDING + int(target.y * minimap_scale),
                )
                target_radius = max(3, int(target.radius * minimap_scale))
                pygame.draw.line(self.screen, HUD_TEXT_COLOR, (player_x, player_y), target_point, width=1)
                pygame.draw.circle(self.screen, HUD_TEXT_COLOR, target_point, max(target_radius + 3, 6), width=1)
        dir_length = max(10, int(tile_pixels * 0.8))
        dir_x = player_x + int(math.cos(self.player.angle) * dir_length)
        dir_y = player_y + int(math.sin(self.player.angle) * dir_length)
        pygame.draw.line(self.screen, PLAYER_DIR_COLOR, (player_x, player_y), (dir_x, dir_y), width=3)

        hint_text = "Agent mode On" if self.current_control is self.agent_control else "Press F4 to enable Agent"
        label = self.font.render(hint_text, True, HUD_TEXT_COLOR)
        self.screen.blit(label, (minimap_rect.left, minimap_rect.bottom + 6))

    def _get_minimap_metrics(self) -> tuple[int, float]:
        cols = len(self.map[0])
        rows = len(self.map)
        area_width = min(MINIMAP_MAX_WIDTH, WIDTH // 3)
        area_height = min(MINIMAP_MAX_HEIGHT, HEIGHT // 2)
        tile_pixels = max(4, min(area_width // cols, area_height // rows))
        return tile_pixels, tile_pixels / TILE_SIZE
