from __future__ import annotations

import math

import pygame

from src.ai.geometry import cast_ray, cast_ray_hit
from src.settings import CEILING_COLOR, FLOOR_COLOR, HALF_FOV, HEIGHT, MAX_DEPTH, RAY_STRIDE, SPRITE_SCALE, TILE_SIZE, WIDTH, FOV


class WorldRenderMixin:
    def _draw_world(self) -> None:
        self.screen.fill(FLOOR_COLOR)
        pygame.draw.rect(self.screen, CEILING_COLOR, (0, 0, WIDTH, HEIGHT // 2))
        self.ray_hits = []
        self.wall_depths = [MAX_DEPTH] * WIDTH
        for column in range(0, WIDTH, RAY_STRIDE):
            ray_angle = self.player.angle - HALF_FOV + (column / WIDTH) * FOV
            raw_distance, hit_vertical, wall_type, wall_offset = cast_ray_hit(
                self.player.x,
                self.player.y,
                ray_angle,
                self.map,
            )
            corrected_distance = max(raw_distance * math.cos(ray_angle - self.player.angle), 0.0001)
            slice_end = min(WIDTH, column + RAY_STRIDE)
            for depth_column in range(column, slice_end):
                self.wall_depths[depth_column] = corrected_distance
            wall_height = int((TILE_SIZE / corrected_distance) * self.proj_plane_dist)
            wall_top = (HEIGHT - wall_height) // 2
            texture = self._wall_texture_for(wall_type)
            texture_width = texture.get_width()
            texture_x = max(0, min(texture_width - 1, int(wall_offset * texture_width)))
            column_surface = texture.subsurface((texture_x, 0, 1, texture.get_height()))
            scaled_column = pygame.transform.scale(
                column_surface,
                (slice_end - column, max(1, wall_height)),
            )
            shade = max(0.28, 1.0 - corrected_distance / MAX_DEPTH)
            if hit_vertical:
                shade *= 0.82
            shaded_column = scaled_column.copy()
            shade_value = max(0, min(255, int(255 * shade)))
            shaded_column.fill((shade_value, shade_value, shade_value), special_flags=pygame.BLEND_RGB_MULT)
            self.screen.blit(shaded_column, (column, wall_top))

            hit_x = self.player.x + math.cos(ray_angle) * raw_distance
            hit_y = self.player.y + math.sin(ray_angle) * raw_distance
            self.ray_hits.append((hit_x, hit_y))
        self._draw_enemies()

    def _draw_enemies(self) -> None:
        ordered = sorted(
            self.enemies,
            key=lambda enemy: (enemy.x - self.player.x) ** 2 + (enemy.y - self.player.y) ** 2,
            reverse=True,
        )
        for enemy in ordered:
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            distance = math.hypot(dx, dy)
            if distance <= 0.0001 or distance >= MAX_DEPTH:
                continue
            sprite_angle = math.atan2(dy, dx)
            relative_angle = (sprite_angle - self.player.angle + math.pi) % math.tau - math.pi
            if abs(relative_angle) > HALF_FOV + 0.35:
                continue
            corrected_distance = distance * math.cos(relative_angle)
            if corrected_distance <= 0.0001:
                continue
            screen_x = WIDTH * 0.5 + math.tan(relative_angle) * self.proj_plane_dist
            sprite_height = int((TILE_SIZE / corrected_distance) * self.proj_plane_dist * SPRITE_SCALE * enemy.size)
            sprite_width = sprite_height
            if sprite_width <= 0 or sprite_height <= 0:
                continue
            left = int(screen_x - sprite_width / 2)
            right = int(screen_x + sprite_width / 2)
            if right < 0 or left >= WIDTH:
                continue
            bottom = HEIGHT // 2 + sprite_height // 2
            top = bottom - sprite_height
            source = self.enemy_hit_surfaces[enemy.kind] if enemy.hit_flash_timer > 0.0 else self.enemy_surfaces[enemy.kind]
            scaled = pygame.transform.smoothscale(source, (sprite_width, sprite_height))
            clip_left = max(0, left)
            clip_right = min(WIDTH, right)
            for screen_column in range(clip_left, clip_right):
                if corrected_distance >= self.wall_depths[screen_column]:
                    continue
                texture_x = int((screen_column - left) / max(sprite_width, 1) * scaled.get_width())
                texture_x = max(0, min(scaled.get_width() - 1, texture_x))
                strip = scaled.subsurface((texture_x, 0, 1, scaled.get_height()))
                self.screen.blit(strip, (screen_column, top))

    def _draw_weapon_overlay(self) -> None:
        recoil_progress = 1.0 - min(1.0, self.weapon_recoil_time_left / max(self.weapon_recoil_duration, 1e-6))
        recoil_curve = math.sin(recoil_progress * math.pi) if self.weapon_recoil_time_left > 0.0 else 0.0
        bob_x = math.sin(self.weapon_bob_phase) * 7.0 * self.weapon_bob_amount
        bob_y = abs(math.cos(self.weapon_bob_phase * 0.5)) * 8.0 * self.weapon_bob_amount
        recoil_y = -18.0 * recoil_curve
        recoil_scale = 1.0 - 0.04 * recoil_curve
        reload_drop = 20.0 + 26.0 * (
            1.0 - self.player.weapon.reload_time_left / max(self.player.weapon.reload_duration, 1e-6)
        ) if self.player.weapon.is_reloading else 0.0
        base_width = int(WIDTH * 0.34)
        scaled_width = max(1, int(base_width * recoil_scale))
        scaled_height = max(1, int(self.weapon_surface.get_height() * (scaled_width / self.weapon_surface.get_width())))
        weapon = pygame.transform.smoothscale(self.weapon_surface, (scaled_width, scaled_height))
        draw_x = WIDTH // 2 - weapon.get_width() // 2 + int(bob_x)
        draw_y = HEIGHT - weapon.get_height() + 42 + int(bob_y + recoil_y + reload_drop)
        self.screen.blit(weapon, (draw_x, draw_y))

    def _pick_crosshair_target(self):
        best_target = None
        best_distance = float("inf")
        for enemy in self.enemies:
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            distance = math.hypot(dx, dy)
            if distance <= 0.0001 or distance >= MAX_DEPTH:
                continue
            enemy_angle = math.atan2(dy, dx)
            relative_angle = (enemy_angle - self.player.angle + math.pi) % math.tau - math.pi
            if abs(relative_angle) > HALF_FOV:
                continue
            angle_window = max(math.atan2(enemy.radius * enemy.size, distance), math.radians(0.35))
            if abs(relative_angle) > angle_window:
                continue
            wall_distance, _ = cast_ray(self.player.x, self.player.y, enemy_angle, self.map)
            if wall_distance < distance - enemy.radius:
                continue
            if distance < best_distance:
                best_distance = distance
                best_target = enemy
        return best_target
