from __future__ import annotations

import math

import pygame

from src.ai.geometry import move_with_slide
from src.entities.weapon import WeaponState
from src.settings import PLAYER_RADIUS, PLAYER_SPEED


class Player:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.angle = 0.0
        self.weapon = WeaponState()

    def update(self, dt: float, level_map: list[str]) -> None:
        keys = pygame.key.get_pressed()
        forward = 0.0
        strafe = 0.0
        if keys[pygame.K_w]:
            forward += 1.0
        if keys[pygame.K_s]:
            forward -= 1.0
        if keys[pygame.K_a]:
            strafe -= 1.0
        if keys[pygame.K_d]:
            strafe += 1.0
        self.move(forward, strafe, dt, level_map)

    def turn(self, yaw_delta: float) -> None:
        self.angle = (self.angle + yaw_delta) % math.tau

    def move(self, forward: float, strafe: float, dt: float, level_map: list[str]) -> None:
        move_x = math.cos(self.angle) * forward - math.sin(self.angle) * strafe
        move_y = math.sin(self.angle) * forward + math.cos(self.angle) * strafe
        if move_x or move_y:
            length = math.hypot(move_x, move_y)
            move_x /= length
            move_y /= length
        step = PLAYER_SPEED * dt
        self._move_axis(move_x * step, 0.0, level_map)
        self._move_axis(0.0, move_y * step, level_map)

    def _move_axis(self, dx: float, dy: float, level_map: list[str]) -> None:
        self.x, self.y = move_with_slide(
            self.x,
            self.y,
            dx,
            dy,
            PLAYER_RADIUS,
            level_map,
        )

    def update_weapon(self, dt: float) -> None:
        self.weapon.update(dt)

    def start_reload(self) -> bool:
        return self.weapon.start_reload()
