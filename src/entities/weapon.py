from __future__ import annotations

from dataclasses import dataclass

from src.settings import AGENT_ACTION_INTERVAL_SECONDS, MAGAZINE_SIZE, RELOAD_SECONDS


@dataclass
class WeaponState:
    magazine_size: int = MAGAZINE_SIZE
    reload_duration: float = RELOAD_SECONDS
    shot_cooldown_duration: float = AGENT_ACTION_INTERVAL_SECONDS
    ammo_in_mag: int = MAGAZINE_SIZE
    is_reloading: bool = False
    reload_time_left: float = 0.0
    shot_cooldown_left: float = 0.0

    def can_shoot(self) -> bool:
        return not self.is_reloading and self.ammo_in_mag > 0 and self.shot_cooldown_left <= 0.0

    def start_reload(self) -> bool:
        if self.is_reloading or self.ammo_in_mag >= self.magazine_size:
            return False
        self.is_reloading = True
        self.reload_time_left = self.reload_duration
        return True

    def update(self, dt: float) -> None:
        self.shot_cooldown_left = max(0.0, self.shot_cooldown_left - dt)
        if not self.is_reloading:
            return
        self.reload_time_left = max(0.0, self.reload_time_left - dt)
        if self.reload_time_left <= 0.0:
            self.is_reloading = False
            self.ammo_in_mag = self.magazine_size

    def consume_round(self) -> bool:
        if not self.can_shoot():
            return False
        self.ammo_in_mag -= 1
        self.shot_cooldown_left = self.shot_cooldown_duration
        return True

    def reset(self) -> None:
        self.ammo_in_mag = self.magazine_size
        self.is_reloading = False
        self.reload_time_left = 0.0
        self.shot_cooldown_left = 0.0
