from __future__ import annotations

from pathlib import Path

import pygame

from src.ai.geometry import cell_center, is_open_cell
from src.settings import BG_COLOR, BGM_VOLUME, ENEMY_TYPES, HUD_TEXT_COLOR, TILE_SIZE


class AudioManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.enabled = False
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self.logged_messages: set[str] = set()
        self.sound_paths = {
            "shoot": project_root / "assets" / "sfx" / "shoot.ogg",
            "empty": project_root / "assets" / "sfx" / "empty.ogg",
            "reload": project_root / "assets" / "sfx" / "reload.ogg",
            "hit": project_root / "assets" / "sfx" / "hit.ogg",
            "enemy_die": project_root / "assets" / "sfx" / "enemy_die.ogg",
            "player_hurt": project_root / "assets" / "sfx" / "player_hurt.ogg",
            "ui_select": project_root / "assets" / "sfx" / "ui_select.ogg",
        }
        self.bgm_path = project_root / "assets" / "music" / "bgm.ogg"
        self._init_mixer()

    def _log_once(self, message: str) -> None:
        if message in self.logged_messages:
            return
        self.logged_messages.add(message)
        print(message)

    def _init_mixer(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.enabled = True
        except pygame.error as exc:
            self._log_once(f"Audio disabled: {exc}")
            self.enabled = False
            return

        for name, path in self.sound_paths.items():
            if not path.exists():
                self._log_once(f"Missing audio file: {path}")
                continue
            try:
                self.sounds[name] = pygame.mixer.Sound(str(path))
            except pygame.error as exc:
                self._log_once(f"Failed to load sound {path}: {exc}")

        if self.bgm_path.exists():
            try:
                pygame.mixer.music.load(str(self.bgm_path))
                pygame.mixer.music.set_volume(BGM_VOLUME)
                pygame.mixer.music.play(-1)
            except pygame.error as exc:
                self._log_once(f"Failed to load BGM {self.bgm_path}: {exc}")
        else:
            self._log_once(f"Missing audio file: {self.bgm_path}")

    def play(self, name: str, volume: float = 1.0) -> None:
        if not self.enabled:
            return
        sound = self.sounds.get(name)
        if sound is None:
            return
        sound.set_volume(max(0.0, min(1.0, volume)))
        sound.play()


class GameResourcesMixin:
    def _build_cover_points(self, level_map: list[str]) -> list[tuple[float, float]]:
        cover_points: list[tuple[float, float]] = []
        for cell_y, row in enumerate(level_map):
            for cell_x, tile in enumerate(row):
                if tile != "0":
                    continue
                if not any(
                    not is_open_cell(cell_x + offset_x, cell_y + offset_y, level_map)
                    for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ):
                    continue
                cover_points.append(cell_center(cell_x, cell_y))
        return cover_points

    def _build_fallback_wall_texture(self, wall_type: str) -> pygame.Surface:
        surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        color_seed = max(1, sum(ord(char) for char in wall_type))
        base_color = (
            70 + (color_seed * 29) % 80,
            68 + (color_seed * 43) % 70,
            64 + (color_seed * 17) % 90,
        )
        accent_color = tuple(max(20, channel - 26) for channel in base_color)
        mortar_color = tuple(max(12, channel - 44) for channel in base_color)
        surface.fill(base_color)
        brick_w = 16 if wall_type == "1" else 12
        brick_h = 12 if wall_type == "1" else 16
        for row_index, top in enumerate(range(0, TILE_SIZE, brick_h)):
            x_offset = 0 if row_index % 2 == 0 else brick_w // 2
            for left in range(-x_offset, TILE_SIZE, brick_w):
                brick_rect = pygame.Rect(left, top, brick_w - 2, brick_h - 2)
                pygame.draw.rect(surface, accent_color, brick_rect)
                pygame.draw.rect(surface, mortar_color, brick_rect, width=1)
        for stripe_x in range(0, TILE_SIZE, 8):
            shade = 10 if (stripe_x // 8) % 2 == 0 else -8
            stripe_color = tuple(max(0, min(255, channel + shade)) for channel in base_color)
            pygame.draw.line(surface, stripe_color, (stripe_x, 0), (stripe_x, TILE_SIZE - 1))
        return surface.convert()

    def _load_wall_textures(self) -> dict[str, pygame.Surface]:
        textures_dir = self.project_root / "assets" / "textures"
        textures: dict[str, pygame.Surface] = {}
        if textures_dir.exists():
            for path in sorted(textures_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                stem = path.stem.lower()
                if not stem.startswith("wall_"):
                    continue
                suffix = stem.split("_", 1)[1]
                texture_key = str(int(suffix)) if suffix.isdigit() else suffix.lstrip("0") or "1"
                try:
                    surface = pygame.image.load(str(path)).convert()
                    textures[texture_key] = pygame.transform.scale(surface, (TILE_SIZE, TILE_SIZE))
                except pygame.error:
                    continue
        if "1" not in textures:
            textures["1"] = self._build_fallback_wall_texture("1")
        if "2" not in textures:
            textures["2"] = self._build_fallback_wall_texture("2")
        return textures

    def _wall_texture_for(self, wall_type: str) -> pygame.Surface:
        texture = self.wall_textures.get(wall_type)
        if texture is not None:
            return texture
        texture = self._build_fallback_wall_texture(wall_type)
        self.wall_textures[wall_type] = texture
        return texture

    def _build_fallback_enemy_surface(self, kind: str) -> pygame.Surface:
        definition = ENEMY_TYPES[kind]
        surface = pygame.Surface((96, 96), pygame.SRCALPHA)
        color = definition["color"]
        if kind == "health_pack":
            pygame.draw.rect(surface, color, (18, 18, 60, 60), border_radius=14)
            pygame.draw.rect(surface, (240, 240, 240), (40, 26, 16, 44), border_radius=6)
            pygame.draw.rect(surface, (240, 240, 240), (26, 40, 44, 16), border_radius=6)
            pygame.draw.rect(surface, (26, 34, 28), (18, 18, 60, 60), width=3, border_radius=14)
            return surface.convert_alpha()
        pygame.draw.ellipse(surface, color, (14, 16, 68, 64))
        pygame.draw.circle(surface, (255, 255, 255), (38, 42), 7)
        pygame.draw.circle(surface, (255, 255, 255), (58, 42), 7)
        pygame.draw.circle(surface, BG_COLOR, (38, 42), 3)
        pygame.draw.circle(surface, BG_COLOR, (58, 42), 3)
        pygame.draw.rect(surface, (245, 245, 245), (34, 62, 28, 8), border_radius=3)
        pygame.draw.rect(surface, color, (8, 58, 18, 24), border_radius=6)
        pygame.draw.rect(surface, color, (70, 58, 18, 24), border_radius=6)
        pygame.draw.rect(surface, color, (28, 76, 14, 16), border_radius=6)
        pygame.draw.rect(surface, color, (54, 76, 14, 16), border_radius=6)
        label = self.font.render(kind[0].upper(), True, HUD_TEXT_COLOR)
        surface.blit(label, (42 - label.get_width() // 2, 2))
        return surface.convert_alpha()

    def _load_enemy_surfaces(self) -> dict[str, pygame.Surface]:
        sprites_dir = self.project_root / "assets" / "sprites"
        surfaces: dict[str, pygame.Surface] = {}
        for kind in ENEMY_TYPES:
            asset_path = sprites_dir / f"{kind}.png"
            if asset_path.exists():
                try:
                    surfaces[kind] = pygame.image.load(str(asset_path)).convert_alpha()
                    continue
                except pygame.error:
                    pass
            surfaces[kind] = self._build_fallback_enemy_surface(kind)
        return surfaces

    def _tint_surface_red(self, source: pygame.Surface) -> pygame.Surface:
        tinted = source.copy()
        overlay = pygame.Surface(source.get_size(), pygame.SRCALPHA)
        overlay.fill((255, 72, 72, 0))
        tinted.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        shaded = pygame.Surface(source.get_size(), pygame.SRCALPHA)
        shaded.fill((220, 80, 80, 180))
        tinted.blit(shaded, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted

    def _build_fallback_weapon_surface(self) -> pygame.Surface:
        surface = pygame.Surface((320, 220), pygame.SRCALPHA)
        body = pygame.Rect(84, 74, 154, 78)
        grip = pygame.Rect(142, 132, 42, 68)
        barrel = pygame.Rect(210, 94, 72, 24)
        pygame.draw.rect(surface, (60, 62, 70), body, border_radius=12)
        pygame.draw.rect(surface, (38, 40, 46), grip, border_radius=10)
        pygame.draw.rect(surface, (88, 92, 104), barrel, border_radius=8)
        pygame.draw.rect(surface, (128, 134, 148), body, width=4, border_radius=12)
        pygame.draw.rect(surface, (150, 156, 168), barrel, width=3, border_radius=8)
        pygame.draw.rect(surface, (192, 140, 64), (96, 84, 46, 18), border_radius=6)
        return surface.convert_alpha()

    def _load_weapon_surface(self) -> pygame.Surface:
        weapon_path = self.project_root / "assets" / "weapons" / "pistol.png"
        if weapon_path.exists():
            try:
                return pygame.image.load(str(weapon_path)).convert_alpha()
            except pygame.error:
                pass
        return self._build_fallback_weapon_surface()

    def _with_wall_variants(self, level_map: list[str]) -> list[str]:
        variant_map: list[str] = []
        for cell_y, row in enumerate(level_map):
            cells = list(row)
            for cell_x, value in enumerate(cells):
                if value != "1":
                    continue
                if (cell_x * 3 + cell_y * 5) % 11 != 0:
                    continue
                if not any(
                    is_open_cell(cell_x + offset_x, cell_y + offset_y, level_map)
                    for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ):
                    continue
                cells[cell_x] = "2"
            variant_map.append("".join(cells))
        return variant_map
