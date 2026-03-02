from __future__ import annotations

import math

import pygame

from src.entities.enemy import Enemy, RLEnemy
from src.game.ui import AgentDebugData, UIData, build_agent_panel_lines, game_over_lines, menu_lines, pause_lines
from src.game.states import (
    INTENT_LABELS,
    PACK_DISTANCE_LABELS,
    RL_DIRECTION_LABELS,
    RL_DISTANCE_LABELS,
    RL_HP_LABELS,
)
from src.settings import (
    CROSSHAIR_COLOR,
    CROSSHAIR_SIZE,
    HEIGHT,
    HUD_PADDING,
    HUD_TEXT_COLOR,
    PLAYER_AGENT_RANGE_MAX,
    PLAYER_AGENT_RANGE_MIN,
    TILE_SIZE,
    WIDTH,
)


class HudMixin:
    def _build_agent_debug_data(self) -> AgentDebugData:
        return_live = self._current_agent_episode_return()
        target_type, target_dist, target_reason = self._agent_target_hint()
        decision_reason = self.agent_control.current_reason or target_reason
        target_ref = self.agent_control.target_lock.target_ref
        if isinstance(target_ref, Enemy):
            target_id = str(target_ref.entity_id)
            target_distance_tiles = self.agent_control.current_target_distance / TILE_SIZE
            target_distance_text = f"{target_distance_tiles:.2f}t"
        else:
            target_id = "none"
            target_distance_text = target_dist
        pack_dist_text = (
            f"{self.agent_control.current_pack_dist_tiles:.2f}t"
            if self.agent_control.current_pack_dist_tiles is not None
            else "N/A"
        )
        enemy_dist_text = (
            f"{self.agent_control.current_enemy_dist_tiles:.2f}t"
            if self.agent_control.current_enemy_dist_tiles is not None
            else "N/A"
        )
        return AgentDebugData(
            mode_label="EVAL" if self.player_q_agent.is_eval_mode else "TRAIN",
            epsilon=self.player_q_agent.epsilon,
            episode_id=self.agent_episode_id,
            episode_return=return_live,
            avg20=self._average_return_20(),
            reward_total=self.agent_reward_pos_total,
            penalty_total=self.agent_penalty_total,
            shoot_ready=self.agent_control.shoot_ready,
            has_los=self.agent_control.current_has_los,
            is_reloading=self.player.weapon.is_reloading,
            aim_error_degrees=math.degrees(self.agent_control.current_aim_error),
            aim_stable_time=self.agent_control.aim_stable_time,
            goal_type=self.agent_control.current_goal_type,
            target_id=target_id,
            target_distance_text=target_distance_text,
            retarget_in_seconds=self.agent_control.retarget_timer,
            emergency_switch=self.agent_control.emergency_retarget_this_tick,
            pack_score=self.agent_control.current_pack_score,
            enemy_score=self.agent_control.current_enemy_score,
            pack_distance_text=pack_dist_text,
            enemy_distance_text=enemy_dist_text,
            intent_label=self.agent_control.debug_intent_label,
            enemy_repulse=self.agent_control.enemy_repulse_active,
            near_wall=self.agent_control.near_wall,
            should_move=self.agent_control.should_move,
            bumped=self.agent_control.current_bumped,
            progress_tiles=self.agent_control.current_progress_tiles,
            stuck_time=self.agent_control.no_progress_time,
            recovering=self.agent_control.recover_time_left > 0.0,
            recovery_side_label=self.agent_control.recovery_side_label,
            target_type=target_type,
            reason=decision_reason,
            no_reason_timer=self.agent_control.no_reason_timer,
            no_reason_cooldown=self.agent_control.no_reason_cooldown,
        )

    def _build_ui_data(self) -> UIData:
        episode_label = None
        if self.game_mode == "arena":
            episode_label = f"EPISODE {self.agent_episode_id} | AVG {self.training_average_reward:.2f}"
        intermission_message = None
        if self.intermission_remaining > 0.0:
            intermission_message = f"Wave Cleared +{self.last_wave_bonus}  Next wave in {self.intermission_remaining:.1f}s"
        return UIData(
            score=self.score,
            wave=self.wave_number,
            episode_label=episode_label,
            hp=self.player_hp,
            kills=self.kill_count,
            ammo_in_mag=self.player.weapon.ammo_in_mag,
            magazine_size=self.player.weapon.magazine_size,
            is_reloading=self.player.weapon.is_reloading,
            reload_time_left=self.player.weapon.reload_time_left,
            feedback_messages=list(self.feedback_messages) if self.feedback_time_left > 0.0 and self.feedback_messages else [],
            intermission_message=intermission_message,
            fps=int(round(self.clock.get_fps())),
        )

    def _panel_height_for_lines(
        self,
        lines: list[str],
        *,
        primary_font: pygame.font.Font,
        secondary_font: pygame.font.Font,
    ) -> int:
        if not lines:
            return 0
        total_height = 0
        for index, _text in enumerate(lines):
            font = primary_font if index == 0 else secondary_font
            total_height += font.get_height()
            if index < len(lines) - 1:
                total_height += 8
        return total_height + 20

    def _draw_hud(self) -> None:
        ui_data = self._build_ui_data()
        top_lines = [f"SCORE {ui_data.score} | WAVE {ui_data.wave}"]
        if ui_data.episode_label is not None:
            top_lines.append(ui_data.episode_label)
        self._draw_hud_block(top_lines, anchor="top_center", primary_font=self.hud_font, secondary_font=self.font)

        left_lines = [f"HP {ui_data.hp}", f"KILLS {ui_data.kills}"]
        if ui_data.feedback_messages:
            left_lines.extend(ui_data.feedback_messages)
        self._draw_hud_block(left_lines, anchor="bottom_left", primary_font=self.hud_font, secondary_font=self.font)

        right_lines = [f"AMMO {ui_data.ammo_in_mag}/{ui_data.magazine_size}"]
        if ui_data.is_reloading:
            right_lines.append(f"RELOADING {ui_data.reload_time_left:.1f}s")
        elif ui_data.ammo_in_mag == 0:
            right_lines.append("EMPTY / RELOAD")
        self._draw_hud_block(right_lines, anchor="bottom_right", primary_font=self.hud_font, secondary_font=self.font)
        self._draw_hud_block([f"FPS {ui_data.fps}"], anchor="top_right", primary_font=self.font, secondary_font=self.font)
        if self.current_control is self.agent_control:
            agent_lines = build_agent_panel_lines(self._build_agent_debug_data())
            self._draw_hud_block(agent_lines, anchor="top_right_below", primary_font=self.font, secondary_font=self.font)
            self._draw_return_sparkline(agent_lines)

        if ui_data.intermission_message is not None:
            label = self.font.render(ui_data.intermission_message, True, HUD_TEXT_COLOR)
            self.screen.blit(label, (WIDTH // 2 - label.get_width() // 2, 56))

        if self.show_rl_debug and self.current_control is self.agent_control:
            self._draw_player_agent_debug()
            self._draw_rl_debug_panel()

        center_x = WIDTH // 2
        center_y = HEIGHT // 2
        pygame.draw.line(self.screen, CROSSHAIR_COLOR, (center_x - CROSSHAIR_SIZE, center_y), (center_x + CROSSHAIR_SIZE, center_y), width=2)
        pygame.draw.line(self.screen, CROSSHAIR_COLOR, (center_x, center_y - CROSSHAIR_SIZE), (center_x, center_y + CROSSHAIR_SIZE), width=2)
        self._draw_weapon_overlay()

    def _draw_hud_block(self, lines: list[str], *, anchor: str, primary_font: pygame.font.Font, secondary_font: pygame.font.Font) -> None:
        if not lines:
            return
        rendered: list[tuple[pygame.Surface, int, int]] = []
        max_width = 0
        total_height = 0
        for index, text in enumerate(lines):
            font = primary_font if index == 0 else secondary_font
            label = font.render(text, True, HUD_TEXT_COLOR)
            rendered.append((label, label.get_width(), label.get_height()))
            max_width = max(max_width, label.get_width())
            total_height += label.get_height()
            if index < len(lines) - 1:
                total_height += 8
        panel_rect = pygame.Rect(0, 0, max_width + 28, total_height + 20)
        if anchor == "top_center":
            panel_rect.midtop = (WIDTH // 2, HUD_PADDING)
        elif anchor == "top_right":
            panel_rect.topright = (WIDTH - HUD_PADDING, HUD_PADDING)
        elif anchor == "top_right_below":
            panel_rect.topright = (WIDTH - HUD_PADDING, HUD_PADDING + 46)
        elif anchor == "bottom_left":
            panel_rect.bottomleft = (HUD_PADDING, HEIGHT - HUD_PADDING)
        else:
            panel_rect.bottomright = (WIDTH - HUD_PADDING, HEIGHT - HUD_PADDING)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel.fill((10, 12, 18, 168))
        pygame.draw.rect(panel, (70, 90, 112, 220), panel.get_rect(), width=2, border_radius=10)
        self.screen.blit(panel, panel_rect.topleft)
        y = panel_rect.y + 10
        for label, label_width, label_height in rendered:
            if anchor in ("bottom_right", "top_right", "top_right_below"):
                x = panel_rect.right - 14 - label_width
            elif anchor == "top_center":
                x = panel_rect.centerx - label_width // 2
            else:
                x = panel_rect.x + 14
            self.screen.blit(label, (x, y))
            y += label_height + 8

    def _draw_return_sparkline(self, agent_panel_lines: list[str]) -> None:
        history = self.agent_episode_returns_history[-30:]
        if len(history) < 2:
            return
        chart_width = 220
        chart_height = 54
        panel_height = self._panel_height_for_lines(
            agent_panel_lines,
            primary_font=self.font,
            secondary_font=self.font,
        )
        chart_top = HUD_PADDING + 46 + panel_height + 8
        chart_rect = pygame.Rect(WIDTH - HUD_PADDING - chart_width, chart_top, chart_width, chart_height)
        panel = pygame.Surface(chart_rect.size, pygame.SRCALPHA)
        panel.fill((10, 12, 18, 148))
        pygame.draw.rect(panel, (70, 90, 112, 210), panel.get_rect(), width=1, border_radius=8)
        self.screen.blit(panel, chart_rect.topleft)
        min_value = min(history)
        max_value = max(history)
        if math.isclose(max_value, min_value, rel_tol=0.0, abs_tol=1e-9):
            mid_y = chart_rect.y + chart_height // 2
            pygame.draw.line(self.screen, HUD_TEXT_COLOR, (chart_rect.x + 10, mid_y), (chart_rect.right - 10, mid_y), width=2)
        else:
            value_span = max_value - min_value
            points: list[tuple[int, int]] = []
            for index, value in enumerate(history):
                x = chart_rect.x + 10 + int(index * (chart_width - 20) / max(1, len(history) - 1))
                normalized = (value - min_value) / value_span
                y = chart_rect.bottom - 10 - int(normalized * (chart_height - 20))
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, HUD_TEXT_COLOR, False, points, width=2)
        label = self.font.render("RET (30)", True, HUD_TEXT_COLOR)
        self.screen.blit(label, (chart_rect.x + 8, chart_rect.y + 4))
        min_label = self.font.render(f"{min_value:.1f}", True, HUD_TEXT_COLOR)
        max_label = self.font.render(f"{max_value:.1f}", True, HUD_TEXT_COLOR)
        self.screen.blit(min_label, (chart_rect.x + 8, chart_rect.bottom - min_label.get_height() - 2))
        self.screen.blit(max_label, (chart_rect.right - max_label.get_width() - 8, chart_rect.y + 4))

    def _draw_game_over_overlay(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 14, 200))
        self.screen.blit(overlay, (0, 0))
        lines = game_over_lines(self.score, self.high_score)
        center_y = HEIGHT // 2 - 72
        for index, (text, size_tag) in enumerate(lines):
            font = self.hud_font if size_tag == "large" else self.font
            label = font.render(text, True, HUD_TEXT_COLOR)
            self.screen.blit(label, (WIDTH // 2 - label.get_width() // 2, center_y + index * 42))

    def _draw_pause_overlay(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 14, 180))
        self.screen.blit(overlay, (0, 0))
        lines = pause_lines()
        center_y = HEIGHT // 2 - 72
        for index, (text, size_tag) in enumerate(lines):
            font = self.hud_font if size_tag == "large" else self.font
            label = font.render(text, True, HUD_TEXT_COLOR)
            self.screen.blit(label, (WIDTH // 2 - label.get_width() // 2, center_y + index * 42))

    def _draw_menu_overlay(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 12, 18, 235))
        self.screen.blit(overlay, (0, 0))
        lines = menu_lines(self.high_score)
        center_y = HEIGHT // 2 - 110
        for index, (text, size_tag) in enumerate(lines):
            font = self.hud_font if size_tag == "large" else self.font
            label = font.render(text, True, HUD_TEXT_COLOR)
            self.screen.blit(label, (WIDTH // 2 - label.get_width() // 2, center_y + index * 48))

    def _draw_player_agent_debug(self) -> None:
        obs = self.player_q_agent.current_obs
        locked_target = self.agent_control.target_lock.target_ref
        target_enemy = locked_target if isinstance(locked_target, Enemy) and locked_target in self.enemies else None
        damage_text = "--" if math.isinf(self.agent_control.last_damage_time) else f"{self.agent_control.last_damage_time:.2f}"
        if obs is None:
            state_text = "state n/a"
        else:
            state_text = (
                f"s={RL_DISTANCE_LABELS.get(obs[0], obs[0])}/"
                f"los={obs[1]}/"
                f"dir={RL_DIRECTION_LABELS.get(obs[2], obs[2])}/"
                f"bump={obs[3]}/"
                f"hp={RL_HP_LABELS.get(obs[4], obs[4])}/"
                f"uf={obs[5]}/"
                f"pack={obs[6]}/"
                f"pack_d={PACK_DISTANCE_LABELS.get(obs[7], obs[7])}"
            )
        action_text = INTENT_LABELS.get(self.agent_control.last_action, "CHASE")
        mode_label = "Train" if self.player_q_agent.training_enabled else "Eval"
        target_text = f"{target_enemy.entity_id}:{target_enemy.kind}" if target_enemy is not None else "none"
        retarget_text = "--" if math.isinf(self.agent_control.last_retarget_time) else f"{self.agent_control.last_retarget_time:.2f}"
        move_len = math.hypot(self.agent_control.current_move_vector[0], self.agent_control.current_move_vector[1])
        debug_text = (
            f"PlayerAgent {mode_label} {state_text}  tgt={target_text}  "
            f"phase={self.agent_control.current_phase}  "
            f"intent={INTENT_LABELS.get(self.agent_control.current_effective_intent, action_text)}  "
            f"los={int(self.agent_control.current_has_los)}  "
            f"dist={self.agent_control.current_target_distance / TILE_SIZE:.1f}t  "
            f"band=({PLAYER_AGENT_RANGE_MIN / TILE_SIZE:.0f},{PLAYER_AGENT_RANGE_MAX / TILE_SIZE:.0f})  "
            f"a={action_text}  "
            f"lock_t={self.agent_control.target_lock.lock_time_left:.2f}  "
            f"retarget_t={retarget_text}  "
            f"retarget_in={self.agent_control.retarget_timer:.2f}  "
            f"emergency={int(self.agent_control.emergency_retarget_this_tick)}  "
            f"nav_cell={self.agent_control.current_nav_target_cell}  "
            f"has_path={int(self.agent_control.has_path)}  "
            f"wp={self.agent_control.next_waypoint}  "
            f"move=({self.agent_control.current_move_vector[0]:.1f},{self.agent_control.current_move_vector[1]:.1f})  "
            f"move_len={move_len:.1f}  "
            f"yaw={self.agent_control.current_yaw_source}  "
            f"yaw_deg={math.degrees(self.player.angle):.1f}  "
            f"target_angle_deg={math.degrees(self.agent_control.current_target_angle):.1f}  "
            f"aim_error_deg={math.degrees(self.agent_control.current_aim_error):+.1f}  "
            f"aim_stable_time={self.agent_control.aim_stable_time:.2f}  "
            f"ready={int(self.agent_control.shoot_ready)}  "
            f"blocked={self.agent_control.blocked_reason}  "
            f"ammo={self.player.weapon.ammo_in_mag}  "
            f"reloading={int(self.player.weapon.is_reloading)}  "
            f"reload_t={self.player.weapon.reload_time_left:.2f}  "
            f"reason={self.agent_control.current_reason}  "
            f"no_reason_t={self.agent_control.no_reason_timer:.2f}  "
            f"nr_cd={self.agent_control.no_reason_cooldown:.2f}  "
            f"near_wall={int(self.agent_control.near_wall)}  "
            f"should_move={int(self.agent_control.should_move)}  "
            f"progress={self.agent_control.current_progress_tiles:.2f}t  "
            f"bumped={int(self.agent_control.current_bumped)}  "
            f"enemy_repulse={int(self.agent_control.enemy_repulse_active)}  "
            f"intent={self.agent_control.debug_intent_label}  "
            f"recover={int(self.agent_control.recover_time_left > 0.0)}  "
            f"side={self.agent_control.recovery_side_label}  "
            f"did_shoot={int(self.agent_control.shoot_triggered_this_tick)}  "
            f"intent_t={self.agent_control.action_time_left:.2f}  "
            f"dmg_t={damage_text}  "
            f"patrol_t={self.agent_control.force_patrol_time_left:.2f}  "
            f"stuck_t={self.agent_control.no_progress_time:.2f}  "
            f"lock={int(self.agent_control.action_locked)}  "
            f"pack_reward={self.agent_control.last_pack_reward:+.2f}  "
            f"r={self.player_q_agent.last_reward_delta:+.2f}  "
            f"eps={self.player_q_agent.epsilon:.3f}"
        )
        label = self.font.render(debug_text, True, HUD_TEXT_COLOR)
        self.screen.blit(label, (HUD_PADDING, HEIGHT - 24))

    def _draw_rl_debug_panel(self) -> None:
        rl_enemy = next((enemy for enemy in self.enemies if isinstance(enemy, RLEnemy)), None)
        panel_width = 340
        panel_height = 292
        panel_x = WIDTH - panel_width - HUD_PADDING
        panel_y = HUD_PADDING
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((10, 12, 18, 210))
        pygame.draw.rect(panel_surface, (70, 90, 112, 235), panel_surface.get_rect(), width=2, border_radius=10)
        self.screen.blit(panel_surface, (panel_x, panel_y))
        lines = [f"RL Debug  epsilon {self.rl_agent.epsilon:.3f}", f"Wave {self.wave_number}"]
        if rl_enemy is None:
            lines.append("RLEnemy unavailable")
        else:
            obs = rl_enemy.current_obs
            obs_text = "n/a"
            if obs is not None:
                obs_text = (
                    f"{RL_DISTANCE_LABELS.get(obs[0], obs[0])}, "
                    f"los={obs[1]}, "
                    f"hp={RL_HP_LABELS.get(obs[2], obs[2])}, "
                    f"dir={RL_DIRECTION_LABELS.get(obs[3], obs[3])}, "
                    f"uf={obs[4]}"
                )
            target_text = str(rl_enemy.target_lock.target_key or "none")
            damage_text = "--" if math.isinf(rl_enemy.last_damage_time) else f"{rl_enemy.last_damage_time:.2f}"
            move_len = math.hypot(rl_enemy.current_move_vector[0], rl_enemy.current_move_vector[1])
            aggro_tiles = 0.5 * max(len(self.map), len(self.map[0]))
            lines.extend((
                f"Intent {INTENT_LABELS.get(rl_enemy.current_effective_intent, 'CHASE')}",
                f"LOS {int(rl_enemy.current_has_los)}  Dist {rl_enemy.current_target_distance / TILE_SIZE:.1f}t",
                f"Aggro {aggro_tiles:.1f}t",
                f"Target {target_text}",
                f"Target lock {rl_enemy.target_lock.lock_time_left:.2f}",
                f"Nav {rl_enemy.current_nav_target_cell}",
                f"Has path {int(rl_enemy.has_path)}  WP {rl_enemy.next_waypoint}",
                f"Move len {move_len:.1f}",
                f"Yaw {rl_enemy.current_yaw_source}",
                f"Intent time {rl_enemy.action_time_left:.2f}",
                f"Last damage {damage_text}",
                f"Locked {int(rl_enemy.action_locked)}",
                f"State {obs_text}",
                f"Last reward {rl_enemy.last_reward_delta:+.2f}",
                f"Pending reward {rl_enemy.pending_reward:+.2f}",
                f"HP {rl_enemy.hp}/{rl_enemy.max_hp}",
            ))
        for index, line in enumerate(lines):
            label = self.font.render(line, True, HUD_TEXT_COLOR)
            self.screen.blit(label, (panel_x + 14, panel_y + 12 + index * 24))
