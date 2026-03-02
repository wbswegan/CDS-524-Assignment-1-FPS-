from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pygame

from src.ai.agent_control import AgentControl, HumanControl
from src.ai.geometry import (
    banded_intent as _banded_intent,
    can_move_to as _can_move_to,
    cell_center as _cell_center,
    distance_bin as _distance_bin,
    flank_point as _flank_point,
    has_line_of_sight as _has_line_of_sight,
    hp_bin as _hp_bin,
    is_open_cell as _is_open_cell,
    is_wall as _is_wall,
    move_with_slide as _move_with_slide,
    normalize_vector as _normalize_vector,
    pack_distance_bin as _pack_distance_bin,
    project_point as _project_point,
    relative_direction_bin as _relative_direction_bin,
    target_angle_and_error as _target_angle_and_error,
    turn_toward_angle as _turn_toward_angle,
    world_to_cell as _world_to_cell,
    wrap_to_pi as _wrap_to_pi
)
from src.ai.pathfinding import (
    build_fallback_path_toward_target as _build_fallback_path_toward_target,
    build_path as _build_path,
    find_free_spawn_position as _find_free_spawn_position,
    reachable_open_cells_within_radius as _reachable_open_cells_within_radius
)
from src.entities.enemy import Enemy, HealthPackTarget, RLEnemy
from src.entities.player import Player
from src.game.hud import HudMixin
from src.game.minimap import MinimapMixin
from src.game.resources import AudioManager, GameResourcesMixin
from src.game.states import (
    APP_STATE_GAME_OVER,
    APP_STATE_MENU,
    APP_STATE_PAUSED,
    APP_STATE_PLAY_MAIN,
    APP_STATE_PLAY_TRAINING,
    INTENT_CHASE,
    INTENT_ENGAGE,
    INTENT_EVADE,
    INTENT_LABELS,
    INTENT_REPOSITION,
    MAIN_STATE_GAME_OVER,
    MAIN_STATE_PLAYING,
    PACK_DISTANCE_LABELS,
    RL_DIRECTION_LABELS,
    RL_DISTANCE_LABELS,
    RL_HP_LABELS,
    SCRIPT_STATE_CHASE,
    SCRIPT_STATE_COLORS,
    SCRIPT_STATE_ENGAGE,
    SCRIPT_STATE_PATROL,
    SCRIPT_STATE_RETURN,
    SCRIPT_STATE_SEARCH
)
from src.game.world_render import WorldRenderMixin
from src.logging import EpisodeStats, RunLogger
from src.rl.features import build_player_agent_obs
from src.rl.policy import enemy_target_score, pack_target_score
from src.rl.player_agent import PlayerQLearningAgent
from src.rl_agent import QLearningAgent
from src.settings import (
    AGENT_ACTION_INTERVAL_SECONDS,
    AIM_DEADZONE_DEGREES,
    AIM_MAX_DEGREES_PER_SECOND,
    AIM_PROPORTIONAL_GAIN,
    BG_COLOR,
    BGM_VOLUME,
    CEILING_COLOR,
    CRITICAL_HP_RATIO,
    CROSSHAIR_COLOR,
    CROSSHAIR_SIZE,
    EMERGENCY_DIST_TILES,
    ENGAGE_IDEAL_T,
    ENGAGE_MAX_T,
    ENGAGE_MIN_T,
    ENEMY_AGGRO_RANGE,
    ENEMY_REPULSE_RADIUS_T,
    ENEMY_REPULSE_WEIGHT,
    ENEMY_PATH_RECALC_INTERVAL,
    ENEMY_SPAWN_CELLS,
    ENEMY_TYPES,
    FLOOR_COLOR,
    FPS,
    FOV,
    GRID_MAP,
    HALF_FOV,
    HEIGHT,
    HEAL_FEEDBACK_SECONDS,
    HEALTH_PACK_HEAL,
    HEALTH_PACK_HP,
    HEALTH_PACK_MIN_SPAWN_DISTANCE_TILES,
    HEALTH_PACK_SCORE,
    HIT_FLASH_SECONDS,
    HUD_PADDING,
    HUD_TEXT_COLOR,
    KILL_HEAL_NORMAL,
    KILL_HEAL_RL,
    MAX_DEPTH,
    MAGAZINE_SIZE,
    MINIMAP_MAX_HEIGHT,
    MINIMAP_MAX_WIDTH,
    MINIMAP_PADDING,
    MOVE_FILTER_DEADZONE,
    MOVE_FILTER_RESPONSE,
    MOUSE_SENSITIVITY,
    NO_REASON_GRACE_SECONDS,
    NO_REASON_PENALTY_COOLDOWN,
    PACK_APPROACH_CLAMP,
    PACK_APPROACH_K,
    PLAYER_COLOR,
    PLAYER_DIR_COLOR,
    PLAYER_MAX_HP,
    PLAYER_RADIUS,
    PLAYER_AGENT_REWARD_BUMP,
    PLAYER_AGENT_REWARD_AIM,
    PLAYER_AGENT_REWARD_BREAK_LOS,
    PLAYER_AGENT_REWARD_DAMAGE,
    PLAYER_AGENT_REWARD_DEATH,
    PLAYER_AGENT_REWARD_HIT,
    PLAYER_AGENT_REWARD_KILL,
    PLAYER_AGENT_REWARD_PACK_HEAL_SCALE,
    PLAYER_AGENT_REWARD_PACK_KILL,
    PLAYER_AGENT_REWARD_PACK_LOW_HP_BONUS,
    PLAYER_AGENT_AIM_THRESHOLD,
    PLAYER_AGENT_AIM_STABLE_SECONDS,
    PLAYER_AGENT_PENALTY_NO_REASON,
    PLAYER_AGENT_RANGE_MAX,
    PLAYER_AGENT_RANGE_MIN,
    PLAYER_AGENT_RANGE_PREFERRED,
    PLAYER_AGENT_PATROL_RADIUS_TILES,
    PLAYER_AGENT_PATROL_REFRESH_SECONDS,
    PLAYER_AGENT_RETARGET_SECONDS,
    PLAYER_AGENT_REWARD_SHAPING,
    PLAYER_AGENT_STRAFE_BURST_SECONDS,
    PLAYER_START,
    PLAYER_SPEED,
    RAY_STRIDE,
    RAY_COLOR,
    RELOAD_SECONDS,
    RETARGET_HYSTERESIS_TILES,
    RETARGET_INTERVAL_SEC,
    RL_DECISION_TICKS,
    RL_ENEMY_DAMAGE_CAP,
    RL_ENEMY_DAMAGE_SCALE,
    RL_ENEMY_HP_CAP,
    RL_ENEMY_RANGE_MAX,
    RL_ENEMY_RANGE_MIN,
    RL_ENEMY_SPEED_CAP,
    RL_REWARD_BREAK_LOS,
    RL_REWARD_DEATH,
    RL_REWARD_GOT_HIT,
    RL_REWARD_HIT_PLAYER,
    RL_REWARD_SHAPING,
    RL_REWARD_WALL_BUMP,
    RL_WAVE_EPSILON_START,
    RL_WAVE_EPSILON_STEP,
    SCRIPT_ALERT_HOLD_SECONDS,
    SCRIPT_HEAR_RANGE_FACTOR,
    SCRIPT_LOST_TARGET_SECONDS,
    SCRIPT_PATROL_RADIUS_MAX_TILES,
    SCRIPT_PATROL_RADIUS_MIN_TILES,
    SCRIPT_PATROL_WAYPOINTS_MAX,
    SCRIPT_PATROL_WAYPOINTS_MIN,
    SCRIPT_SIGHT_RANGE_TILES,
    SFX_VOLUME_EMPTY,
    SFX_VOLUME_ENEMY_DIE,
    SFX_VOLUME_HIT,
    SFX_VOLUME_PLAYER_HURT,
    SFX_VOLUME_RELOAD,
    SFX_VOLUME_SHOOT,
    SFX_VOLUME_UI_SELECT,
    SMOOTH_ALPHA,
    SPRITE_SCALE,
    STUCK_GRACE_SEC,
    STUCK_MIN_PROGRESS_TILES,
    STUCK_RECOVER_SEC,
    TILE_SIZE,
    TITLE,
    TRAINING_ARENA_ENEMY_CELL,
    TRAINING_ARENA_ENEMY_CELLS,
    TRAINING_ARENA_ENEMY_KINDS,
    TRAINING_ARENA_MAP,
    TRAINING_ARENA_PLAYER_ANGLES,
    TRAINING_ARENA_PLAYER_START,
    INTENT_HOLD_SECONDS,
    TARGET_SCORE_HYST,
    UNDER_FIRE_WINDOW_SECONDS,
    WALL_COLOR,
    WALL_REPULSE_RADIUS_TILES,
    WALL_REPULSE_WEIGHT,
    WALL_SHADE_COLOR,
    W_ENEMY_CLOSE,
    W_ENEMY_EMERGENCY,
    WAVE_BASE_COUNT,
    WAVE_CLEAR_BONUS_BASE,
    WAVE_COUNT_GROWTH,
    WAVE_HP_SCALE,
    WAVE_INTERMISSION_SECONDS,
    WAVE_SPEED_SCALE,
    WEAPON_DAMAGE,
    W_PACK_BASE,
    W_PACK_CLOSE,
    W_PACK_LOWHP,
    WIDTH
)


class Game(GameResourcesMixin, WorldRenderMixin, HudMixin, MinimapMixin):
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.hud_font = pygame.font.SysFont("consolas", 28)
        self.running = True
        self.project_root = Path.cwd()
        self.map = GRID_MAP
        self.player = Player(*PLAYER_START)
        self.player_hp = PLAYER_MAX_HP
        self.player_prev_hp = PLAYER_MAX_HP
        self.player_bumped = False
        self.player_under_fire_timer = 0.0
        self.player_prev_threat_los = False
        self.skip_player_shaping_once = True
        self.score = 0
        self.kill_count = 0
        self.game_mode = "main"
        self.wave_number = 1
        self.intermission_remaining = 0.0
        self.last_wave_bonus = 0
        self.training_episode_count = 0
        self.training_total_reward = 0.0
        self.training_episode_reward = 0.0
        self.training_average_reward = 0.0
        self.training_run_id = ""
        self.training_score_total = 0
        self.training_kills_total = 0
        self.zero_reward_episode_streak = 0
        self.episode_stats = EpisodeStats()
        self.agent_reward_pos_total = 0.0
        self.agent_penalty_total = 0.0
        self.agent_episode_id = 1
        self.agent_last_episode_return = 0.0
        self.agent_episode_returns_history: list[float] = []
        self.agent_episode_steps = 0
        self.agent_episode_epsilon_used = 0.0
        self.training_log_path = self.project_root / "logs" / "training_log.csv"
        self.run_logger = RunLogger(self.training_log_path)
        self.feedback_messages: list[str] = []
        self.feedback_time_left = 0.0
        self.app_state = APP_STATE_MENU
        self.current_play_state = APP_STATE_PLAY_MAIN
        self.paused_return_state = APP_STATE_PLAY_MAIN
        self.main_state = MAIN_STATE_PLAYING
        self.high_score_path = Path("highscore.json")
        self.high_score = self._load_high_score()
        self.human_control = HumanControl()
        self.player_agent_path = Path("models/player_q_v2.json")
        self.player_q_agent = PlayerQLearningAgent(self.player_agent_path)
        self.agent_control = AgentControl(self.player_q_agent)
        self.current_control: HumanControl | AgentControl = self.human_control
        self.mouse_captured = False
        self.show_path_debug = False
        self.show_rl_debug = False
        self.show_script_ai_debug = False
        self.random = random.Random()
        self.next_entity_id = 1
        self.rl_agent_path = Path("models/rl_enemy_q.json")
        self.rl_agent = self._load_rl_agent()
        self.audio = AudioManager(self.project_root)
        self.proj_plane_dist = (WIDTH / 2) / math.tan(HALF_FOV)
        self.ray_hits: list[tuple[float, float]] = []
        self.wall_depths = [MAX_DEPTH] * WIDTH
        self.wall_textures = self._load_wall_textures()
        self.enemy_surfaces = self._load_enemy_surfaces()
        self.enemy_hit_surfaces = {kind: self._tint_surface_red(surface) for kind, surface in self.enemy_surfaces.items()}
        self.weapon_surface = self._load_weapon_surface()
        self.weapon_bob_phase = 0.0
        self.weapon_bob_amount = 0.0
        self.weapon_recoil_time_left = 0.0
        self.weapon_recoil_duration = 0.08
        self.player_prev_draw_pos = (self.player.x, self.player.y)
        self.cover_points: list[tuple[float, float]] = []
        self.enemies: list[Enemy] = []
        self._set_level_map(GRID_MAP)
        self._set_mouse_capture(False)

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
            self._update_title()

        self.player_q_agent.save()
        self._save_rl_agent()
        pygame.quit()

    def run_training_episodes(self, episode_count: int, *, render_every: int = 0) -> None:
        if episode_count <= 0:
            return
        self.player_q_agent.set_eval_mode(False)
        self.player_q_agent.set_training_enabled(True)
        self._start_training_arena()
        step_index = 0
        fixed_dt = 1.0 / FPS

        while self.running and self.training_episode_count < episode_count:
            pygame.event.pump()
            self._update(fixed_dt)
            step_index += 1
            if render_every > 0 and step_index % render_every == 0:
                self._draw()
                self._update_title()

        self.player_q_agent.save()
        self._save_rl_agent()
        self._print_training_sanity_summary()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.app_state == APP_STATE_MENU:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                    self.audio.play("ui_select", SFX_VOLUME_UI_SELECT)
                    self._start_main_mode()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                    self.audio.play("ui_select", SFX_VOLUME_UI_SELECT)
                    self._start_training_arena()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                continue
            elif self.app_state == APP_STATE_PAUSED:
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    self._resume_from_pause()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    self._return_to_menu()
                continue
            elif self.app_state == APP_STATE_GAME_OVER:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    self._restart_current_mode()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    self._return_to_menu()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                continue
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._pause_game()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self._set_mouse_capture(not self.mouse_captured)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F2:
                self._set_control_mode(self.human_control)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                self._set_control_mode(self.agent_control)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F4:
                if self.current_control is self.agent_control:
                    self._set_control_mode(self.human_control)
                else:
                    self._set_control_mode(self.agent_control)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F5:
                self.player_q_agent.set_training_enabled(
                    not self.player_q_agent.training_enabled
                )
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F8:
                self.player_q_agent.set_eval_mode(not self.player_q_agent.is_eval_mode)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._request_player_reload()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F7:
                self.show_rl_debug = not self.show_rl_debug
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F6:
                self.show_script_ai_debug = not self.show_script_ai_debug
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.show_path_debug = not self.show_path_debug
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.current_control is self.human_control
            ):
                self.request_player_shoot(agent_request=False)

    def _update(self, dt: float) -> None:
        if self.app_state in (APP_STATE_MENU, APP_STATE_PAUSED, APP_STATE_GAME_OVER):
            self.player_prev_hp = self.player_hp
            return

        self.player_under_fire_timer = max(0.0, self.player_under_fire_timer - dt)
        self.feedback_time_left = max(0.0, self.feedback_time_left - dt)
        self.player.update_weapon(dt)
        self.weapon_recoil_time_left = max(0.0, self.weapon_recoil_time_left - dt)
        self.agent_control.shoot_triggered_this_tick = False
        self.current_control.apply(self, dt)
        if self.current_control is self.agent_control:
            self.episode_stats.record_step()
            self._sync_episode_stats_mirrors()
        move_delta = math.hypot(self.player.x - self.player_prev_draw_pos[0], self.player.y - self.player_prev_draw_pos[1])
        move_strength = min(1.0, move_delta / max(1.0, PLAYER_SPEED * max(dt, 1e-6)))
        if move_strength > 0.01:
            self.weapon_bob_phase = (self.weapon_bob_phase + dt * 8.0 * (0.45 + move_strength)) % math.tau
        self.weapon_bob_amount += (move_strength - self.weapon_bob_amount) * min(1.0, dt * 8.0)
        self.player_prev_draw_pos = (self.player.x, self.player.y)
        if self.current_control is self.agent_control and self.player_bumped:
            self._add_player_agent_reward(PLAYER_AGENT_REWARD_BUMP)
        if self.game_mode == "arena":
            self._update_enemies(dt)
            self._update_player_agent_shaping()
            if self.player_hp <= 0:
                self._add_player_agent_reward(PLAYER_AGENT_REWARD_DEATH)
                self._finish_training_episode(False)
            elif not self._hostile_enemies():
                self._finish_training_episode(True)
        else:
            if self.intermission_remaining > 0.0:
                self.intermission_remaining = max(0.0, self.intermission_remaining - dt)
                if self.intermission_remaining == 0.0:
                    self.wave_number += 1
                    self.enemies = self._spawn_wave(self.wave_number)
            else:
                self._update_enemies(dt)
                if not self._hostile_enemies():
                    self._start_wave_intermission()
            self._update_player_agent_shaping()
        if self.game_mode == "main":
            if self.player_hp <= 0:
                self._add_player_agent_reward(PLAYER_AGENT_REWARD_DEATH)
                self._enter_game_over()
        self.player_prev_hp = self.player_hp

    def _draw(self) -> None:
        if self.app_state == APP_STATE_MENU:
            self.screen.fill(BG_COLOR)
            self._draw_menu_overlay()
            pygame.display.flip()
            return
        self._draw_world()
        self._draw_minimap()
        self._draw_hud()
        if self.app_state == APP_STATE_PAUSED:
            self._draw_pause_overlay()
        if self.app_state == APP_STATE_GAME_OVER:
            self._draw_game_over_overlay()
        pygame.display.flip()

    def _update_title(self) -> None:
        fps = self.clock.get_fps()
        capture_state = "LOCKED" if self.mouse_captured else "FREE"
        pygame.display.set_caption(f"{TITLE} | FPS: {fps:.1f} | Mouse: {capture_state}")

    def _load_rl_agent(self) -> QLearningAgent:
        if self.rl_agent_path.exists():
            try:
                agent = QLearningAgent.load(self.rl_agent_path, seed=7)
                if agent.num_actions == 4:
                    return agent
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return QLearningAgent(
            4,
            alpha=0.12,
            gamma=0.92,
            epsilon=0.9,
            epsilon_decay=0.999,
            epsilon_min=0.08,
            seed=7,
        )

    def _save_rl_agent(self) -> None:
        try:
            self.rl_agent.save(self.rl_agent_path)
        except OSError:
            pass

    def _load_high_score(self) -> int:
        if self.high_score_path.exists():
            try:
                payload = json.loads(self.high_score_path.read_text(encoding="utf-8"))
                return max(0, int(payload.get("high_score", 0)))
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                return 0
        return 0

    def _save_high_score(self) -> None:
        try:
            self.high_score_path.write_text(
                json.dumps({"high_score": self.high_score}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _show_feedback(self, *messages: str) -> None:
        self.feedback_messages = [message for message in messages if message]
        self.feedback_time_left = HEAL_FEEDBACK_SECONDS

    def _sync_episode_stats_mirrors(self) -> None:
        self.agent_reward_pos_total = self.episode_stats.pos_reward
        self.agent_penalty_total = self.episode_stats.penalty
        self.agent_episode_steps = self.episode_stats.steps
        self.agent_episode_epsilon_used = self.episode_stats.eps_used

    def _ensure_run_logger(self) -> None:
        if self.run_logger.path != self.training_log_path:
            self.run_logger = RunLogger(self.training_log_path)

    def _reset_agent_reward_monitor(self, *, reset_history: bool = False) -> None:
        self.episode_stats.begin(episode_id=self.agent_episode_id, eps_used=0.0)
        self._sync_episode_stats_mirrors()
        self.training_episode_reward = 0.0
        if reset_history:
            self.agent_episode_id = 1
            self.agent_last_episode_return = 0.0
            self.agent_episode_returns_history = []
            self.zero_reward_episode_streak = 0

    def _new_training_run_id(self) -> str:
        return RunLogger.new_run_id()

    def _finish_player_agent_learning_only(self) -> None:
        self.player_q_agent.end_episode()

    def _print_training_sanity_summary(self) -> None:
        self._ensure_run_logger()
        selected_run_id, rows = self.run_logger.latest_run_rows()
        if selected_run_id is None or not rows:
            print("sanity summary: no training rows found")
            return
        first_eps = [row["epsilon"] for row in rows[:3]]
        last_eps = [row["epsilon"] for row in rows[-3:]]
        print(f"sanity summary run_id={selected_run_id}")
        print(f"first 3 eps: {first_eps}")
        print(f"last 3 eps: {last_eps}")
        print("last 5 csv rows:")
        for row in rows[-5:]:
            print(
                f"{row['run_id']},{row['episode_id']},{row['mode']},{row['epsilon']},"
                f"{row['return']},{row['pos_reward']},{row['penalty']},{row['steps']},"
                f"{row['kills']},{row['score_delta']},{row['hp_end']}"
            )

    def _current_agent_episode_return(self) -> float:
        return self.episode_stats.episode_return

    def _finalize_agent_episode_metrics(self) -> float | None:
        if (
            self.player_q_agent.current_state is None
            and self.episode_stats.pos_reward == 0.0
            and self.episode_stats.penalty == 0.0
        ):
            return None
        self.episode_stats.finish(hp_end=self.player_hp)
        pos_total = self.episode_stats.pos_reward
        penalty_total = self.episode_stats.penalty
        steps_total = self.episode_stats.steps
        score_delta = self.episode_stats.score_delta if self.current_play_state == APP_STATE_PLAY_TRAINING else self.score
        kills_delta = self.episode_stats.kills if self.current_play_state == APP_STATE_PLAY_TRAINING else self.kill_count
        hp_end = self.player_hp
        mode_label = "EVAL" if self.player_q_agent.is_eval_mode else "TRAIN"
        epsilon_value = 0.0 if self.player_q_agent.is_eval_mode else self.episode_stats.eps_used
        return_final = pos_total - penalty_total
        self.agent_last_episode_return = return_final
        if self.current_play_state == APP_STATE_PLAY_TRAINING:
            print(
                f"EP end: id={self.agent_episode_id}, eps_logged={epsilon_value:.4f}, "
                f"eps_next={0.0 if self.player_q_agent.is_eval_mode else max(self.player_q_agent.epsilon_min_value, self.player_q_agent.training_epsilon * self.player_q_agent.epsilon_decay_value):.4f}"
            )
            if self.agent_episode_id % 20 == 0 or self.agent_episode_id == 1:
                print(
                    "episode_end",
                    f"episode_id={self.agent_episode_id}",
                    f"steps={steps_total}",
                    f"pos_total={pos_total:.4f}",
                    f"penalty_total={penalty_total:.4f}",
                    f"return_final={return_final:.4f}",
                    f"mode={mode_label}",
                    f"epsilon={epsilon_value:.4f}",
                )
            if steps_total < 10:
                print(
                    f"Warning: short episode logged (episode_id={self.agent_episode_id}, steps={steps_total}). "
                    "Check episode termination conditions."
                )
            self._ensure_run_logger()
            self.run_logger.append_episode(
                run_id=self.training_run_id,
                mode=mode_label,
                stats=self.episode_stats,
                kills_total=self.training_kills_total,
                score_total=self.training_score_total,
            )
        self.agent_episode_returns_history.append(return_final)
        if steps_total > 0 and pos_total == 0.0 and penalty_total == 0.0:
            self.zero_reward_episode_streak += 1
            if self.zero_reward_episode_streak >= 3:
                print(
                    f"Warning: {self.zero_reward_episode_streak} consecutive episodes logged "
                    f"with steps>0 but zero reward/penalty. Check reward pipeline."
                )
        else:
            self.zero_reward_episode_streak = 0
        self.player_q_agent.advance_exploration_episode()
        self.agent_episode_id += 1
        return return_final

    def _average_return_20(self) -> float:
        if not self.agent_episode_returns_history:
            return 0.0
        recent = self.agent_episode_returns_history[-20:]
        return sum(recent) / len(recent)

    def _agent_target_hint(self) -> tuple[str, str, str]:
        target = self._get_valid_player_agent_target(self.agent_control) if self.current_control is self.agent_control else None
        if target is None:
            return "NONE", "N/A", "no_target"

        distance_tiles = math.hypot(target.x - self.player.x, target.y - self.player.y) / TILE_SIZE
        if isinstance(target, HealthPackTarget):
            return "PACK", f"{distance_tiles:.2f}t", self.agent_control.current_reason or "pack_value"

        has_los = _has_line_of_sight(self.player.x, self.player.y, target.x, target.y, self.map)
        if has_los and PLAYER_AGENT_RANGE_MIN <= distance_tiles * TILE_SIZE <= PLAYER_AGENT_RANGE_MAX:
            reason = "engage_enemy"
        else:
            reason = self.agent_control.current_reason or "chase_enemy"
        return "ENEMY", f"{distance_tiles:.2f}t", reason

    def _finish_player_agent_episode(self) -> float | None:
        episode_return = self._finalize_agent_episode_metrics()
        self.player_q_agent.finish_episode()
        return episode_return

    def _request_player_reload(self) -> bool:
        started = self.player.start_reload()
        if started:
            self.audio.play("reload", SFX_VOLUME_RELOAD)
        return started

    def _enter_game_over(self) -> None:
        if self.app_state == APP_STATE_GAME_OVER:
            return
        self.app_state = APP_STATE_GAME_OVER
        self.main_state = MAIN_STATE_GAME_OVER
        self.player_hp = 0
        self._finish_player_agent_episode()
        if self.score > self.high_score:
            self.high_score = self.score
            self._save_high_score()
        self._set_mouse_capture(False)

    def _pause_game(self) -> None:
        if self.app_state not in (APP_STATE_PLAY_MAIN, APP_STATE_PLAY_TRAINING):
            return
        self.paused_return_state = self.app_state
        self.app_state = APP_STATE_PAUSED
        self._set_mouse_capture(False)

    def _resume_from_pause(self) -> None:
        if self.app_state != APP_STATE_PAUSED:
            return
        self.app_state = self.paused_return_state
        self._set_mouse_capture(True)

    def _restart_current_mode(self) -> None:
        self.audio.play("ui_select", SFX_VOLUME_UI_SELECT)
        if self.current_play_state == APP_STATE_PLAY_TRAINING:
            self._start_training_arena()
        else:
            self._start_main_mode()

    def _return_to_menu(self) -> None:
        self.audio.play("ui_select", SFX_VOLUME_UI_SELECT)
        self._finish_player_agent_learning_only()
        self.app_state = APP_STATE_MENU
        self.main_state = MAIN_STATE_PLAYING
        self.feedback_messages = []
        self.feedback_time_left = 0.0
        self.intermission_remaining = 0.0
        self.enemies = []
        self._set_mouse_capture(False)

    def _allocate_entity_id(self) -> int:
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id

    def _set_level_map(self, level_map: list[str]) -> None:
        self.map = self._with_wall_variants(level_map)
        self.cover_points = self._build_cover_points(self.map)

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
                    _is_open_cell(cell_x + offset_x, cell_y + offset_y, level_map)
                    for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ):
                    continue
                cells[cell_x] = "2"
            variant_map.append("".join(cells))
        return variant_map

    def _script_hear_range_tiles(self) -> float:
        return SCRIPT_HEAR_RANGE_FACTOR * max(len(self.map), len(self.map[0]))

    def _patrol_radius_tiles(self) -> int:
        max_dim = max(len(self.map), len(self.map[0]))
        auto_radius = max_dim // 4
        return max(
            min(SCRIPT_PATROL_RADIUS_MAX_TILES, auto_radius),
            min(SCRIPT_PATROL_RADIUS_MIN_TILES, max_dim // 2),
        )

    def _generate_patrol_route(
        self,
        start_cell: tuple[int, int],
        kind: str,
    ) -> list[tuple[int, int]]:
        radius_tiles = self._patrol_radius_tiles()
        reachable_cells = _reachable_open_cells_within_radius(start_cell, radius_tiles, self.map)
        if not reachable_cells:
            return [start_cell]

        min_points = min(SCRIPT_PATROL_WAYPOINTS_MIN, len(reachable_cells))
        max_points = min(SCRIPT_PATROL_WAYPOINTS_MAX, len(reachable_cells))
        if max_points <= 1:
            return [start_cell]

        seed = (
            start_cell[0] * 73856093
            ^ start_cell[1] * 19349663
            ^ sum(ord(char) for char in kind) * 83492791
        )
        patrol_rng = random.Random(seed)
        candidates = [
            cell
            for cell in reachable_cells
            if cell != start_cell and math.dist(cell, start_cell) >= 2.0
        ]
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
        enemy.engage_min_range = TILE_SIZE * float(
            definition.get("engage_min_tiles", 3.5 if enemy.attack_style == "ranged" else 0.6)
        )
        enemy.engage_max_range = TILE_SIZE * float(
            definition.get("engage_max_tiles", 5.5 if enemy.attack_style == "ranged" else 1.4)
        )

    def _start_main_mode(self) -> None:
        self._finish_player_agent_learning_only()
        self.app_state = APP_STATE_PLAY_MAIN
        self.current_play_state = APP_STATE_PLAY_MAIN
        self.paused_return_state = APP_STATE_PLAY_MAIN
        self.game_mode = "main"
        self.main_state = MAIN_STATE_PLAYING
        self._set_level_map(GRID_MAP)
        self.score = 0
        self.kill_count = 0
        self.wave_number = 1
        self.intermission_remaining = 0.0
        self.last_wave_bonus = 0
        self._reset_agent_reward_monitor(reset_history=True)
        self._reset_player_after_death()
        self.enemies = self._spawn_wave(self.wave_number)
        self._set_mouse_capture(True)

    def _start_training_arena(self) -> None:
        self._finish_player_agent_learning_only()
        self.player_q_agent.set_eval_mode(False)
        self.app_state = APP_STATE_PLAY_TRAINING
        self.current_play_state = APP_STATE_PLAY_TRAINING
        self.paused_return_state = APP_STATE_PLAY_TRAINING
        self.game_mode = "arena"
        self.main_state = MAIN_STATE_PLAYING
        self._set_level_map(TRAINING_ARENA_MAP)
        self.score = 0
        self.kill_count = 0
        self.intermission_remaining = 0.0
        self.last_wave_bonus = 0
        self.training_episode_count = 0
        self.training_total_reward = 0.0
        self.training_episode_reward = 0.0
        self.training_average_reward = 0.0
        self.training_run_id = self._new_training_run_id()
        self.player_q_agent.reset_for_new_run(self.training_run_id)
        self.training_score_total = 0
        self.training_kills_total = 0
        self._reset_agent_reward_monitor(reset_history=True)
        self._set_control_mode(self.agent_control)
        self._reset_training_episode()
        self._set_mouse_capture(True)

    def _spawn_training_enemy(self) -> Enemy:
        variant_index = self.training_episode_count % max(1, len(TRAINING_ARENA_ENEMY_CELLS))
        enemy_kind = TRAINING_ARENA_ENEMY_KINDS[variant_index] if TRAINING_ARENA_ENEMY_KINDS else "soldier"
        definition = ENEMY_TYPES[enemy_kind]
        spawn_cell = TRAINING_ARENA_ENEMY_CELLS[variant_index] if TRAINING_ARENA_ENEMY_CELLS else TRAINING_ARENA_ENEMY_CELL
        spawn_world_x = (spawn_cell[0] + 0.5) * TILE_SIZE
        spawn_world_y = (spawn_cell[1] + 0.5) * TILE_SIZE
        spawn_x, spawn_y = _find_free_spawn_position(
            spawn_world_x,
            spawn_world_y,
            definition["radius"],
            self.map,
            0,
        )
        hp = definition["hp"]
        enemy = Enemy(
            kind=enemy_kind,
            x=spawn_x,
            y=spawn_y,
            hp=hp,
            max_hp=hp,
            speed=definition["speed"],
            score=definition["score"],
            radius=definition["radius"],
            size=definition["size"],
            color=definition["color"],
            aggro_range=ENEMY_AGGRO_RANGE,
            damage=definition["damage"],
            shot_cooldown_max=definition["shoot_cooldown"],
            entity_id=self._allocate_entity_id(),
        )
        self._configure_script_enemy(enemy, spawn_cell)
        return enemy

    def _reset_training_episode(self) -> None:
        self.player.x, self.player.y = TRAINING_ARENA_PLAYER_START
        variant_index = self.training_episode_count % max(1, len(TRAINING_ARENA_PLAYER_ANGLES))
        self.player.angle = TRAINING_ARENA_PLAYER_ANGLES[variant_index] if TRAINING_ARENA_PLAYER_ANGLES else 0.0
        self.score = 0
        self.kill_count = 0
        self.player.weapon.reset()
        self.weapon_bob_phase = 0.0
        self.weapon_bob_amount = 0.0
        self.weapon_recoil_time_left = 0.0
        self.player_prev_draw_pos = (self.player.x, self.player.y)
        self.player_hp = PLAYER_MAX_HP
        self.feedback_messages = []
        self.feedback_time_left = 0.0
        self.player_prev_hp = PLAYER_MAX_HP
        self.player_bumped = False
        self.player_under_fire_timer = 0.0
        self.player_prev_threat_los = False
        self.skip_player_shaping_once = True
        self._reset_agent_reward_monitor()
        self.episode_stats.begin(
            episode_id=self.agent_episode_id,
            eps_used=0.0 if self.player_q_agent.is_eval_mode else self.player_q_agent.begin_episode(),
        )
        self._sync_episode_stats_mirrors()
        print(f"EP start: id={self.episode_stats.episode_id}, eps_used={self.episode_stats.eps_used:.4f}")
        self.agent_control.action_time_left = 0.0
        self.agent_control.action_locked = False
        self.agent_control.emergency_unlock = False
        self.agent_control.reset_navigation()
        self.enemies = [self._spawn_training_enemy()]

    def _finish_training_episode(self, player_won: bool) -> None:
        episode_return = self._finish_player_agent_episode() or 0.0
        self.training_episode_count += 1
        self.training_total_reward += episode_return
        self.training_average_reward = self.training_total_reward / self.training_episode_count
        self._reset_training_episode()

    def _set_mouse_capture(self, captured: bool) -> None:
        self.mouse_captured = captured
        pygame.event.set_grab(captured)
        pygame.mouse.set_visible(not captured)
        pygame.mouse.get_rel()

    def _set_control_mode(self, control: HumanControl | AgentControl) -> None:
        if self.current_control is self.agent_control and control is not self.agent_control:
            self._finish_player_agent_learning_only()
        self.current_control = control
        if control is self.agent_control:
            self.agent_control.action_time_left = 0.0
            self.agent_control.action_locked = False
            self.agent_control.emergency_unlock = False
            self.agent_control.reset_navigation()
            self.skip_player_shaping_once = True
            self._reset_agent_reward_monitor()

    def _update_mouse_look(self) -> None:
        rel_x, _ = pygame.mouse.get_rel()
        if not self.mouse_captured:
            return
        self.player.angle = (self.player.angle + rel_x * MOUSE_SENSITIVITY) % math.tau

    def _add_player_agent_reward(self, amount: float) -> None:
        if self.current_control is self.agent_control:
            self.episode_stats.record_reward(amount)
            self._sync_episode_stats_mirrors()
            self.player_q_agent.record_reward(amount)
            if self.game_mode == "arena":
                self.training_episode_reward += amount

    def _add_player_agent_pack_reward(self, amount: float) -> None:
        if amount == 0.0:
            return
        if self.current_control is self.agent_control:
            self.agent_control.last_pack_reward += amount
        self._add_player_agent_reward(amount)

    def _current_health_pack_target(self) -> HealthPackTarget | None:
        for enemy in self.enemies:
            if isinstance(enemy, HealthPackTarget) and enemy.hp > 0:
                return enemy
        return None

    def _update_player_agent_shaping(self) -> None:
        if self.current_control is not self.agent_control:
            self.player_prev_threat_los = False
            self.agent_control.prev_pack_dist_tiles = None
            return

        self.agent_control.last_pack_reward = 0.0
        pack_target = self._current_health_pack_target()
        if pack_target is None:
            self.agent_control.prev_pack_dist_tiles = None
        else:
            pack_distance_tiles = math.hypot(pack_target.x - self.player.x, pack_target.y - self.player.y) / TILE_SIZE
            previous_distance_tiles = self.agent_control.prev_pack_dist_tiles
            if previous_distance_tiles is not None:
                delta = previous_distance_tiles - pack_distance_tiles
                shaping = max(-PACK_APPROACH_CLAMP, min(PACK_APPROACH_CLAMP, PACK_APPROACH_K * delta))
                if abs(shaping) > 1e-6:
                    self._add_player_agent_pack_reward(shaping)
            self.agent_control.prev_pack_dist_tiles = pack_distance_tiles

        target_enemy = self._get_player_agent_target(self.agent_control, allow_search=False)
        if target_enemy is None:
            self.player_prev_threat_los = False
            self.skip_player_shaping_once = False
            return

        has_los = _has_line_of_sight(
            self.player.x,
            self.player.y,
            target_enemy.x,
            target_enemy.y,
            self.map,
        )
        if self.skip_player_shaping_once:
            self.player_prev_threat_los = has_los
            self.skip_player_shaping_once = False
            return
        distance = math.hypot(target_enemy.x - self.player.x, target_enemy.y - self.player.y)
        if has_los and TILE_SIZE * 2.5 <= distance <= TILE_SIZE * 5.0:
            self._add_player_agent_reward(PLAYER_AGENT_REWARD_SHAPING)
        if self.player_under_fire_timer > 0.0 and self.player_prev_threat_los and not has_los:
            self._add_player_agent_reward(PLAYER_AGENT_REWARD_BREAK_LOS)
        self.player_prev_threat_los = has_los

    def _reset_player_after_death(self) -> None:
        self.player.x, self.player.y = PLAYER_START
        self.player.angle = 0.0
        self.player.weapon.reset()
        self.weapon_bob_phase = 0.0
        self.weapon_bob_amount = 0.0
        self.weapon_recoil_time_left = 0.0
        self.player_prev_draw_pos = (self.player.x, self.player.y)
        self.player_hp = PLAYER_MAX_HP
        self.feedback_messages = []
        self.feedback_time_left = 0.0
        self.player_under_fire_timer = 0.0
        self.player_prev_threat_los = False
        self.skip_player_shaping_once = True
        self.agent_control.action_time_left = 0.0
        self.agent_control.action_locked = False
        self.agent_control.emergency_unlock = False
        self.agent_control.reset_navigation()

    def _move_player(self, forward: float, strafe: float, dt: float) -> bool:
        move_x = (math.cos(self.player.angle) * forward - math.sin(self.player.angle) * strafe) * PLAYER_SPEED * dt
        move_y = (math.sin(self.player.angle) * forward + math.cos(self.player.angle) * strafe) * PLAYER_SPEED * dt
        return self._move_player_by_vector(move_x, move_y)

    def _player_hits_enemy(self, world_x: float, world_y: float) -> bool:
        player_radius = PLAYER_RADIUS
        for enemy in self.enemies:
            if enemy.hp <= 0:
                continue
            combined_radius = player_radius + enemy.radius
            if math.hypot(enemy.x - world_x, enemy.y - world_y) < combined_radius:
                return True
        return False

    def _move_player_by_vector(self, move_x: float, move_y: float) -> bool:
        intended = math.hypot(move_x, move_y)
        next_x, next_y = _move_with_slide(
            self.player.x,
            self.player.y,
            move_x,
            move_y,
            PLAYER_RADIUS,
            self.map,
        )
        if self._player_hits_enemy(next_x, next_y):
            next_x, next_y = self.player.x, self.player.y
        actual = math.hypot(next_x - self.player.x, next_y - self.player.y)
        self.player.x, self.player.y = next_x, next_y
        return intended > 0.01 and actual < intended * 0.35

    def _update_human_player(self, dt: float) -> bool:
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
        return self._move_player(forward, strafe, dt)

    def _player_target_score(self, enemy: Enemy) -> float | None:
        player_cell = _world_to_cell(self.player.x, self.player.y)
        enemy_cell = _world_to_cell(enemy.x, enemy.y)
        distance = math.hypot(enemy.x - self.player.x, enemy.y - self.player.y)
        has_los = _has_line_of_sight(self.player.x, self.player.y, enemy.x, enemy.y, self.map)
        path = _build_path(player_cell, enemy_cell, self.map)
        path_distance = len(path) * TILE_SIZE if path or enemy_cell == player_cell else None
        if isinstance(enemy, HealthPackTarget):
            hp_ratio = self.player_hp / max(PLAYER_MAX_HP, 1)
            if hp_ratio > 0.95 and distance > TILE_SIZE * 6.0:
                return None
            base_distance = path_distance if path_distance is not None else distance
            pack_score = base_distance + distance * 0.1
            if has_los:
                pack_score -= TILE_SIZE * 0.25
            need_factor = 1.0 + max(0.0, 0.75 - hp_ratio) * 2.5
            if hp_ratio < 0.35:
                pack_score -= TILE_SIZE * 6.0
            return max(TILE_SIZE * 0.25, pack_score / need_factor)
        base_distance = path_distance if path_distance is not None else distance
        score = base_distance + distance * 0.05
        if has_los:
            score -= TILE_SIZE * 0.25
        return score

    def _pack_target_value(self) -> tuple[HealthPackTarget | None, float, float | None, str]:
        pack_target = self._current_health_pack_target()
        if pack_target is None:
            return None, 0.0, None, "no_pack"
        hp_ratio = self.player_hp / max(PLAYER_MAX_HP, 1)
        distance_tiles = math.hypot(pack_target.x - self.player.x, pack_target.y - self.player.y) / TILE_SIZE
        score, reason = pack_target_score(
            hp_ratio=hp_ratio,
            distance_tiles=distance_tiles,
            base_weight=W_PACK_BASE,
            close_weight=W_PACK_CLOSE,
            low_hp_bonus=W_PACK_LOWHP,
        )
        return pack_target, score, distance_tiles, reason

    def _enemy_target_value(self) -> tuple[Enemy | None, float, float | None, str]:
        nearest_enemy, nearest_distance = self._nearest_hostile_enemy_threat(prefer_path=True)
        if nearest_enemy is None:
            return None, 0.0, None, "no_enemy"
        distance_tiles = nearest_distance / TILE_SIZE
        score, reason = enemy_target_score(
            distance_tiles=distance_tiles,
            under_fire=self.player_under_fire_timer > 0.0,
            emergency_distance_tiles=EMERGENCY_DIST_TILES,
            close_weight=W_ENEMY_CLOSE,
            emergency_weight=W_ENEMY_EMERGENCY,
        )
        return nearest_enemy, score, distance_tiles, reason

    def _player_target_distance_value(self, enemy: Enemy, *, prefer_path: bool = True) -> float:
        euclidean_distance = math.hypot(enemy.x - self.player.x, enemy.y - self.player.y)
        if not prefer_path or isinstance(enemy, HealthPackTarget):
            return euclidean_distance
        player_cell = _world_to_cell(self.player.x, self.player.y)
        enemy_cell = _world_to_cell(enemy.x, enemy.y)
        path = _build_path(player_cell, enemy_cell, self.map)
        if path or enemy_cell == player_cell:
            return len(path) * TILE_SIZE
        return euclidean_distance

    def _nearest_hostile_enemy_threat(self, *, prefer_path: bool) -> tuple[Enemy | None, float]:
        best_enemy: Enemy | None = None
        best_distance = float("inf")
        for enemy in self._hostile_enemies():
            distance = self._player_target_distance_value(enemy, prefer_path=prefer_path)
            if distance < best_distance:
                best_distance = distance
                best_enemy = enemy
        return best_enemy, best_distance

    def _assign_player_agent_target(
        self,
        control: AgentControl,
        target: Enemy,
        *,
        score: float | None = None,
        emergency: bool,
    ) -> None:
        target_score = score if score is not None else (self._player_target_score(target) or self._player_target_distance_value(target, prefer_path=True))
        control.target_lock.assign(target.entity_id, target, target_score)
        control.path.clear()
        control.path_target_cell = None
        control.repath_timer = 0.0
        control.last_seen_target_cell = None
        control.mark_target_switch(emergency=emergency)

    def _maybe_retarget_player_agent(self, control: AgentControl) -> None:
        if self.current_control is not self.agent_control:
            return
        if self.app_state not in (APP_STATE_PLAY_MAIN, APP_STATE_PLAY_TRAINING):
            return

        current_target = self._get_valid_player_agent_target(control)
        pack_target, pack_score, pack_dist_tiles, pack_reason = self._pack_target_value()
        enemy_target, enemy_score, enemy_dist_tiles, enemy_reason = self._enemy_target_value()
        control.current_pack_score = pack_score
        control.current_enemy_score = enemy_score
        control.current_pack_dist_tiles = pack_dist_tiles
        control.current_enemy_dist_tiles = enemy_dist_tiles

        emergency_enemy = (
            enemy_target is not None
            and enemy_dist_tiles is not None
            and enemy_dist_tiles <= EMERGENCY_DIST_TILES
        )
        if emergency_enemy and enemy_target is not current_target:
            self._assign_player_agent_target(control, enemy_target, emergency=True)
            control.current_goal_type = "ENEMY"
            control.current_reason = "emergency_enemy"
            return

        if current_target is None and control.retarget_timer > 0.0:
            control.retarget_timer = 0.0
        if control.retarget_timer > 0.0 and current_target is not None:
            control.current_goal_type = "PACK" if isinstance(current_target, HealthPackTarget) else "ENEMY"
            return

        desired_target: Enemy | None = None
        desired_goal = "NONE"
        desired_reason = "no_target"
        desired_score = 0.0
        if pack_target is not None and (pack_score > enemy_score or enemy_target is None):
            desired_target = pack_target
            desired_goal = "PACK"
            desired_reason = pack_reason
            desired_score = pack_score
        elif enemy_target is not None:
            desired_target = enemy_target
            desired_goal = "ENEMY"
            desired_reason = enemy_reason
            desired_score = enemy_score

        if desired_target is None:
            control.current_goal_type = "NONE"
            control.retarget_timer = RETARGET_INTERVAL_SEC
            return

        current_goal = "PACK" if isinstance(current_target, HealthPackTarget) else "ENEMY" if current_target is not None else "NONE"
        if current_goal == "PACK":
            current_score = pack_score
        elif current_goal == "ENEMY" and current_target is not None:
            current_target_dist_tiles = self._player_target_distance_value(current_target, prefer_path=True) / TILE_SIZE
            current_score = W_ENEMY_CLOSE / (current_target_dist_tiles + 1.0)
            if current_target_dist_tiles <= EMERGENCY_DIST_TILES or self.player_under_fire_timer > 0.0:
                current_score += W_ENEMY_EMERGENCY
        else:
            current_score = float("-inf")
        should_switch = current_target is None or current_target is not desired_target
        if current_target is not None and current_target is not desired_target:
            if desired_goal != current_goal and desired_score <= current_score + TARGET_SCORE_HYST:
                should_switch = False
            elif desired_goal == current_goal:
                current_distance_tiles = (
                    pack_dist_tiles if desired_goal == "PACK"
                    else self._player_target_distance_value(current_target, prefer_path=True) / TILE_SIZE
                )
                desired_distance_tiles = pack_dist_tiles if desired_goal == "PACK" else enemy_dist_tiles
                if (
                    current_distance_tiles is not None
                    and desired_distance_tiles is not None
                    and desired_distance_tiles + RETARGET_HYSTERESIS_TILES >= current_distance_tiles
                ):
                    should_switch = False

        if should_switch:
            self._assign_player_agent_target(control, desired_target, score=desired_score, emergency=False)
        else:
            control.retarget_timer = RETARGET_INTERVAL_SEC
        control.current_goal_type = desired_goal
        control.current_reason = desired_reason

    def _get_valid_player_agent_target(self, control: AgentControl) -> Enemy | None:
        current_target = control.target_lock.target_ref
        if isinstance(current_target, Enemy) and current_target in self.enemies and current_target.hp > 0:
            if isinstance(current_target, HealthPackTarget):
                _pack, pack_score, _pack_dist, _pack_reason = self._pack_target_value()
                control.target_lock.score = pack_score
            else:
                current_target_dist_tiles = self._player_target_distance_value(current_target, prefer_path=True) / TILE_SIZE
                current_score = W_ENEMY_CLOSE / (current_target_dist_tiles + 1.0)
                if current_target_dist_tiles <= EMERGENCY_DIST_TILES or self.player_under_fire_timer > 0.0:
                    current_score += W_ENEMY_EMERGENCY
                control.target_lock.score = current_score
            return current_target

        control.target_lock.clear()
        return None

    def _get_player_agent_target(self, control: AgentControl, *, allow_search: bool = True) -> Enemy | None:
        current_target = self._get_valid_player_agent_target(control)
        if current_target is not None:
            return current_target

        if not allow_search or control.last_retarget_time < PLAYER_AGENT_RETARGET_SECONDS:
            return None

        pack_target, pack_score, pack_dist_tiles, pack_reason = self._pack_target_value()
        enemy_target, enemy_score, enemy_dist_tiles, enemy_reason = self._enemy_target_value()
        control.current_pack_score = pack_score
        control.current_enemy_score = enemy_score
        control.current_pack_dist_tiles = pack_dist_tiles
        control.current_enemy_dist_tiles = enemy_dist_tiles
        if pack_target is not None and (pack_score > enemy_score or enemy_target is None):
            best_target: Enemy | None = pack_target
            best_score = pack_score
            control.current_goal_type = "PACK"
            control.current_reason = pack_reason
        else:
            best_target = enemy_target
            best_score = enemy_score
            control.current_goal_type = "ENEMY" if enemy_target is not None else "NONE"
            control.current_reason = enemy_reason

        if best_target is None:
            return None

        self._assign_player_agent_target(control, best_target, score=best_score, emergency=False)

        locked_target = control.target_lock.target_ref
        if isinstance(locked_target, Enemy):
            return locked_target
        control.target_lock.clear()
        return None

    def _build_player_agent_state(self, control: AgentControl) -> tuple[int, int, int, int, int, int, int, int]:
        pack_target = self._current_health_pack_target()
        pack_distance = (
            math.hypot(pack_target.x - self.player.x, pack_target.y - self.player.y)
            if pack_target is not None
            else None
        )
        target_enemy = self._get_player_agent_target(control, allow_search=False)
        return build_player_agent_obs(
            player_x=self.player.x,
            player_y=self.player.y,
            player_angle=self.player.angle,
            player_bumped=self.player_bumped,
            player_hp=self.player_hp,
            player_max_hp=PLAYER_MAX_HP,
            under_fire=self.player_under_fire_timer > 0.0,
            level_map=self.map,
            target_x=target_enemy.x if target_enemy is not None else None,
            target_y=target_enemy.y if target_enemy is not None else None,
            pack_distance=pack_distance,
        )

    def _nearest_enemy_to_player(self) -> Enemy | None:
        return min(
            (enemy for enemy in self.enemies if not isinstance(enemy, HealthPackTarget)),
            key=lambda enemy: (enemy.x - self.player.x) ** 2 + (enemy.y - self.player.y) ** 2,
            default=None,
        )

    def _hostile_enemies(self) -> list[Enemy]:
        return [enemy for enemy in self.enemies if not isinstance(enemy, HealthPackTarget)]

    def _find_cover_point(
        self,
        source_x: float,
        source_y: float,
        actor_radius: float,
        threat_x: float,
        threat_y: float,
        anchor_x: float,
        anchor_y: float,
    ) -> tuple[float, float] | None:
        best_point: tuple[float, float] | None = None
        best_score = float("inf")
        max_distance = TILE_SIZE * 10.0

        for cover_x, cover_y in self.cover_points:
            if math.hypot(cover_x - source_x, cover_y - source_y) > max_distance:
                continue
            if _has_line_of_sight(threat_x, threat_y, cover_x, cover_y, self.map):
                continue
            if not _can_move_to(cover_x, cover_y, actor_radius, self.map):
                continue

            source_distance = math.hypot(cover_x - source_x, cover_y - source_y)
            anchor_distance = math.hypot(cover_x - anchor_x, cover_y - anchor_y)
            score = source_distance + anchor_distance * 0.45
            if score < best_score:
                best_score = score
                best_point = (cover_x, cover_y)

        return best_point

    def _is_critical_hp(self, current_hp: int, max_hp: int) -> bool:
        return current_hp <= max(1, int(max_hp * CRITICAL_HP_RATIO))

    def _resolve_navigation_waypoint(
        self,
        actor_x: float,
        actor_y: float,
        actor_radius: float,
        target_x: float,
        target_y: float,
        path: list[tuple[int, int]],
        path_target_cell: tuple[int, int] | None,
        repath_timer: float,
        dt: float,
    ) -> tuple[tuple[float, float], list[tuple[int, int]], tuple[int, int] | None, float]:
        target_x, target_y = _find_free_spawn_position(
            target_x,
            target_y,
            actor_radius,
            self.map,
            0,
        )
        target_cell = _world_to_cell(target_x, target_y)
        if _has_line_of_sight(actor_x, actor_y, target_x, target_y, self.map):
            return (target_x, target_y), [], None, 0.0

        repath_timer -= dt
        actor_cell = _world_to_cell(actor_x, actor_y)
        if repath_timer <= 0.0 or path_target_cell != target_cell or not path:
            path = _build_path(actor_cell, target_cell, self.map)
            if not path:
                path, fallback_cell = _build_fallback_path_toward_target(actor_cell, target_cell, self.map)
                if fallback_cell is not None:
                    path_target_cell = fallback_cell
                else:
                    path_target_cell = target_cell
            else:
                path_target_cell = target_cell
            repath_timer = ENEMY_PATH_RECALC_INTERVAL

        while path:
            waypoint_x, waypoint_y = _cell_center(*path[0])
            if math.hypot(waypoint_x - actor_x, waypoint_y - actor_y) > max(actor_radius, TILE_SIZE * 0.18):
                break
            path.pop(0)

        if path:
            return _cell_center(*path[0]), path, path_target_cell, repath_timer
        return (target_x, target_y), path, path_target_cell, repath_timer

    def _intent_target_point(
        self,
        source_x: float,
        source_y: float,
        actor_radius: float,
        target_x: float,
        target_y: float,
        intent: int,
        side_seed: int,
    ) -> tuple[tuple[float, float], bool]:
        dir_x, dir_y, distance = _normalize_vector(target_x - source_x, target_y - source_y)
        side_sign = 1 if side_seed % 2 == 0 else -1
        desired_range = TILE_SIZE * 3.5

        if intent == INTENT_CHASE:
            return (target_x, target_y), False
        if intent == INTENT_ENGAGE:
            if distance > desired_range + TILE_SIZE:
                return (target_x, target_y), True
            if distance < desired_range - TILE_SIZE * 0.75:
                return _project_point(source_x, source_y, -dir_x, -dir_y, TILE_SIZE * 2.5), True
            return (source_x, source_y), True
        if intent == INTENT_EVADE:
            retreat_point = _project_point(source_x, source_y, -dir_x, -dir_y, TILE_SIZE * 3.0)
            cover_point = self._find_cover_point(
                source_x,
                source_y,
                actor_radius,
                target_x,
                target_y,
                retreat_point[0],
                retreat_point[1],
            )
            return cover_point or retreat_point, False

        flank_point = _flank_point(source_x, source_y, target_x, target_y, side_sign, TILE_SIZE * 2.5)
        cover_point = self._find_cover_point(
            source_x,
            source_y,
            actor_radius,
            target_x,
            target_y,
            flank_point[0],
            flank_point[1],
        )
        return cover_point or flank_point, False

    def _player_intent_target_point(
        self,
        target_enemy: Enemy,
        intent: int,
        side_seed: int,
    ) -> tuple[tuple[float, float], bool]:
        dir_x, dir_y, distance = _normalize_vector(
            target_enemy.x - self.player.x,
            target_enemy.y - self.player.y,
        )
        side_sign = 1 if side_seed % 2 == 0 else -1
        band_mid = PLAYER_AGENT_RANGE_PREFERRED

        if intent == INTENT_CHASE:
            if distance > PLAYER_AGENT_RANGE_MAX:
                return (target_enemy.x, target_enemy.y), False
            if distance < PLAYER_AGENT_RANGE_MIN:
                return _project_point(
                    self.player.x,
                    self.player.y,
                    -dir_x,
                    -dir_y,
                    max(TILE_SIZE * 1.5, PLAYER_AGENT_RANGE_MIN - distance + TILE_SIZE * 0.5),
                ), False
            return (self.player.x, self.player.y), False

        if intent == INTENT_ENGAGE:
            if distance > PLAYER_AGENT_RANGE_MAX:
                return (target_enemy.x, target_enemy.y), True
            if distance < PLAYER_AGENT_RANGE_MIN:
                return _project_point(
                    self.player.x,
                    self.player.y,
                    -dir_x,
                    -dir_y,
                    max(TILE_SIZE * 1.5, band_mid - distance),
                ), True
            return (self.player.x, self.player.y), True

        if intent == INTENT_EVADE:
            retreat_anchor = _project_point(
                self.player.x,
                self.player.y,
                -dir_x,
                -dir_y,
                max(TILE_SIZE * 2.0, PLAYER_AGENT_RANGE_MAX - distance + TILE_SIZE),
            )
            cover_point = self._find_cover_point(
                self.player.x,
                self.player.y,
                PLAYER_RADIUS,
                target_enemy.x,
                target_enemy.y,
                retreat_anchor[0],
                retreat_anchor[1],
            )
            return cover_point or retreat_anchor, False

        flank_point = _flank_point(
            self.player.x,
            self.player.y,
            target_enemy.x,
            target_enemy.y,
            side_sign,
            max(TILE_SIZE * 2.0, abs(distance - band_mid) + TILE_SIZE),
        )
        cover_point = self._find_cover_point(
            self.player.x,
            self.player.y,
            PLAYER_RADIUS,
            target_enemy.x,
            target_enemy.y,
            flank_point[0],
            flank_point[1],
        )
        return cover_point or flank_point, False

    def _player_chase_nav_target(
        self,
        control: AgentControl,
        target_enemy: Enemy,
    ) -> tuple[tuple[float, float], tuple[int, int], bool]:
        has_los = _has_line_of_sight(
            self.player.x,
            self.player.y,
            target_enemy.x,
            target_enemy.y,
            self.map,
        )
        current_target_cell = _world_to_cell(target_enemy.x, target_enemy.y)
        if has_los:
            control.last_seen_target_cell = current_target_cell
            dir_x, dir_y, _ = _normalize_vector(target_enemy.x - self.player.x, target_enemy.y - self.player.y)
            if dir_x == 0.0 and dir_y == 0.0:
                dir_x = math.cos(self.player.angle)
                dir_y = math.sin(self.player.angle)
            orbit_radius = ENGAGE_IDEAL_T * TILE_SIZE
            base_x = target_enemy.x - dir_x * orbit_radius
            base_y = target_enemy.y - dir_y * orbit_radius
            angles = (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2)
            for angle_offset in angles:
                rot_cos = math.cos(angle_offset)
                rot_sin = math.sin(angle_offset)
                cand_dir_x = dir_x * rot_cos - dir_y * rot_sin
                cand_dir_y = dir_x * rot_sin + dir_y * rot_cos
                candidate_x = target_enemy.x - cand_dir_x * orbit_radius
                candidate_y = target_enemy.y - cand_dir_y * orbit_radius
                if _can_move_to(candidate_x, candidate_y, PLAYER_RADIUS, self.map):
                    candidate_cell = _world_to_cell(candidate_x, candidate_y)
                    return (candidate_x, candidate_y), candidate_cell, True
            return (base_x, base_y), _world_to_cell(base_x, base_y), True

        nav_target_cell = control.last_seen_target_cell or current_target_cell
        nav_target_world = _cell_center(*nav_target_cell)
        return nav_target_world, nav_target_cell, False

    def _choose_player_patrol_cell(self, control: AgentControl) -> tuple[int, int]:
        player_cell = _world_to_cell(self.player.x, self.player.y)
        reachable_cells = _reachable_open_cells_within_radius(
            player_cell,
            PLAYER_AGENT_PATROL_RADIUS_TILES,
            self.map,
        )
        candidates = [
            cell
            for cell in reachable_cells
            if cell != player_cell and math.dist(cell, player_cell) >= 2.0
        ]
        if not candidates:
            candidates = [cell for cell in reachable_cells if cell != player_cell]
        if not candidates:
            return player_cell
        return self.random.choice(candidates)

    def _player_patrol_nav_target(
        self,
        control: AgentControl,
        *,
        force_refresh: bool = False,
    ) -> tuple[tuple[float, float], tuple[int, int]]:
        player_cell = _world_to_cell(self.player.x, self.player.y)
        refresh_needed = force_refresh or control.patrol_target_cell is None or control.patrol_time_left <= 0.0
        if control.patrol_target_cell is not None and control.patrol_target_cell != player_cell:
            patrol_center = _cell_center(*control.patrol_target_cell)
            if math.hypot(patrol_center[0] - self.player.x, patrol_center[1] - self.player.y) <= TILE_SIZE * 0.2:
                refresh_needed = True
        if refresh_needed:
            control.patrol_target_cell = self._choose_player_patrol_cell(control)
            control.patrol_time_left = PLAYER_AGENT_PATROL_REFRESH_SECONDS
            control.path.clear()
            control.path_target_cell = None
            control.repath_timer = 0.0
        patrol_cell = control.patrol_target_cell or player_cell
        return _cell_center(*patrol_cell), patrol_cell

    def _player_strafe_nav_target(
        self,
        control: AgentControl,
        target_enemy: Enemy,
        target_distance: float,
    ) -> tuple[tuple[float, float], tuple[int, int]]:
        if control.strafe_time_left <= 0.0:
            control.strafe_time_left = PLAYER_AGENT_STRAFE_BURST_SECONDS
            control.strafe_sign *= -1
        dir_x, dir_y, _ = _normalize_vector(
            target_enemy.x - self.player.x,
            target_enemy.y - self.player.y,
        )
        side_x = -dir_y * control.strafe_sign
        side_y = dir_x * control.strafe_sign
        desired_forward = max(-TILE_SIZE * 0.75, min(TILE_SIZE * 0.75, PLAYER_AGENT_RANGE_PREFERRED - target_distance))
        target_point = (
            self.player.x + side_x * TILE_SIZE * 2.0 - dir_x * desired_forward,
            self.player.y + side_y * TILE_SIZE * 2.0 - dir_y * desired_forward,
        )
        cover_point = self._find_cover_point(
            self.player.x,
            self.player.y,
            PLAYER_RADIUS,
            target_enemy.x,
            target_enemy.y,
            target_point[0],
            target_point[1],
        )
        resolved_target = cover_point or target_point
        return resolved_target, _world_to_cell(resolved_target[0], resolved_target[1])

    def _player_backoff_strafe_nav_target(
        self,
        control: AgentControl,
        target_enemy: Enemy,
        target_distance: float,
    ) -> tuple[tuple[float, float], tuple[int, int]]:
        dir_x, dir_y, _ = _normalize_vector(
            target_enemy.x - self.player.x,
            target_enemy.y - self.player.y,
        )
        if dir_x == 0.0 and dir_y == 0.0:
            dir_x = math.cos(self.player.angle)
            dir_y = math.sin(self.player.angle)
        side_x = -dir_y * control.strafe_sign
        side_y = dir_x * control.strafe_sign
        backoff_distance = max(TILE_SIZE * 1.5, (ENGAGE_IDEAL_T * TILE_SIZE) - target_distance)
        target_point = (
            self.player.x - dir_x * backoff_distance + side_x * TILE_SIZE * 1.2,
            self.player.y - dir_y * backoff_distance + side_y * TILE_SIZE * 1.2,
        )
        cover_point = self._find_cover_point(
            self.player.x,
            self.player.y,
            PLAYER_RADIUS,
            target_enemy.x,
            target_enemy.y,
            target_point[0],
            target_point[1],
        )
        resolved_target = cover_point or target_point
        return resolved_target, _world_to_cell(resolved_target[0], resolved_target[1])

    def _player_wall_repulsion(
        self,
        desired_move_x: float,
        desired_move_y: float,
    ) -> tuple[float, float, bool]:
        desired_length = math.hypot(desired_move_x, desired_move_y)
        if desired_length <= 0.0001:
            return 0.0, 0.0, False

        probe_distances = (
            TILE_SIZE * 0.6,
            TILE_SIZE * 0.8,
            TILE_SIZE * 1.0,
            TILE_SIZE * 1.2,
        )
        probe_directions = (
            (1.0, 0.0),
            (math.sqrt(0.5), math.sqrt(0.5)),
            (0.0, 1.0),
            (-math.sqrt(0.5), math.sqrt(0.5)),
            (-1.0, 0.0),
            (-math.sqrt(0.5), -math.sqrt(0.5)),
            (0.0, -1.0),
            (math.sqrt(0.5), -math.sqrt(0.5)),
        )
        repulse_x = 0.0
        repulse_y = 0.0
        near_wall = False
        repulse_radius = WALL_REPULSE_RADIUS_TILES * TILE_SIZE

        for dir_x, dir_y in probe_directions:
            hit_distance = None
            for probe_distance in probe_distances:
                probe_x = self.player.x + dir_x * probe_distance
                probe_y = self.player.y + dir_y * probe_distance
                if _is_wall(probe_x, probe_y, self.map):
                    hit_distance = probe_distance
                    break
            if hit_distance is None or hit_distance > repulse_radius:
                continue
            near_wall = True
            weight = max(0.0, (repulse_radius - hit_distance) / max(repulse_radius, 1e-6))
            repulse_x -= dir_x * weight
            repulse_y -= dir_y * weight

        if not near_wall:
            return desired_move_x, desired_move_y, False

        desired_dir_x = desired_move_x / desired_length
        desired_dir_y = desired_move_y / desired_length
        combined_x = desired_dir_x + WALL_REPULSE_WEIGHT * repulse_x
        combined_y = desired_dir_y + WALL_REPULSE_WEIGHT * repulse_y
        comb_dir_x, comb_dir_y, comb_length = _normalize_vector(combined_x, combined_y)
        if comb_length <= 0.0001:
            return desired_move_x, desired_move_y, True
        return comb_dir_x * desired_length, comb_dir_y * desired_length, True

    def _player_enemy_repulsion(
        self,
        desired_move_x: float,
        desired_move_y: float,
    ) -> tuple[float, float, bool]:
        desired_length = math.hypot(desired_move_x, desired_move_y)
        if desired_length <= 0.0001:
            return desired_move_x, desired_move_y, False
        repulse_radius = ENEMY_REPULSE_RADIUS_T * TILE_SIZE
        repulse_x = 0.0
        repulse_y = 0.0
        active = False
        for enemy in self._hostile_enemies():
            dist_x = self.player.x - enemy.x
            dist_y = self.player.y - enemy.y
            away_x, away_y, distance = _normalize_vector(dist_x, dist_y)
            if distance <= 0.0001 or distance >= repulse_radius:
                continue
            weight = (repulse_radius - distance) / repulse_radius
            repulse_x += away_x * weight
            repulse_y += away_y * weight
            active = True
        if not active:
            return desired_move_x, desired_move_y, False
        desired_dir_x = desired_move_x / desired_length
        desired_dir_y = desired_move_y / desired_length
        combined_x = desired_dir_x + ENEMY_REPULSE_WEIGHT * repulse_x
        combined_y = desired_dir_y + ENEMY_REPULSE_WEIGHT * repulse_y
        dir_x, dir_y, length = _normalize_vector(combined_x, combined_y)
        if length <= 0.0001:
            return desired_move_x, desired_move_y, True
        return dir_x * desired_length, dir_y * desired_length, True

    def _smooth_player_move_vector(
        self,
        control: AgentControl,
        move_x: float,
        move_y: float,
    ) -> tuple[float, float]:
        if math.hypot(move_x, move_y) <= 0.0001:
            control.move_smooth_x += (0.0 - control.move_smooth_x) * SMOOTH_ALPHA
            control.move_smooth_y += (0.0 - control.move_smooth_y) * SMOOTH_ALPHA
        else:
            control.move_smooth_x += (move_x - control.move_smooth_x) * SMOOTH_ALPHA
            control.move_smooth_y += (move_y - control.move_smooth_y) * SMOOTH_ALPHA
        if math.hypot(control.move_smooth_x, control.move_smooth_y) < MOVE_FILTER_DEADZONE * 0.5:
            control.move_smooth_x = 0.0
            control.move_smooth_y = 0.0
        return control.move_smooth_x, control.move_smooth_y

    def _player_recovery_move(
        self,
        control: AgentControl,
        target_x: float | None,
        target_y: float | None,
        dt: float,
    ) -> tuple[float, float]:
        if target_x is None or target_y is None:
            dir_x = math.cos(self.player.angle)
            dir_y = math.sin(self.player.angle)
        else:
            dir_x, dir_y, _ = _normalize_vector(target_x - self.player.x, target_y - self.player.y)
            if dir_x == 0.0 and dir_y == 0.0:
                dir_x = math.cos(self.player.angle)
                dir_y = math.sin(self.player.angle)
        tangent_x = -dir_y * control.recover_sign
        tangent_y = dir_x * control.recover_sign
        recover_x = tangent_x - 0.35 * dir_x
        recover_y = tangent_y - 0.35 * dir_y
        norm_x, norm_y, _ = _normalize_vector(recover_x, recover_y)
        step_distance = PLAYER_SPEED * dt * 0.85
        return norm_x * step_distance, norm_y * step_distance

    def _update_player_stuck_fallback(
        self,
        control: AgentControl,
        moved_distance: float,
        bumped: bool,
        should_be_moving: bool,
        dt: float,
    ) -> None:
        control.current_bumped = bumped
        control.current_progress_tiles = moved_distance / TILE_SIZE
        control.should_move = should_be_moving
        if control.recover_time_left > 0.0:
            control.no_progress_time = 0.0
            return
        if control.replan_after_recover:
            control.path.clear()
            control.path_target_cell = None
            control.repath_timer = 0.0
            control.force_patrol_time_left = 0.0
            control.replan_after_recover = False
        if not should_be_moving:
            control.no_progress_time = 0.0
            return
        min_progress_world = STUCK_MIN_PROGRESS_TILES * TILE_SIZE
        if bumped or moved_distance < min_progress_world:
            control.no_progress_time += dt
        else:
            control.no_progress_time = 0.0
        if control.no_progress_time < STUCK_GRACE_SEC:
            return
        control.no_progress_time = 0.0
        control.recover_time_left = STUCK_RECOVER_SEC
        control.recover_sign *= -1
        control.replan_after_recover = True
        control.arbiter.clear_filtered_move()
        control.move_smooth_x = 0.0
        control.move_smooth_y = 0.0

    def _update_player_no_reason_penalty(
        self,
        control: AgentControl,
        *,
        target_exists: bool,
        dt: float,
    ) -> None:
        if self.current_control is not self.agent_control:
            control.no_reason_timer = 0.0
            return
        if self.app_state not in (APP_STATE_PLAY_MAIN, APP_STATE_PLAY_TRAINING):
            control.no_reason_timer = 0.0
            return
        if not target_exists or control.current_reason != "no_reason" or self.player.weapon.is_reloading:
            control.no_reason_timer = 0.0
            return
        control.no_reason_timer += dt
        if (
            control.no_reason_cooldown <= 0.0
            and control.no_reason_timer >= NO_REASON_GRACE_SECONDS
        ):
            self._add_player_agent_reward(PLAYER_AGENT_PENALTY_NO_REASON)
            control.no_reason_cooldown = NO_REASON_PENALTY_COOLDOWN

    def _player_shoot_conditions_met(
        self,
        control: AgentControl,
        *,
        target_exists: bool,
        has_los: bool,
        aim_delta: float,
    ) -> bool:
        return (
            target_exists
            and has_los
            and abs(aim_delta) <= PLAYER_AGENT_AIM_THRESHOLD
            and control.aim_stable_time >= PLAYER_AGENT_AIM_STABLE_SECONDS
            and not self.player.weapon.is_reloading
            and self.player.weapon.ammo_in_mag > 0
            and self.player.weapon.shot_cooldown_left <= 0.0
        )

    def _apply_player_intent(self, control: AgentControl, intent: int, dt: float) -> bool:
        control.shoot_triggered_this_tick = False
        control.current_reason = "no_reason"
        self._maybe_retarget_player_agent(control)
        target_enemy = self._get_player_agent_target(control, allow_search=False)
        if target_enemy is None:
            control.current_phase = "SEARCH"
            target_enemy = self._get_player_agent_target(control, allow_search=True)
        control.blocked_reason = self._player_shot_blocked_reason(agent_request=True) or "none"
        control.shoot_ready = control.blocked_reason == "none"

        if target_enemy is None:
            control.current_effective_intent = INTENT_CHASE
            control.debug_intent_label = "CHASE"
            control.current_target_distance = 0.0
            control.current_has_los = False
            control.current_target_angle = self.player.angle
            control.current_aim_error = math.pi
            control.aim_stable_time = 0.0
            control.current_reason = "search_patrol"
            patrol_target, nav_target_cell = self._player_patrol_nav_target(control)
            control.current_phase = "SEARCH"
            nav_point, control.path, control.path_target_cell, control.repath_timer = self._resolve_navigation_waypoint(
                self.player.x,
                self.player.y,
                PLAYER_RADIUS,
                patrol_target[0],
                patrol_target[1],
                control.path,
                control.path_target_cell,
                control.repath_timer,
                dt,
            )
            control.current_nav_target_cell = control.path_target_cell or nav_target_cell
            control.has_path = bool(control.path)
            control.next_waypoint = control.path[0] if control.path else None
            self._update_player_no_reason_penalty(control, target_exists=False, dt=dt)
            return self._navigate_player_toward(
                control,
                nav_point[0],
                nav_point[1],
                dt,
                False,
                None,
                "nav",
            )

        player_cell = _world_to_cell(self.player.x, self.player.y)
        has_los = _has_line_of_sight(
            self.player.x,
            self.player.y,
            target_enemy.x,
            target_enemy.y,
            self.map,
        )
        target_distance = math.hypot(target_enemy.x - self.player.x, target_enemy.y - self.player.y)
        control.current_target_distance = target_distance
        control.current_has_los = has_los
        aim_angle, aim_delta = _target_angle_and_error(
            self.player.x,
            self.player.y,
            self.player.angle,
            target_enemy.x,
            target_enemy.y,
        )
        engage_min_world = ENGAGE_MIN_T * TILE_SIZE
        engage_max_world = ENGAGE_MAX_T * TILE_SIZE
        if has_los and abs(aim_delta) <= PLAYER_AGENT_AIM_THRESHOLD:
            control.aim_stable_time += dt
            self._add_player_agent_reward(PLAYER_AGENT_REWARD_AIM)
        else:
            control.aim_stable_time = 0.0

        shoot_conditions_met = self._player_shoot_conditions_met(
            control,
            target_exists=True,
            has_los=has_los,
            aim_delta=aim_delta,
        )
        if shoot_conditions_met:
            did_shoot = self.request_player_shoot(agent_request=True)
            control.shoot_triggered_this_tick = did_shoot
            control.blocked_reason = self._player_shot_blocked_reason(agent_request=True) or "none"
            control.shoot_ready = control.blocked_reason == "none"
            control.current_phase = "SHOOT" if did_shoot else "MOVE"
            if did_shoot:
                control.current_reason = "shoot_enemy"
        else:
            control.current_phase = "MOVE"

        if control.recover_time_left > 0.0:
            control.current_phase = "MOVE"
            control.current_effective_intent = INTENT_REPOSITION
            control.current_reason = "stuck_recover"
            control.debug_intent_label = "BACKOFF"
            control.path.clear()
            control.path_target_cell = None
            control.repath_timer = 0.0
            target_point = (self.player.x, self.player.y)
            nav_target_cell = _world_to_cell(self.player.x, self.player.y)
            should_shoot = has_los
            yaw_source = "recover"
        elif isinstance(target_enemy, HealthPackTarget):
            control.current_effective_intent = INTENT_CHASE
            control.debug_intent_label = "CHASE"
            target_point = (target_enemy.x, target_enemy.y)
            nav_target_cell = _world_to_cell(target_enemy.x, target_enemy.y)
            should_shoot = has_los
            yaw_source = "aim" if has_los else "nav"
        elif control.force_patrol_time_left > 0.0:
            control.current_phase = "MOVE"
            control.current_effective_intent = INTENT_REPOSITION
            control.current_reason = "stuck_patrol"
            control.debug_intent_label = "STRAFE"
            target_point, nav_target_cell = self._player_patrol_nav_target(control, force_refresh=control.patrol_target_cell is None)
            should_shoot = False
            yaw_source = "nav"
        elif not has_los or target_distance > engage_max_world:
            control.current_effective_intent = INTENT_CHASE
            if not isinstance(target_enemy, HealthPackTarget):
                control.current_reason = "chase_enemy"
            control.debug_intent_label = "CHASE"
            target_point, nav_target_cell, should_shoot = self._player_chase_nav_target(control, target_enemy)
            yaw_source = "nav"
        elif target_distance < engage_min_world:
            control.current_effective_intent = INTENT_EVADE
            if not isinstance(target_enemy, HealthPackTarget):
                control.current_reason = "backoff_enemy"
            control.debug_intent_label = "BACKOFF"
            target_point, nav_target_cell = self._player_backoff_strafe_nav_target(control, target_enemy, target_distance)
            should_shoot = has_los
            yaw_source = "aim"
        else:
            control.current_effective_intent = INTENT_ENGAGE
            if control.current_reason == "no_reason" and not isinstance(target_enemy, HealthPackTarget):
                control.current_reason = "strafe_enemy"
            elif not isinstance(target_enemy, HealthPackTarget):
                control.current_reason = "strafe_enemy"
            control.debug_intent_label = "STRAFE"
            target_point, nav_target_cell = self._player_strafe_nav_target(control, target_enemy, target_distance)
            should_shoot = True
            yaw_source = "aim"

        nav_point, control.path, control.path_target_cell, control.repath_timer = self._resolve_navigation_waypoint(
            self.player.x,
            self.player.y,
            PLAYER_RADIUS,
            target_point[0],
            target_point[1],
            control.path,
            control.path_target_cell,
            control.repath_timer,
            dt,
        )
        control.current_nav_target_cell = control.path_target_cell or nav_target_cell
        control.has_path = bool(control.path)
        control.next_waypoint = control.path[0] if control.path else None
        self._update_player_no_reason_penalty(control, target_exists=True, dt=dt)
        return self._navigate_player_toward(
            control,
            nav_point[0],
            nav_point[1],
            dt,
            should_shoot or has_los,
            target_enemy,
            yaw_source,
        )

    def _navigate_player_toward(
        self,
        control: AgentControl,
        target_x: float,
        target_y: float,
        dt: float,
        should_shoot: bool,
        target_enemy: Enemy | None,
        yaw_source: str,
    ) -> bool:
        control.arbiter.reset()
        control.near_wall = False
        dir_x, dir_y, distance = _normalize_vector(target_x - self.player.x, target_y - self.player.y)
        move_angle = self.player.angle
        if distance > 0.0001:
            move_angle = math.atan2(dir_y, dir_x)

        aim_angle = move_angle
        has_enemy_los = False
        aim_delta = math.pi
        if target_enemy is not None:
            enemy_dir_x, enemy_dir_y, enemy_distance = _normalize_vector(
                target_enemy.x - self.player.x,
                target_enemy.y - self.player.y,
            )
            if enemy_distance > 0.0001:
                aim_angle, aim_delta = _target_angle_and_error(
                    self.player.x,
                    self.player.y,
                    self.player.angle,
                    target_enemy.x,
                    target_enemy.y,
                )
                has_enemy_los = _has_line_of_sight(
                    self.player.x,
                    self.player.y,
                    target_enemy.x,
                    target_enemy.y,
                    self.map,
                )

        should_prefer_aim = target_enemy is not None and (
            should_shoot
            or yaw_source == "nav"
            or not self._is_critical_hp(self.player_hp, PLAYER_MAX_HP)
        )
        control.arbiter.suggest_yaw(yaw_source, move_angle, 1)
        if should_prefer_aim:
            control.arbiter.suggest_yaw("aim", aim_angle, 3 if should_shoot or yaw_source == "nav" else 2)
        if control.recover_time_left > 0.0:
            recover_yaw = (self.player.angle + math.radians(10.0 * control.recover_sign)) % math.tau
            control.arbiter.suggest_yaw("recover", recover_yaw, 1)
            recover_target_x = target_enemy.x if target_enemy is not None else target_x
            recover_target_y = target_enemy.y if target_enemy is not None else target_y
            recover_x, recover_y = self._player_recovery_move(control, recover_target_x, recover_target_y, dt)
            control.arbiter.desired_move_x = recover_x
            control.arbiter.desired_move_y = recover_y
        elif distance > TILE_SIZE * 0.2:
            control.arbiter.suggest_move_toward(
                self.player.x,
                self.player.y,
                target_x,
                target_y,
                PLAYER_SPEED,
                dt,
            )
        if control.arbiter.yaw_target is not None:
            self.player.angle, control.current_yaw_rate, _ = control.aim_controller.step(
                self.player.angle,
                control.arbiter.yaw_target,
                dt,
            )
        else:
            control.current_yaw_rate = 0.0
        control.current_yaw_source = control.arbiter.yaw_source
        if target_enemy is not None:
            aim_angle, aim_delta = _target_angle_and_error(
                self.player.x,
                self.player.y,
                self.player.angle,
                target_enemy.x,
                target_enemy.y,
            )
        else:
            aim_angle = self.player.angle
            aim_delta = 0.0
        control.current_aim_error = aim_delta
        control.current_target_angle = aim_angle
        desired_move_length = math.hypot(control.arbiter.desired_move_x, control.arbiter.desired_move_y)
        move_x, move_y = control.arbiter.finalize_move(dt)
        move_x, move_y, enemy_repulse_active = self._player_enemy_repulsion(move_x, move_y)
        control.enemy_repulse_active = enemy_repulse_active
        move_x, move_y, near_wall = self._player_wall_repulsion(move_x, move_y)
        control.near_wall = near_wall
        move_x, move_y = self._smooth_player_move_vector(control, move_x, move_y)
        control.current_move_vector = (move_x, move_y)
        start_x = self.player.x
        start_y = self.player.y
        bumped = self._move_player_by_vector(move_x, move_y)
        moved_distance = math.hypot(self.player.x - start_x, self.player.y - start_y)
        should_be_moving = (
            self.current_control is self.agent_control
            and self.app_state in (APP_STATE_PLAY_MAIN, APP_STATE_PLAY_TRAINING)
            and target_enemy is not None
        )
        self._update_player_stuck_fallback(
            control,
            moved_distance,
            bumped,
            should_be_moving,
            dt,
        )
        return bumped

    def _build_agent_control_state(self) -> dict[str, object]:
        nearest_enemy = min(
            self.enemies,
            key=lambda enemy: (enemy.x - self.player.x) ** 2 + (enemy.y - self.player.y) ** 2,
            default=None,
        )
        state: dict[str, object] = {
            "player_cell": _world_to_cell(self.player.x, self.player.y),
            "player_hp": self.player_hp,
            "wave": self.wave_number,
            "enemy_count": len(self.enemies),
        }
        if nearest_enemy is None:
            state["nearest_enemy"] = None
            return state

        distance = math.hypot(nearest_enemy.x - self.player.x, nearest_enemy.y - self.player.y)
        state["nearest_enemy"] = {
            "kind": nearest_enemy.kind,
            "distance_bin": _distance_bin(distance),
            "los": _has_line_of_sight(
                self.player.x,
                self.player.y,
                nearest_enemy.x,
                nearest_enemy.y,
                self.map,
            ),
            "relative_direction": _relative_direction_bin(
                nearest_enemy.x,
                nearest_enemy.y,
                self.player.x,
                self.player.y,
                self.player.angle,
            ),
        }
        return state

    def _enemy_chase_target(
        self,
        enemy: Enemy,
        dt: float,
        player_cell: tuple[int, int],
    ) -> tuple[tuple[float, float] | None, bool]:
        has_los = _has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map)
        if has_los:
            enemy.last_seen_player_cell = player_cell
            enemy.path.clear()
            enemy.path_target_cell = None
            return (self.player.x, self.player.y), True

        enemy.repath_timer -= dt
        target_cell = enemy.last_seen_player_cell or player_cell
        if enemy.repath_timer <= 0.0 or enemy.path_target_cell != target_cell:
            enemy.path = _build_path(_world_to_cell(enemy.x, enemy.y), target_cell, self.map)
            enemy.path_target_cell = target_cell
            enemy.repath_timer = ENEMY_PATH_RECALC_INTERVAL

        while enemy.path:
            waypoint_x, waypoint_y = _cell_center(*enemy.path[0])
            if math.hypot(waypoint_x - enemy.x, waypoint_y - enemy.y) > enemy.radius:
                break
            enemy.path.pop(0)

        if enemy.path:
            return _cell_center(*enemy.path[0]), False
        if enemy.last_seen_player_cell is not None:
            return _cell_center(*enemy.last_seen_player_cell), False
        return None, False

    def _move_enemy_toward_point(
        self,
        enemy: Enemy,
        target_x: float,
        target_y: float,
        dt: float,
    ) -> bool:
        steer_x = target_x - enemy.x
        steer_y = target_y - enemy.y
        steer_distance = math.hypot(steer_x, steer_y)
        if steer_distance <= 0.0001:
            return False

        move_scale = min(enemy.speed * dt, steer_distance) / steer_distance
        move_x = steer_x * move_scale
        move_y = steer_y * move_scale
        return self._move_enemy_by_vector(enemy, move_x, move_y)

    def _move_enemy_by_vector(self, enemy: Enemy, move_x: float, move_y: float) -> bool:
        intended = math.hypot(move_x, move_y)
        next_x, next_y = _move_with_slide(
            enemy.x,
            enemy.y,
            move_x,
            move_y,
            enemy.radius,
            self.map,
        )
        actual = math.hypot(next_x - enemy.x, next_y - enemy.y)
        enemy.x, enemy.y = next_x, next_y
        return intended > 0.01 and actual < intended * 0.35

    def _enemy_shoot_player(self, enemy: Enemy) -> bool:
        if enemy.shot_cooldown > 0.0 or enemy.damage <= 0:
            return False
        if not _has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map):
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

    def _advance_patrol_waypoint(self, enemy: Enemy) -> tuple[int, int] | None:
        if not enemy.patrol_points:
            return enemy.home_cell

        current_waypoint = enemy.patrol_points[enemy.patrol_index % len(enemy.patrol_points)]
        waypoint_x, waypoint_y = _cell_center(*current_waypoint)
        if math.hypot(waypoint_x - enemy.x, waypoint_y - enemy.y) <= TILE_SIZE * 0.2:
            enemy.patrol_index = (enemy.patrol_index + 1) % len(enemy.patrol_points)
            current_waypoint = enemy.patrol_points[enemy.patrol_index]
        return current_waypoint

    def _navigate_script_enemy(
        self,
        enemy: Enemy,
        target_x: float,
        target_y: float,
        dt: float,
    ) -> bool:
        nav_point, enemy.path, enemy.path_target_cell, enemy.repath_timer = self._resolve_navigation_waypoint(
            enemy.x,
            enemy.y,
            enemy.radius,
            target_x,
            target_y,
            enemy.path,
            enemy.path_target_cell,
            enemy.repath_timer,
            dt,
        )
        enemy.current_nav_target_cell = enemy.path_target_cell or _world_to_cell(target_x, target_y)
        enemy.has_path = bool(enemy.path)
        enemy.next_waypoint = enemy.path[0] if enemy.path else None

        steer_x = nav_point[0] - enemy.x
        steer_y = nav_point[1] - enemy.y
        steer_distance = math.hypot(steer_x, steer_y)
        if steer_distance <= 0.0001:
            enemy.current_move_vector = (0.0, 0.0)
            return False

        move_scale = min(enemy.speed * dt, steer_distance) / steer_distance
        move_x = steer_x * move_scale
        move_y = steer_y * move_scale
        enemy.current_move_vector = (move_x, move_y)
        if math.hypot(move_x, move_y) > 0.0001:
            enemy.angle = math.atan2(move_y, move_x)
        return self._move_enemy_by_vector(enemy, move_x, move_y)

    def _update_script_enemy(self, enemy: Enemy, dt: float) -> None:
        player_cell = _world_to_cell(self.player.x, self.player.y)
        distance = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
        dist_tiles = distance / TILE_SIZE
        has_los = _has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map)
        in_sight = has_los and dist_tiles <= enemy.sight_range_tiles

        enemy.current_target_distance = distance
        enemy.current_has_los = has_los
        enemy.alert_time_left = max(0.0, enemy.alert_time_left - dt)
        enemy.strafe_time_left = max(0.0, enemy.strafe_time_left - dt)
        if has_los:
            enemy.last_seen_player_cell = player_cell
            enemy.search_target_cell = player_cell
            enemy.no_los_time = 0.0
            enemy.alert_time_left = max(enemy.alert_time_left, SCRIPT_ALERT_HOLD_SECONDS)
        else:
            enemy.no_los_time += dt
            if dist_tiles <= enemy.hear_range_tiles:
                enemy.search_target_cell = player_cell
                enemy.alert_time_left = max(enemy.alert_time_left, SCRIPT_ALERT_HOLD_SECONDS)

        if in_sight:
            if enemy.attack_style == "melee":
                enemy.state = SCRIPT_STATE_ENGAGE if distance <= enemy.melee_range * 1.35 else SCRIPT_STATE_CHASE
            elif enemy.engage_min_range <= distance <= enemy.engage_max_range:
                enemy.state = SCRIPT_STATE_ENGAGE
            else:
                enemy.state = SCRIPT_STATE_CHASE
        elif dist_tiles <= enemy.hear_range_tiles:
            enemy.state = SCRIPT_STATE_SEARCH
        elif (
            enemy.state in (SCRIPT_STATE_SEARCH, SCRIPT_STATE_CHASE, SCRIPT_STATE_ENGAGE)
            and enemy.alert_time_left > 0.0
            and enemy.no_los_time < SCRIPT_LOST_TARGET_SECONDS
        ):
            enemy.state = SCRIPT_STATE_SEARCH
        elif enemy.no_los_time >= SCRIPT_LOST_TARGET_SECONDS and dist_tiles > enemy.hear_range_tiles:
            enemy.state = SCRIPT_STATE_RETURN
        elif enemy.state == SCRIPT_STATE_RETURN and enemy.home_cell is not None:
            enemy.state = SCRIPT_STATE_RETURN
        else:
            enemy.state = SCRIPT_STATE_PATROL

        enemy.current_move_vector = (0.0, 0.0)
        enemy.current_nav_target_cell = None
        enemy.has_path = False
        enemy.next_waypoint = None

        if enemy.state == SCRIPT_STATE_PATROL:
            patrol_cell = self._advance_patrol_waypoint(enemy)
            if patrol_cell is not None:
                patrol_x, patrol_y = _cell_center(*patrol_cell)
                self._navigate_script_enemy(enemy, patrol_x, patrol_y, dt)
            return

        if enemy.state == SCRIPT_STATE_RETURN:
            return_cell = self._advance_patrol_waypoint(enemy) or enemy.home_cell
            if return_cell is not None:
                return_x, return_y = _cell_center(*return_cell)
                self._navigate_script_enemy(enemy, return_x, return_y, dt)
                if math.hypot(return_x - enemy.x, return_y - enemy.y) <= TILE_SIZE * 0.25:
                    enemy.state = SCRIPT_STATE_PATROL
            return

        if enemy.state == SCRIPT_STATE_SEARCH:
            search_cell = enemy.search_target_cell or enemy.last_seen_player_cell or player_cell
            enemy.current_nav_target_cell = search_cell
            search_x, search_y = _cell_center(*search_cell)
            self._navigate_script_enemy(enemy, search_x, search_y, dt)
            if math.hypot(search_x - enemy.x, search_y - enemy.y) <= TILE_SIZE * 0.3 and not has_los:
                enemy.state = SCRIPT_STATE_RETURN if dist_tiles > enemy.hear_range_tiles else SCRIPT_STATE_PATROL
            return

        if enemy.state == SCRIPT_STATE_CHASE:
            chase_cell = player_cell if has_los else (enemy.last_seen_player_cell or enemy.search_target_cell or player_cell)
            enemy.current_nav_target_cell = chase_cell
            chase_x, chase_y = _cell_center(*chase_cell)
            self._navigate_script_enemy(enemy, chase_x, chase_y, dt)
            return

        if enemy.attack_style == "melee":
            if distance > enemy.melee_range * 0.9:
                self._navigate_script_enemy(enemy, self.player.x, self.player.y, dt)
            if distance <= enemy.melee_range:
                self._enemy_melee_attack_player(enemy)
            return

        if distance > enemy.engage_max_range:
            target_x, target_y = self.player.x, self.player.y
        elif distance < enemy.engage_min_range:
            dir_x, dir_y, _ = _normalize_vector(enemy.x - self.player.x, enemy.y - self.player.y)
            target_x, target_y = _project_point(
                enemy.x,
                enemy.y,
                dir_x,
                dir_y,
                max(TILE_SIZE * 1.5, enemy.engage_min_range - distance + TILE_SIZE * 0.4),
            )
        else:
            if enemy.strafe_time_left <= 0.0:
                enemy.strafe_sign *= -1
                enemy.strafe_time_left = 0.9 + (enemy.entity_id % 3) * 0.2
            target_x, target_y = _flank_point(
                enemy.x,
                enemy.y,
                self.player.x,
                self.player.y,
                enemy.strafe_sign,
                TILE_SIZE * 1.8,
            )
        self._navigate_script_enemy(enemy, target_x, target_y, dt)
        if has_los:
            self._enemy_shoot_player(enemy)

    def _get_rl_enemy_target(self, enemy: RLEnemy) -> Player | None:
        aggro_tiles = 0.5 * max(len(self.map), len(self.map[0]))
        aggro_range = aggro_tiles * TILE_SIZE
        hysteresis_range = (aggro_tiles + 2.0) * TILE_SIZE
        distance = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
        allowed_range = hysteresis_range if enemy.target_lock.target_key == "player" else aggro_range
        if self.player_hp <= 0 or distance > allowed_range:
            enemy.target_lock.clear()
            enemy.current_target_distance = 0.0
            enemy.current_has_los = False
            return None

        has_los = _has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map)
        enemy.current_target_distance = distance
        enemy.current_has_los = has_los
        if has_los:
            enemy.last_seen_player_cell = _world_to_cell(self.player.x, self.player.y)
        else:
            path = _build_path(_world_to_cell(enemy.x, enemy.y), _world_to_cell(self.player.x, self.player.y), self.map)
            if not path:
                enemy.target_lock.clear()
                return None

        if enemy.target_lock.target_key != "player":
            enemy.target_lock.assign("player", self.player, distance)
        else:
            enemy.target_lock.score = distance
        return self.player

    def _apply_enemy_intent(self, enemy: Enemy, intent: int, dt: float) -> bool:
        enemy_target: Player | None
        if isinstance(enemy, RLEnemy):
            enemy_target = self._get_rl_enemy_target(enemy)
        else:
            enemy_target = self.player

        if enemy_target is None:
            if isinstance(enemy, RLEnemy):
                enemy.current_effective_intent = INTENT_CHASE
                enemy.current_nav_target_cell = None
                enemy.has_path = False
                enemy.next_waypoint = None
                enemy.current_move_vector = (0.0, 0.0)
                enemy.current_yaw_source = "nav"
                enemy.current_yaw_rate = 0.0
                enemy.arbiter.reset()
                enemy.arbiter.clear_filtered_move()
            return False

        effective_intent = intent
        if isinstance(enemy, RLEnemy):
            effective_intent = _banded_intent(
                enemy.current_target_distance,
                enemy.current_has_los,
                RL_ENEMY_RANGE_MIN,
                RL_ENEMY_RANGE_MAX,
            )
            enemy.current_effective_intent = effective_intent

        if isinstance(enemy, RLEnemy) and effective_intent == INTENT_CHASE and not enemy.current_has_los:
            nav_target_cell = enemy.last_seen_player_cell or _world_to_cell(enemy_target.x, enemy_target.y)
            nav_target_point = _cell_center(*nav_target_cell)
            target_point = nav_target_point
            should_shoot = False
        elif isinstance(enemy, RLEnemy) and effective_intent == INTENT_ENGAGE and enemy.current_has_los:
            side_sign = 1 if enemy.entity_id % 2 == 0 else -1
            target_point = _flank_point(
                enemy.x,
                enemy.y,
                enemy_target.x,
                enemy_target.y,
                side_sign,
                TILE_SIZE * 1.5,
            )
            nav_target_cell = _world_to_cell(target_point[0], target_point[1])
            should_shoot = True
        else:
            target_point, should_shoot = self._intent_target_point(
                enemy.x,
                enemy.y,
                enemy.radius,
                enemy_target.x,
                enemy_target.y,
                effective_intent,
                _world_to_cell(enemy.x, enemy.y)[0] + _world_to_cell(enemy.x, enemy.y)[1],
            )
            nav_target_cell = _world_to_cell(target_point[0], target_point[1])
        nav_point, enemy.path, enemy.path_target_cell, enemy.repath_timer = self._resolve_navigation_waypoint(
            enemy.x,
            enemy.y,
            enemy.radius,
            target_point[0],
            target_point[1],
            enemy.path,
            enemy.path_target_cell,
            enemy.repath_timer,
            dt,
        )
        if isinstance(enemy, RLEnemy):
            enemy.current_nav_target_cell = enemy.path_target_cell or nav_target_cell
            enemy.has_path = bool(enemy.path)
            enemy.next_waypoint = enemy.path[0] if enemy.path else None
        move_angle = enemy.angle
        dir_x, dir_y, nav_distance = _normalize_vector(nav_point[0] - enemy.x, nav_point[1] - enemy.y)
        if nav_distance > 0.0001:
            move_angle = math.atan2(dir_y, dir_x)
        aim_angle = math.atan2(enemy_target.y - enemy.y, enemy_target.x - enemy.x)
        yaw_source = "evade" if effective_intent in (INTENT_EVADE, INTENT_REPOSITION) else "nav"
        if isinstance(enemy, RLEnemy):
            enemy.arbiter.reset()
            enemy.arbiter.suggest_yaw(yaw_source, move_angle, 1)
            should_prefer_aim = (
                effective_intent in (INTENT_CHASE, INTENT_ENGAGE)
                or not self._is_critical_hp(enemy.hp, enemy.max_hp)
            )
            if should_prefer_aim:
                enemy.arbiter.suggest_yaw("aim", aim_angle, 3 if should_shoot else 2)
            if nav_distance > TILE_SIZE * 0.1:
                enemy.arbiter.suggest_move_toward(enemy.x, enemy.y, nav_point[0], nav_point[1], enemy.speed, dt)
            if enemy.arbiter.yaw_target is not None:
                enemy.angle, enemy.current_yaw_rate, _ = enemy.aim_controller.step(
                    enemy.angle,
                    enemy.arbiter.yaw_target,
                    dt,
                )
            else:
                enemy.current_yaw_rate = 0.0
            enemy.current_yaw_source = enemy.arbiter.yaw_source
            move_x, move_y = enemy.arbiter.finalize_move(dt)
            enemy.current_move_vector = (move_x, move_y)
            bumped = self._move_enemy_by_vector(enemy, move_x, move_y)
        else:
            bumped = self._move_enemy_toward_point(enemy, nav_point[0], nav_point[1], dt)
        if should_shoot:
            self._enemy_shoot_player(enemy)
        return bumped

    def _add_rl_reward(self, enemy: RLEnemy, amount: float) -> None:
        enemy.pending_reward += amount
        enemy.last_reward_delta = amount

    def _rl_enemy_observation(self, enemy: RLEnemy, has_los: bool) -> tuple[int, int, int, int, int]:
        distance = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
        return (
            _distance_bin(distance),
            int(has_los),
            _hp_bin(enemy.hp, enemy.max_hp),
            _relative_direction_bin(enemy.x, enemy.y, self.player.x, self.player.y, self.player.angle),
            int(enemy.under_fire_timer > 0.0),
        )

    def _update_rl_enemy(self, enemy: RLEnemy, dt: float) -> None:
        enemy.target_lock.tick(dt)
        enemy.last_damage_time += dt
        has_los_before = _has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map)
        current_obs = self._rl_enemy_observation(enemy, has_los_before)
        enemy.current_obs = current_obs
        current_state = self.rl_agent.get_state(current_obs)
        enemy.decision_time_left = max(0.0, enemy.decision_time_left - dt)
        enemy.action_time_left = max(0.0, enemy.action_time_left - dt)
        enemy.action_locked = enemy.action_time_left > 0.0 and not enemy.emergency_unlock
        if enemy.emergency_unlock:
            if enemy.current_state is not None and enemy.current_action is not None:
                self.rl_agent.update(
                    enemy.current_state,
                    enemy.current_action,
                    enemy.pending_reward,
                    current_state,
                    False,
                )
            enemy.current_state = current_state
            enemy.current_action = self.rl_agent.select_action(current_state)
            enemy.pending_reward = 0.0
            enemy.last_reward_delta = 0.0
            enemy.decision_time_left = enemy.decision_interval_seconds
            enemy.action_time_left = INTENT_HOLD_SECONDS
            enemy.action_locked = True
            enemy.emergency_unlock = False
        elif enemy.current_action is None:
            if enemy.current_state is not None and enemy.current_action is not None:
                self.rl_agent.update(
                    enemy.current_state,
                    enemy.current_action,
                    enemy.pending_reward,
                    current_state,
                    False,
                )
            enemy.current_state = current_state
            enemy.current_action = self.rl_agent.select_action(current_state)
            enemy.pending_reward = 0.0
            enemy.last_reward_delta = 0.0
            enemy.decision_time_left = enemy.decision_interval_seconds
            enemy.action_time_left = INTENT_HOLD_SECONDS
            enemy.action_locked = True
        elif enemy.decision_time_left <= 0.0:
            enemy.decision_time_left = enemy.decision_interval_seconds
            if enemy.action_time_left <= 0.0:
                if enemy.current_state is not None and enemy.current_action is not None:
                    self.rl_agent.update(
                        enemy.current_state,
                        enemy.current_action,
                        enemy.pending_reward,
                        current_state,
                        False,
                    )
                enemy.current_state = current_state
                enemy.current_action = self.rl_agent.select_action(current_state)
                enemy.pending_reward = 0.0
                enemy.last_reward_delta = 0.0
                enemy.action_time_left = INTENT_HOLD_SECONDS
                enemy.action_locked = True

        bumped = self._apply_enemy_intent(enemy, enemy.current_action or INTENT_CHASE, dt)
        has_los_after = _has_line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y, self.map)
        distance = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
        if has_los_after and TILE_SIZE * 2.5 <= distance <= TILE_SIZE * 5.0:
            self._add_rl_reward(enemy, RL_REWARD_SHAPING)
        if enemy.under_fire_timer > 0.0 and has_los_before and not has_los_after:
            self._add_rl_reward(enemy, RL_REWARD_BREAK_LOS)
        if enemy.current_effective_intent == INTENT_ENGAGE and enemy.shot_cooldown == enemy.shot_cooldown_max:
            self._add_rl_reward(enemy, RL_REWARD_HIT_PLAYER)

        if bumped:
            self._add_rl_reward(enemy, RL_REWARD_WALL_BUMP)
        enemy.last_has_los = has_los_after

    def _update_enemies(self, dt: float) -> None:
        for enemy in self.enemies:
            enemy.shot_cooldown = max(0.0, enemy.shot_cooldown - dt)
            enemy.under_fire_timer = max(0.0, enemy.under_fire_timer - dt)
            enemy.hit_flash_timer = max(0.0, enemy.hit_flash_timer - dt)
            if isinstance(enemy, HealthPackTarget):
                enemy.current_move_vector = (0.0, 0.0)
                enemy.current_nav_target_cell = _world_to_cell(enemy.x, enemy.y)
                enemy.has_path = False
                enemy.next_waypoint = None
                continue
            if isinstance(enemy, RLEnemy):
                self._update_rl_enemy(enemy, dt)
                continue
            self._update_script_enemy(enemy, dt)

    def _apply_destroy_reward(self, enemy: Enemy) -> None:
        if isinstance(enemy, HealthPackTarget):
            hp_before = self.player_hp
            self.score += enemy.score
            if self.game_mode == "arena":
                self.training_score_total += enemy.score
                self.episode_stats.record_score(enemy.score)
            self.player_hp = min(PLAYER_MAX_HP, self.player_hp + enemy.heal_amount)
            hp_gained = self.player_hp - hp_before
            pack_reward = PLAYER_AGENT_REWARD_PACK_KILL + hp_gained * PLAYER_AGENT_REWARD_PACK_HEAL_SCALE
            if hp_gained > 0 and hp_before / max(PLAYER_MAX_HP, 1) < 0.35:
                pack_reward += PLAYER_AGENT_REWARD_PACK_LOW_HP_BONUS
            self._add_player_agent_pack_reward(pack_reward)
            self._show_feedback(f"+{enemy.score}", f"+{enemy.heal_amount} HP")
            return

        self.score += enemy.score
        self.kill_count += 1
        if self.game_mode == "arena":
            self.training_score_total += enemy.score
            self.training_kills_total += 1
            self.episode_stats.record_kill(enemy.score)
        self.player_hp = min(PLAYER_MAX_HP, self.player_hp + KILL_HEAL_NORMAL)
        self._show_feedback(f"+{KILL_HEAL_NORMAL} HP")

    def _player_shot_blocked_reason(self, agent_request: bool) -> str | None:
        if self.player.weapon.is_reloading:
            return "reloading"
        if self.player.weapon.ammo_in_mag == 0:
            return "empty"
        if self.player.weapon.shot_cooldown_left > 0.0:
            return "cooldown"
        return None

    def request_player_shoot(self, *, agent_request: bool) -> bool:
        if self.app_state == APP_STATE_MENU:
            if agent_request:
                self.agent_control.blocked_reason = "menu"
                self.agent_control.shoot_ready = False
            return False
        if self.app_state == APP_STATE_PAUSED:
            if agent_request:
                self.agent_control.blocked_reason = "paused"
                self.agent_control.shoot_ready = False
            return False
        if self.app_state == APP_STATE_GAME_OVER:
            if agent_request:
                self.agent_control.blocked_reason = "game_over"
                self.agent_control.shoot_ready = False
            return False

        blocked_reason = self._player_shot_blocked_reason(agent_request)
        if blocked_reason is not None:
            if blocked_reason == "empty":
                self.audio.play("empty", SFX_VOLUME_EMPTY)
                self._request_player_reload()
            if agent_request:
                self.agent_control.blocked_reason = blocked_reason
                self.agent_control.shoot_ready = False
            return False

        fired = self._fire_player_shot()
        if fired and agent_request:
            if self.player.weapon.ammo_in_mag == 0:
                self._request_player_reload()
            self.agent_control.blocked_reason = self._player_shot_blocked_reason(agent_request=True) or "none"
            self.agent_control.shoot_ready = False
        return fired

    def _fire_player_shot(self) -> bool:
        if not self.player.weapon.consume_round():
            return False

        self.audio.play("shoot", SFX_VOLUME_SHOOT)
        target = self._pick_crosshair_target()
        if target is None:
            return True

        self.audio.play("hit", SFX_VOLUME_HIT)
        self._add_player_agent_reward(PLAYER_AGENT_REWARD_HIT)
        if isinstance(target, RLEnemy):
            self._add_rl_reward(target, RL_REWARD_GOT_HIT)
            target.under_fire_timer = UNDER_FIRE_WINDOW_SECONDS
            target.emergency_unlock = True
            target.decision_time_left = 0.0
            target.action_time_left = 0.0
            target.action_locked = False
            target.last_damage_time = 0.0
        target.hit_flash_timer = HIT_FLASH_SECONDS
        target.hp -= WEAPON_DAMAGE
        if target.hp <= 0:
            self._add_player_agent_reward(PLAYER_AGENT_REWARD_KILL)
            if isinstance(target, RLEnemy) and target.current_state is not None and target.current_action is not None:
                self._add_rl_reward(target, RL_REWARD_DEATH)
                self.rl_agent.update(
                    target.current_state,
                    target.current_action,
                    target.pending_reward,
                    target.current_state,
                    True,
                )
            self.audio.play("enemy_die", SFX_VOLUME_ENEMY_DIE)
            self._apply_destroy_reward(target)
            self.enemies.remove(target)
            if self.game_mode == "arena" and not self._hostile_enemies():
                self._finish_training_episode(True)
        self.weapon_recoil_time_left = self.weapon_recoil_duration
        return True

    def _start_wave_intermission(self) -> None:
        self.last_wave_bonus = WAVE_CLEAR_BONUS_BASE * self.wave_number
        self.score += self.last_wave_bonus
        self.intermission_remaining = WAVE_INTERMISSION_SECONDS

    def _wave_enemy_count(self, wave_number: int) -> int:
        return min(
            max(0, len(ENEMY_SPAWN_CELLS) - 1),
            WAVE_BASE_COUNT + wave_number * WAVE_COUNT_GROWTH,
        )

    def _rl_epsilon_for_wave(self, wave_number: int) -> float:
        return max(
            self.rl_agent.epsilon_min,
            RL_WAVE_EPSILON_START - (wave_number - 1) * RL_WAVE_EPSILON_STEP,
        )

    def _wave_type_weights(self, wave_number: int) -> dict[str, float]:
        imp_weight = max(0.28, 0.72 - 0.05 * (wave_number - 1))
        soldier_weight = min(0.46, 0.22 + 0.04 * (wave_number - 1))
        tank_weight = min(0.26, 0.03 * max(0, wave_number - 2))
        total = imp_weight + soldier_weight + tank_weight
        return {
            "imp": imp_weight / total,
            "soldier": soldier_weight / total,
            "tank": tank_weight / total,
        }

    def _spawn_wave(self, wave_number: int) -> list[Enemy]:
        hp_multiplier = 1.0 + (wave_number - 1) * WAVE_HP_SCALE
        speed_multiplier = 1.0 + (wave_number - 1) * WAVE_SPEED_SCALE
        normal_count = self._wave_enemy_count(wave_number)
        weights = self._wave_type_weights(wave_number)
        spawn_cells = ENEMY_SPAWN_CELLS[:]
        self.random.shuffle(spawn_cells)
        enemies: list[Enemy] = []
        used_spawn_cells: set[tuple[int, int]] = set()

        for spawn_index, (cell_x, cell_y) in enumerate(spawn_cells[:normal_count]):
            used_spawn_cells.add((cell_x, cell_y))
            kind = self.random.choices(
                population=list(weights.keys()),
                weights=list(weights.values()),
                k=1,
            )[0]
            definition = ENEMY_TYPES[kind]
            spawn_world_x = (cell_x + 0.5) * TILE_SIZE
            spawn_world_y = (cell_y + 0.5) * TILE_SIZE
            spawn_x, spawn_y = _find_free_spawn_position(
                spawn_world_x,
                spawn_world_y,
                definition["radius"],
                self.map,
                spawn_index,
            )
            scaled_hp = max(1, int(round(definition["hp"] * hp_multiplier)))
            enemy_cls = Enemy
            damage = definition.get("damage", 0)
            shot_cooldown = definition.get("shoot_cooldown", 0.0)
            scaled_speed = definition["speed"] * speed_multiplier
            enemies.append(
                enemy_cls(
                    kind=kind,
                    x=spawn_x,
                    y=spawn_y,
                    hp=scaled_hp,
                    max_hp=scaled_hp,
                    speed=scaled_speed,
                    score=definition["score"],
                    radius=definition["radius"],
                    size=definition["size"],
                    color=definition["color"],
                    aggro_range=ENEMY_AGGRO_RANGE,
                    damage=damage,
                    shot_cooldown_max=shot_cooldown,
                    entity_id=self._allocate_entity_id(),
                )
            )
            self._configure_script_enemy(enemies[-1], (cell_x, cell_y))

        health_pack = self._spawn_health_pack_target(used_spawn_cells)
        if health_pack is not None:
            enemies.append(health_pack)
        return enemies

    def _spawn_health_pack_target(
        self,
        used_spawn_cells: set[tuple[int, int]],
    ) -> HealthPackTarget | None:
        definition = ENEMY_TYPES["health_pack"]
        min_distance = HEALTH_PACK_MIN_SPAWN_DISTANCE_TILES * TILE_SIZE
        player_cell = _world_to_cell(self.player.x, self.player.y)

        candidate_cells = [
            cell
            for cell in ENEMY_SPAWN_CELLS
            if cell not in used_spawn_cells and _is_open_cell(cell[0], cell[1], self.map)
        ]
        if not candidate_cells:
            candidate_cells = [
                (cell_x, cell_y)
                for cell_y, row in enumerate(self.map)
                for cell_x, tile in enumerate(row)
                if tile == "0"
            ]

        far_cells = [
            cell
            for cell in candidate_cells
            if math.hypot(_cell_center(*cell)[0] - self.player.x, _cell_center(*cell)[1] - self.player.y) >= min_distance
        ]
        if far_cells:
            candidate_cells = far_cells
        if not candidate_cells:
            return None

        best_cell = max(
            candidate_cells,
            key=lambda cell: (
                math.hypot(_cell_center(*cell)[0] - self.player.x, _cell_center(*cell)[1] - self.player.y),
                math.dist(cell, player_cell),
            ),
        )
        spawn_world_x, spawn_world_y = _cell_center(*best_cell)
        spawn_x, spawn_y = _find_free_spawn_position(
            spawn_world_x,
            spawn_world_y,
            definition["radius"],
            self.map,
            0,
        )
        return HealthPackTarget(
            kind="health_pack",
            x=spawn_x,
            y=spawn_y,
            hp=HEALTH_PACK_HP,
            max_hp=HEALTH_PACK_HP,
            speed=0.0,
            score=HEALTH_PACK_SCORE,
            radius=definition["radius"],
            size=definition["size"],
            color=definition["color"],
            aggro_range=0.0,
            entity_id=self._allocate_entity_id(),
            home_cell=best_cell,
            current_nav_target_cell=best_cell,
        )

def run(*, train_episodes: int | None = None, render_every: int = 0) -> None:
    game = Game()
    if train_episodes is not None:
        game.run_training_episodes(train_episodes, render_every=render_every)
        return
    game.run()
