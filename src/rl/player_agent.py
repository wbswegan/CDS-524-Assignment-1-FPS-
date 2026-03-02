from __future__ import annotations

import json
from pathlib import Path

from src.rl.q_agent import QLearningAgent


class PlayerQLearningAgent:
    action_count = 4
    schema_version = 2
    epsilon_start = 1.0
    epsilon_decay = 0.985
    epsilon_min = 0.05

    def __init__(self, path: Path) -> None:
        self.path = path
        self.agent = self._load()
        self.training_enabled = True
        self.training_epsilon = self.agent.epsilon
        self.current_state: int | None = None
        self.current_obs: tuple[int, int, int, int, int, int, int, int] | None = None
        self.current_action: int = 0
        self.pending_reward = 0.0
        self.last_reward_delta = 0.0

    def _load(self) -> QLearningAgent:
        legacy_path = self.path.with_name("player_q.json")
        backup_path = self.path.with_name("player_q_backup.json")
        if not self.path.exists() and legacy_path.exists():
            try:
                if backup_path.exists():
                    backup_path.unlink()
                legacy_path.replace(backup_path)
            except OSError:
                pass
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == self.schema_version:
                    agent = QLearningAgent.load(self.path, seed=11)
                    if agent.num_actions == self.action_count:
                        return agent
                if backup_path.exists():
                    backup_path.unlink()
                self.path.replace(backup_path)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return QLearningAgent(
            self.action_count,
            alpha=0.12,
            gamma=0.94,
            epsilon=self.epsilon_start,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
            seed=11,
        )

    def save(self) -> None:
        try:
            saved_path = self.agent.save(self.path)
            payload = json.loads(saved_path.read_text(encoding="utf-8"))
            payload["schema_version"] = self.schema_version
            saved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def reload(self) -> None:
        self.agent = self._load()
        self.training_epsilon = self.agent.epsilon
        if not self.training_enabled:
            self.agent.epsilon = 0.0

    def reset_exploration(self) -> None:
        self.training_enabled = True
        self.training_epsilon = self.epsilon_start
        self.agent.epsilon = self.epsilon_start
        self.agent.epsilon_decay = self.epsilon_decay
        self.agent.epsilon_min = self.epsilon_min

    def reset_for_new_run(
        self,
        _run_id: str,
        *,
        eps_start: float | None = None,
        eps_decay: float | None = None,
        eps_min: float | None = None,
    ) -> None:
        if eps_start is not None:
            self.epsilon_start = eps_start
        if eps_decay is not None:
            self.epsilon_decay = eps_decay
        if eps_min is not None:
            self.epsilon_min = eps_min
        self.reset_exploration()

    def add_reward(self, amount: float) -> None:
        self.pending_reward += amount
        self.last_reward_delta = amount

    def record_reward(self, amount: float) -> None:
        self.add_reward(amount)

    def set_training_enabled(self, enabled: bool) -> None:
        self.training_enabled = enabled
        if enabled:
            self.agent.epsilon = max(self.agent.epsilon_min, self.training_epsilon)
        else:
            self.training_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0

    def set_eval_mode(self, is_eval: bool) -> None:
        self.set_training_enabled(not is_eval)

    @property
    def is_eval_mode(self) -> bool:
        return not self.training_enabled

    def select_action(self, obs: tuple[int, int, int, int, int, int, int, int]) -> int:
        if self.training_enabled:
            self.agent.epsilon = self.training_epsilon
        state = self.agent.get_state(obs)
        if self.current_state is not None:
            self.agent.update(
                self.current_state,
                self.current_action,
                self.pending_reward,
                state,
                False,
            )
            if not self.training_enabled:
                self.agent.epsilon = 0.0
            else:
                self.agent.epsilon = self.training_epsilon
        self.current_obs = obs
        self.current_state = state
        self.current_action = self.agent.select_action(state)
        self.pending_reward = 0.0
        self.last_reward_delta = 0.0
        return self.current_action

    def begin_episode(self) -> float:
        if self.training_enabled:
            self.agent.epsilon = self.training_epsilon
            return self.training_epsilon
        self.agent.epsilon = 0.0
        return 0.0

    def finish_episode(self) -> None:
        if self.current_state is None:
            return
        if self.training_enabled:
            self.agent.epsilon = self.training_epsilon
        self.agent.update(
            self.current_state,
            self.current_action,
            self.pending_reward,
            self.current_state,
            True,
        )
        if not self.training_enabled:
            self.agent.epsilon = 0.0
        else:
            self.agent.epsilon = self.training_epsilon
        self.current_state = None
        self.current_obs = None
        self.pending_reward = 0.0
        self.save()

    def advance_exploration_episode(self) -> None:
        if not self.training_enabled:
            self.agent.epsilon = 0.0
            return
        self.training_epsilon = max(self.epsilon_min, self.training_epsilon * self.epsilon_decay)
        self.agent.epsilon = self.training_epsilon

    def end_episode(self) -> None:
        self.finish_episode()

    @property
    def epsilon(self) -> float:
        return self.agent.epsilon

    @property
    def epsilon_decay_value(self) -> float:
        return self.epsilon_decay

    @property
    def epsilon_min_value(self) -> float:
        return self.epsilon_min
