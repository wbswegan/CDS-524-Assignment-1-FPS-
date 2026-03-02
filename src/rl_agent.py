from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class QLearningAgent:
    def __init__(
        self,
        num_actions: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._rng = random.Random(seed)
        self._state_to_id: dict[str, int] = {}
        self._id_to_state: dict[int, str] = {}
        self._q_table: dict[int, list[float]] = {}

    def get_state(self, obs: Any) -> int:
        key = self._serialize_obs(obs)
        if key not in self._state_to_id:
            state_id = len(self._state_to_id)
            self._state_to_id[key] = state_id
            self._id_to_state[state_id] = key
            self._q_table[state_id] = [0.0] * self.num_actions
        return self._state_to_id[key]

    def select_action(self, state: int) -> int:
        q_values = self._ensure_state(state)
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.num_actions)

        max_q = max(q_values)
        best_actions = [action for action, value in enumerate(q_values) if value == max_q]
        return self._rng.choice(best_actions)

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        if not 0 <= a < self.num_actions:
            raise ValueError(f"action index {a} is out of bounds for {self.num_actions} actions")

        q_values = self._ensure_state(s)
        next_q_values = self._ensure_state(s2)
        current_q = q_values[a]
        target = r if done else r + self.gamma * max(next_q_values)
        q_values[a] = current_q + self.alpha * (target - current_q)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_actions": self.num_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_to_id": self._state_to_id,
            "q_table": {str(state): values for state, values in self._q_table.items()},
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> QLearningAgent:
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        agent = cls(
            payload["num_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            seed=seed,
        )
        agent._state_to_id = {str(key): int(value) for key, value in payload["state_to_id"].items()}
        agent._id_to_state = {
            state_id: state_key for state_key, state_id in agent._state_to_id.items()
        }
        agent._q_table = {
            int(state): [float(value) for value in values]
            for state, values in payload["q_table"].items()
        }
        for state_id in agent._state_to_id.values():
            agent._q_table.setdefault(state_id, [0.0] * agent.num_actions)
        return agent

    def q_values(self, state: int) -> list[float]:
        return list(self._ensure_state(state))

    def _ensure_state(self, state: int) -> list[float]:
        if state not in self._q_table:
            self._q_table[state] = [0.0] * self.num_actions
        return self._q_table[state]

    def _serialize_obs(self, obs: Any) -> str:
        normalized = self._normalize_obs(obs)
        return json.dumps(normalized, separators=(",", ":"), sort_keys=True)

    def _normalize_obs(self, obs: Any) -> Any:
        if isinstance(obs, dict):
            return {str(key): self._normalize_obs(value) for key, value in sorted(obs.items())}
        if isinstance(obs, (list, tuple)):
            return [self._normalize_obs(value) for value in obs]
        if isinstance(obs, bool):
            return obs
        if isinstance(obs, int):
            return obs
        if isinstance(obs, float):
            return round(obs, 6)
        if obs is None:
            return None
        return str(obs)
