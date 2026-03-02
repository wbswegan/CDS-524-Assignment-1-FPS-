from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeStats:
    episode_id: int = 1
    eps_used: float = 0.0
    steps: int = 0
    kills: int = 0
    score_delta: int = 0
    pos_reward: float = 0.0
    penalty: float = 0.0
    hp_end: int = 0

    def begin(self, *, episode_id: int, eps_used: float) -> None:
        self.episode_id = episode_id
        self.eps_used = eps_used
        self.steps = 0
        self.kills = 0
        self.score_delta = 0
        self.pos_reward = 0.0
        self.penalty = 0.0
        self.hp_end = 0

    def record_step(self) -> None:
        self.steps += 1

    def record_reward(self, amount: float) -> None:
        if amount > 0.0:
            self.pos_reward += amount
        elif amount < 0.0:
            self.penalty += -amount

    def record_kill(self, score_delta: int) -> None:
        self.kills += 1
        self.score_delta += score_delta

    def record_score(self, score_delta: int) -> None:
        self.score_delta += score_delta

    def finish(self, *, hp_end: int) -> None:
        self.hp_end = hp_end

    @property
    def episode_return(self) -> float:
        return self.pos_reward - self.penalty
