from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.logging.episode_stats import EpisodeStats


TRAINING_LOG_FIELDS = [
    "run_id",
    "episode_id",
    "mode",
    "epsilon",
    "return",
    "pos_reward",
    "penalty",
    "steps",
    "kills",
    "score_delta",
    "hp_end",
    "kills_total",
    "score_total",
]


@dataclass
class RunLogger:
    path: Path

    @staticmethod
    def new_run_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def append_episode(
        self,
        *,
        run_id: str,
        mode: str,
        stats: EpisodeStats,
        kills_total: int | None = None,
        score_total: int | None = None,
    ) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_header = self._ensure_schema_header()
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRAINING_LOG_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "run_id": run_id,
                        "episode_id": stats.episode_id,
                        "mode": mode,
                        "epsilon": f"{stats.eps_used:.4f}",
                        "return": f"{stats.episode_return:.4f}",
                        "pos_reward": f"{stats.pos_reward:.4f}",
                        "penalty": f"{stats.penalty:.4f}",
                        "steps": stats.steps,
                        "kills": stats.kills,
                        "score_delta": stats.score_delta,
                        "hp_end": stats.hp_end,
                        "kills_total": "" if kills_total is None else kills_total,
                        "score_total": "" if score_total is None else score_total,
                    }
                )
        except OSError as exc:
            print(f"Warning: failed to append training log row to {self.path}: {exc}")

    def load_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                return [row for row in reader if row.get("episode_id")]
        except OSError as exc:
            print(f"Warning: failed to load training log at {self.path}: {exc}")
            return []

    def latest_run_rows(self) -> tuple[str | None, list[dict[str, str]]]:
        rows = self.load_rows()
        run_ids = sorted({row.get("run_id", "").strip() for row in rows if row.get("run_id", "").strip()})
        if not run_ids:
            return None, []
        latest_run_id = run_ids[-1]
        return latest_run_id, [row for row in rows if row.get("run_id", "").strip() == latest_run_id]

    def _ensure_schema_header(self) -> bool:
        expected_header = ",".join(TRAINING_LOG_FIELDS)
        write_header = not self.path.exists()
        if write_header:
            return True
        try:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                current_header = handle.readline().strip()
            if current_header == expected_header:
                return False
            legacy_path = self.path.with_name(f"{self.path.stem}_legacy.csv")
            if legacy_path.exists():
                legacy_path.unlink()
            self.path.replace(legacy_path)
            return True
        except OSError as exc:
            print(f"Warning: failed to inspect training log header at {self.path}: {exc}")
            return False
