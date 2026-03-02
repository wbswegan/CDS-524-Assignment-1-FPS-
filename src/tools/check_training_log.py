from __future__ import annotations

import csv
from pathlib import Path

from src.tools.plot_training_log import LOG_PATH, select_last_run_or_session


def main() -> None:
    rows = list(csv.DictReader(LOG_PATH.open("r", encoding="utf-8", newline="")))
    filtered_rows, selection_label = select_last_run_or_session(rows)
    if not filtered_rows:
        raise SystemExit("No rows found for latest run")

    episode_ids = [int(row["episode_id"]) for row in filtered_rows]
    epsilons = [float(row["epsilon"]) for row in filtered_rows]
    kills = [int(row["kills"]) for row in filtered_rows]

    if episode_ids != list(range(1, len(filtered_rows) + 1)):
        raise SystemExit("episode_id sequence is not contiguous within the selected run")
    if not epsilons or epsilons[0] < 0.9:
        raise SystemExit("epsilon does not start near 1.0 in the selected run")
    if epsilons[-1] > epsilons[0]:
        raise SystemExit("epsilon did not decay over the selected run")
    if all(kill == index for index, kill in enumerate(kills, start=1)):
        raise SystemExit("kills column appears cumulative (kills == episode_id pattern)")

    print(selection_label)
    print(f"rows checked: {len(filtered_rows)}")
    print(f"eps first/last: {epsilons[0]:.4f} -> {epsilons[-1]:.4f}")
    print("training log self-check passed")


if __name__ == "__main__":
    main()
