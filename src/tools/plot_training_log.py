from __future__ import annotations

import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "matplotlib is required for plot_training_log.py. Install it with: pip install matplotlib"
    ) from exc


LOG_PATH = Path("logs/training_log.csv")
RETURN_CURVE_PATH = Path("logs/return_curve.png")
RETURN_BY_MODE_PATH = Path("logs/return_by_mode.png")
EPSILON_CURVE_PATH = Path("logs/epsilon_curve.png")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"training log not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            episode_id = row.get("episode_id")
            episode_return = row.get("return")
            if not episode_id or not episode_return:
                continue
            rows.append(row)
        return rows


def split_sessions(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    sessions: list[list[dict[str, str]]] = []
    current_session: list[dict[str, str]] = []
    previous_episode_id: int | None = None

    for row in rows:
        episode_id = int(row["episode_id"])
        if previous_episode_id is not None and episode_id < previous_episode_id and current_session:
            sessions.append(current_session)
            current_session = []
        current_session.append(row)
        previous_episode_id = episode_id

    if current_session:
        sessions.append(current_session)
    return sessions


def select_last_run_or_session(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    run_ids = [row.get("run_id", "").strip() for row in rows if row.get("run_id", "").strip()]
    if run_ids:
        newest_run_id = max(set(run_ids))
        return [row for row in rows if row.get("run_id", "").strip() == newest_run_id], f"selected run_id: {newest_run_id}"
    sessions = split_sessions(rows)
    if not sessions:
        raise RuntimeError("no training sessions found")
    return sessions[-1], f"last session length: {len(sessions[-1])}"


def rolling_mean(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        output.append(sum(chunk) / len(chunk))
    return output


def plot_return_curve(rows: list[dict[str, str]]) -> None:
    episodes = [int(row["episode_id"]) for row in rows]
    returns = [float(row["return"]) for row in rows]
    avg20 = rolling_mean(returns, 20)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, returns, label="Return", linewidth=1.8, color="#2c7fb8")
    plt.plot(episodes, avg20, label="Avg20", linewidth=2.2, color="#d95f0e")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Return")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    RETURN_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RETURN_CURVE_PATH, dpi=160)
    plt.close()


def plot_return_by_mode(rows: list[dict[str, str]]) -> None:
    train_eps = [int(row["episode_id"]) for row in rows if row["mode"] == "TRAIN"]
    train_returns = [float(row["return"]) for row in rows if row["mode"] == "TRAIN"]
    eval_eps = [int(row["episode_id"]) for row in rows if row["mode"] == "EVAL"]
    eval_returns = [float(row["return"]) for row in rows if row["mode"] == "EVAL"]

    plt.figure(figsize=(10, 5))
    if train_eps:
        plt.scatter(train_eps, train_returns, label="TRAIN", s=22, alpha=0.75, color="#1b9e77")
    if eval_eps:
        plt.scatter(eval_eps, eval_returns, label="EVAL", s=28, alpha=0.8, color="#d95f02")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return by Mode")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    RETURN_BY_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RETURN_BY_MODE_PATH, dpi=160)
    plt.close()


def plot_epsilon_curve(rows: list[dict[str, str]]) -> None:
    episodes = [int(row["episode_id"]) for row in rows]
    epsilons = [float(row["epsilon"]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, label="Epsilon", linewidth=2.0, color="#756bb1")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon vs Episode")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    EPSILON_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(EPSILON_CURVE_PATH, dpi=160)
    plt.close()


def main() -> None:
    rows = load_rows(LOG_PATH)
    if not rows:
        raise RuntimeError(f"training log is empty: {LOG_PATH}")
    session_rows, selection_label = select_last_run_or_session(rows)
    plot_return_curve(session_rows)
    plot_return_by_mode(session_rows)
    plot_epsilon_curve(session_rows)
    print(selection_label)
    print(f"using rows: {len(session_rows)}")
    print(f"saved {RETURN_CURVE_PATH}")
    print(f"saved {RETURN_BY_MODE_PATH}")
    print(f"saved {EPSILON_CURVE_PATH}")


if __name__ == "__main__":
    main()
