from __future__ import annotations


def menu_lines(high_score: int) -> tuple[tuple[str, str], ...]:
    return (
        ("Doomlike FPS", "large"),
        ("1) Main Mode (Large Map)", "small"),
        ("2) Training Mode (Small Arena)", "small"),
        (f"High Score {high_score}", "small"),
        ("Press 1 or 2 to start. ESC to quit.", "small"),
    )


def pause_lines() -> tuple[tuple[str, str], ...]:
    return (
        ("Paused", "large"),
        ("Resume (ENTER)", "small"),
        ("Back to Menu (M)", "small"),
        ("Press ESC to resume", "small"),
    )


def game_over_lines(score: int, high_score: int) -> tuple[tuple[str, str], ...]:
    return (
        ("Try one more!", "large"),
        (f"Current Score {score}", "small"),
        (f"High Score {high_score}", "small"),
        ("Press ENTER to restart current mode", "small"),
        ("Press M to return to Menu", "small"),
        ("Press ESC to quit", "small"),
    )
