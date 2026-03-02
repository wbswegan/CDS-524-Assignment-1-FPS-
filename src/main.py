import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=None)
    parser.add_argument("--render_every", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless or args.train_episodes is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from src.game.game import run

    run(train_episodes=args.train_episodes, render_every=args.render_every)


if __name__ == "__main__":
    main()
