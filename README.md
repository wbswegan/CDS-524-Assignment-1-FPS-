# Pygame Doom-like

Minimal Doom-like FPS demo in Pygame with:
- raycast walls
- sprite enemies
- wave progression
- an `RLEnemy`
- player-side tabular Q-learning
- a Training Arena mode

## Run

```powershell
python -B main.py
```

## Player Q-Table

The player Q-table is stored at:

```text
models/player_q.json
```

Behavior:
- auto-load on startup if the file exists
- auto-save on exit
- auto-save at the end of each player-agent episode

## Hotkeys

- `F2`: Human control
- `F3`: Agent control
- `F4`: toggle Training Arena / main game
- `F5`: toggle player agent `Training` / `Eval`
- `F1`: toggle RL enemy debug panel
- `P`: toggle enemy path debug on minimap
- `Esc`: toggle mouse capture

## Demo Flow

1. Start the game with `python -B main.py`.
2. Press `F4` to enter `Training Arena`.
3. Press `F3` to ensure `Agent` control is active.
4. Leave the game in arena mode for a while so the player agent can accumulate episodes.
   Watch `Episode`, `AvgReward`, and the `PlayerQ` HUD line.
5. Press `F5` to switch to `Eval` when you want pure exploitation (`epsilon = 0`).
6. Press `F4` to return to the main map.
7. Stay in `Agent` mode and observe how the learned policy behaves on the larger map.

Suggested short test:
- Train for 50-100 arena episodes.
- Switch to `Eval`.
- Return to the main map and observe movement/engagement decisions.
