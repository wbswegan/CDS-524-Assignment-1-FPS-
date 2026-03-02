from __future__ import annotations

import math

from src.ai.geometry import distance_bin, has_line_of_sight, hp_bin, pack_distance_bin, relative_direction_bin


def build_player_agent_obs(
    *,
    player_x: float,
    player_y: float,
    player_angle: float,
    player_bumped: bool,
    player_hp: int,
    player_max_hp: int,
    under_fire: bool,
    level_map: list[str],
    target_x: float | None,
    target_y: float | None,
    pack_distance: float | None,
) -> tuple[int, int, int, int, int, int, int, int]:
    pack_exists = int(pack_distance is not None)
    pack_dist_bin = pack_distance_bin(pack_distance)
    if target_x is None or target_y is None:
        return (
            2,
            0,
            3,
            int(player_bumped),
            hp_bin(player_hp, player_max_hp),
            int(under_fire),
            pack_exists,
            pack_dist_bin,
        )

    distance = math.hypot(target_x - player_x, target_y - player_y)
    return (
        distance_bin(distance),
        int(has_line_of_sight(player_x, player_y, target_x, target_y, level_map)),
        relative_direction_bin(target_x, target_y, player_x, player_y, player_angle),
        int(player_bumped),
        hp_bin(player_hp, player_max_hp),
        int(under_fire),
        pack_exists,
        pack_dist_bin,
    )
