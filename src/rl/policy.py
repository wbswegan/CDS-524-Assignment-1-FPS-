from __future__ import annotations


def pack_target_score(
    *,
    hp_ratio: float,
    distance_tiles: float,
    base_weight: float,
    close_weight: float,
    low_hp_bonus: float,
    low_hp_ratio: float = 0.35,
) -> tuple[float, str]:
    score = base_weight + close_weight / (distance_tiles + 1.0)
    reason = "pack_value"
    if hp_ratio < low_hp_ratio:
        score += low_hp_bonus
        reason = "lowHP_pack"
    return score, reason


def enemy_target_score(
    *,
    distance_tiles: float,
    under_fire: bool,
    emergency_distance_tiles: float,
    close_weight: float,
    emergency_weight: float,
) -> tuple[float, str]:
    score = close_weight / (distance_tiles + 1.0)
    reason = "chase_enemy"
    if distance_tiles <= emergency_distance_tiles or under_fire:
        score += emergency_weight
        reason = "emergency_enemy" if distance_tiles <= emergency_distance_tiles else "under_fire_enemy"
    return score, reason
