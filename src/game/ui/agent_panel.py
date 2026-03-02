from __future__ import annotations

from src.game.ui.models import AgentDebugData


def build_agent_panel_lines(data: AgentDebugData) -> list[str]:
    return [
        f"MODE: {data.mode_label}",
        f"eps={data.epsilon:.2f}",
        f"EP: {data.episode_id}",
        f"Return: {data.episode_return:.1f}",
        f"Avg20: {data.avg20:.1f}",
        f"Reward: {data.reward_total:.1f}",
        f"Penalty: {data.penalty_total:.1f}",
        f"shoot_ready={int(data.shoot_ready)}  los={int(data.has_los)}  reloading={int(data.is_reloading)}",
        f"aim_err={data.aim_error_degrees:+.1f}  aim_stable={data.aim_stable_time:.2f}",
        f"Goal: {data.goal_type}",
        f"tgt={data.target_id}  tgt_d={data.target_distance_text}",
        f"retarget_in={data.retarget_in_seconds:.2f}s  emergency={int(data.emergency_switch)}",
        f"pack_score={data.pack_score:.2f}  enemy_score={data.enemy_score:.2f}",
        f"d_pack={data.pack_distance_text}  d_enemy={data.enemy_distance_text}",
        f"intent={data.intent_label}  eng_band=[3.0,6.0]  enemy_repulse={int(data.enemy_repulse)}",
        f"near_wall={int(data.near_wall)}  should_move={int(data.should_move)}  bumped={int(data.bumped)}",
        f"progress={data.progress_tiles:.2f}t  stuck_t={data.stuck_time:.2f}  recover={int(data.recovering)}  side={data.recovery_side_label}",
        f"Target: {data.target_type}",
        f"Tdist: {data.target_distance_text}",
        f"Reason: {data.reason}",
        f"NR t/cd: {data.no_reason_timer:.2f}/{data.no_reason_cooldown:.2f}",
    ]
