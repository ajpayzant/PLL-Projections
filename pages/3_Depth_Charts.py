"""
Page 3 — Depth Charts
Mark players active / inactive, designate goalie starters, adjust usage multipliers.
Changes here flow into the next projection run.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from _engine_state import (
    get_engine, init_session,
    team_color, team_name, pos_badge,
    get_depth_chart, set_player_override,
)

st.set_page_config(page_title="Depth Charts · PLL", page_icon="🥍", layout="wide")
init_session()

st.markdown("""
<style>
  .main .block-container { padding-top:1rem; max-width:1800px; }
  .note-text { color:#64748b; font-size:.80rem; font-style:italic; }
</style>
""", unsafe_allow_html=True)

engine = get_engine()
result = st.session_state.get("last_result")

if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id = result.home_proj.team_id
away_id = result.away_proj.team_id
home_nm = team_name(home_id)
away_nm = team_name(away_id)

game = st.session_state.selected_game or {}
st.title("📋 Depth Charts")
st.markdown(f"**{away_nm} @ {home_nm}** · Game {game.get('game_number','—')}")
st.markdown("""
Adjust rosters below. Changes apply when you re-run the projection on the **Projections** page.

- **Active** — uncheck to mark a player as scratched / DNP
- **Starter (G)** — mark exactly one goalie as the starting goalie
- **Usage** — 1.0 = normal. 1.2 = 20% more ice time / usage. 0.0 = inactive
""")
st.markdown("---")

def _render_team(team_id: str, team_nm: str, players):
    st.markdown(f"### {team_nm}")

    dc = get_depth_chart(team_id)
    active_players = [p for p in players]
    if not active_players:
        st.info(f"No players found for {team_nm}.")
        return

    # Sort by position priority then proj_points desc
    POS_ORDER = {"A": 0, "M": 1, "FO": 2, "D": 3, "SSDM": 4, "LSM": 5, "G": 6}
    active_players.sort(key=lambda p: (POS_ORDER.get(p.position, 9), -p.proj_points))

    goalies = [p for p in active_players if p.position == "G"]
    # Determine current starter
    current_starter = None
    for p in goalies:
        if dc.get(p.player_id, {}).get("is_starter", False):
            current_starter = p.player_id
            break
    if current_starter is None and goalies:
        # Default: highest save%
        current_starter = max(goalies, key=lambda p: p.proj_save_pct).player_id

    # Header row
    cols = st.columns([3, 1, 1, 1, 2, 2, 2])
    cols[0].markdown("**Player**")
    cols[1].markdown("**Pos**")
    cols[2].markdown("**Active**")
    cols[3].markdown("**Starter (G)**")
    cols[4].markdown("**Usage**")
    cols[5].markdown("**Proj Goals**")
    cols[6].markdown("**Proj Pts**")
    st.markdown('<hr style="margin:4px 0 8px 0; border-color:rgba(148,163,184,.15);">', unsafe_allow_html=True)

    for p in active_players:
        pid = p.player_id
        existing = dc.get(pid, {})
        is_currently_active = existing.get("active", True)
        current_usage = existing.get("usage_multiplier", 1.0)
        is_goalie = p.position == "G"

        col1, col2, col3, col4, col5, col6, col7 = st.columns([3, 1, 1, 1, 2, 2, 2])

        with col1:
            nm = p.full_name or pid
            style = "" if is_currently_active else "color:#64748b;text-decoration:line-through;"
            st.markdown(f'<span style="{style}">{nm}</span>', unsafe_allow_html=True)

        with col2:
            st.markdown(pos_badge(p.position), unsafe_allow_html=True)

        with col3:
            new_active = st.checkbox(
                "Active", value=is_currently_active,
                key=f"active_{team_id}_{pid}",
                label_visibility="collapsed",
            )
            if new_active != is_currently_active:
                set_player_override(team_id, pid, "active", new_active)

        with col4:
            if is_goalie:
                is_starter_now = (current_starter == pid)
                new_starter = st.checkbox(
                    "Starter", value=is_starter_now,
                    key=f"starter_{team_id}_{pid}",
                    label_visibility="collapsed",
                )
                if new_starter and not is_starter_now:
                    # Clear other goalies' starter flag
                    for g in goalies:
                        set_player_override(team_id, g.player_id, "is_starter", False)
                    set_player_override(team_id, pid, "is_starter", True)
                    current_starter = pid
            else:
                st.write("")

        with col5:
            new_usage = st.number_input(
                "Usage", min_value=0.0, max_value=2.0, step=0.05,
                value=float(current_usage),
                key=f"usage_{team_id}_{pid}",
                label_visibility="collapsed",
                disabled=not is_currently_active,
            )
            if abs(new_usage - current_usage) > 0.001:
                set_player_override(team_id, pid, "usage_multiplier", new_usage)

        with col6:
            st.write(f"{p.proj_goals:.3f}")

        with col7:
            st.write(f"{p.proj_points:.3f}")

    st.markdown("")

    # Bulk actions
    with st.expander(f"Bulk actions — {team_nm}"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button(f"Activate all — {team_nm}", key=f"activate_all_{team_id}"):
                for p in active_players:
                    set_player_override(team_id, p.player_id, "active", True)
                st.rerun()
        with col_b:
            if st.button(f"Reset all usage — {team_nm}", key=f"reset_usage_{team_id}"):
                for p in active_players:
                    set_player_override(team_id, p.player_id, "usage_multiplier", 1.0)
                st.rerun()
        with col_c:
            if st.button(f"Clear depth chart — {team_nm}", key=f"clear_{team_id}"):
                st.session_state.depth_charts[team_id] = {}
                st.rerun()


# Render both teams
_render_team(away_id, away_nm, result.away_players)
st.markdown("---")
_render_team(home_id, home_nm, result.home_players)

st.markdown("---")
st.markdown('<span class="note-text">Changes take effect when you click **▶ Run Projection** on the Projections page.</span>',
            unsafe_allow_html=True)

if st.button("↩ Return to Projections", use_container_width=False):
    st.switch_page("pages/1_Projections.py")
