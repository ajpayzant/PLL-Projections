"""Page 3 — Depth Charts"""
from __future__ import annotations

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from _engine_state import (
    SHARED_CSS, pos_badge,
    get_engine, init_session,
    team_name,
    get_depth_chart, set_player_override,
)

st.set_page_config(page_title="Depth Charts · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id = result.home_proj.team_id
away_id = result.away_proj.team_id
home_nm = team_name(home_id)
away_nm = team_name(away_id)
game    = st.session_state.selected_game or {}

st.title("📋 Depth Charts")
st.markdown(f"**{away_nm} @ {home_nm}** · Game {game.get('game_number','—')}")
st.markdown("""
Adjust rosters here. Re-run the projection on the **Projections** page to apply changes.

- **Active** — uncheck to scratch a player (DNP / injured)
- **Starter** — select exactly one goalie per team as the starting goalie
- **Usage** — 1.0 = normal, 1.2 = 20% more involvement, 0.8 = limited role
""")
st.markdown("---")

POS_ORDER = {"A": 0, "M": 1, "FO": 2, "D": 3, "SSDM": 4, "LSM": 5, "G": 6}

def _render_team(team_id: str, team_nm: str, players):
    st.markdown(f"### {team_nm}")
    dc = get_depth_chart(team_id)

    sorted_players = sorted(players, key=lambda p: (POS_ORDER.get(p.position, 9), -p.proj_points))
    goalies = [p for p in sorted_players if p.position == "G"]

    # Determine starter goalie
    current_starter = next(
        (p.player_id for p in goalies if dc.get(p.player_id, {}).get("is_starter", False)),
        max(goalies, key=lambda p: p.proj_save_pct).player_id if goalies else None,
    )

    hdr = st.columns([3, 1, 1, 1, 2, 2, 2])
    for col, lbl in zip(hdr, ["Player","Pos","Active","Starter","Usage","Proj G","Proj Pts"]):
        col.markdown(f"**{lbl}**")
    st.markdown('<hr style="margin:4px 0 8px;border-color:rgba(148,163,184,.15);">',
                unsafe_allow_html=True)

    for p in sorted_players:
        pid       = p.player_id
        existing  = dc.get(pid, {})
        is_active = existing.get("active", True)
        usage_val = float(existing.get("usage_multiplier", 1.0))
        is_goalie = p.position == "G"

        c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 1, 1, 1, 2, 2, 2])

        with c1:
            nm = p.full_name or pid
            style = "" if is_active else "color:#64748b;text-decoration:line-through;"
            st.markdown(f'<span style="{style}">{nm}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown(pos_badge(p.position), unsafe_allow_html=True)
        with c3:
            new_active = st.checkbox("", value=is_active,
                                     key=f"act_{team_id}_{pid}",
                                     label_visibility="collapsed")
            if new_active != is_active:
                set_player_override(team_id, pid, "active", new_active)
        with c4:
            if is_goalie:
                is_starter_now = (current_starter == pid)
                new_starter = st.checkbox("", value=is_starter_now,
                                          key=f"start_{team_id}_{pid}",
                                          label_visibility="collapsed")
                if new_starter and not is_starter_now:
                    for g in goalies:
                        set_player_override(team_id, g.player_id, "is_starter", False)
                    set_player_override(team_id, pid, "is_starter", True)
                    current_starter = pid
            else:
                st.write("")
        with c5:
            new_usage = st.number_input("", min_value=0.0, max_value=2.0, step=0.05,
                                        value=usage_val,
                                        key=f"use_{team_id}_{pid}",
                                        label_visibility="collapsed",
                                        disabled=not is_active)
            if abs(new_usage - usage_val) > 0.001:
                set_player_override(team_id, pid, "usage_multiplier", new_usage)
        with c6:
            st.write(f"{p.proj_goals:.3f}")
        with c7:
            st.write(f"{p.proj_points:.3f}")

    st.markdown("")
    with st.expander(f"Bulk actions — {team_nm}"):
        ca, cb, cc = st.columns(3)
        with ca:
            if st.button(f"Activate all", key=f"act_all_{team_id}"):
                for p in sorted_players:
                    set_player_override(team_id, p.player_id, "active", True)
                st.rerun()
        with cb:
            if st.button(f"Reset usage", key=f"rst_use_{team_id}"):
                for p in sorted_players:
                    set_player_override(team_id, p.player_id, "usage_multiplier", 1.0)
                st.rerun()
        with cc:
            if st.button(f"Clear all overrides", key=f"clr_{team_id}"):
                st.session_state.depth_charts[team_id] = {}
                st.rerun()


_render_team(away_id, away_nm, result.away_players)
st.markdown("---")
_render_team(home_id, home_nm, result.home_players)

st.markdown("---")
st.markdown('<span class="note-text">Changes take effect when you click ▶ Run Projection '
            'on the Projections page.</span>', unsafe_allow_html=True)
