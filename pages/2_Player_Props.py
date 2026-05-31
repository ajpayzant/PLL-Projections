"""
Page 2 — Player Props
Full prop market pricing for every active player.
Supports custom line overrides, milestone props, and goalie/FO specialist views.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    get_engine, init_session,
    card, fmt_odds, fmt_prob,
    team_color, team_name, pos_badge,
    build_overrides, build_active_players,
)
from projection_engine_v3 import PricingEngine, GameSimulator

st.set_page_config(page_title="Player Props · PLL", page_icon="🥍", layout="wide")
init_session()

st.markdown("""
<style>
  .main .block-container { padding-top:1rem; max-width:1800px; }
  .pll-card { border:1px solid rgba(148,163,184,.2); border-radius:12px; padding:12px 16px;
    background:linear-gradient(160deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
    box-shadow:0 4px 16px rgba(0,0,0,.10); margin-bottom:8px; }
  .pll-card-label { color:#94a3b8; font-size:.78rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
  .pll-card-value { font-size:1.5rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
  .pll-card-sub { color:#94a3b8; font-size:.78rem; margin-top:3px; }
  .odds-fav { background:#16a34a;color:#fff;border-radius:6px;padding:2px 8px;font-weight:700;font-size:.85rem; }
  .odds-dog { background:#2563eb;color:#fff;border-radius:6px;padding:2px 8px;font-weight:700;font-size:.85rem; }
  .odds-even { background:#475569;color:#fff;border-radius:6px;padding:2px 8px;font-weight:700;font-size:.85rem; }
  .note-text { color:#64748b; font-size:.80rem; font-style:italic; }
  .over-badge { color:#16a34a; font-weight:700; }
  .under-badge { color:#dc2626; font-weight:700; }
</style>
""", unsafe_allow_html=True)

engine = get_engine()
result = st.session_state.get("last_result")

if result is None:
    st.warning("No projection loaded. Go to **Projections** and run a game first.")
    st.stop()

game = st.session_state.selected_game or {}
home_id = result.home_proj.team_id
away_id = result.away_proj.team_id
home_nm = team_name(home_id)
away_nm = team_name(away_id)

st.title("👤 Player Prop Markets")
st.markdown(f"**{away_nm} @ {home_nm}** · Game {game.get('game_number','—')} · {str(game.get('game_date',''))[:10]}")

hold_pct = st.session_state.get("hold_pct", 0.045)
pricing = PricingEngine(hold_pct=hold_pct)

# ── Sidebar controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    show_team = st.radio("Team", [f"Both", away_nm, home_nm], key="prop_team_filter")
    show_pos = st.multiselect(
        "Positions", ["A", "M", "D", "FO", "SSDM", "LSM", "G"],
        default=["A", "M", "FO", "G"], key="prop_pos_filter",
    )
    min_pts = st.slider("Min projected points", 0.0, 3.0, 0.3, 0.1, key="prop_min_pts")
    show_milestones = st.checkbox("Show milestone props (1+, 2+, 3+)", value=True)
    st.markdown("---")
    st.markdown("### Manual Line Override")
    st.markdown('<span class="note-text">Enter a custom line for a player/stat. Leave blank to use model line.</span>', unsafe_allow_html=True)
    override_player = st.text_input("Player name (partial)", key="override_player_name")
    override_stat = st.selectbox("Stat", ["goals", "assists", "points", "shots_on_goal", "saves", "faceoff_wins"], key="override_stat")
    override_line = st.number_input("Line value", min_value=0.0, max_value=20.0, step=0.5, value=0.5, key="override_line_val")

# ── Collect all player simulations ────────────────────────────────────────
all_sims = result.home_player_sims + result.away_player_sims
all_projs = {p.player_id: p for p in result.home_players + result.away_players}
markets = result.player_markets

# Filter
def _keep(pid: str) -> bool:
    pm = markets.get(pid, {})
    pv = pm.get("proj_values", {})
    pts = pv.get("points", 0) or pv.get("saves", 0) or pv.get("faceoff_wins", 0)
    if pts < min_pts:
        return False
    proj = all_projs.get(pid)
    if proj is None:
        return False
    if not proj.active:
        return False
    pos = proj.position
    if pos not in show_pos:
        return False
    tid = proj.team_id
    if show_team == away_nm and tid != away_id:
        return False
    if show_team == home_nm and tid != home_id:
        return False
    return True

filtered_sims = [s for s in all_sims if _keep(s.player_id)]
filtered_sims.sort(
    key=lambda s: markets.get(s.player_id, {}).get("proj_values", {}).get("points", 0),
    reverse=True,
)

if not filtered_sims:
    st.info("No players match the current filters.")
    st.stop()

st.markdown(f"**{len(filtered_sims)} players shown** (adjust filters in sidebar)")
st.markdown("---")

# ── Player prop cards ─────────────────────────────────────────────────────
STAT_LABELS = {
    "goals": "Goals",
    "assists": "Assists",
    "points": "Points",
    "shots": "Shots",
    "shots_on_goal": "SOG",
    "two_pt_goals": "2PT Goals",
    "saves": "Saves",
    "faceoff_wins": "FO Wins",
    "one_pt_goals": "1PT Goals",
}

FIELD_STATS = ["goals", "assists", "points", "shots_on_goal", "two_pt_goals"]
GOALIE_STATS = ["saves"]
FO_STATS = ["faceoff_wins"]
MILESTONE_STATS = {"goals": [1, 2, 3], "assists": [1, 2], "saves": [10, 12, 14]}

for ps in filtered_sims:
    pid = ps.player_id
    pm = markets.get(pid, {})
    proj = all_projs.get(pid)
    if proj is None:
        continue

    pv = pm.get("proj_values", {})
    mkt = pm.get("markets", {})
    nm = proj.full_name or pid
    pos = proj.position
    tid = proj.team_id

    with st.expander(
        f"{nm}  ·  {pos}  ·  {team_name(tid)}  "
        f"| Proj: {pv.get('points', pv.get('saves', pv.get('faceoff_wins', 0))):.2f} pts",
        expanded=False,
    ):
        col_info, col_dist = st.columns([1, 2])

        with col_info:
            st.markdown(f"**Position:** {pos_badge(pos)}", unsafe_allow_html=True)
            st.markdown(f"**Team:** {team_name(tid)}")
            if pos != "G":
                st.markdown(f"**Proj Goals:** {proj.proj_goals:.3f}")
                st.markdown(f"**Proj Assists:** {proj.proj_assists:.3f}")
                st.markdown(f"**Proj Points:** {proj.proj_points:.3f}")
                st.markdown(f"**Proj Shots:** {proj.proj_shots:.2f}  SOG: {proj.proj_sog:.2f}")
                if proj.proj_2pt_goals > 0.02:
                    rate = proj.proj_2pt_goals / max(proj.proj_goals, 0.01)
                    st.markdown(f"**2PT Rate:** {rate:.1%}  ({proj.proj_2pt_goals:.3f} proj)")
            else:
                st.markdown(f"**Proj Saves:** {proj.proj_saves:.2f}")
                st.markdown(f"**Proj Save%:** {proj.proj_save_pct:.3f}")
            if pos == "FO":
                st.markdown(f"**Proj FO Wins:** {proj.proj_faceoff_wins:.2f}  ({proj.proj_faceoff_pct:.3f})")

        with col_dist:
            # Show distribution for primary stat
            primary_stat = "saves" if pos == "G" else ("faceoff_wins" if pos == "FO" else "points")
            if primary_stat in ps.stat_distributions:
                dist = ps.stat_distributions[primary_stat]
                fig = go.Figure(go.Histogram(
                    x=dist, nbinsx=20,
                    marker_color=team_color(tid), opacity=0.75,
                ))
                proj_val = pv.get(primary_stat, 0)
                fig.add_vline(x=proj_val, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Proj: {proj_val:.2f}")
                auto_line = pm.get("prop_lines", {}).get(primary_stat, proj_val)
                if abs(auto_line - proj_val) > 0.1:
                    fig.add_vline(x=auto_line, line_dash="dot", line_color="#94a3b8",
                                  annotation_text=f"Line: {auto_line}")
                fig.update_layout(
                    height=180, margin=dict(l=0, r=0, t=4, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"), showlegend=False,
                    xaxis_title=STAT_LABELS.get(primary_stat, primary_stat),
                    yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Prop lines table
        stat_list = GOALIE_STATS if pos == "G" else FO_STATS if pos == "FO" else FIELD_STATS

        rows = []
        for stat in stat_list:
            if stat not in ps.stat_distributions:
                continue
            dist = ps.stat_distributions[stat]

            # Check if user has a manual override for this player+stat
            custom_line = None
            if override_player and override_player.lower() in nm.lower() and override_stat == stat:
                custom_line = override_line

            ml = pricing.price_prop(ps, stat, line=custom_line)
            proj_v = pv.get(stat, 0)
            rows.append({
                "Stat": STAT_LABELS.get(stat, stat),
                "Proj": f"{proj_v:.3f}",
                "Line": ml.line,
                "P(Over)": f"{ml.fair_over_prob:.3f}",
                "Over": ml.over_odds,
                "P(Under)": f"{ml.fair_under_prob:.3f}",
                "Under": ml.under_odds,
                "P10": f"{np.percentile(dist,10):.1f}",
                "P50": f"{np.percentile(dist,50):.1f}",
                "P90": f"{np.percentile(dist,90):.1f}",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Milestone props
        if show_milestones:
            for stat, levels in MILESTONE_STATS.items():
                if stat not in ps.stat_distributions:
                    continue
                mile_rows = []
                for lvl in levels:
                    ml_m = pricing.price_prop(ps, stat, line=lvl - 0.5)
                    ml_m.stat = f"{stat}_{lvl}+"
                    dist = ps.stat_distributions[stat]
                    p_hit = float(np.mean(dist >= lvl))
                    mile_rows.append({
                        "Milestone": f"{STAT_LABELS.get(stat, stat)} {lvl}+",
                        "P(Hit)": f"{p_hit:.3f}",
                        "Yes (Over)": ml_m.over_odds,
                        "No (Under)": ml_m.under_odds,
                    })
                if mile_rows:
                    st.markdown(f"**Milestones — {STAT_LABELS.get(stat, stat)}**")
                    st.dataframe(pd.DataFrame(mile_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<span class="note-text">Props priced using Monte Carlo simulation distributions (20,000 sims). '
            f'Hold: {hold_pct*100:.1f}%. Adjust hold in Projections sidebar.</span>', unsafe_allow_html=True)
