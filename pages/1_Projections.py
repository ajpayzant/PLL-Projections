"""Page 1 — Game Projections"""
from __future__ import annotations

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    SHARED_CSS, card, fmt_prob, fmt_goals,
    get_engine, init_session,
    team_color, team_name,
    build_overrides, build_active_players,
)

st.set_page_config(page_title="Projections · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

engine  = get_engine()
upcoming = engine.upcoming_games()

st.title("📊 Game Projections")

if not upcoming:
    st.warning("No upcoming games in the schedule. The data warehouse may need to be refreshed.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Select Game")
    labels = [
        f"G{g['game_number']} · {team_name(g['away_team_id'])} @ {team_name(g['home_team_id'])} · {str(g.get('game_date',''))[:10]}"
        for g in upcoming
    ]
    idx = st.selectbox("Game", range(len(upcoming)), format_func=lambda i: labels[i], key="game_idx")
    game = upcoming[idx]
    st.session_state.selected_game = game

    home_id = game["home_team_id"]
    away_id = game["away_team_id"]
    home_nm = team_name(home_id)
    away_nm = team_name(away_id)

    st.markdown("---")
    st.markdown("### Team Adjustments")
    st.markdown('<span class="note-text">1.0 = no change</span>', unsafe_allow_html=True)

    st.markdown(f"**{home_nm} (Home)**")
    h_off = st.slider("Offense", 0.70, 1.30, 1.00, 0.01, key="h_off")
    h_def = st.slider("Defense vs opp", 0.70, 1.30, 1.00, 0.01, key="h_def")

    st.markdown(f"**{away_nm} (Away)**")
    a_off = st.slider("Offense", 0.70, 1.30, 1.00, 0.01, key="a_off")
    a_def = st.slider("Defense vs opp", 0.70, 1.30, 1.00, 0.01, key="a_def")

    st.markdown("---")
    hold_pct = st.slider("Hold %", 2.0, 8.0, 4.5, 0.5, key="hold_slider") / 100.0
    st.session_state.hold_pct = hold_pct

    run_btn = st.button("▶  Run Projection", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────
team_adj = {
    home_id: {"off_mult": h_off, "def_mult_opp": h_def},
    away_id: {"off_mult": a_off, "def_mult_opp": a_def},
}
st.session_state.team_adjustments = team_adj

if run_btn or st.session_state.last_result is None:
    with st.spinner("Running 20,000 simulations…"):
        ov = build_overrides()
        ac = build_active_players()
        result = engine.project(
            home_team_id=home_id,
            away_team_id=away_id,
            player_overrides=ov or None,
            active_players=ac or None,
            team_adjustments=team_adj,
        )
        st.session_state.last_result = result

result = st.session_state.last_result
if result is None:
    st.info("Click **▶ Run Projection** in the sidebar.")
    st.stop()

hp = result.home_proj
ap = result.away_proj
gs = result.game_sim
gm = result.game_market

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    f'<h2 style="text-align:center;">'
    f'<span style="color:{team_color(away_id)}">{away_nm}</span>'
    f' &nbsp;@&nbsp; '
    f'<span style="color:{team_color(home_id)}">{home_nm}</span>'
    f'</h2>'
    f'<p style="text-align:center;color:#94a3b8;">Game {game["game_number"]} · {str(game.get("game_date",""))[:10]}</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Win prob row ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
with c1:
    st.markdown(card(f"{away_nm} Win Prob", fmt_prob(gm.away_win_prob), f"ML: {gm.away_ml}"),
                unsafe_allow_html=True)
with c2:
    st.markdown(card("Spread", f"{gm.spread_home:+.1f}", f"{gm.spread_home_odds}/{gm.spread_away_odds}"),
                unsafe_allow_html=True)
with c3:
    st.markdown(card("Total Line", str(gm.total_line), f"O{gm.over_odds} / U{gm.under_odds}"),
                unsafe_allow_html=True)
with c4:
    st.markdown(card("Exp. Total", f"{gs.expected_total:.1f}", "sim median"),
                unsafe_allow_html=True)
with c5:
    st.markdown(card(f"{home_nm} Win Prob", fmt_prob(gm.home_win_prob), f"ML: {gm.home_ml}"),
                unsafe_allow_html=True)

fig_wp = go.Figure(go.Bar(
    x=[gm.away_win_prob * 100, gm.home_win_prob * 100],
    y=[away_nm, home_nm], orientation="h",
    marker_color=[team_color(away_id), team_color(home_id)],
    text=[f"{gm.away_win_prob*100:.1f}%", f"{gm.home_win_prob*100:.1f}%"],
    textposition="auto",
))
fig_wp.update_layout(
    height=110, margin=dict(l=0, r=0, t=2, b=0),
    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
    yaxis=dict(showgrid=False),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"), showlegend=False,
)
st.plotly_chart(fig_wp, use_container_width=True)
st.markdown("---")

# ── Team stats table ──────────────────────────────────────────────────────
st.markdown("### Team Projections")
proj_df = pd.DataFrame([
    {"Team": away_nm,
     "Goals": round(ap.proj_goals, 1), "Score": round(ap.proj_scores, 1),
     "Shots": round(ap.proj_shots, 1), "SOG": round(ap.proj_sog, 1),
     "FO%": f"{ap.proj_faceoff_pct:.3f}", "FO Wins": round(ap.proj_faceoff_wins, 1),
     "2PT": round(ap.proj_2pt_goals, 2), "Assists": round(ap.proj_assists, 1),
     "Saves": round(ap.proj_saves, 1), "Save%": f"{ap.proj_save_pct:.3f}",
     "TOs": round(ap.proj_turnovers, 1), "GBs": round(ap.proj_ground_balls, 1)},
    {"Team": home_nm,
     "Goals": round(hp.proj_goals, 1), "Score": round(hp.proj_scores, 1),
     "Shots": round(hp.proj_shots, 1), "SOG": round(hp.proj_sog, 1),
     "FO%": f"{hp.proj_faceoff_pct:.3f}", "FO Wins": round(hp.proj_faceoff_wins, 1),
     "2PT": round(hp.proj_2pt_goals, 2), "Assists": round(hp.proj_assists, 1),
     "Saves": round(hp.proj_saves, 1), "Save%": f"{hp.proj_save_pct:.3f}",
     "TOs": round(hp.proj_turnovers, 1), "GBs": round(hp.proj_ground_balls, 1)},
]).set_index("Team")
st.dataframe(proj_df, use_container_width=True)
st.markdown("---")

# ── Sim distributions ─────────────────────────────────────────────────────
st.markdown("### Simulation Distributions")
tab1, tab2, tab3 = st.tabs(["Score Total", "Margin", "Goals by Team"])

with tab1:
    tot = gs.total_distribution
    fig = go.Figure(go.Histogram(x=tot, nbinsx=40, marker_color="#3b82f6", opacity=0.7))
    fig.add_vline(x=gm.total_line, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Line: {gm.total_line}")
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=6,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#f1f5f9"),
                      xaxis_title="Combined Score", yaxis_title="Sims")
    st.plotly_chart(fig, use_container_width=True)
    p_ov = float(np.mean(tot > gm.total_line))
    a, b, c = st.columns(3)
    a.metric("P(Over)",  f"{p_ov:.1%}")
    b.metric("P(Under)", f"{1-p_ov:.1%}")
    c.metric("P10 / P90", f"{np.percentile(tot,10):.0f} / {np.percentile(tot,90):.0f}")

with tab2:
    mar = gs.margin_distribution
    fig2 = go.Figure(go.Histogram(x=mar, nbinsx=40, marker_color="#8b5cf6", opacity=0.7))
    fig2.add_vline(x=0, line_color="#64748b", line_dash="dash")
    fig2.add_vline(x=gs.spread_home, line_color="#f59e0b", line_dash="dot",
                   annotation_text=f"Spread: {gs.spread_home:+.1f}")
    fig2.update_layout(height=280, margin=dict(l=0,r=0,t=6,b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#f1f5f9"),
                       xaxis_title=f"Margin (+ = {home_nm})", yaxis_title="Sims")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=gs.away_goals, name=away_nm,
                                marker_color=team_color(away_id), opacity=0.6, nbinsx=25))
    fig3.add_trace(go.Histogram(x=gs.home_goals, name=home_nm,
                                marker_color=team_color(home_id), opacity=0.6, nbinsx=25))
    fig3.update_layout(barmode="overlay", height=280, margin=dict(l=0,r=0,t=6,b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#f1f5f9"),
                       xaxis_title="Goals", yaxis_title="Sims")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Player summary ────────────────────────────────────────────────────────
st.markdown("### Player Projection Summary")
st.markdown('<span class="note-text">See Player Props page for full lines. '
            'Use Depth Charts to mark players inactive.</span>', unsafe_allow_html=True)

for nm, players in [(away_nm, result.away_players), (home_nm, result.home_players)]:
    active = [p for p in players if p.active]
    if not active:
        continue
    st.markdown(f"**{nm}**")
    rows = [
        {"Player": p.full_name or p.player_id, "Pos": p.position,
         "Proj G": round(p.proj_goals, 2), "Proj A": round(p.proj_assists, 2),
         "Proj Pts": round(p.proj_points, 2), "Proj Sh": round(p.proj_shots, 1),
         "Proj SOG": round(p.proj_sog, 1),
         "2PT Rate": f"{p.proj_2pt_goals/max(p.proj_goals,0.01):.0%}" if p.proj_goals > 0.05 else "—",
         "Proj SV": round(p.proj_saves, 1) if p.position == "G" else "—"}
        for p in sorted(active, key=lambda x: x.proj_points, reverse=True)[:14]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
