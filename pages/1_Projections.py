"""
Page 1 — Game Projections
Select an upcoming game, configure team-level adjustments, run projections,
and view team totals, win probability, and the simulation distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from _engine_state import (
    get_engine, init_session,
    card, fmt_odds, fmt_prob, fmt_goals,
    team_color, team_name,
    build_overrides, build_active_players,
)

st.set_page_config(page_title="Projections · PLL", page_icon="🥍", layout="wide")
init_session()

# ── Inject shared CSS ─────────────────────────────────────────────────────
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
  .note-text { color:#64748b;font-size:.80rem;font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ── Load engine ───────────────────────────────────────────────────────────
engine = get_engine()
upcoming = engine.upcoming_games()

st.title("📊 Game Projections")

if not upcoming:
    st.warning("No upcoming games found in the schedule.")
    st.stop()

# ── Sidebar: game selection + team adjustments ────────────────────────────
with st.sidebar:
    st.markdown("### Game Selection")

    game_labels = []
    for g in upcoming:
        ht = team_name(g["home_team_id"])
        at = team_name(g["away_team_id"])
        date = str(g.get("game_date", ""))[:10]
        game_labels.append(f"G{g['game_number']} · {at} @ {ht} · {date}")

    selected_idx = st.selectbox(
        "Upcoming game",
        options=range(len(upcoming)),
        format_func=lambda i: game_labels[i],
        key="game_select_idx",
    )
    selected_game = upcoming[selected_idx]
    st.session_state.selected_game = selected_game

    home_id = selected_game["home_team_id"]
    away_id = selected_game["away_team_id"]
    home_nm = team_name(home_id)
    away_nm = team_name(away_id)

    st.markdown("---")
    st.markdown("### Team Rating Adjustments")
    st.markdown('<span class="note-text">1.0 = no change. Bump up/down to reflect injuries, travel, etc.</span>', unsafe_allow_html=True)

    st.markdown(f"**{home_nm} (Home)**")
    h_off = st.slider("Offense multiplier", 0.70, 1.30, 1.00, 0.01, key="h_off_mult")
    h_def = st.slider("Def quality vs opp", 0.70, 1.30, 1.00, 0.01, key="h_def_mult",
                      help="Increases/decreases how much the opponent's defense suppresses this team")

    st.markdown(f"**{away_nm} (Away)**")
    a_off = st.slider("Offense multiplier", 0.70, 1.30, 1.00, 0.01, key="a_off_mult")
    a_def = st.slider("Def quality vs opp", 0.70, 1.30, 1.00, 0.01, key="a_def_mult")

    st.markdown("---")
    hold_pct = st.slider("Market hold %", 2.0, 8.0, 4.5, 0.5, key="hold_pct_slider") / 100.0
    st.session_state.hold_pct = hold_pct

    run_btn = st.button("▶ Run Projection", type="primary", use_container_width=True)

# ── Run projection ────────────────────────────────────────────────────────
team_adj = {
    home_id: {"off_mult": h_off, "def_mult_opp": h_def},
    away_id: {"off_mult": a_off, "def_mult_opp": a_def},
}
st.session_state.team_adjustments = team_adj

if run_btn or st.session_state.last_result is None:
    with st.spinner("Running 20,000 simulations…"):
        overrides = build_overrides()
        active = build_active_players()
        result = engine.project(
            home_team_id=home_id,
            away_team_id=away_id,
            player_overrides=overrides if overrides else None,
            active_players=active if active else None,
            team_adjustments=team_adj,
        )
        st.session_state.last_result = result

result = st.session_state.last_result
if result is None:
    st.info("Click **▶ Run Projection** in the sidebar to generate projections.")
    st.stop()

hp = result.home_proj
ap = result.away_proj
gs = result.game_sim
gm = result.game_market

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<h2 style="text-align:center; margin-bottom:4px;">
  <span style="color:{team_color(away_id)}">{away_nm}</span>
  &nbsp;@&nbsp;
  <span style="color:{team_color(home_id)}">{home_nm}</span>
</h2>
<p style="text-align:center;color:#94a3b8;margin-top:0;">
  Game {selected_game['game_number']} · {str(selected_game.get('game_date',''))[:10]}
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Win probability & market ──────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])

with c1:
    st.markdown(card(
        f"{away_nm} Win Prob",
        fmt_prob(gm.away_win_prob),
        f"ML: {gm.away_ml}",
    ), unsafe_allow_html=True)

with c2:
    st.markdown(card("Spread", f"{gm.spread_home:+.1f}", f"{gm.spread_home_odds} / {gm.spread_away_odds}"),
                unsafe_allow_html=True)

with c3:
    st.markdown(card("Total Line", f"{gm.total_line}", f"O{gm.over_odds} / U{gm.under_odds}"),
                unsafe_allow_html=True)

with c4:
    st.markdown(card("Exp. Total", f"{gs.expected_total:.1f}", "median simulated score"),
                unsafe_allow_html=True)

with c5:
    st.markdown(card(
        f"{home_nm} Win Prob",
        fmt_prob(gm.home_win_prob),
        f"ML: {gm.home_ml}",
    ), unsafe_allow_html=True)

# Win prob bar
fig_wp = go.Figure(go.Bar(
    x=[gm.away_win_prob * 100, gm.home_win_prob * 100],
    y=[away_nm, home_nm],
    orientation="h",
    marker_color=[team_color(away_id), team_color(home_id)],
    text=[f"{gm.away_win_prob*100:.1f}%", f"{gm.home_win_prob*100:.1f}%"],
    textposition="auto",
))
fig_wp.update_layout(
    height=120, margin=dict(l=0, r=0, t=4, b=0),
    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
    yaxis=dict(showgrid=False),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"),
    showlegend=False,
)
st.plotly_chart(fig_wp, use_container_width=True)

st.markdown("---")

# ── Team projections table ────────────────────────────────────────────────
st.markdown("### Team Projections")

proj_df = pd.DataFrame([
    {
        "Team": away_nm,
        "Goals": round(ap.proj_goals, 1),
        "Score": round(ap.proj_scores, 1),
        "Shots": round(ap.proj_shots, 1),
        "SOG": round(ap.proj_sog, 1),
        "FO%": f"{ap.proj_faceoff_pct:.3f}",
        "FO Wins": round(ap.proj_faceoff_wins, 1),
        "2PT Goals": round(ap.proj_2pt_goals, 2),
        "Assists": round(ap.proj_assists, 1),
        "Saves": round(ap.proj_saves, 1),
        "Save%": f"{ap.proj_save_pct:.3f}",
        "TOs": round(ap.proj_turnovers, 1),
        "GBs": round(ap.proj_ground_balls, 1),
    },
    {
        "Team": home_nm,
        "Goals": round(hp.proj_goals, 1),
        "Score": round(hp.proj_scores, 1),
        "Shots": round(hp.proj_shots, 1),
        "SOG": round(hp.proj_sog, 1),
        "FO%": f"{hp.proj_faceoff_pct:.3f}",
        "FO Wins": round(hp.proj_faceoff_wins, 1),
        "2PT Goals": round(hp.proj_2pt_goals, 2),
        "Assists": round(hp.proj_assists, 1),
        "Saves": round(hp.proj_saves, 1),
        "Save%": f"{hp.proj_save_pct:.3f}",
        "TOs": round(hp.proj_turnovers, 1),
        "GBs": round(hp.proj_ground_balls, 1),
    },
])
proj_df = proj_df.set_index("Team")
st.dataframe(proj_df, use_container_width=True)

st.markdown("---")

# ── Simulation distributions ──────────────────────────────────────────────
st.markdown("### Simulation Distributions (20,000 sims)")

tab1, tab2, tab3 = st.tabs(["Score Total", "Score Margin", "Goals by Team"])

with tab1:
    fig_tot = go.Figure()
    total_arr = gs.total_distribution
    fig_tot.add_trace(go.Histogram(
        x=total_arr, nbinsx=40,
        name="Total Score", marker_color="#3b82f6", opacity=0.7,
    ))
    fig_tot.add_vline(x=gm.total_line, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Line: {gm.total_line}", annotation_position="top right")
    fig_tot.add_vline(x=gs.expected_total, line_dash="dot", line_color="#94a3b8",
                      annotation_text=f"Median: {gs.expected_total:.1f}", annotation_position="top left")
    fig_tot.update_layout(
        height=300, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        xaxis_title="Combined Score (incl. 2pt bonus)", yaxis_title="Simulations",
    )
    st.plotly_chart(fig_tot, use_container_width=True)

    p_over = float(np.mean(total_arr > gm.total_line))
    col_o, col_u, col_m = st.columns(3)
    col_o.metric("P(Over)", f"{p_over:.1%}", delta=f"Line: {gm.total_line}")
    col_u.metric("P(Under)", f"{1-p_over:.1%}")
    col_m.metric("P10 / P90", f"{np.percentile(total_arr,10):.0f} / {np.percentile(total_arr,90):.0f}")

with tab2:
    margin_arr = gs.margin_distribution
    fig_mar = go.Figure()
    colors_margin = [team_color(home_id) if v > 0 else team_color(away_id) for v in margin_arr]
    fig_mar.add_trace(go.Histogram(
        x=margin_arr, nbinsx=40,
        name="Margin", marker_color="#3b82f6", opacity=0.7,
    ))
    fig_mar.add_vline(x=0, line_color="#64748b", line_dash="dash")
    fig_mar.add_vline(x=gs.spread_home, line_color="#f59e0b", line_dash="dot",
                      annotation_text=f"Spread: {gs.spread_home:+.1f}")
    fig_mar.update_layout(
        height=300, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        xaxis_title=f"Score Margin (+ = {home_nm} wins)", yaxis_title="Simulations",
    )
    st.plotly_chart(fig_mar, use_container_width=True)

with tab3:
    fig_g = go.Figure()
    fig_g.add_trace(go.Histogram(
        x=gs.away_goals, name=away_nm,
        marker_color=team_color(away_id), opacity=0.6, nbinsx=30,
    ))
    fig_g.add_trace(go.Histogram(
        x=gs.home_goals, name=home_nm,
        marker_color=team_color(home_id), opacity=0.6, nbinsx=30,
    ))
    fig_g.update_layout(
        barmode="overlay", height=300, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        xaxis_title="Goals", yaxis_title="Simulations",
    )
    st.plotly_chart(fig_g, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{away_nm} median goals", f"{np.median(gs.away_goals):.1f}")
    c2.metric(f"{away_nm} P10/P90", f"{np.percentile(gs.away_goals,10):.0f} / {np.percentile(gs.away_goals,90):.0f}")
    c3.metric(f"{home_nm} median goals", f"{np.median(gs.home_goals):.1f}")
    c4.metric(f"{home_nm} P10/P90", f"{np.percentile(gs.home_goals,10):.0f} / {np.percentile(gs.home_goals,90):.0f}")

st.markdown("---")

# ── Top player projections (quick view) ──────────────────────────────────
st.markdown("### Player Projection Summary")
st.markdown('<span class="note-text">Go to Player Props for full prop markets. Go to Depth Charts to mark players inactive.</span>', unsafe_allow_html=True)

for side_nm, side_players in [(away_nm, result.away_players), (home_nm, result.home_players)]:
    active_players = [p for p in side_players if p.active]
    if not active_players:
        continue
    st.markdown(f"**{side_nm}**")
    rows = []
    for p in sorted(active_players, key=lambda x: x.proj_points, reverse=True)[:12]:
        rows.append({
            "Player": p.full_name or p.player_id,
            "Pos": p.position,
            "Proj Goals": round(p.proj_goals, 2),
            "Proj Assists": round(p.proj_assists, 2),
            "Proj Points": round(p.proj_points, 2),
            "Proj Shots": round(p.proj_shots, 1),
            "Proj SOG": round(p.proj_sog, 1),
            "2PT Rate": f"{p.proj_2pt_goals/max(p.proj_goals,0.01):.0%}" if p.proj_goals > 0.05 else "—",
            "Proj Saves": round(p.proj_saves, 1) if p.position == "G" else "—",
        })
    df_p = pd.DataFrame(rows)
    st.dataframe(df_p, use_container_width=True, hide_index=True)
