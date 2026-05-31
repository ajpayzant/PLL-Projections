"""
Page 4 — Game Lines
Final market output: moneyline, spread, total with manual line overrides.
Shows the full priced market and allows line adjustment to see probability shifts.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    get_engine, init_session,
    card, fmt_odds, fmt_prob,
    team_color, team_name,
)
from projection_engine_v3 import PricingEngine

st.set_page_config(page_title="Game Lines · PLL", page_icon="🥍", layout="wide")
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
  .note-text { color:#64748b;font-size:.80rem;font-style:italic; }
  .line-table td { padding: 6px 12px; }
  .line-table th { padding: 6px 12px; color: #94a3b8; }
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
gs = result.game_sim
gm = result.game_market
hold_pct = st.session_state.get("hold_pct", 0.045)

st.title("💰 Game Lines")
st.markdown(f"**{away_nm} @ {home_nm}** · Game {game.get('game_number','—')} · {str(game.get('game_date',''))[:10]}")

# ── Sidebar: line overrides + hold ────────────────────────────────────────
with st.sidebar:
    st.markdown("### Line Overrides")
    st.markdown('<span class="note-text">Override any market line to see how probabilities shift at that price.</span>',
                unsafe_allow_html=True)

    use_custom_total = st.checkbox("Override total line", value=False, key="override_total_chk")
    custom_total = st.number_input(
        "Total line", min_value=10.0, max_value=40.0, step=0.5,
        value=float(gm.total_line), key="custom_total_line",
        disabled=not use_custom_total,
    )

    use_custom_spread = st.checkbox("Override spread", value=False, key="override_spread_chk")
    custom_spread = st.number_input(
        f"Spread ({home_nm})", min_value=-15.0, max_value=15.0, step=0.5,
        value=float(gm.spread_home), key="custom_spread_line",
        disabled=not use_custom_spread,
    )

    st.markdown("---")
    hold_pct_slider = st.slider("Hold %", 2.0, 8.0, hold_pct * 100, 0.5, key="gl_hold") / 100.0
    pricing = PricingEngine(hold_pct=hold_pct_slider)

    st.markdown("---")
    st.markdown("### Export")
    if st.button("Copy lines to clipboard (CSV)", key="copy_csv"):
        st.info("Use the table download button below to export.")

# ── Market lines ──────────────────────────────────────────────────────────
total_arr = gs.total_distribution
margin_arr = gs.margin_distribution

# Recompute with potential overrides
actual_total_line = custom_total if use_custom_total else gm.total_line
actual_spread = custom_spread if use_custom_spread else gm.spread_home

p_over = float(np.mean(total_arr > actual_total_line))
p_under = 1.0 - p_over
over_adj = (p_over / (p_over + p_under)) * (1 + hold_pct_slider)
under_adj = (p_under / (p_over + p_under)) * (1 + hold_pct_slider)

def _am(prob):
    prob = min(max(prob, 0.001), 0.999)
    if prob >= 0.50:
        return str(int(-round((prob / (1 - prob)) * 100)))
    return "+" + str(int(round(((1 - prob) / prob) * 100)))

over_odds = _am(over_adj)
under_odds = _am(under_adj)

p_home_cover = float(np.mean(margin_arr > actual_spread))
p_away_cover = 1.0 - p_home_cover
home_cover_adj = (p_home_cover / (p_home_cover + p_away_cover)) * (1 + hold_pct_slider)
away_cover_adj = (p_away_cover / (p_home_cover + p_away_cover)) * (1 + hold_pct_slider)
home_spread_odds = _am(home_cover_adj)
away_spread_odds = _am(away_cover_adj)

st.markdown("---")
st.markdown("## Final Market Lines")

# Moneyline
st.markdown("### Moneyline")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(card(
        f"{away_nm} ML",
        gm.away_ml,
        f"Win prob: {fmt_prob(gm.away_win_prob)}",
    ), unsafe_allow_html=True)
with c2:
    st.markdown(card("Model", "—", f"Based on {gs.n_sims:,} sims"), unsafe_allow_html=True)
with c3:
    st.markdown(card(
        f"{home_nm} ML",
        gm.home_ml,
        f"Win prob: {fmt_prob(gm.home_win_prob)}",
    ), unsafe_allow_html=True)

# Spread
st.markdown("### Spread")
c1, c2, c3 = st.columns(3)
with c1:
    spread_away = -actual_spread
    label_a = f"{away_nm} {spread_away:+.1f}" if spread_away != 0 else f"{away_nm} PK"
    st.markdown(card(label_a, away_spread_odds, f"P(cover): {p_away_cover:.1%}"), unsafe_allow_html=True)
with c2:
    override_note = " ⚡ OVERRIDDEN" if use_custom_spread else ""
    st.markdown(card("Spread Line", f"{actual_spread:+.1f}{override_note}", "Model spread"), unsafe_allow_html=True)
with c3:
    label_h = f"{home_nm} {actual_spread:+.1f}" if actual_spread != 0 else f"{home_nm} PK"
    st.markdown(card(label_h, home_spread_odds, f"P(cover): {p_home_cover:.1%}"), unsafe_allow_html=True)

# Total
st.markdown("### Total")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(card("Over", over_odds, f"P(over): {p_over:.1%}"), unsafe_allow_html=True)
with c2:
    override_note = " ⚡ OVERRIDDEN" if use_custom_total else ""
    st.markdown(card(f"Total Line{override_note}", f"{actual_total_line}", f"Model median: {gs.expected_total:.1f}"),
                unsafe_allow_html=True)
with c3:
    st.markdown(card("Under", under_odds, f"P(under): {p_under:.1%}"), unsafe_allow_html=True)

st.markdown("---")

# ── Full market summary table ─────────────────────────────────────────────
st.markdown("### Full Market Summary")

market_rows = [
    {"Market": f"{away_nm} Moneyline", "Line": "—", "Odds": gm.away_ml,
     "Fair Prob": fmt_prob(gm.away_win_prob), "Hold": f"{hold_pct_slider*100:.1f}%"},
    {"Market": f"{home_nm} Moneyline", "Line": "—", "Odds": gm.home_ml,
     "Fair Prob": fmt_prob(gm.home_win_prob), "Hold": f"{hold_pct_slider*100:.1f}%"},
    {"Market": f"Spread {away_nm} {spread_away:+.1f}", "Line": f"{spread_away:+.1f}",
     "Odds": away_spread_odds,
     "Fair Prob": fmt_prob(p_away_cover), "Hold": f"{hold_pct_slider*100:.1f}%"},
    {"Market": f"Spread {home_nm} {actual_spread:+.1f}", "Line": f"{actual_spread:+.1f}",
     "Odds": home_spread_odds,
     "Fair Prob": fmt_prob(p_home_cover), "Hold": f"{hold_pct_slider*100:.1f}%"},
    {"Market": "Total Over", "Line": f"{actual_total_line}", "Odds": over_odds,
     "Fair Prob": fmt_prob(p_over), "Hold": f"{hold_pct_slider*100:.1f}%"},
    {"Market": "Total Under", "Line": f"{actual_total_line}", "Odds": under_odds,
     "Fair Prob": fmt_prob(p_under), "Hold": f"{hold_pct_slider*100:.1f}%"},
]
st.dataframe(pd.DataFrame(market_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Score probability table ───────────────────────────────────────────────
st.markdown("### Score Probability Grid")
st.markdown('<span class="note-text">Probability of each exact score combination.</span>', unsafe_allow_html=True)

# Build a score grid from the simulation
home_scores_arr = gs.home_scores.astype(int)
away_scores_arr = gs.away_scores.astype(int)
n = len(home_scores_arr)

h_range = sorted(set(home_scores_arr.clip(0, 30)))[:16]
a_range = sorted(set(away_scores_arr.clip(0, 30)))[:16]

grid = np.zeros((len(a_range), len(h_range)))
for i, av in enumerate(a_range):
    for j, hv in enumerate(h_range):
        grid[i, j] = np.mean((home_scores_arr == hv) & (away_scores_arr == av))

fig_grid = go.Figure(go.Heatmap(
    z=grid * 100,
    x=[str(v) for v in h_range],
    y=[str(v) for v in a_range],
    colorscale="Blues",
    text=[[f"{grid[i,j]*100:.1f}%" for j in range(len(h_range))] for i in range(len(a_range))],
    texttemplate="%{text}",
    textfont=dict(size=9),
    showscale=True,
    colorbar=dict(title="Probability %"),
))
fig_grid.update_layout(
    height=420,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title=f"{home_nm} Score",
    yaxis_title=f"{away_nm} Score",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"),
    title=f"Score Probability Grid ({n//1000}k sims)",
)
st.plotly_chart(fig_grid, use_container_width=True)

st.markdown("---")

# ── Total distribution with line marker ──────────────────────────────────
st.markdown("### Score Total Distribution")
fig_tot = go.Figure()
fig_tot.add_trace(go.Histogram(
    x=total_arr, nbinsx=35,
    marker_color="#3b82f6", opacity=0.7, name="Total",
))
fig_tot.add_vline(
    x=actual_total_line, line_dash="dash", line_color="#f59e0b", line_width=2,
    annotation_text=f"Line: {actual_total_line} | O{over_odds} / U{under_odds}",
    annotation_position="top right",
)
fig_tot.update_layout(
    height=260, margin=dict(l=0, r=0, t=8, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9"),
    xaxis_title="Combined Score", yaxis_title="Simulations",
)
st.plotly_chart(fig_tot, use_container_width=True)

st.markdown('<span class="note-text">'
            f'Model: v3 possession-chain · {gs.n_sims:,} Monte Carlo simulations · '
            f'Hold: {hold_pct_slider*100:.1f}% · '
            f'Bias: ~0 goals (calibrated)'
            '</span>', unsafe_allow_html=True)
