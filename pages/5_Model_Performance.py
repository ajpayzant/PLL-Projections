"""Page 5 -- Model Performance"""
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

from _engine_state import SHARED_CSS, get_engine, init_session, get_data_freshness

st.set_page_config(page_title="Model Performance · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.title("📈 Model Performance")
st.markdown(
    "Backtested on all historical PLL games from 2023 onward. "
    "The model projects each game using only data available *before* that game was played."
)

engine = get_engine()

# -- Freshness indicator ---------------------------------------------------
fresh = get_data_freshness()
if fresh.get("available"):
    if fresh["stale"]:
        st.warning(f"⚠️ Data last updated {fresh['last_updated']} ({fresh['age_hours']:.0f}h ago). Run the Update PLL Data Warehouse Action to refresh.")
    else:
        st.markdown(
            f'<span class="note-text">Data updated: {fresh["last_updated"]} ({fresh["age_hours"]:.0f}h ago)</span>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# -- Run backtest lazily (only when this page is visited) ------------------
# The backtest takes ~25-35s but is cached for the session via cache_data.
# It also fits the calibrator so win probabilities are calibrated for the
# rest of the session after this page has been visited once.
@st.cache_data(show_spinner="Running backtest (~30s, once per session)…", ttl=7200)
def _get_backtest_result(_engine_id: int):
    from projection_engine_v3 import Backtester
    bt = Backtester(engine.loader, n_sims=5_000)
    result = bt.run()
    return result, bt.raw_rows

with st.spinner("Running backtest analysis — this takes ~30 seconds the first time…"):
    bt, raw_rows = _get_backtest_result(id(engine))

# Fit the calibrator on the backtest results so win probabilities are
# calibrated for the rest of this session.
if raw_rows and not engine.calibrator._fitted:
    engine.calibrator.fit(raw_rows)

if bt.n_games == 0:
    st.warning("No backtest games available. Check that historical data is loaded.")
    st.stop()

# -- Summary metrics -------------------------------------------------------
st.markdown("### Summary Metrics")
st.markdown(
    f'<span class="note-text">Based on {bt.n_games} historical games (2023–present). '
    f'All predictions use only data from before each game was played.</span>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Games tested", bt.n_games)
c2.metric("MAE total goals", f"{bt.mae_total_goals:.2f}",
          help="Mean absolute error on combined goals per game. Lower = better.")
c3.metric("Correct winner %", f"{bt.correct_winner_pct:.1%}",
          help="% of games where the model correctly identified the winner.")
c4.metric("Brier score", f"{bt.brier_score:.4f}",
          help="Probability calibration score (0=perfect, 0.25=coin-flip). Lower = better.")
c5.metric("Bias (total goals)", f"{bt.bias_total_goals:+.2f}",
          help="Average (predicted − actual). Positive = model systematically over-predicts scoring.")

st.markdown("---")

# -- Calibration chart -----------------------------------------------------
st.markdown("### Win Probability Calibration")
st.markdown(
    '<span class="note-text">A well-calibrated model has bars close to the diagonal. '
    'If the bar for "60-70% predicted" reaches ~65% actual, the model is accurate.</span>',
    unsafe_allow_html=True,
)

cal_df = bt.calibration_table.dropna()
if not cal_df.empty and "mean_pred" in cal_df.columns and "actual_rate" in cal_df.columns:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cal_df["mean_pred"],
        y=cal_df["actual_rate"],
        name="Actual win rate",
        marker_color="#3b82f6",
        opacity=0.8,
        width=0.08,
        text=[f"n={int(n)}" for n in cal_df["n"]],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(color="#f59e0b", dash="dash", width=1.5),
    ))
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Predicted win probability", range=[0, 1],
                   tickformat=".0%", showgrid=True, gridcolor="rgba(148,163,184,.15)"),
        yaxis=dict(title="Actual win rate", range=[0, 1],
                   tickformat=".0%", showgrid=True, gridcolor="rgba(148,163,184,.15)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"), legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -- MAE by season ---------------------------------------------------------
st.markdown("### Goal Prediction Error Detail")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Home vs Away MAE**")
    hv_df = pd.DataFrame([
        {"Side": "Home goals", "MAE": bt.mae_home_goals},
        {"Side": "Away goals", "MAE": bt.mae_away_goals},
        {"Side": "Total goals", "MAE": bt.mae_total_goals},
        {"Side": "Total score", "MAE": bt.mae_total_scores},
    ])
    st.dataframe(hv_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**RMSE & Bias**")
    rb_df = pd.DataFrame([
        {"Metric": "RMSE (total goals)", "Value": f"{bt.rmse_total_goals:.3f}"},
        {"Metric": "Bias (total goals)", "Value": f"{bt.bias_total_goals:+.3f}"},
        {"Metric": "Brier score", "Value": f"{bt.brier_score:.4f}"},
        {"Metric": "Correct winner %", "Value": f"{bt.correct_winner_pct:.1%}"},
    ])
    st.dataframe(rb_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    '<span class="note-text">'
    'Backtest uses 5,000 simulations per game for speed. '
    'Calibrator is fitted on these results to adjust win probabilities on the Projections page.'
    '</span>',
    unsafe_allow_html=True,
)
