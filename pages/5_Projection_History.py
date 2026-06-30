"""Page 5 -- Projection History & Model Performance"""
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
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from _engine_state import SHARED_CSS, init_session

st.set_page_config(page_title="Projection History · PLL", page_icon="📊", layout="wide")
init_session()

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.title("📊 Projection History & Model Performance")

# ── Load saved games from Google Sheets ──────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading saved projections from Google Sheets...")
def _load_all_games():
    """
    Read all saved game tabs from the master sheet.
    Returns a list of dicts, each with player_props and team_projections DataFrames.
    """
    try:
        from gsheets_writer import list_saved_games, read_game_tab
    except ImportError:
        return [], "gsheets_writer module not found."

    games = list_saved_games()
    if not games:
        return [], None

    loaded = []
    for g in games:
        try:
            sections = read_game_tab(g["tab_name"])
            g["player_props"]      = sections.get("player_props", pd.DataFrame())
            g["team_projections"]  = sections.get("team_projections", pd.DataFrame())
            g["game_lines"]        = sections.get("game_lines", pd.DataFrame())
            loaded.append(g)
        except Exception:
            continue
    return loaded, None


try:
    all_games, load_err = _load_all_games()
except Exception as e:
    all_games, load_err = [], str(e)

if load_err:
    st.error(f"Could not load from Google Sheets: {load_err}")
    st.stop()

if not all_games:
    st.info(
        "No saved projections found yet. Run a projection on the **Projections** page "
        "and click **☁️ Save to Google Sheets** to start tracking."
    )
    st.stop()

# ── Filter to games that have actuals filled in ───────────────────────────────

def _has_actuals(g: dict) -> bool:
    pp = g.get("player_props", pd.DataFrame())
    if pp.empty or "Actual Result" not in pp.columns:
        return False
    return pp["Actual Result"].replace("", np.nan).notna().any()

games_with_actuals = [g for g in all_games if _has_actuals(g)]
games_pending      = [g for g in all_games if not _has_actuals(g)]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.markdown(
        f"**{len(all_games)}** saved games · "
        f"**{len(games_with_actuals)}** with actuals"
    )
    if games_with_actuals:
        st.markdown("---")
        pos_filter = st.multiselect(
            "Positions", ["A", "M", "FO", "SSDM", "LSM", "D", "G"],
            default=["A", "M", "FO", "G"],
            key="hist_pos",
        )
        stat_filter = st.multiselect(
            "Stats", ["Goals", "Assists", "Points", "SOG", "Saves", "FO Wins"],
            default=["Goals", "Assists", "Points", "SOG", "Saves", "FO Wins"],
            key="hist_stat",
        )

    st.markdown("---")
    if st.button("🔄 Refresh data", key="hist_refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Pending games notice ──────────────────────────────────────────────────────
if games_pending:
    with st.expander(f"⏳ {len(games_pending)} game(s) pending actuals", expanded=False):
        for g in games_pending:
            st.markdown(
                f"- **{g['away']} @ {g['home']}** · "
                f"Game {g['game_number']} · {g['game_date']} · "
                f"Use **🔄 Sync Actuals** on the Projections page after the game completes"
            )

if not games_with_actuals:
    st.info(
        "No games with actuals yet. After a game completes, go to the **Projections** page, "
        "select the game, and click **🔄 Sync Actuals**."
    )
    st.stop()

# ── Build master player props DataFrame ──────────────────────────────────────

def _build_master_props(games: list) -> pd.DataFrame:
    frames = []
    for g in games:
        pp = g.get("player_props", pd.DataFrame()).copy()
        if pp.empty:
            continue
        pp["game_date"]   = g["game_date"]
        pp["game_tab"]    = g["tab_name"]
        pp["away"]        = g["away"]
        pp["home"]        = g["home"]
        frames.append(pp)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    for col in ["Projection", "Main Line", "Fair P(Over)", "P10", "P50", "P90", "Actual Result"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute error columns
    df["error"]     = df["Actual Result"] - df["Projection"]
    df["abs_error"] = df["error"].abs()
    df["hit"]       = df["Hit/Miss"].str.strip().str.lower() == "hit"
    df["has_line"]  = df["Main Line"].notna() & df["Main Line"].ne(0)

    return df


master = _build_master_props(games_with_actuals)

if master.empty:
    st.warning("Could not parse player props data from saved sheets.")
    st.stop()

# Apply filters
if "Pos" in master.columns:
    master = master[master["Pos"].isin(pos_filter)]
if "Stat" in master.columns:
    master = master[master["Stat"].isin(stat_filter)]

# Drop rows with no actual
master = master[master["Actual Result"].notna()]

# ── Section 1: Summary scorecards ─────────────────────────────────────────────
st.markdown("## Overall Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

total_props = len(master[master["has_line"]])
hit_rate    = float(master[master["has_line"]]["hit"].mean()) if total_props > 0 else 0.0
mae_all     = float(master["abs_error"].mean()) if len(master) > 0 else 0.0
bias        = float(master["error"].mean()) if len(master) > 0 else 0.0
games_count = len(games_with_actuals)

with col1:
    st.metric("Games Tracked", games_count)
with col2:
    st.metric("Props Graded", total_props)
with col3:
    st.metric("Hit Rate (main line)", f"{hit_rate*100:.1f}%",
              delta=f"{(hit_rate-0.50)*100:+.1f}% vs 50%")
with col4:
    st.metric("Avg Error (MAE)", f"{mae_all:.3f}")
with col5:
    bias_dir = "high" if bias > 0 else "low"
    st.metric("Projection Bias", f"{bias:+.3f}",
              delta=f"Model runs {bias_dir}" if abs(bias) > 0.05 else "No bias detected")

st.markdown("---")

# ── Section 2: Accuracy by stat ───────────────────────────────────────────────
st.markdown("## Accuracy by Stat")

stat_summary = (
    master.groupby("Stat")
    .agg(
        Games=("game_tab", "nunique"),
        Props=("Projection", "count"),
        MAE=("abs_error", "mean"),
        Bias=("error", "mean"),
        Hit_Rate=("hit", lambda x: x[master.loc[x.index, "has_line"]].mean()
                  if master.loc[x.index, "has_line"].any() else np.nan),
    )
    .round(3)
    .reset_index()
    .rename(columns={"Hit_Rate": "Hit Rate"})
)

if not stat_summary.empty:
    fig_mae = px.bar(
        stat_summary, x="Stat", y="MAE",
        color="MAE",
        color_continuous_scale=["#34d399", "#fbbf24", "#f87171"],
        title="Mean Absolute Error by Stat",
        labels={"MAE": "MAE (lower = better)"},
    )
    fig_mae.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", showlegend=False,
        coloraxis_showscale=False,
    )
    fig_mae.update_xaxes(showgrid=False)
    fig_mae.update_yaxes(gridcolor="rgba(148,163,184,.15)")

    fig_bias = px.bar(
        stat_summary, x="Stat", y="Bias",
        color="Bias",
        color_continuous_scale=["#f87171", "#e2e8f0", "#34d399"],
        color_continuous_midpoint=0,
        title="Projection Bias by Stat (positive = model ran high)",
        labels={"Bias": "Mean Error"},
    )
    fig_bias.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", showlegend=False,
        coloraxis_showscale=False,
    )
    fig_bias.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,.5)")
    fig_bias.update_xaxes(showgrid=False)
    fig_bias.update_yaxes(gridcolor="rgba(148,163,184,.15)")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_mae, use_container_width=True)
    with c2:
        st.plotly_chart(fig_bias, use_container_width=True)

    st.dataframe(
        stat_summary.style.format({
            "MAE": "{:.3f}", "Bias": "{:+.3f}",
            "Hit Rate": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
        }),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")

# ── Section 3: Accuracy by position ──────────────────────────────────────────
st.markdown("## Accuracy by Position")

if "Pos" in master.columns:
    pos_summary = (
        master.groupby("Pos")
        .agg(
            Props=("Projection", "count"),
            MAE=("abs_error", "mean"),
            Bias=("error", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("MAE")
    )
    st.dataframe(
        pos_summary.style.format({"MAE": "{:.3f}", "Bias": "{:+.3f}"}),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")

# ── Section 4: Per-player accuracy ───────────────────────────────────────────
st.markdown("## Per-Player Accuracy")
st.caption("Players with at least 3 graded props")

if "Player" in master.columns:
    player_summary = (
        master.groupby(["Player", "Pos"])
        .agg(
            Props=("Projection", "count"),
            MAE=("abs_error", "mean"),
            Bias=("error", "mean"),
            Hit_Rate=("hit", lambda x: x[master.loc[x.index, "has_line"]].mean()
                      if master.loc[x.index, "has_line"].any() else np.nan),
        )
        .round(3)
        .reset_index()
        .rename(columns={"Hit_Rate": "Hit Rate"})
    )
    player_summary = player_summary[player_summary["Props"] >= 3].sort_values("MAE")

    if not player_summary.empty:
        st.dataframe(
            player_summary.style.format({
                "MAE": "{:.3f}", "Bias": "{:+.3f}",
                "Hit Rate": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Not enough data yet — need at least 3 graded props per player.")

st.markdown("---")

# ── Section 5: Calibration check ─────────────────────────────────────────────
st.markdown("## Probability Calibration")
st.caption(
    "Does a 60% model probability actually hit ~60% of the time? "
    "A well-calibrated model follows the diagonal line."
)

cal_df = master[master["has_line"] & master["Fair P(Over)"].notna()].copy()
if len(cal_df) >= 20:
    cal_df["prob_bucket"] = pd.cut(
        cal_df["Fair P(Over)"],
        bins=[0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=["<45%", "45-50%", "50-55%", "55-60%", "60-65%", "65-70%", ">70%"],
    )
    calib = (
        cal_df.groupby("prob_bucket", observed=True)
        .agg(Count=("hit", "count"), Actual_Hit_Rate=("hit", "mean"))
        .reset_index()
        .rename(columns={"prob_bucket": "Model Probability Bucket"})
    )
    calib["Actual_Hit_Rate"] = calib["Actual_Hit_Rate"].round(3)

    mid_map = {"<45%": 0.42, "45-50%": 0.475, "50-55%": 0.525,
               "55-60%": 0.575, "60-65%": 0.625, "65-70%": 0.675, ">70%": 0.75}
    calib["mid"] = calib["Model Probability Bucket"].map(mid_map)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="rgba(148,163,184,.5)"),
        name="Perfect calibration",
    ))
    fig_cal.add_trace(go.Scatter(
        x=calib["mid"], y=calib["Actual_Hit_Rate"],
        mode="lines+markers",
        marker=dict(size=10, color="#34d399"),
        line=dict(color="#34d399"),
        name="Model",
        text=calib["Count"].apply(lambda n: f"n={n}"),
        textposition="top center",
    ))
    fig_cal.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="Model P(Over)",
        yaxis_title="Actual Hit Rate",
        xaxis=dict(range=[0.3, 0.85], tickformat=".0%", showgrid=False),
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(148,163,184,.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.dataframe(
        calib[["Model Probability Bucket", "Count", "Actual_Hit_Rate"]]
        .rename(columns={"Actual_Hit_Rate": "Actual Hit Rate"})
        .style.format({"Actual Hit Rate": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )
else:
    st.info(f"Need at least 20 graded props for calibration chart. Currently have {len(cal_df)}.")

st.markdown("---")

# ── Section 6: Game-by-game results ──────────────────────────────────────────
st.markdown("## Game-by-Game Results")

for g in games_with_actuals:
    pp = g.get("player_props", pd.DataFrame()).copy()
    if pp.empty:
        continue
    pp["Actual Result"] = pd.to_numeric(pp.get("Actual Result", pd.Series()), errors="coerce")
    pp["Projection"]    = pd.to_numeric(pp.get("Projection", pd.Series()), errors="coerce")
    pp = pp[pp["Actual Result"].notna()]
    if pp.empty:
        continue

    mae = float((pp["Actual Result"] - pp["Projection"]).abs().mean())
    hit_r = float((pp["Hit/Miss"].str.strip().str.lower() == "hit").mean()) if "Hit/Miss" in pp.columns else 0.0

    with st.expander(
        f"**{g['away']} @ {g['home']}** · Game {g['game_number']} · {g['game_date']} "
        f"· MAE {mae:.3f} · Hit Rate {hit_r*100:.0f}%",
        expanded=False,
    ):
        display_cols = [c for c in ["Player", "Pos", "Stat", "Projection",
                                     "Main Line", "Fair P(Over)", "P50",
                                     "Actual Result", "Hit/Miss"]
                        if c in pp.columns]
        if display_cols:
            st.dataframe(
                pp[display_cols]
                .sort_values(["Pos", "Player", "Stat"])
                .style.format({
                    "Projection": "{:.2f}", "P50": "{:.2f}",
                    "Actual Result": "{:.0f}",
                    "Fair P(Over)": lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—",
                }),
                use_container_width=True, hide_index=True,
            )
