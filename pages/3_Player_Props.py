"""Page 3 -- Player Props"""
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
    SHARED_CSS, pos_badge, fmt_prob,
    get_engine, init_session,
    team_color, team_name,
    render_update_projection_btn,
)
from projection_engine_v3 import PricingEngine

st.set_page_config(page_title="Player Props · PLL", page_icon="🥍", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** and run a game first.")
    st.stop()

game    = st.session_state.selected_game or {}
home_id = result.home_proj.team_id
away_id = result.away_proj.team_id
home_nm = team_name(home_id)
away_nm = team_name(away_id)
hold_pct = st.session_state.get("hold_pct", 0.075)
pricing  = PricingEngine(hold_pct=hold_pct)

st.title("👤 Player Prop Markets")
st.markdown(
    f"**{away_nm} @ {home_nm}** · "
    f"Game {game.get('game_number','--')} · "
    f"{str(game.get('game_date',''))[:10]}"
)

STAT_LABELS = {
    "goals": "Goals", "assists": "Assists", "points": "Points",
    "shots": "Shots", "shots_on_goal": "SOG", "two_pt_goals": "2PT Goals",
    "one_pt_goals": "1PT Goals", "saves": "Saves", "faceoff_wins": "FO Wins",
    "ground_balls": "Ground Balls",
}
FIELD_STATS  = ["goals", "assists", "points", "shots_on_goal", "two_pt_goals", "ground_balls"]
GOALIE_STATS = ["saves"]
FO_STATS     = ["faceoff_wins"]
MILE_DEFS    = {
    "goals":         [1, 2, 3],
    "assists":       [1, 2],
    "points":        [1, 2, 3, 4],
    "saves":         [10, 12, 14],
    "shots_on_goal": [2, 3, 4],
}

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### View Mode")
    view_mode = st.radio(
        "Layout",
        ["Table (all players)", "Expander (per player)"],
        key="prop_view_mode",
        help="Table view shows all players and their main lines in one sortable grid.",
    )

    st.markdown("---")
    st.markdown("### Filters")
    show_team = st.radio("Team", ["Both", away_nm, home_nm], key="prop_team")
    show_pos  = st.multiselect(
        "Positions", ["A","M","D","FO","SSDM","LSM","G"],
        default=["A","M","FO","G"], key="prop_pos",
    )
    min_pts   = st.number_input("Min projected points", 0.0, 3.0, 0.3, 0.1, key="prop_min_pts")
    show_miles = st.checkbox("Show milestone props (1+, 2+, 3+)", value=True)
    show_alt   = st.checkbox("Show alternate line pricing", value=False)

    st.markdown("---")
    st.markdown("### Market Margin %")
    hold_num = st.number_input(
        "Market margin %",
        min_value=2.0, max_value=15.0,
        value=float(st.session_state.get("hold_pct", 0.075) * 100),
        step=0.5, key="pp_hold_num",
        help="Vig/margin applied to all priced props. Updates across all pages.",
    )
    st.markdown("---")
    st.markdown("### Market Line Comparison")
    st.markdown('<span class="note-text">Enter a market line to see model edge vs market.</span>',
                unsafe_allow_html=True)
    mkt_player = st.text_input("Player name (partial)", key="mkt_player")
    mkt_stat   = st.selectbox("Stat", ["goals","assists","points","shots_on_goal",
                                        "saves","faceoff_wins"], key="mkt_stat")
    mkt_line   = st.number_input("Market line", 0.5, 25.5, 0.5, 0.5, key="mkt_line",
                                  help="Enter the sportsbook's line. Model shows edge = fair prob minus market implied prob.")
    mkt_over_odds = st.number_input("Market over odds", -500, 500, -110, 5, key="mkt_over_odds",
                                     help="Enter as integer, e.g. -115 or +105")

    st.markdown("---")
    st.markdown("### Quick Line Override")
    st.markdown('<span class="note-text">Price any player at a custom line.</span>',
                unsafe_allow_html=True)
    ov_player = st.text_input("Player name (partial)", key="ov_player")
    ov_stat   = st.selectbox("Stat", ["goals","assists","points","shots_on_goal",
                                       "saves","faceoff_wins"], key="ov_stat")
    ov_line   = st.number_input(
        "Line", 0.5, 25.5, 0.5, 1.0, key="ov_line",
        help="Lines are forced to x.5 values to avoid pushes."
    )

    st.markdown("---")
    engine = get_engine()

    # -- This week's other games (quick switch) ----------------------------
    import datetime as _dt
    _today = _dt.date.today()
    _all_games = engine.upcoming_games()
    _week_games = []
    for _g in _all_games:
        try:
            _gd = _dt.date.fromisoformat(str(_g.get("game_date", ""))[:10])
            if -1 <= (_gd - _today).days <= 7:
                _week_games.append(_g)
        except Exception:
            pass
    # Only show if there are other games this week besides the current one
    _other = [_g for _g in _week_games
              if _g.get("home_team_id") != home_id or _g.get("away_team_id") != away_id]
    if _other:
        st.markdown("### This week's games")
        for _g in _other:
            _ht = team_name(_g.get("home_team_id", ""))
            _at = team_name(_g.get("away_team_id", ""))
            _lbl = f"Game {_g.get('game_number','?')} · {_at} @ {_ht}"
            if st.button(_lbl, key=f"sw_{_g.get('home_team_id')}_{_g.get('away_team_id')}",
                         use_container_width=True):
                from _engine_state import run_projection_for_game, _autosave
                with st.spinner("Projecting…"):
                    run_projection_for_game(engine, _g)
                    _autosave()
                st.rerun()

    render_update_projection_btn(engine, key="p2")

# hold_pct synced globally via session state
new_hold_pct = hold_num / 100.0
st.session_state.hold_pct = new_hold_pct
pricing = PricingEngine(hold_pct=new_hold_pct)

# -- Collect sims ----------------------------------------------------------
all_projs = {p.player_id: p for p in result.home_players + result.away_players}
markets   = result.player_markets

def _half_only_lines(lo: float, hi: float):
    """Generate alternate prop lines with decimal .5 only; whole numbers are excluded."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return [0.5]
    if hi < lo:
        lo, hi = hi, lo
    lo = max(0.5, lo)
    start = np.floor(lo) + 0.5
    if start < lo - 1e-9:
        start += 1.0
    end = np.ceil(hi) + 0.5
    return [round(float(v), 1) for v in np.arange(start, end + 1e-9, 1.0)]

def _alt_width(stat: str) -> float:
    return 5.0 if stat in {"saves", "faceoff_wins"} else 3.0

def _keep(pid: str) -> bool:
    pm  = markets.get(pid, {})
    pv  = pm.get("proj_values", {})
    pts = max(pv.get("points",0), pv.get("saves",0), pv.get("faceoff_wins",0))
    if pts < min_pts:
        return False
    proj = all_projs.get(pid)
    if proj is None or not proj.active:
        return False
    if proj.position not in show_pos:
        return False
    if show_team == away_nm and proj.team_id != away_id:
        return False
    if show_team == home_nm and proj.team_id != home_id:
        return False
    return True

sims_filtered = sorted(
    [s for s in (result.home_player_sims + result.away_player_sims) if _keep(s.player_id)],
    key=lambda s: markets.get(s.player_id, {}).get("proj_values", {}).get("points", 0),
    reverse=True,
)

if not sims_filtered:
    st.info("No players match the current filters.")
    st.stop()

st.markdown(f"**{len(sims_filtered)} players shown** · hold: {new_hold_pct*100:.1f}%")
st.markdown("---")


# ===========================================================================
# TABLE VIEW  — stat-grouped prop sheet
# ===========================================================================
if view_mode == "Table (all players)":

    # Extra CSS for the prop sheet
    st.markdown("""
    <style>
    .prop-section-header {
        font-size:.78rem; font-weight:700; letter-spacing:.08em;
        text-transform:uppercase; color:#94a3b8;
        border-bottom:1px solid rgba(148,163,184,.20);
        padding-bottom:4px; margin:18px 0 6px;
    }
    .prop-subhead {
        font-size:.70rem; font-weight:600; color:#64748b;
        text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Prop Sheet — All Players")
    st.markdown(
        '<span class="note-text">Sorted by projection. '
        'Proj = model mean · Line = balanced x.5 line · '
        'P10/P90 = 10th/90th percentile range · '
        'Click any column header to re-sort.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Build per-player data once, reuse across stat tables ────────────
    player_data = []
    for ps in sims_filtered:
        pid  = ps.player_id
        proj = all_projs.get(pid)
        if proj is None:
            continue
        pm  = markets.get(pid, {})
        pv  = pm.get("proj_values", {})
        ms  = pm.get("markets", {})
        player_data.append({
            "pid": pid, "ps": ps, "proj": proj,
            "pv": pv, "ms": ms,
            "nm": proj.full_name or pid,
            "pos": proj.position,
            "tid": proj.team_id,
        })

    # Map stat key -> PlayerProjection attribute for deterministic proj values
    _DET_ATTR = {
        "goals": "proj_goals", "assists": "proj_assists", "points": "proj_points",
        "shots": "proj_shots", "shots_on_goal": "proj_sog",
        "two_pt_goals": "proj_2pt_goals", "one_pt_goals": "proj_1pt_goals",
        "saves": "proj_saves", "faceoff_wins": "proj_faceoff_wins",
        "ground_balls": "proj_ground_balls",
    }

    def _stat_table(stat: str, label: str, pd_list: list, sort_col: str = "Proj") -> pd.DataFrame:
        """Build one clean stat table: Player | Team | Pos | Proj | P10 | P90 | Line | Over | Under | P(Over)"""
        rows = []
        for d in pd_list:
            ps  = d["ps"]
            pv  = d["pv"]
            ms  = d["ms"]
            proj = d["proj"]
            if stat not in ps.stat_distributions:
                continue
            dist = ps.stat_distributions[stat]
            # Use deterministic PlayerProjection value so Proj column matches
            # the left panel exactly. Fall back to sim mean only if missing.
            attr = _DET_ATTR.get(stat)
            proj_val = float(getattr(proj, attr, None) or pv.get(stat, 0))
            if proj_val < 0.02:
                continue
            m = ms.get(stat, {})
            line = m.get("line")
            rows.append({
                "Player":   d["nm"],
                "Team":     team_name(d["tid"]),
                "Pos":      d["pos"],
                "Proj":     round(proj_val, 1),
                "P10":      round(float(np.percentile(dist, 10)), 1),
                "Median":   round(float(np.percentile(dist, 50)), 1),
                "P90":      round(float(np.percentile(dist, 90)), 1),
                "Line":     f"{line:.1f}" if isinstance(line, (int, float)) else "--",
                "Over":     m.get("over_odds", "--"),
                "Under":    m.get("under_odds", "--"),
                "P(Over)":  f"{m.get('fair_over_prob', 0):.1%}" if m else "--",
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values(sort_col, ascending=False).reset_index(drop=True)
        return df

    # ── Field player stat sections ───────────────────────────────────────
    field = [d for d in player_data if d["pos"] not in ("G", "FO")]
    special = [d for d in player_data if d["pos"] in ("G", "FO")]

    FIELD_SECTIONS = [
        ("points",        "Points"),
        ("goals",         "Goals"),
        ("assists",       "Assists"),
        ("shots_on_goal", "Shots on Goal"),
    ]

    if field:
        st.markdown('<div class="prop-section-header">Field Players</div>', unsafe_allow_html=True)
        tabs = st.tabs([lbl for _, lbl in FIELD_SECTIONS])
        for tab, (stat, lbl) in zip(tabs, FIELD_SECTIONS):
            with tab:
                df = _stat_table(stat, lbl, field)
                if df.empty:
                    st.caption(f"No {lbl} data available.")
                else:
                    # applymap was removed in pandas 2.1+; use map instead.
                    def _style_odds(val):
                        if isinstance(val, str) and val.startswith("+"):
                            return "color:#34d399;font-weight:600"
                        if isinstance(val, str) and val.startswith("-"):
                            return "color:#f1f5f9"
                        return ""

                    styled = df.style.map(
                        _style_odds, subset=["Over", "Under"]
                    ).format(precision=2)
                    st.dataframe(styled, width="stretch", hide_index=True)

    # ── Goalies ──────────────────────────────────────────────────────────
    goalies = [d for d in special if d["pos"] == "G"]
    fo_players = [d for d in special if d["pos"] == "FO"]

    if goalies or fo_players:
        st.markdown('<div class="prop-section-header">Goalies & Faceoff Specialists</div>',
                    unsafe_allow_html=True)
        spec_cols = st.columns(2)

        with spec_cols[0]:
            st.markdown('<div class="prop-subhead">Saves</div>', unsafe_allow_html=True)
            df_sv = _stat_table("saves", "Saves", goalies)
            if df_sv.empty:
                st.caption("No goalie data.")
            else:
                st.dataframe(df_sv, width="stretch", hide_index=True)

        with spec_cols[1]:
            st.markdown('<div class="prop-subhead">Faceoff Wins</div>', unsafe_allow_html=True)
            df_fo = _stat_table("faceoff_wins", "FO Wins", fo_players)
            if df_fo.empty:
                st.caption("No FO data.")
            else:
                st.dataframe(df_fo, width="stretch", hide_index=True)

    # ── Milestones summary ───────────────────────────────────────────────
    if show_miles and field:
        st.markdown('<div class="prop-section-header">Milestones</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="note-text">P(Hit) = model probability of reaching that threshold.</span>',
            unsafe_allow_html=True,
        )
        MILE_SECTIONS = [
            ("points",        "Points", [1, 2, 3, 4]),
            ("goals",         "Goals",  [1, 2, 3]),
            ("assists",       "Assists",[1, 2]),
            ("shots_on_goal", "SOG",    [2, 3, 4]),
        ]
        mile_tabs = st.tabs([lbl for _, lbl, _ in MILE_SECTIONS])
        for tab, (stat, lbl, levels) in zip(mile_tabs, MILE_SECTIONS):
            with tab:
                mile_rows = []
                for d in field:
                    ps  = d["ps"]
                    pv  = d["pv"]
                    if stat not in ps.stat_distributions:
                        continue
                    dist = ps.stat_distributions[stat]
                    proj_val = float(pv.get(stat, 0))
                    if proj_val < 0.02:
                        continue
                    row = {
                        "Player": d["nm"],
                        "Team":   team_name(d["tid"]),
                        "Pos":    d["pos"],
                        "Proj":   round(proj_val, 2),
                    }
                    for lvl in levels:
                        ml_m = pricing.price_prop(ps, stat, line=lvl - 0.5)
                        p_hit = float(np.mean(dist >= lvl))
                        row[f"{lvl}+ odds"] = ml_m.over_odds
                        row[f"{lvl}+ P"]    = f"{p_hit:.1%}"
                    mile_rows.append(row)
                if mile_rows:
                    df_m = pd.DataFrame(mile_rows).sort_values("Proj", ascending=False).reset_index(drop=True)
                    st.dataframe(df_m, width="stretch", hide_index=True)
                else:
                    st.caption(f"No {lbl} milestone data.")

    st.markdown("---")
    st.markdown(
        f'<span class="note-text">'
        f'Margin: {new_hold_pct*100:.1f}% · 20,000 sims · '
        f'Switch to Expander view for distributions, alt lines, and market comparison'
        f'</span>',
        unsafe_allow_html=True,
    )
    st.stop()


# ===========================================================================
# EXPANDER VIEW (original, per-player)
# ===========================================================================

for ps in sims_filtered:
    pid  = ps.player_id
    pm   = markets.get(pid, {})
    proj = all_projs.get(pid)
    if proj is None:
        continue

    pv  = pm.get("proj_values", {})
    nm  = proj.full_name or pid
    pos = proj.position
    tid = proj.team_id

    # -- Get primary stat line for collapsed summary row ------------------
    if pos == "G":
        pri_stat, pri_proj = "saves", proj.proj_saves
    elif pos == "FO":
        pri_stat, pri_proj = "faceoff_wins", proj.proj_faceoff_wins
    else:
        pri_stat, pri_proj = "points", proj.proj_points

    pm_data = markets.get(pid, {})
    pri_market = pm_data.get("markets", {}).get(pri_stat, {})
    if pri_market:
        line_val = pri_market.get("line")
        line_str = f"{line_val:.1f}" if isinstance(line_val, (int, float)) else "?"
        over_str = pri_market.get("over_odds", "--")
        under_str = pri_market.get("under_odds", "--")
        expander_label = (
            f"{nm}  ·  {pos}  ·  {team_name(tid)}  |  "
            f"Proj: {pri_proj:.2f}  |  Line: {line_str}  |  "
            f"O {over_str} / U {under_str}"
        )
    else:
        expander_label = f"{nm}  ·  {pos}  ·  {team_name(tid)}  |  Proj: {pri_proj:.2f}"

    with st.expander(expander_label, expanded=False):
        col_info, col_dist = st.columns([1, 2])

        with col_info:
            st.markdown(f"**Pos:** {pos_badge(pos)}", unsafe_allow_html=True)
            st.markdown(f"**Team:** {team_name(tid)}")
            if pos == "G":
                st.markdown(f"Proj Saves: **{proj.proj_saves:.2f}**")
                st.markdown(f"Save%: **{proj.proj_save_pct:.3f}**")
            elif pos == "FO":
                st.markdown(f"FO Wins: **{proj.proj_faceoff_wins:.2f}**")
                st.markdown(f"FO%: **{proj.proj_faceoff_pct:.3f}**")
            else:
                st.markdown(f"Goals: **{proj.proj_goals:.3f}**")
                st.markdown(f"Assists: **{proj.proj_assists:.3f}**")
                st.markdown(f"Points: **{proj.proj_points:.3f}**")
                st.markdown(f"Shots: **{proj.proj_shots:.2f}**  SOG: **{proj.proj_sog:.2f}**")
                if proj.proj_2pt_goals > 0.02:
                    rate = proj.proj_2pt_goals / max(proj.proj_goals, 0.01)
                    st.markdown(f"2PT Rate: **{rate:.1%}**")
                st.markdown(f"Zero-score prob: **{proj.zero_prob_goals:.1%}**")

        with col_dist:
            pri = "saves" if pos == "G" else ("faceoff_wins" if pos == "FO" else "points")
            if pri in ps.stat_distributions:
                dist = ps.stat_distributions[pri]
                fig  = go.Figure(go.Histogram(x=dist, nbinsx=20,
                                              marker_color=team_color(tid), opacity=0.75))
                pv_val = pv.get(pri, 0)
                fig.add_vline(x=pv_val, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Proj: {pv_val:.2f}")
                fig.update_layout(
                    height=170, margin=dict(l=0,r=0,t=4,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"), showlegend=False,
                    xaxis_title=STAT_LABELS.get(pri, pri), yaxis_title="",
                )
                st.plotly_chart(fig, width="stretch")

        # -- Model line table ----------------------------------------------
        # Proj column uses the deterministic PlayerProjection values (same as
        # the left panel) so the numbers are identical. Pricing still uses the
        # full sim distributions — only the display label is standardised.
        det_proj_map = {
            "goals":         proj.proj_goals,
            "assists":       proj.proj_assists,
            "points":        proj.proj_points,
            "shots":         proj.proj_shots,
            "shots_on_goal": proj.proj_sog,
            "two_pt_goals":  proj.proj_2pt_goals,
            "one_pt_goals":  proj.proj_1pt_goals,
            "saves":         proj.proj_saves,
            "faceoff_wins":  proj.proj_faceoff_wins,
            "ground_balls":  proj.proj_ground_balls,
        }
        stat_list = GOALIE_STATS if pos == "G" else (FO_STATS if pos == "FO" else FIELD_STATS)
        rows = []
        for stat in stat_list:
            if stat not in ps.stat_distributions:
                continue
            dist = ps.stat_distributions[stat]
            custom_line = None
            if ov_player and ov_player.lower() in nm.lower() and ov_stat == stat:
                custom_line = ov_line
            ml  = pricing.price_prop(ps, stat, line=custom_line)
            pct = float(np.percentile(dist, 75)) - float(np.percentile(dist, 25))
            rows.append({
                "Stat":     STAT_LABELS.get(stat, stat),
                "Proj":     f"{det_proj_map.get(stat, pv.get(stat, 0)):.1f}",
                "Line":     f"{ml.line:.1f}",
                "P(Over)":  f"{ml.fair_over_prob:.3f}",
                "Over":     ml.over_odds,
                "P(Under)": f"{ml.fair_under_prob:.3f}",
                "Under":    ml.under_odds,
                "IQR":      f"{pct:.2f}",
                "P10":      f"{np.percentile(dist,10):.1f}",
                "Median":   f"{np.percentile(dist,50):.1f}",
                "P90":      f"{np.percentile(dist,90):.1f}",
            })
        if rows:
            st.markdown("**Model Lines**")
            # Add market comparison row if this player/stat matches the sidebar input
            if (mkt_player and mkt_player.lower() in nm.lower()):
                for row in rows:
                    if row["Stat"] == STAT_LABELS.get(mkt_stat, mkt_stat):
                        dist_key = mkt_stat
                        if dist_key in ps.stat_distributions:
                            dist_mkt = ps.stat_distributions[dist_key]
                            fair_p = float(np.mean(dist_mkt > mkt_line))
                            try:
                                mo = int(mkt_over_odds)
                                mkt_implied = (-mo / (-mo + 100)) if mo < 0 else (100 / (mo + 100))
                            except Exception:
                                mkt_implied = 0.5
                            edge = fair_p - mkt_implied
                            edge_str = f"+{edge:.1%}" if edge > 0 else f"{edge:.1%}"
                            edge_color = "green" if edge > 0.02 else ("red" if edge < -0.02 else "gray")
                            st.markdown(
                                f'<div style="background:rgba(8,145,178,.12);border-left:3px solid #0891b2;'
                                f'padding:4px 10px;border-radius:0 4px 4px 0;margin:4px 0;">'
                                f'<span style="font-size:.80rem;color:#7dd3fc;">Market comparison -- '
                                f'{STAT_LABELS.get(mkt_stat, mkt_stat)} {mkt_line:.1f}: '
                                f'Fair P(Over)={fair_p:.3f} | Market implied={mkt_implied:.3f} | '
                                f'<b style="color:{edge_color};">Edge: {edge_str}</b></span></div>',
                                unsafe_allow_html=True,
                            )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # -- Alternate line pricing -----------------------------------------
        if show_alt:
            for stat in stat_list:
                if stat not in ps.stat_distributions:
                    continue
                dist = ps.stat_distributions[stat]
                proj_v = pv.get(stat, 0)
                st.markdown(f"**Alternate Lines -- {STAT_LABELS.get(stat, stat)}**")

                main_ml = pricing.price_prop(ps, stat)
                width = _alt_width(stat)
                lo = main_ml.line - width
                hi = main_ml.line + width
                alt_lines = _half_only_lines(lo, hi)
                alt_rows = []
                for al in alt_lines:
                    ml_a = pricing.price_prop(ps, stat, line=al)
                    alt_rows.append({
                        "Line": f"{al:.1f}",
                        "P(Over)":  f"{ml_a.fair_over_prob:.3f}",
                        "Over Odds":  ml_a.over_odds,
                        "P(Under)": f"{ml_a.fair_under_prob:.3f}",
                        "Under Odds": ml_a.under_odds,
                        "Main Line": f"{main_ml.line:.1f}",
                        "Model Proj": f"{proj_v:.3f}",
                    })
                if alt_rows:
                    st.dataframe(pd.DataFrame(alt_rows), width="stretch", hide_index=True)

        # -- Milestone props ------------------------------------------------
        if show_miles:
            mile_stats = (
                ["saves"] if pos == "G"
                else ["goals", "assists", "points"] if pos not in ("FO",)
                else []
            )
            any_mile = False
            for stat in mile_stats:
                levels = MILE_DEFS.get(stat)
                if not levels or stat not in ps.stat_distributions:
                    continue
                dist = ps.stat_distributions[stat]
                m_rows = []
                for lvl in levels:
                    ml_m = pricing.price_prop(ps, stat, line=lvl - 0.5)
                    m_rows.append({
                        "Milestone": f"{STAT_LABELS.get(stat,stat)} {lvl}+",
                        "P(Hit)":    f"{float(np.mean(dist >= lvl)):.3f}",
                        "Yes odds":  ml_m.over_odds,
                        "No odds":   ml_m.under_odds,
                    })
                if m_rows:
                    if not any_mile:
                        st.markdown("**Milestones**")
                        any_mile = True
                    st.dataframe(pd.DataFrame(m_rows), width="stretch", hide_index=True)

            # SOG milestones separately
            if pos not in ("G", "FO") and "shots_on_goal" in ps.stat_distributions:
                sog_levels = MILE_DEFS.get("shots_on_goal", [])
                sog_dist   = ps.stat_distributions["shots_on_goal"]
                sog_rows   = []
                for lvl in sog_levels:
                    ml_m = pricing.price_prop(ps, "shots_on_goal", line=lvl - 0.5)
                    sog_rows.append({
                        "Milestone": f"SOG {lvl}+",
                        "P(Hit)":    f"{float(np.mean(sog_dist >= lvl)):.3f}",
                        "Yes odds":  ml_m.over_odds,
                        "No odds":   ml_m.under_odds,
                    })
                if sog_rows:
                    st.dataframe(pd.DataFrame(sog_rows), width="stretch", hide_index=True)

st.markdown("---")
st.markdown(
    f'<span class="note-text">'
    f'Margin: {new_hold_pct*100:.1f}% · 20,000 sims · '
    f'Enable "Alternate line pricing" in sidebar for full line grids'
    f'</span>',
    unsafe_allow_html=True,
)
