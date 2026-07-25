"""
PLL Live Trading — standalone Streamlit app.

Runs SEPARATELY from the main PLL Projections app (so heavy live polling never
slows the projections UI) but stays CONNECTED to it through the shared
``data/session_autosave.json`` file: the live app inherits the game selection,
per-market margins (holds), team-rating overrides and depth-chart overrides the
user set in the projections app, then re-projects and re-simulates in-play.

Pipeline (all in live/):
    live_schedule  -> auto-detect the live game (+ manual override)
    live_feed      -> poll play-by-play every few seconds
    live_state     -> reconstruct banked (certain) per-player stats
    live_model     -> re-simulate ONLY the time remaining, add banked
    live_pricing   -> fair prices + EV edge vs manually-entered book lines

Run:  streamlit run live/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# -- path bootstrap: repo root + this dir on sys.path --------------------------
_LIVE_DIR = Path(__file__).resolve().parent
_ROOT = _LIVE_DIR.parent
for _p in (str(_ROOT), str(_LIVE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_feed import LiveFeed, fetch_state                      # noqa: E402
from live_state import reconstruct                               # noqa: E402
from live_model import LiveModel                                 # noqa: E402
from live_pricing import quote_edge, TRADED_STATS                # noqa: E402
from live_schedule import list_games, detect_live, find_game     # noqa: E402
from projection_engine_v3 import ProjectionEngine, DEFAULT_HOLDS, DEFAULT_GLOBAL_HOLD  # noqa: E402

_AUTOSAVE_PATH = _ROOT / "data" / "session_autosave.json"
_DB_PATH = os.getenv("PLL_DB_PATH",
                     str(_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"))
DEFAULT_YEAR = 2026
TRADE_FOCUS = ("points", "goals", "assists")  # the markets the user trades

TEAM_NAMES = {
    "ATL": "Atlas", "OUT": "Outlaws", "CAN": "Cannons", "RED": "Redwoods",
    "WAT": "Waterdogs", "WHP": "Whipsnakes", "CHA": "Chaos", "ARC": "Archers",
}


# -- engine (cached; its own resource so the projections app is untouched) -----
@st.cache_resource(show_spinner="Loading projection engine…")
def get_engine() -> ProjectionEngine:
    eng = ProjectionEngine(db_path=_DB_PATH)
    eng.load()
    eng.fit(run_backtest=False)
    return eng


@st.cache_resource(show_spinner="Running pregame simulation…")
def get_pregame(_engine: ProjectionEngine, home_id: str, away_id: str,
                game_date: str | None, cfg_token: str):
    """Pregame ProjectionResult for this matchup, inheriting the projections
    app's overrides/holds. Cached on (matchup, config token) so it re-runs only
    when the game or the inherited config changes — NOT on every live poll."""
    cfg = _load_autosave()
    p_ov, active, starters = _overrides_from_depth_charts(cfg)
    return _engine.project(
        home_team_id=home_id, away_team_id=away_id, game_date=game_date,
        player_overrides=p_ov or None,
        active_players=active or None,
        starter_goalies=starters or None,
        team_rating_overrides=cfg.get("team_rating_overrides") or None,
    )


def _load_autosave() -> dict:
    """Read the projections app's shared state (holds, overrides, selected game).
    Best-effort: an empty dict just means the live app uses model defaults."""
    try:
        p = json.loads(_AUTOSAVE_PATH.read_text(encoding="utf-8"))
        if p.get("version") != 1:
            return {}
        return p
    except Exception:
        return {}


def _overrides_from_depth_charts(cfg: dict):
    """Reconstruct the engine.project() override dicts from the shared depth_charts.

    The projections app stores raw depth_charts in the autosave file and builds
    the project() args (player_overrides/active_players/starter_goalies) at call
    time via pages/_engine_state.py. We replicate that mapping here so the live
    app applies the SAME per-player usage/rating/starter overrides the user set.
    """
    depth = cfg.get("depth_charts") or {}
    player_overrides: dict = {}
    active_players: dict = {}
    starter_goalies: dict = {}
    for team_id, team_dc in depth.items():
        for pid, settings in (team_dc or {}).items():
            entry: dict = {}
            if "active" in settings:
                entry["active"] = settings["active"]
                active_players[pid] = settings["active"]
            if "usage_multiplier" in settings:
                um = float(settings["usage_multiplier"])
                entry["usage_multiplier"] = um
                if um == 0.0:
                    entry["active"] = False
                    active_players[pid] = False
            if "is_starter" in settings:
                entry["is_starter"] = settings["is_starter"]
                if settings["is_starter"]:
                    starter_goalies.setdefault(str(team_id), pid)
            override_keys: list = []
            if "position_override" in settings:
                entry["pos_norm"] = settings["position_override"]
                override_keys.append("pos_norm")
            for rk, rv in (settings.get("rating_overrides") or {}).items():
                entry[rk] = rv
                override_keys.append(rk)
            if override_keys:
                entry["_override_keys"] = override_keys
            if entry:
                player_overrides[pid] = entry
    return player_overrides, active_players, starter_goalies


def _cfg_token(cfg: dict) -> str:
    """Stable hash of the inherited config so the pregame cache invalidates when
    the user changes holds/overrides in the projections app."""
    import hashlib
    keys = ("depth_charts", "team_rating_overrides", "hold_pct", "hold_by_stat")
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def _holds_from_cfg(cfg: dict):
    holds = dict(DEFAULT_HOLDS)
    stored = cfg.get("hold_by_stat") or {}
    if isinstance(stored, dict):
        holds.update({k: float(v) for k, v in stored.items()})
    glob = float(cfg.get("hold_pct", DEFAULT_GLOBAL_HOLD))
    return holds, glob


def _tname(tid: str) -> str:
    return TEAM_NAMES.get(str(tid).upper(), str(tid))


def _stub_game(slug: str, yr: int):
    """Fallback GameInfo when a manual slug isn't in the schedule list."""
    from live_schedule import GameInfo
    return GameInfo(slug=slug, game_number=None, home_team_id="", away_team_id="",
                    home_name="?", away_name="?", status=1, home_score=0,
                    visitor_score=0, period=0, clock_minutes=0, clock_seconds=0,
                    start_time=None, year=yr)


def _infer_home(state):
    for e in state.events:
        if e.get("eventType") == "goal" and e.get("homeScore", 0):
            return e.get("teamId")
    return ""


def _infer_away(state, home_id):
    tids = [e.get("teamId") for e in state.events if e.get("teamId")]
    for t in dict.fromkeys(tids):
        if t and t != home_id:
            return t
    return ""


# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="PLL Live Trading", page_icon="🥍", layout="wide")
st.title("🥍 PLL Live Trading")
st.caption("Live rest-of-game re-simulation. Inherits your pregame setup from the "
           "Projections app (margins, overrides) via the shared session file.")

# -- sidebar: game selection ---------------------------------------------------
with st.sidebar:
    st.header("Game")
    year = st.number_input("Season", min_value=2019, max_value=2100,
                           value=DEFAULT_YEAR, step=1)
    refresh_secs = st.slider("Auto-refresh (seconds)", 5, 30, 8)

    if st.button("🔄 Rescan for live games", use_container_width=True):
        st.cache_data.clear()

    @st.cache_data(ttl=20, show_spinner=False)
    def _games(yr: int):
        return list_games(int(yr))

    games = _games(year)
    live_games = [g for g in games if g.is_live]

    if live_games:
        st.success(f"{len(live_games)} live game(s) detected.")
    else:
        st.info("No live game detected right now.")

    # Auto-detect + confirm, with manual override (per the chosen design).
    mode = st.radio("Game source",
                    ["Auto-detected live", "Pick from schedule", "Manual slug"],
                    index=0 if live_games else 1)

    selected = None
    if mode == "Auto-detected live" and live_games:
        opts = {g.label: g for g in live_games}
        pick = st.selectbox("Confirm live game", list(opts.keys()))
        selected = opts[pick]
    elif mode == "Pick from schedule":
        # live first, then upcoming, then finals
        ordered = ([g for g in games if g.is_live]
                   + [g for g in games if g.is_scheduled]
                   + [g for g in games if g.is_final])
        opts = {g.label: g for g in ordered}
        if opts:
            pick = st.selectbox("Game", list(opts.keys()))
            selected = opts[pick]
    else:  # Manual slug
        slug = st.text_input("Play-by-play slug", value="2026-ev-36")
        if slug:
            selected = find_game(int(year), slug) or _stub_game(slug, int(year))

    st.divider()
    pace_weight = st.slider(
        "Pace weight", 0.0, 1.0, 0.5, 0.05,
        help="How much the rest-of-game projection leans on the pace observed so "
             "far vs the pregame model. Blend grows with game elapsed; 0 = ignore "
             "in-game pace, 1 = full trust once the game is late.")

if selected is None:
    st.warning("No game selected. Detect a live game or enter a slug in the sidebar.")
    st.stop()


# -- header: matchup + clock ---------------------------------------------------
st.subheader(f"{_tname(selected.away_team_id)} @ {_tname(selected.home_team_id)}"
             f"  ·  {selected.slug}")

engine = get_engine()
cfg = _load_autosave()
holds, glob_hold = _holds_from_cfg(cfg)
# apply inherited holds to the engine's pricing so live prices match the app
engine.pricing.hold_by_stat = dict(holds)
engine.pricing.hold_pct = glob_hold


@st.fragment(run_every=refresh_secs)
def live_board():
    """Polls the feed and re-renders the board every `refresh_secs`. Isolated in a
    fragment so only this section reruns on the timer, not the whole app."""
    state = LiveFeed(selected.slug).poll()
    if not state.ok:
        st.error(f"Feed error: {state.error}")
        return

    banked = reconstruct(state.events)
    frac_rem = state.fraction_remaining

    # clock / score row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{state.away_score} – {state.home_score}",
              help=f"{_tname(selected.away_team_id)} – {_tname(selected.home_team_id)}")
    c2.metric("Period / Clock", f"P{state.period}  {state.clock_minutes:02d}:{state.clock_seconds:02d}")
    c3.metric("Game remaining", f"{frac_rem:.0%}")
    c4.metric("Events", state.n_events)

    if state.is_final:
        st.info("Game reads FINAL — projections below are the settled banked totals.")

    home_id = selected.home_team_id or _infer_home(state)
    away_id = selected.away_team_id or _infer_away(state, home_id)
    if not home_id or not away_id:
        st.warning("Could not determine team IDs for this game; use 'Pick from "
                   "schedule' so home/away are known.")
        return

    # pregame projection (cached) + live re-sim (every poll)
    pregame = get_pregame(engine, home_id, away_id, selected.game_date,
                          _cfg_token(cfg))
    live = LiveModel(engine).resimulate(
        pregame, banked.by_player, frac_rem, pace_weight=pace_weight,
        home_team_id=home_id, away_team_id=away_id,
        team_of=banked.team_of, events=state.events)

    # index sims by player for the props table
    sims = {str(ps.player_id): ps for ps in live.all_sims()}

    st.markdown("### Live player projections & edge")
    stat = st.radio("Market", list(TRADE_FOCUS), horizontal=True, key="mkt")

    rows = []
    for pid, ps in sims.items():
        if stat not in ps.stat_distributions:
            continue
        dist = ps.stat_distributions[stat]
        proj = float(np.mean(dist))
        if proj < 0.3 and float(np.max(dist)) < 1:
            continue
        line = ps.prop_lines.get(stat, np.floor(np.median(dist)) + 0.5)
        bk = banked.get(pid, stat if stat != "points" else "points")
        rows.append({"pid": pid, "name": ps.full_name, "line": float(line),
                     "proj": proj, "banked": bk, "dist": dist})
    rows.sort(key=lambda r: r["proj"], reverse=True)

    st.caption("Enter the book's American odds to see EV per $1 staked. Blank = "
               "just show the model's fair line.")

    # header
    h = st.columns([3, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.4])
    for col, lbl in zip(h, ["Player", "Line", "Proj", "Bank", "P(over)",
                            "Fair O", "Book O", "Book U", "Edge"]):
        col.markdown(f"**{lbl}**")

    for r in rows[:24]:
        q = quote_edge(r["pid"], r["name"], stat, r["dist"], r["line"],
                       banked=r["banked"])
        cols = st.columns([3, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.4])
        cols[0].write(r["name"])
        cols[1].write(f'{r["line"]:.1f}')
        cols[2].write(f'{r["proj"]:.2f}')
        cols[3].write(f'{r["banked"]:.0f}')
        cols[4].write("LOCK" if q.is_settled else f'{q.model_prob_over:.0%}')
        cols[5].write(q.model_fair_over)
        bo = cols[6].text_input("bo", key=f"bo_{stat}_{r['pid']}",
                                label_visibility="collapsed", placeholder="-110")
        bu = cols[7].text_input("bu", key=f"bu_{stat}_{r['pid']}",
                                label_visibility="collapsed", placeholder="+120")
        # recompute EV if the user entered odds
        eq = quote_edge(r["pid"], r["name"], stat, r["dist"], r["line"],
                        banked=r["banked"],
                        book_over_odds=bo.strip() or None,
                        book_under_odds=bu.strip() or None)
        edge_txt = ""
        if eq.best_ev is not None:
            side = eq.best_side or "—"
            edge_txt = f"{side} {eq.best_ev:+.1%}" if side != "—" else "no +EV"
        cols[8].write(edge_txt)

    st.caption(f"Last poll: {state.n_events} events · pace_weight={pace_weight:.2f} · "
               f"holds inherited from projections app")


live_board()
