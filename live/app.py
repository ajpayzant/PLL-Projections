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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from live_pricing import (                                       # noqa: E402
    prob_over, prob_to_american, american_to_profit, TRADED_STATS)
from live_schedule import list_games, detect_live, find_game     # noqa: E402
from projection_engine_v3 import ProjectionEngine, DEFAULT_HOLDS, DEFAULT_GLOBAL_HOLD  # noqa: E402

_AUTOSAVE_PATH = _ROOT / "data" / "session_autosave.json"
_DB_PATH = os.getenv("PLL_DB_PATH",
                     str(_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"))
DEFAULT_YEAR = 2026
TRADE_FOCUS = ("points", "goals", "assists")  # the markets the user trades
LIVE_SIMS = 8000  # live re-sim count: enough for prop probs, keeps the board snappy


# -- Bootstrap DB from parquet if missing (mirrors pages/_engine_state.py) -----
# The .duckdb warehouse is gitignored (too large), so on a fresh Streamlit Cloud
# deploy or clone it doesn't exist. scripts/bootstrap_db.py rebuilds it from the
# committed parquet files in ~10s. The main projections app does this at startup;
# the live app must too, or the engine constructor raises FileNotFoundError.
def _db_is_valid() -> bool:
    """True only if the DB file exists AND its clean schema is populated."""
    p = Path(_DB_PATH)
    if not p.exists() or p.stat().st_size < 4096:
        return False
    con = None
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        return con.execute("SELECT COUNT(*) FROM clean.team_game_stats").fetchone()[0] > 0
    except Exception:
        return False
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _ensure_db() -> None:
    if _db_is_valid():
        return
    bootstrap = _ROOT / "scripts" / "bootstrap_db.py"
    if not bootstrap.exists():
        st.error("Database not found and scripts/bootstrap_db.py is missing.")
        st.stop()
    with st.spinner("Building database from data files — first load only, ~10 seconds…"):
        try:
            result = subprocess.run(
                [sys.executable, str(bootstrap), "--force"],
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            st.error("Database build timed out — press R to rerun or reboot the app.")
            st.stop()
        except Exception as e:
            st.error(f"Database build could not start: {e}")
            st.stop()
    if result.returncode != 0:
        st.error(
            f"Database bootstrap failed.\n\n```\n{result.stderr[-2000:]}\n```\n\n"
            "Run the GitHub Action (Update PLL Data Warehouse) to populate data/.")
        st.stop()
    if not _db_is_valid():
        st.error("Database was rebuilt but still isn't valid — re-run the data "
                 "GitHub Action, then reboot the app.")
        st.stop()

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

_ensure_db()  # rebuild the warehouse from parquet on a fresh deploy, before engine load
engine = get_engine()
cfg = _load_autosave()
holds, glob_hold = _holds_from_cfg(cfg)
# apply inherited holds to the engine's pricing so live prices match the app
engine.pricing.hold_by_stat = dict(holds)
engine.pricing.hold_pct = glob_hold


def _parse_odds(txt):
    """Parse a typed American-odds string ('-110', '+120', '110') -> float or None."""
    if txt is None:
        return None
    s = str(txt).strip().replace("+", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if abs(v) >= 100 else None  # ignore obviously-bad entries


def _ev(prob: float, odds) -> float | None:
    """EV per $1 staked at American `odds` given model win prob."""
    o = _parse_odds(odds)
    if o is None:
        return None
    return prob * american_to_profit(o) - (1.0 - prob)


@st.cache_data(ttl=600, show_spinner=False, max_entries=8)
def build_board(_engine, slug: str, home_id: str, away_id: str,
                game_date: str | None, cfg_token: str, pace_weight: float,
                tick: int) -> dict:
    """Poll the feed, reconstruct banked stats, run the cached pregame projection
    and the live rest-of-game re-sim, and return a fully-computed, picklable board.

    Cached on a per-refresh `tick` so the EXPENSIVE work (poll + Monte Carlo re-sim)
    runs at most once per refresh window. Every keystroke in the book-odds boxes
    reruns the fragment but hits this cache instead of re-simulating — that's what
    keeps typing responsive. Returns plain floats/strings only (no numpy/objects)."""
    state = LiveFeed(slug).poll()
    if not state.ok:
        return {"ok": False, "error": state.error}

    banked = reconstruct(state.events)
    frac_rem = state.fraction_remaining

    pregame = get_pregame(_engine, home_id, away_id, game_date, cfg_token)
    pre_by_pid = {str(ps.player_id): ps
                  for ps in (pregame.home_player_sims + pregame.away_player_sims)}

    live = LiveModel(_engine, n_sims=LIVE_SIMS).resimulate(
        pregame, banked.by_player, frac_rem, pace_weight=pace_weight,
        home_team_id=home_id, away_team_id=away_id,
        team_of=banked.team_of, events=state.events)

    stats: dict[str, list] = {}
    for stat in TRADE_FOCUS:
        rows = []
        for ps in live.all_sims():
            if stat not in ps.stat_distributions:
                continue
            pid = str(ps.player_id)
            dist = ps.stat_distributions[stat]
            live_proj = float(np.mean(dist))
            bk = banked.get(pid, stat)
            if live_proj < 0.3 and bk <= 0:
                continue
            # One market line per player-stat (the live balanced line) so pregame
            # and live odds are quoted at the SAME line and are directly comparable.
            line = float(ps.prop_lines.get(stat, np.floor(np.median(dist)) + 0.5))
            settled = bool(dist.size and float(np.std(dist)) < 1e-9)
            p_over = prob_over(dist, line)
            live_odds = "LOCKED" if settled else prob_to_american(p_over)

            pre_ps = pre_by_pid.get(pid)
            if pre_ps is not None and stat in pre_ps.stat_distributions:
                pre_dist = pre_ps.stat_distributions[stat]
                pre_proj = float(np.mean(pre_dist))
                pre_odds = prob_to_american(prob_over(pre_dist, line))
            else:
                pre_proj, pre_odds = None, "—"  # call-up: no pregame projection

            rows.append({
                "pid": pid, "Player": ps.full_name, "Line": line,
                "Now": float(bk), "PreProj": pre_proj, "PreOddsO": pre_odds,
                "LiveProj": round(live_proj, 2), "LiveOddsO": live_odds,
                "Pover": round(p_over, 4), "settled": settled,
            })
        rows.sort(key=lambda r: r["LiveProj"], reverse=True)
        stats[stat] = rows

    return {
        "ok": True,
        "meta": {
            "away_score": state.away_score, "home_score": state.home_score,
            "period": state.period, "cm": state.clock_minutes, "cs": state.clock_seconds,
            "frac_rem": frac_rem, "n_events": state.n_events, "is_final": state.is_final,
        },
        "stats": stats,
    }


@st.fragment(run_every=refresh_secs)
def live_board():
    """Renders the board every `refresh_secs`. The heavy compute is cached in
    build_board(); this function only formats + prices typed odds (cheap)."""
    import time
    home_id = selected.home_team_id
    away_id = selected.away_team_id
    if not home_id or not away_id:
        # manual-slug path with unknown teams: peek one poll to infer
        peek = LiveFeed(selected.slug).poll()
        home_id = home_id or _infer_home(peek)
        away_id = away_id or _infer_away(peek, home_id)
    if not home_id or not away_id:
        st.warning("Could not determine team IDs. Use 'Pick from schedule' so "
                   "home/away are known.")
        return

    # per-refresh cache tick: the same within a refresh window, so keystrokes
    # reuse the cached sim instead of recomputing it.
    tick = int(time.time() // refresh_secs)
    board = build_board(engine, selected.slug, home_id, away_id,
                        selected.game_date, _cfg_token(cfg), pace_weight, tick)
    if not board.get("ok"):
        st.error(f"Feed error: {board.get('error')}")
        return

    m = board["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{m['away_score']} – {m['home_score']}",
              help=f"{_tname(away_id)} (away) – {_tname(home_id)} (home)")
    c2.metric("Period / Clock", f"P{m['period']}  {m['cm']:02d}:{m['cs']:02d}")
    c3.metric("Game remaining", f"{m['frac_rem']:.0%}")
    c4.metric("Events", m["n_events"])
    if m["is_final"]:
        st.info("Game reads FINAL — projections below are the settled banked totals.")

    st.markdown("### Live player board")
    stat = st.radio("Market", list(TRADE_FOCUS), horizontal=True, key="mkt")
    st.caption("**Now** = stats already banked · **Pre** = pregame projection & fair "
               "over-odds · **Live** = current full-game projection & fair over-odds "
               "(all at the same line). Type the book's odds into **Book O/U** to see "
               "**Edge** (EV per $1). Editing odds does NOT re-run the simulation.")

    rows = board["stats"].get(stat, [])
    if not rows:
        st.info("No tradeable players for this market yet.")
        return

    # Persist typed odds across refreshes, keyed by stat+player, in session_state.
    ostore = st.session_state.setdefault("book_odds", {})

    df = pd.DataFrame(rows).set_index("pid")
    df["Book O"] = [ostore.get(f"{stat}:{pid}:o", "") for pid in df.index]
    df["Book U"] = [ostore.get(f"{stat}:{pid}:u", "") for pid in df.index]
    show = df[["Player", "Line", "Now", "PreProj", "PreOddsO",
               "LiveProj", "LiveOddsO", "Pover", "Book O", "Book U"]].copy()
    show["Pover"] = (show["Pover"] * 100).round(0)

    edited = st.data_editor(
        show, hide_index=True, use_container_width=True, key=f"editor_{stat}",
        column_config={
            "Player": st.column_config.TextColumn("Player", disabled=True, width="medium"),
            "Line": st.column_config.NumberColumn("Line", disabled=True, format="%.1f"),
            "Now": st.column_config.NumberColumn("Now", disabled=True, format="%.0f",
                                                 help="Banked so far (certain)"),
            "PreProj": st.column_config.NumberColumn("Pre Proj", disabled=True, format="%.2f",
                                                     help="Pregame projection"),
            "PreOddsO": st.column_config.TextColumn("Pre O", disabled=True,
                                                    help="Pregame fair over-odds at this line"),
            "LiveProj": st.column_config.NumberColumn("Live Proj", disabled=True, format="%.2f",
                                                      help="Current full-game projection"),
            "LiveOddsO": st.column_config.TextColumn("Live O", disabled=True,
                                                     help="Current fair over-odds at this line"),
            "Pover": st.column_config.NumberColumn("P(o)%", disabled=True, format="%.0f",
                                                   help="Model prob of going over the line"),
            "Book O": st.column_config.TextColumn("Book O", help="Book's over odds, e.g. -110"),
            "Book U": st.column_config.TextColumn("Book U", help="Book's under odds, e.g. +120"),
        },
    )

    # Persist edits + compute edge (cheap: arithmetic on the cached prob).
    edge_rows = []
    for pid, r in edited.iterrows():
        bo, bu = str(r["Book O"] or ""), str(r["Book U"] or "")
        ostore[f"{stat}:{pid}:o"] = bo
        ostore[f"{stat}:{pid}:u"] = bu
        p_over = float(df.loc[pid, "Pover"])
        if bool(df.loc[pid, "settled"]):
            continue
        ev_o = _ev(p_over, bo)
        ev_u = _ev(1.0 - p_over, bu)
        cands = [(s, e) for s, e in (("Over", ev_o), ("Under", ev_u)) if e is not None]
        if not cands:
            continue
        side, ev = max(cands, key=lambda x: x[1])
        edge_rows.append({"Player": r["Player"], "Line": float(df.loc[pid, "Line"]),
                          "Bet": side, "Book": bo if side == "Over" else bu,
                          "Edge %": round(ev * 100, 1)})

    if edge_rows:
        st.markdown("#### Edge (from your entered odds)")
        edf = pd.DataFrame(edge_rows).sort_values("Edge %", ascending=False)
        st.dataframe(
            edf, hide_index=True, use_container_width=True,
            column_config={"Edge %": st.column_config.NumberColumn(
                "Edge %", format="%.1f",
                help="EV per $1 staked. Positive = +EV bet at your odds.")},
        )
        pos = edf[edf["Edge %"] > 0]
        if not pos.empty:
            st.success(f"{len(pos)} +EV opportunit{'y' if len(pos)==1 else 'ies'} "
                       f"— best: {pos.iloc[0]['Player']} {pos.iloc[0]['Bet']} "
                       f"{pos.iloc[0]['Line']:.1f} ({pos.iloc[0]['Edge %']:+.1f}%)")

    st.caption(f"Last poll: {m['n_events']} events · pace_weight={pace_weight:.2f} · "
               f"{LIVE_SIMS:,} live sims · holds inherited from projections app")


live_board()
