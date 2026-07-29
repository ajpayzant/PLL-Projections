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
from live_pricing import TRADED_STATS                            # noqa: E402
from live_schedule import list_games, detect_live, find_game     # noqa: E402
from projection_engine_v3 import ProjectionEngine, DEFAULT_HOLDS, DEFAULT_GLOBAL_HOLD  # noqa: E402

_AUTOSAVE_PATH = _ROOT / "data" / "session_autosave.json"
_DB_PATH = os.getenv("PLL_DB_PATH",
                     str(_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"))
DEFAULT_YEAR = 2026
TRADE_FOCUS = ("points", "goals", "assists")  # the markets the user trades
LIVE_SIMS = 8000  # live re-sim count: enough for prop probs, keeps the board snappy

# X+ thresholds to post per market ("N+" == over N-0.5). Points ladders higher
# than goals/assists because point totals run higher.
LADDER = {
    "points":  (1, 2, 3, 4, 5, 6),
    "goals":   (1, 2, 3, 4),
    "assists": (1, 2, 3, 4),
}


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
    """Return the pregame setup (holds, overrides, selected game) the live board
    should inherit from the Projections app.

    Resolution order:
      1. A setup file the user UPLOADED in the sidebar (stored in session_state).
         This is the reliable path when the two apps are SEPARATE deployments —
         they don't share a filesystem, so the local autosave below is never
         written by the projections container.
      2. The local ``data/session_autosave.json`` — only present when both apps
         run from the same folder on one machine (local dev).
    Best-effort: an empty dict just means the live app uses model defaults."""
    uploaded = st.session_state.get("_uploaded_cfg")
    if isinstance(uploaded, dict) and uploaded.get("version") == 1:
        return uploaded
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
st.caption("Live rest-of-game re-simulation. Upload the setup file you exported "
           "from the Projections app so the pregame lines here match your "
           "finalized projections, margins and overrides exactly.")

# -- sidebar: pregame setup file -----------------------------------------------
with st.sidebar:
    st.header("Pregame setup")
    st.caption("Export a setup file from the Projections app "
               "(**💾 Save for Live Trading**) and upload it here. The live board "
               "reproduces your finalized pregame lines from it, then updates as "
               "the game develops.")
    _setup = st.file_uploader(
        "Setup file (.json)", type="json", key="live_setup_file",
        help="The JSON exported from the Projections app: game selection, depth "
             "chart, rating overrides and per-market margins.")
    if _setup is not None:
        try:
            _cfg_in = json.loads(_setup.read().decode("utf-8"))
            if _cfg_in.get("version") == 1:
                # Only re-apply (and clear caches) when the uploaded content
                # actually changes, so re-runs don't thrash the pregame cache.
                _tok = _cfg_token(_cfg_in)
                if st.session_state.get("_uploaded_cfg_token") != _tok:
                    st.session_state["_uploaded_cfg"] = _cfg_in
                    st.session_state["_uploaded_cfg_token"] = _tok
                    get_pregame.clear()
                st.success("Setup loaded — pregame lines will match the "
                           "Projections app.")
            else:
                st.error("Unrecognised setup file (expected version 1).")
        except Exception as _e:
            st.error(f"Could not read setup file: {_e}")
    elif st.session_state.get("_uploaded_cfg"):
        st.caption("✓ Using a previously uploaded setup file this session.")

    st.divider()
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

# -- pregame-setup provenance banner -------------------------------------------
# Make it unmistakable whether the board is running on your finalized Projections
# setup or on raw model defaults — the latter is exactly the failure mode where
# "the live lines don't match my finalized projections".
_src = st.session_state.get("_uploaded_cfg")
_n_overrides = sum(len(v or {}) for v in (cfg.get("depth_charts") or {}).values())
_n_teamrat = sum(len(v or {}) for v in (cfg.get("team_rating_overrides") or {}).values())
if _src:
    st.success(
        f"Pregame setup loaded from your uploaded file — "
        f"{_n_overrides} player override(s), {_n_teamrat} team-rating override(s), "
        f"holds inherited. Pregame lines below reproduce your finalized projections.")
elif cfg:
    st.info(
        f"Using the local session file — {_n_overrides} player override(s), "
        f"{_n_teamrat} team-rating override(s). (Upload a setup file in the sidebar "
        "if this deployment can't see the Projections app's local state.)")
else:
    st.warning(
        "⚠️ No pregame setup loaded — the board is using RAW MODEL DEFAULTS "
        "(no overrides, default margins). Export a setup file from the Projections "
        "app (💾 Save for Live Trading) and upload it in the sidebar so the pregame "
        "lines match your finalized projections.")


@st.cache_data(ttl=600, show_spinner=False, max_entries=8)
def build_board(_engine, slug: str, home_id: str, away_id: str,
                game_date: str | None, cfg_token: str, pace_weight: float,
                tick: int, sched_is_final: bool = False) -> dict:
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

    # Build the live model. n_sims trims the live re-sim for responsiveness, but
    # tolerate an older LiveModel (e.g. a stale module still cached in a warm
    # Streamlit process) that predates the n_sims parameter — fall back rather
    # than crash the whole board with a TypeError.
    try:
        model = LiveModel(_engine, n_sims=LIVE_SIMS)
    except TypeError:
        model = LiveModel(_engine)
    live = model.resimulate(
        pregame, banked.by_player, frac_rem, pace_weight=pace_weight,
        home_team_id=home_id, away_team_id=away_id,
        team_of=banked.team_of, events=state.events)

    pricing = _engine.pricing

    # ---- player props: moving O/U line + posted X+ price ladder --------------
    stats: dict[str, list] = {}
    for stat in TRADE_FOCUS:
        thresholds = LADDER[stat]
        rows = []
        for ps in live.all_sims():
            if stat not in ps.stat_distributions:
                continue
            pid = str(ps.player_id)
            dist = np.asarray(ps.stat_distributions[stat], dtype=float)
            live_proj = float(np.mean(dist))
            bk = float(banked.get(pid, stat))
            if live_proj < 0.3 and bk <= 0:
                continue
            settled = bool(dist.size and float(np.std(dist)) < 1e-9)

            # Balanced O/U line NOW (the moving reference the user prices off).
            live_line = float(ps.prop_lines.get(stat, np.floor(np.median(dist)) + 0.5))
            # Pregame balanced line + projection, for the "how far has it moved".
            pre_ps = pre_by_pid.get(pid)
            if pre_ps is not None and stat in pre_ps.stat_distributions:
                pre_dist = np.asarray(pre_ps.stat_distributions[stat], dtype=float)
                pre_line = float(pre_ps.prop_lines.get(
                    stat, np.floor(np.median(pre_dist)) + 0.5))
                pre_proj = round(float(np.mean(pre_dist)), 2)
            else:
                pre_line, pre_proj = None, None  # call-up: no pregame projection

            # Posted X+ ladder: "N+" == over (N-0.5). Use the engine's PricingEngine
            # so the held (juiced) price reflects the inherited per-market hold —
            # this is the number the trader posts into BOSS for "N or more".
            ladder = {}
            for n in thresholds:
                p = float(np.mean(dist >= n))
                if p >= 0.995:
                    ladder[n] = "LOCK"      # already banked — a certain winner
                elif p <= 0.02:
                    ladder[n] = ""          # too unlikely to bother posting
                else:
                    ml = pricing.price_distribution(stat, dist, line=n - 0.5)
                    ladder[n] = ml.over_odds
            row = {
                "pid": pid, "Player": ps.full_name, "Now": bk,
                "PreLine": pre_line, "LiveLine": live_line,
                "LiveProj": round(live_proj, 2), "settled": settled,
            }
            for n in thresholds:
                row[f"{n}+"] = ladder[n]
            rows.append(row)
        rows.sort(key=lambda r: r["LiveProj"], reverse=True)
        stats[stat] = rows

    # ---- game markets: live ML / spread / total, vs pregame ------------------
    def _mkt(gm):
        # home gets + when underdog; away lays - when favored (projections-app convention)
        return {
            "home_ml": gm.home_ml, "away_ml": gm.away_ml,
            "home_wp": round(gm.home_win_prob * 100, 1),
            "away_wp": round(gm.away_win_prob * 100, 1),
            "spread_home_disp": round(-gm.spread_home, 1), "spread_home_odds": gm.spread_home_odds,
            "spread_away_disp": round(gm.spread_home, 1), "spread_away_odds": gm.spread_away_odds,
            "total_line": gm.total_line, "over": gm.over_odds, "under": gm.under_odds,
        }
    game = {"home_name": _tname(home_id), "away_name": _tname(away_id)}
    try:
        game["live"] = _mkt(pricing.price_game(live.game_sim)) if live.game_sim else None
    except Exception:
        game["live"] = None
    try:
        game["pre"] = _mkt(pricing.price_game(pregame.game_sim))
    except Exception:
        game["pre"] = None

    # Orientation guard: the scoreboard (feed homeScore/visitorScore) and the
    # priced game markets (which sum banked goals by team_of vs the SCHEDULE's
    # home/away ids) must refer to the same team as "home". If the feed's home
    # score disagrees with the banked-goal home total by more than a rounding
    # slack, home/away are likely flipped for this slug — surface it rather than
    # price the wrong side. (Scores can exceed goals via 2-pt, so compare goals.)
    orient_ok = True
    orient_msg = ""
    if home_id and away_id:
        bh_goals = _team_banked_all(banked.by_player, banked.team_of, home_id, "goals")
        ba_goals = _team_banked_all(banked.by_player, banked.team_of, away_id, "goals")
        # feed scores count 2-pt as 2, so only compare when we can back those out;
        # use goals-only banked vs feed score direction (which side leads) as a
        # cheap, robust orientation check.
        if (bh_goals + ba_goals) > 0 and (state.home_score + state.away_score) > 0:
            feed_home_leads = state.home_score > state.away_score
            bank_home_leads = bh_goals > ba_goals
            if state.home_score != state.away_score and bh_goals != ba_goals \
                    and feed_home_leads != bank_home_leads:
                orient_ok = False
                orient_msg = (
                    f"Feed score {state.away_score}-{state.home_score} (away-home) "
                    f"disagrees with banked goals ({_tname(away_id)} {ba_goals:.0f} / "
                    f"{_tname(home_id)} {bh_goals:.0f}). Home/away may be flipped for "
                    "this slug — game-market sides may be reversed.")

    return {
        "ok": True,
        "meta": {
            "away_score": state.away_score, "home_score": state.home_score,
            "period": state.period, "cm": state.clock_minutes, "cs": state.clock_seconds,
            "frac_rem": frac_rem, "n_events": state.n_events,
            # Authoritative FINAL = schedule eventStatus; feed clock is the fallback.
            "is_final": bool(sched_is_final or state.is_final),
            "is_overtime": bool(getattr(state, "is_overtime", False)),
            "orient_ok": orient_ok, "orient_msg": orient_msg,
        },
        "stats": stats, "game": game,
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
                        selected.game_date, _cfg_token(cfg), pace_weight, tick,
                        sched_is_final=bool(getattr(selected, "is_final", False)))
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
    if m.get("is_overtime") and not m["is_final"]:
        st.warning("Overtime (sudden death) — regulation is fully banked; the board "
                   "treats all stats as settled and does not simulate more time.")
    if m["is_final"]:
        st.info("Game reads FINAL — lines below are the settled result.")
    if not m.get("orient_ok", True):
        st.error("⚠️ " + m.get("orient_msg", "Home/away orientation mismatch — "
                 "game-market sides may be reversed. Verify before trading."))

    # ---- game markets: suggested live prices to post -------------------------
    g = board.get("game") or {}
    live_g, pre_g = g.get("live"), g.get("pre")
    st.markdown("### Game markets — suggested live prices")
    if live_g:
        an, hn = g["away_name"], g["home_name"]
        grows = [
            {"Market": f"{an} ML", "Live": live_g["away_ml"],
             "Pregame": pre_g["away_ml"] if pre_g else "—",
             "Win %": live_g["away_wp"]},
            {"Market": f"{hn} ML", "Live": live_g["home_ml"],
             "Pregame": pre_g["home_ml"] if pre_g else "—",
             "Win %": live_g["home_wp"]},
            {"Market": f"{an} {live_g['spread_away_disp']:+.1f}", "Live": live_g["spread_away_odds"],
             "Pregame": (f"{pre_g['spread_away_disp']:+.1f} ({pre_g['spread_away_odds']})"
                         if pre_g else "—"), "Win %": None},
            {"Market": f"{hn} {live_g['spread_home_disp']:+.1f}", "Live": live_g["spread_home_odds"],
             "Pregame": (f"{pre_g['spread_home_disp']:+.1f} ({pre_g['spread_home_odds']})"
                         if pre_g else "—"), "Win %": None},
            {"Market": f"Over {live_g['total_line']:.1f}", "Live": live_g["over"],
             "Pregame": (f"{pre_g['total_line']:.1f} ({pre_g['over']})" if pre_g else "—"),
             "Win %": None},
            {"Market": f"Under {live_g['total_line']:.1f}", "Live": live_g["under"],
             "Pregame": (f"{pre_g['total_line']:.1f} ({pre_g['under']})" if pre_g else "—"),
             "Win %": None},
        ]
        st.dataframe(
            pd.DataFrame(grows), hide_index=True, use_container_width=True,
            column_config={
                "Live": st.column_config.TextColumn("Live (post this)"),
                "Pregame": st.column_config.TextColumn("Pregame"),
                "Win %": st.column_config.NumberColumn("Win %", format="%.1f"),
            })
    else:
        st.info("Game markets unavailable (team IDs unknown for this game).")

    # ---- player props: moving line + X+ ladder to post -----------------------
    st.markdown("### Player props — suggested X+ prices")
    stat = st.radio("Market", list(TRADE_FOCUS), horizontal=True, key="mkt")
    stat_label = stat.title()
    thresholds = LADDER[stat]
    st.caption(f"**{stat_label} (live)** = current {stat_label.lower()} in the game so far · "
               "**Live Line** = the model's balanced O/U line right now (moves as the "
               "game develops) · **N+** columns = the held price to POST for "
               f"“N or more {stat_label.lower()}”. LOCK = already clinched. Adjust these "
               "into BOSS as events occur. No prices are bet here — this is your book-side monitor.")

    rows = board["stats"].get(stat, [])
    if not rows:
        st.info("No tradeable players for this market yet.")
        return

    df = pd.DataFrame(rows)
    ladder_cols = [f"{n}+" for n in thresholds]
    show_cols = ["Player", "Now", "PreLine", "LiveLine", "LiveProj"] + ladder_cols
    show = df[show_cols].copy()

    colcfg = {
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Now": st.column_config.NumberColumn(
            f"{stat_label} (live)", format="%.0f",
            help=f"Current {stat_label.lower()} in this game so far (banked)"),
        "PreLine": st.column_config.NumberColumn(
            "Pre Line", format="%.1f", help="Pregame balanced O/U line"),
        "LiveLine": st.column_config.NumberColumn(
            "Live Line", format="%.1f",
            help="Current balanced O/U line — moves with the game; price your X+ off this"),
        "LiveProj": st.column_config.NumberColumn(
            "Live Proj", format="%.2f", help="Current projected full-game total"),
    }
    for n in thresholds:
        colcfg[f"{n}+"] = st.column_config.TextColumn(
            f"{n}+", help=f"Suggested price to post for {n}+ {stat_label.lower()}")

    st.dataframe(show, hide_index=True, use_container_width=True, column_config=colcfg)

    st.caption(f"Last poll: {m['n_events']} events · pace_weight={pace_weight:.2f} · "
               f"{LIVE_SIMS:,} live sims · holds inherited from projections app")


live_board()
