"""
Shared engine state -- loaded once per session via st.cache_resource.
NOT a Streamlit page -- kept in pages/ so imports work, but hidden from nav
via the leading underscore (Streamlit ≥ 1.28 respects _prefix convention).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# -- Path bootstrap --------------------------------------------------------
_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


from projection_engine_v3 import (   # noqa: E402
    ProjectionEngine,
    ProjectionResult,
    TeamProjection,
    PlayerProjection,
    PricingEngine,
    _norm_pos,
    LG_GOALS, LG_SHOTS, LG_SHOT_PCT, LG_SOG_RATE,
    LG_FO_PCT, LG_SAVE_PCT, LG_2PT_RATE,
    LG_SHOTS_PER_TOUCH, LG_ASSIST_CONV,
    LG_2PT_SHOT_RATE, LG_PASS_PER_TOUCH, LG_CLEAN_SAVE_RATE,
)

# -- DB path ---------------------------------------------------------------
DB_PATH = os.getenv(
    "PLL_DB_PATH",
    str(_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"),
)

# -- Data freshness helper -------------------------------------------------
def get_data_freshness() -> dict:
    """
    Return info about when the warehouse data was last updated.
    Reads the modification time of the schedule parquet — the most recently
    written file after a warehouse build — as a proxy for last update time.
    """
    import datetime as _dt
    parquet = _ROOT / "data" / "curated_data" / "all_requested_seasons" / "game_schedule_all.parquet"
    try:
        mtime = parquet.stat().st_mtime
        last_updated = _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc)
        age_hours = (_dt.datetime.now(_dt.timezone.utc) - last_updated).total_seconds() / 3600
        stale = age_hours > 72  # warn if data is more than 3 days old
        return {
            "last_updated": last_updated.strftime("%Y-%m-%d %H:%M UTC"),
            "age_hours": round(age_hours, 1),
            "stale": stale,
            "available": True,
        }
    except Exception:
        return {"available": False, "stale": False}


# -- Autosave path ---------------------------------------------------------
# Written on every meaningful state change; restored silently on startup.
# Lives outside git-tracked folders so it never gets committed.
_AUTOSAVE_PATH = _ROOT / "data" / "session_autosave.json"


# -- Bootstrap DB from parquets if missing or incomplete ------------------
def _db_is_valid() -> bool:
    """Return True only if the DB file exists AND has the clean schema populated."""
    p = Path(DB_PATH)
    if not p.exists() or p.stat().st_size < 4096:
        return False
    con = None
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM clean.team_game_stats").fetchone()[0]
        return n > 0
    except Exception:
        return False
    finally:
        # Always release the file handle. A leaked read connection on Windows
        # can block the next bootstrap's --force overwrite, which shows up as an
        # app that "won't load until you reboot it a second time".
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _ensure_db() -> None:
    if _db_is_valid():
        return
    # DB is missing, empty, or the clean schema was never created — rebuild.
    # This handles: fresh Streamlit Cloud deploy, empty file left by a prior
    # failed bootstrap, or DB created by an incompatible DuckDB version.
    bootstrap = _ROOT / "scripts" / "bootstrap_db.py"
    if not bootstrap.exists():
        st.error(
            "Database not found and bootstrap script is missing. "
            "Ensure scripts/bootstrap_db.py is in the repository."
        )
        st.stop()
    with st.spinner("Building database from data files -- first load only, ~10 seconds…"):
        try:
            result = subprocess.run(
                [sys.executable, str(bootstrap), "--force"],
                capture_output=True, text=True,
                timeout=300,  # never hang the app forever on a stuck build
            )
        except subprocess.TimeoutExpired:
            st.error(
                "Database build timed out. This is usually a transient Streamlit "
                "Cloud cold-start issue — click **Rerun** (press R) or reboot the "
                "app once more. If it persists, re-run the **Update PLL Data "
                "Warehouse** GitHub Action."
            )
            st.stop()
        except Exception as e:
            st.error(f"Database build could not start: {e}")
            st.stop()
    if result.returncode != 0:
        st.error(
            f"Database bootstrap failed.\n\n```\n{result.stderr[-2000:]}\n```\n\n"
            "Run the GitHub Action (Update PLL Data Warehouse) to populate data/."
        )
        st.stop()
    # Verify the rebuild actually produced a valid DB before continuing, so a
    # silent bad build surfaces as a clear message instead of a downstream crash.
    if not _db_is_valid():
        st.error(
            "Database was rebuilt but still isn't valid. The data files may be "
            "missing or incomplete — re-run the **Update PLL Data Warehouse** "
            "GitHub Action, then reboot the app."
        )
        st.stop()


_ensure_db()


# -- Roster freshness token -------------------------------------------------
# The engine reads current_rosters.csv once, at load(). Because the engine is a
# cached resource, a running app would otherwise keep the roster snapshot from
# whenever it was first cached — so roster adds/drops committed by the GitHub
# Action (e.g. a player added to a team) never appeared until a manual reboot.
# We key the cache on the roster files' modification times: when the scraper
# updates current_rosters.csv (or a new gameday roster lands), the token changes
# and Streamlit rebuilds the engine with the fresh rosters automatically.
def _roster_fingerprint() -> str:
    import hashlib
    parts = []
    paths = [
        _ROOT / "data" / "reference_tables" / "current_rosters.csv",
        _ROOT / "data" / "reference_tables" / "gameday_rosters" / "gameday_latest.csv",
    ]
    gd_dir = _ROOT / "data" / "reference_tables" / "gameday_rosters"
    if gd_dir.exists():
        paths += sorted(gd_dir.glob("gameday_2026_week*.csv"))
    for p in paths:
        try:
            parts.append(f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}")
        except Exception:
            parts.append(f"{p.name}:missing")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# -- Engine cache ----------------------------------------------------------
# `roster_token` is part of the cache key: when it changes (rosters updated on
# disk), Streamlit builds a fresh engine instead of returning the stale one.
@st.cache_resource(show_spinner="Loading projection engine…")
def _build_engine(roster_token: str) -> ProjectionEngine:
    engine = ProjectionEngine(db_path=DB_PATH)
    engine.load()
    # run_backtest=False keeps startup fast (~2-3s).
    # The calibrator is fitted lazily when the Model Performance page is visited.
    engine.fit(run_backtest=False)
    return engine


def get_engine() -> ProjectionEngine:
    return _build_engine(_roster_fingerprint())


def refresh_rosters() -> None:
    """Force the engine to rebuild from the latest roster files on disk AND
    re-run the projection so the fresh rosters actually reach the UI.

    Clearing the engine cache alone is not enough: the depth chart and
    projection pages render from st.session_state["last_result"], a projection
    computed by a *prior* engine.project() call. That result is only recomputed
    when it is None (see 1_Projections.py re-projection guard). So we must also
    invalidate last_result and the baseline caches, and request a re-run, or the
    rebuilt engine is never queried and the display stays stale."""
    try:
        _build_engine.clear()
    except Exception:
        st.cache_resource.clear()
    # Invalidate the baseline cache and remember the new fingerprint so the
    # auto-detect path doesn't immediately re-trigger.
    for k in ("_baseline_result", "_baseline_result_key"):
        st.session_state.pop(k, None)
    st.session_state["_roster_fingerprint_seen"] = _roster_fingerprint()
    # Re-project the currently selected game NOW against the freshly rebuilt
    # engine, so both the Projections and Depth Charts pages show fresh rosters
    # immediately. If no game is selected yet, just null the result and let the
    # Projections page's auto-run handle it.
    game = st.session_state.get("selected_game") or {}
    if game.get("home_team_id") and game.get("away_team_id"):
        try:
            run_projection_for_game(_build_engine(_roster_fingerprint()), game)
            return
        except Exception:
            pass
    st.session_state.pop("last_result", None)
    st.session_state["_run_after_load"] = True


def maybe_refresh_on_roster_change() -> bool:
    """Auto-detect a roster file change (e.g. an unattended gameday scrape that
    landed while the app was open) and invalidate the cached projection so it
    re-runs against fresh rosters — no manual button press needed.

    Returns True if a change was detected and caches were invalidated. Call this
    early in each page render, before reading st.session_state["last_result"]."""
    current = _roster_fingerprint()
    seen = st.session_state.get("_roster_fingerprint_seen")
    if seen is None:
        # First render this session: record the baseline, don't force a re-run.
        st.session_state["_roster_fingerprint_seen"] = current
        return False
    if current != seen:
        st.session_state["_roster_fingerprint_seen"] = current
        for k in ("last_result", "_baseline_result", "_baseline_result_key"):
            st.session_state.pop(k, None)
        st.session_state["_run_after_load"] = True
        return True
    return False


# -- Autosave / autorestore ------------------------------------------------

def _autosave() -> None:
    """Write current session state to disk. Called after every meaningful change."""
    try:
        game = st.session_state.get("selected_game") or {}
        saved_season = st.session_state.get("season_filter") or (
            int(str(game.get("game_date", ""))[:4]) if game.get("game_date") else None
        )
        payload = {
            "selected_game":         game,
            "depth_charts":          st.session_state.get("depth_charts", {}),
            "team_rating_overrides": st.session_state.get("team_rating_overrides", {}),
            "hold_pct":              st.session_state.get("hold_pct", 0.075),
            "season_filter":         saved_season,
            "version":               1,
        }
        _AUTOSAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _AUTOSAVE_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass  # autosave is best-effort; never crash the app


def _autorestore() -> bool:
    """
    Restore session state from the autosave file if it exists.
    Returns True if state was restored, False otherwise.
    Only runs once per session (guarded by _autorestore_done flag).
    """
    if st.session_state.get("_autorestore_done"):
        return False
    st.session_state["_autorestore_done"] = True

    if not _AUTOSAVE_PATH.exists():
        return False
    try:
        payload = json.loads(_AUTOSAVE_PATH.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            return False

        restored = False
        if payload.get("selected_game"):
            st.session_state["selected_game"] = payload["selected_game"]
            restored = True
        if payload.get("depth_charts"):
            st.session_state["depth_charts"] = payload["depth_charts"]
            restored = True
        if payload.get("team_rating_overrides"):
            st.session_state["team_rating_overrides"] = payload["team_rating_overrides"]
            restored = True
        if "hold_pct" in payload:
            st.session_state["hold_pct"] = float(payload["hold_pct"])
        if payload.get("season_filter") is not None:
            st.session_state["season_filter"] = int(payload["season_filter"])

        # Clear widget seed keys so they re-init from restored values
        stale = [k for k in st.session_state
                 if k.startswith(("tr_num_", "pr_num_", "hold_num_", "pp_hold_num"))]
        for k in stale:
            del st.session_state[k]
        for k in ("game_idx_p1",):
            st.session_state.pop(k, None)

        if restored:
            st.session_state["_run_after_load"] = True
            st.session_state["last_result"] = None
        return restored
    except Exception:
        return False


# -- Session state ---------------------------------------------------------
def init_session() -> None:
    defaults: Dict = {
        "selected_game":            None,
        "last_result":              None,
        # depth_charts: {team_id: {player_id: {active, usage_multiplier, is_starter,
        #   rating_overrides: {rating_key: float}}}}
        "depth_charts":             {},
        # team_rating_overrides: {team_id: {rating_key: float}}
        "team_rating_overrides":    {},
        "line_overrides":           {},
        "hold_pct":                 0.075,
        "last_projection_updated_at": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Silently restore last session on first load of a new browser session.
    # _autorestore is guarded internally so it only fires once.
    _autorestore()


# -- Depth chart helpers ---------------------------------------------------
def get_depth_chart(team_id: str) -> Dict:
    if team_id not in st.session_state.depth_charts:
        st.session_state.depth_charts[team_id] = {}
    return st.session_state.depth_charts[team_id]


def set_player_override(team_id: str, player_id: str, key: str, value) -> None:
    dc = get_depth_chart(team_id)
    if player_id not in dc:
        dc[player_id] = {}
    dc[player_id][key] = value
    _autosave()


def set_player_rating(team_id: str, player_id: str, rating_key: str, value: float) -> None:
    """Override a specific model rating for a player (e.g. share_goals_ewm)."""
    dc = get_depth_chart(team_id)
    if player_id not in dc:
        dc[player_id] = {}
    if "rating_overrides" not in dc[player_id]:
        dc[player_id]["rating_overrides"] = {}
    dc[player_id]["rating_overrides"][rating_key] = value
    _autosave()


def build_overrides() -> Dict:
    """Build full override dict for ProjectionEngine.project()."""
    merged: Dict = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            entry: Dict = {}
            if "active" in settings:
                entry["active"] = settings["active"]
            if "usage_multiplier" in settings:
                um = float(settings["usage_multiplier"])
                entry["usage_multiplier"] = um
                # Treat usage=0.0 as inactive so player is fully excluded
                if um == 0.0:
                    entry["active"] = False
            if "is_starter" in settings:
                entry["is_starter"] = settings["is_starter"]
            # Position override: inject as pos_norm so _project_player uses the
            # overridden position for POS_DEFAULTS, POS_CAPS, and zero-inflation.
            override_keys: List[str] = []
            if "position_override" in settings:
                entry["pos_norm"] = settings["position_override"]
                override_keys.append("pos_norm")
            for rk, rv in settings.get("rating_overrides", {}).items():
                entry[rk] = rv
                override_keys.append(rk)
            # Pass the set of user-overridden keys so the engine can
            # bypass credibility blending for explicitly set values.
            if override_keys:
                entry["_override_keys"] = override_keys
            if entry:
                merged[pid] = entry
    return merged


def build_active_players() -> Dict:
    out: Dict = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            if "active" in settings:
                out[pid] = settings["active"]
            # usage=0.0 → inactive
            if float(settings.get("usage_multiplier", 1.0)) == 0.0:
                out[pid] = False
    return out


def build_starter_goalies() -> Dict:
    """Return {team_id: player_id} for any team where a goalie starter was manually set."""
    out: Dict = {}
    for team_id, team_dc in st.session_state.depth_charts.items():
        for pid, settings in team_dc.items():
            if settings.get("is_starter"):
                out[team_id] = pid
                break  # only one starter per team
    return out


# -- Session save/restore --------------------------------------------------

def session_to_json() -> str:
    """Serialize all user overrides and selected game to a JSON string."""
    import json
    game = st.session_state.get("selected_game") or {}
    # Persist the season so the season selectbox re-seeds to the correct year on
    # restore (otherwise it defaults to index=0 / most-recent season and the game
    # lookup finds no match, causing the wrong game to be projected).
    saved_season = st.session_state.get("season_filter") or (
        int(str(game.get("game_date", ""))[:4]) if game.get("game_date") else None
    )
    payload = {
        "selected_game":         game,
        "depth_charts":          st.session_state.get("depth_charts", {}),
        "team_rating_overrides": st.session_state.get("team_rating_overrides", {}),
        "hold_pct":              st.session_state.get("hold_pct", 0.075),
        "season_filter":         saved_season,
        "version":               1,
    }
    return json.dumps(payload, indent=2, default=str)


def session_from_json(json_str: str) -> bool:
    """Restore session state from a previously saved JSON string. Returns True on success."""
    import json
    try:
        payload = json.loads(json_str)
        if payload.get("version") != 1:
            return False
        if payload.get("selected_game"):
            st.session_state["selected_game"] = payload["selected_game"]
        if payload.get("depth_charts"):
            st.session_state["depth_charts"] = payload["depth_charts"]
        if payload.get("team_rating_overrides"):
            st.session_state["team_rating_overrides"] = payload["team_rating_overrides"]
        if "hold_pct" in payload:
            st.session_state["hold_pct"] = float(payload["hold_pct"])

        # Clear all widget keys that need to re-seed from restored state:
        # - tr_num_*: team-rating number inputs (must re-init from team_rating_overrides)
        # - pr_num_*: player-rating number inputs in the depth chart panel
        # - hold_num_* / pp_hold_num: hold% inputs
        # - game_idx_p1: game selectbox -- cleared so it re-seeds to the restored game
        stale_keys = [k for k in st.session_state if k.startswith("tr_num_")
                      or k.startswith("pr_num_")
                      or k.startswith("hold_num_") or k.startswith("pp_hold_num")]
        for k in stale_keys:
            del st.session_state[k]
        if "game_idx_p1" in st.session_state:
            del st.session_state["game_idx_p1"]
        # Re-seed the season selectbox to the saved season so the game list is
        # filtered correctly before we look up the restored game.
        if payload.get("season_filter") is not None:
            st.session_state["season_filter"] = int(payload["season_filter"])

        # Signal the Projections page to auto-run after the rerun.
        st.session_state["_run_after_load"] = True
        # Clear stale result so projection reruns with restored state
        st.session_state["last_result"] = None
        return True
    except Exception:
        return False


# -- New helper functions --------------------------------------------------

def run_projection_for_game(engine, game: Dict) -> Optional[ProjectionResult]:
    """Run projection for a specific game dict using current session state."""
    home_id = str(game.get("home_team_id", ""))
    away_id = str(game.get("away_team_id", ""))
    if not home_id or not away_id:
        return None
    team_rating_overrides = {}
    for tid in [home_id, away_id]:
        ov = get_team_rating_overrides(tid)
        if ov:
            team_rating_overrides[tid] = ov
    result = engine.project(
        home_team_id=home_id,
        away_team_id=away_id,
        game_date=game.get("game_date"),
        player_overrides=build_overrides() or None,
        active_players=build_active_players() or None,
        starter_goalies=build_starter_goalies() or None,
        team_rating_overrides=team_rating_overrides or None,
    )
    st.session_state.last_result = result
    st.session_state.selected_game = game
    _autosave()
    return result


def render_update_projection_btn(engine, key: str = "upd") -> bool:
    """Render Update Projection button in sidebar. Returns True if clicked."""
    game = st.session_state.get("selected_game")
    if not game:
        return False
    clicked = st.sidebar.button(
        "🔄 Update Projection",
        type="primary",
        use_container_width=True,
        key=f"upd_btn_{key}",
        help="Rerun projection with current depth chart, usage, and rating settings.",
    )
    if clicked:
        with st.spinner("Running 20,000 simulations…"):
            run_projection_for_game(engine, game)
        st.rerun()
    return clicked


# -- Universal projection runner -------------------------------------------
def _season_from_game_dict(g: Dict) -> int:
    for key in ("game_number_season", "season"):
        val = g.get(key)
        if val:
            try:
                return int(val)
            except Exception:
                pass
    gdate = str(g.get("game_date", "") or "")
    if len(gdate) >= 4:
        try:
            return int(gdate[:4])
        except Exception:
            pass
    return 0


def get_selected_game_or_default(engine: Optional[ProjectionEngine] = None) -> Optional[Dict]:
    """Return the persisted selected game, or default to the next upcoming game."""
    game = st.session_state.get("selected_game")
    if isinstance(game, dict) and game.get("home_team_id") and game.get("away_team_id"):
        return game

    engine = engine or get_engine()
    games = engine.upcoming_games()
    if not games:
        return None

    for g in games:
        g["game_number_season"] = _season_from_game_dict(g)

    games = sorted_upcoming(games)
    idx = default_game_index(games)
    game = games[idx] if games else None

    if game:
        st.session_state.selected_game = game
    return game


def build_team_rating_overrides_for_game(game: Optional[Dict]) -> Dict:
    if not game:
        return {}
    out: Dict = {}
    for tid in [
        str(game.get("home_team_id", "") or ""),
        str(game.get("away_team_id", "") or ""),
    ]:
        if not tid:
            continue
        ov = get_team_rating_overrides(tid)
        if ov:
            out[tid] = ov
    return out


def run_selected_projection(
    engine: Optional[ProjectionEngine] = None,
    game: Optional[Dict] = None,
) -> Optional[ProjectionResult]:
    """
    Rerun projection for the currently selected game. Captures all current
    session state: depth chart, usage multipliers, player rating overrides,
    goalie starter, team rating overrides, and hold %.
    """
    engine = engine or get_engine()
    game = game or get_selected_game_or_default(engine)
    if not game:
        return None

    home_id = str(game.get("home_team_id", "") or "")
    away_id = str(game.get("away_team_id", "") or "")
    if not home_id or not away_id:
        return None

    game_date_raw = str(game.get("game_date", "") or "")
    game_date = game_date_raw[:10] if game_date_raw else None

    team_rating_overrides = build_team_rating_overrides_for_game(game)

    result = engine.project(
        home_team_id=home_id,
        away_team_id=away_id,
        game_date=game_date,
        player_overrides=build_overrides() or None,
        active_players=build_active_players() or None,
        starter_goalies=build_starter_goalies() or None,
        team_rating_overrides=team_rating_overrides or None,
    )

    st.session_state.selected_game = game
    st.session_state.last_result = result
    st.session_state.last_projection_updated_at = date.today().isoformat()
    _autosave()
    return result


def render_global_projection_runner(
    engine: Optional[ProjectionEngine] = None,
    key_prefix: str = "global",
    show_hold: bool = False,
) -> None:
    """
    Sidebar Update Projection button for every page.
    Place after page-specific sidebar widgets so their state is captured.
    """
    engine = engine or get_engine()
    game = get_selected_game_or_default(engine)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Projection")

        flash = st.session_state.pop("_projection_update_flash", None)
        if flash:
            st.success(flash)

        if game:
            home_id = str(game.get("home_team_id", "") or "")
            away_id = str(game.get("away_team_id", "") or "")
            game_number = game.get("game_number", "--")
            game_date = str(game.get("game_date", "") or "")[:10]
            st.markdown(
                f'<span class="note-text">'
                f'<b>{team_name(away_id)} @ {team_name(home_id)}</b><br>'
                f'Game {game_number} · {game_date}'
                f'</span>',
                unsafe_allow_html=True,
            )
            last_upd = st.session_state.get("last_projection_updated_at")
            if last_upd:
                st.markdown(
                    f'<span class="note-text">Last run: {last_upd}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No upcoming game available.")

        if show_hold:
            hold_pct = st.session_state.get("hold_pct", 0.045)
            new_hold = st.slider(
                "Hold %", 2.0, 8.0, float(hold_pct * 100), 0.5,
                key=f"{key_prefix}_hold_pct",
            ) / 100.0
            st.session_state.hold_pct = new_hold

        if st.button(
            "▶ Update Projection",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_update_projection",
            disabled=game is None,
            help="Rerun with current depth chart, usage, starter, and rating overrides.",
        ):
            with st.spinner("Running 20,000 simulations…"):
                result = run_selected_projection(engine=engine, game=game)
            if result is None:
                st.error("Projection could not be updated.")
            else:
                st.session_state["_projection_update_flash"] = "✓ Projection updated."
                st.rerun()


# -- Team rating override helpers ------------------------------------------
def get_team_rating_overrides(team_id: str) -> Dict:
    return st.session_state.team_rating_overrides.get(team_id, {})


def set_team_rating_override(team_id: str, key: str, value: float) -> None:
    if team_id not in st.session_state.team_rating_overrides:
        st.session_state.team_rating_overrides[team_id] = {}
    st.session_state.team_rating_overrides[team_id][key] = value
    _autosave()


def build_team_adjustments() -> Dict:
    out: Dict = {}
    for tid, overrides in st.session_state.team_rating_overrides.items():
        if not overrides:
            continue
        out[tid] = {"off_mult": overrides.get("off_mult", 1.0),
                    "def_mult_opp": overrides.get("def_mult_opp", 1.0)}
    return out


# -- Game selector helpers -------------------------------------------------
def sorted_upcoming(games: List[Dict]) -> List[Dict]:
    import datetime as dt
    today = dt.date.today()
    current_year = today.year

    def _sort_key(g):
        season = int(g.get("game_number_season") or g.get("season") or 0)
        gdate_raw = g.get("game_date", "") or ""
        try:
            gdate = str(gdate_raw)[:10]
        except Exception:
            gdate = "9999-12-31"
        season_rank = 0 if season == current_year else (1 if season > current_year else 2)
        return (season_rank, gdate)

    return sorted(games, key=_sort_key)


def default_game_index(games: List[Dict]) -> int:
    import datetime as dt
    today = dt.date.today()
    current_year = today.year

    for i, g in enumerate(games):
        season = int(g.get("game_number_season") or
                     _extract_season_from_game(g) or 0)
        if season != current_year:
            continue
        gdate_raw = str(g.get("game_date", ""))[:10]
        try:
            gd = dt.date.fromisoformat(gdate_raw)
            if gd >= today:
                return i
        except Exception:
            pass
    return 0


def _extract_season_from_game(g: Dict) -> int:
    for key in ("season", "game_number_season"):
        v = g.get(key)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    gdate = str(g.get("game_date", ""))
    if len(gdate) >= 4:
        try:
            return int(gdate[:4])
        except Exception:
            pass
    return 0


# -- Constants exposed for UI ----------------------------------------------
TEAM_RATING_DEFS = {
    "goals_ewm": {
        "label": "Goals scored per game (count)",
        "help": (
            "The team's recent average number of goals scored per game — a count "
            "of made shots, where a 2-point goal counts as 1 (not 2). This is NOT "
            "the scoreboard total; the projection converts goals into points "
            "separately by valuing 2-point goals at 2. League avg ~11.2 goals "
            "≈ ~11.8 scoreboard points. Raise if the offense is hot; lower if a "
            "key scorer is out."
        ),
        "min": 5.0, "max": 20.0, "step": 0.1, "fmt": "{:.1f}",
    },
    "two_pt_rate_ewm": {
        "label": "2-point goal rate (share of goals worth 2)",
        "help": (
            "The fraction of the team's goals that are 2-pointers. This does NOT "
            "change how many goals the team scores — it splits the existing goal "
            "count into 1s and 2s, which changes the scoreboard total (each 2pt "
            "goal adds an extra point). Example: at 12 goals, a rate of 0.08 = "
            "~1 two-pointer (~13 pts); 0.20 = ~2.4 two-pointers (~14.4 pts). "
            "League avg ~0.07. Raise for a heavy 2pt-shooting team; lower for a "
            "team that rarely takes them."
        ),
        "min": 0.0, "max": 0.40, "step": 0.005, "fmt": "{:.3f}",
    },
    "shot_pct_ewm": {
        "label": "Finishing rate (goals per shot attempt)",
        "help": (
            "Goals divided by total shot attempts — how often any shot becomes a "
            "goal. This is NOT shots-on-goal % / accuracy (how often a shot hits "
            "the cage); it is the finishing conversion on all attempts. League "
            "avg ~0.274."
        ),
        "min": 0.15, "max": 0.45, "step": 0.005, "fmt": "{:.3f}",
    },
    "shots_ewm": {
        "label": "Shot attempts per game",
        "help": (
            "Total shot attempts per game, counting both on-cage and off-cage "
            "shots. This is not shots on goal (on-target only). League avg ~41."
        ),
        "min": 25.0, "max": 60.0, "step": 0.5, "fmt": "{:.1f}",
    },
    "sog_rate_ewm": {
        "label": "Shots-on-goal rate (share of shots on target)",
        "help": (
            "The fraction of the team's shot attempts that are on goal (on-cage). "
            "This is shot accuracy, NOT finishing: shots on goal per game = shot "
            "attempts × this rate. Raising it lifts projected shots on goal, which "
            "modestly raises projected goals (more on-target shots convert). It "
            "also raises the opposing goalie's shots faced. League avg ~0.64. "
            "Engine uses values between 0.40 and 0.85."
        ),
        "min": 0.40, "max": 0.85, "step": 0.005, "fmt": "{:.3f}",
    },
    "assists_ewm": {
        "label": "Assists per game",
        "help": (
            "Team's recent avg assists per game. League avg ~7.3. "
            "Some teams assist on most goals (high assist culture); others shoot more unassisted. "
            "Raise to boost all player assist projections proportionally; lower to reduce them. "
            "Affects proj_assists for every active player via team-level scaling."
        ),
        "min": 2.0, "max": 14.0, "step": 0.1, "fmt": "{:.1f}",
    },
    # bayes_fo_pct removed from team-level adjustments.
    # FO win rate is now driven entirely by active FO players' individual ratings
    # via _apply_fo_correction() in ProjectionEngine.project(). Adjusting the
    # team-level FO% here had no effect because the roster-derived rate overwrote
    # it. Adjust FO player ratings directly in Depth Charts instead.
    "bayes_save_pct": {
        "label": "Goalie save % (saves ÷ shots on goal faced)",
        "help": (
            "The starting goalie's Bayesian-shrunk save rate: saves ÷ shots on "
            "goal faced, where shots on goal faced = saves + goals allowed. The "
            "denominator is on-cage shots only (a save can only happen on an "
            "on-target shot), not total opponent shot attempts. League avg ~0.537."
        ),
        "min": 0.35, "max": 0.75, "step": 0.005, "fmt": "{:.3f}",
    },
    "goals_against_ewm": {
        "label": "Goals allowed per game (count, defense)",
        "help": (
            "The team's recent average number of goals allowed per game — a "
            "count, where a 2-point goal allowed counts as 1 (not 2). LOWER = "
            "better defense. This does not distinguish whether goals allowed were "
            "1s or 2s, so it is not the same as points allowed. League avg ~11.2."
        ),
        "min": 5.0, "max": 20.0, "step": 0.1, "fmt": "{:.1f}",
    },
}

# Each rating carries a "group" so the depth-chart edit panel can organise the
# ratings under stat-category headers (Goal Ratings, Assist Ratings, …) so it's
# obvious which lever affects which projection.
PLAYER_RATING_DEFS = {
    # ── GOAL RATINGS ──────────────────────────────────────────────────────────
    "share_goals_ewm": {
        "label": "Goal share", "group": "Goal Ratings",
        "help": "Player's share of team goals (0.20 = 20% of all team goals). Overriding pins the player to the fraction you set; teammates' goals adjust so the team total holds.",
        "min": 0.0, "max": 0.50, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    "shot_pct_ewm": {
        "label": "Shooting %", "group": "Goal Ratings",
        "help": "Goals per shot. League avg ~0.27. Raising it scales the player's goal projection up (relative to their own baseline); teammates absorb the difference.",
        "min": 0.05, "max": 0.60, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "var_index_goals": {
        "label": "Goals volatility (var/mean)", "group": "Goal Ratings",
        "help": "Dispersion of the player's goal distribution: variance ÷ mean. 1.0 = steady (Poisson); higher = more boom-or-bust, raising the deep X+ (3+/4+) goal prices. Range is tighter for goals than other stats because goals are tied to the team total each simulation — pushing past ~1.3 would start to pull the goal mean/O-U line down.",
        "min": 1.00, "max": 1.30, "step": 0.05, "fmt": "{:.2f}",
        "positions": ["A", "M", "SSDM", "LSM", "FO", "D"],
    },
    # ── ASSIST RATINGS ────────────────────────────────────────────────────────
    "share_assists_ewm": {
        "label": "Assist share", "group": "Assist Ratings",
        "help": "Player's share of team assists. Overriding pins the player to the fraction you set; teammates' assists adjust so the team total holds.",
        "min": 0.0, "max": 0.50, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    "assist_conv_ewm": {
        "label": "Assist conversion", "group": "Assist Ratings",
        "help": "Assists per assist opportunity. Raising it scales the player's assist projection up (vs their own baseline); teammates absorb the difference. League avg ~0.31 (A), 0.25 (M).",
        "min": 0.00, "max": 1.00, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "pass_per_touch_ewm": {
        "label": "Pass per touch", "group": "Assist Ratings",
        "help": "Passes per possession touch — distributor signal. Raising it scales the player's assist projection up (vs their own baseline); teammates absorb the difference. League avg ~0.73.",
        "min": 0.30, "max": 1.00, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "var_index_assists": {
        "label": "Assists volatility (var/mean)", "group": "Assist Ratings",
        "help": "Dispersion of the player's assist distribution: variance ÷ mean. 1.0 = steady; higher = more boom-or-bust, raising the deep assist X+ prices. Reshapes the milestone tails only — the mean / O/U line is held fixed.",
        "min": 1.00, "max": 3.00, "step": 0.05, "fmt": "{:.2f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    # ── SHOT RATINGS ──────────────────────────────────────────────────────────
    "share_shots_ewm": {
        "label": "Shot share", "group": "Shot Ratings",
        "help": "Player's share of team shots (0.18 = 18%). Overriding pins the player to the fraction you set; teammates' shots adjust so the team total holds.",
        "min": 0.0, "max": 0.50, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    "shots_per_touch_ewm": {
        "label": "Shots per touch", "group": "Shot Ratings",
        "help": "Shots generated per touch. Raising it scales the player's shot projection up (vs their own baseline); teammates absorb the difference. League avg ~0.21 (A/M).",
        "min": 0.02, "max": 0.60, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "sog_rate_ewm": {
        "label": "Shots on goal %", "group": "Shot Ratings",
        "help": "Fraction of shots that are on goal. Affects the player's SOG projection only (not shots or goals); SOG is capped at their shot count. League avg ~0.63.",
        "min": 0.20, "max": 1.00, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "var_index_shots": {
        "label": "Shots volatility (var/mean)", "group": "Shot Ratings",
        "help": "Dispersion of the player's shot distribution: variance ÷ mean. 1.0 = steady; higher = more boom-or-bust, raising the deep shot X+ prices. Mean / O/U line held fixed.",
        "min": 1.00, "max": 3.00, "step": 0.05, "fmt": "{:.2f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    "var_index_sog": {
        "label": "SOG volatility (var/mean)", "group": "Shot Ratings",
        "help": "Dispersion of the player's shots-on-goal distribution: variance ÷ mean. 1.0 = steady; higher = more boom-or-bust, raising the deep SOG X+ prices. Range capped at 2.0 because SOG is capped at the player's shot count, which limits how far the tail can widen.",
        "min": 1.00, "max": 2.00, "step": 0.05, "fmt": "{:.2f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    # ── 2-POINT RATINGS ───────────────────────────────────────────────────────
    "two_pt_rate_ewm": {
        "label": "2PT goal rate", "group": "2-Point Ratings",
        "help": "Fraction of the player's goals that are 2-pointers (outcome). Shifts the 1pt/2pt split of their goals. League avg ~7%.",
        "min": 0.0, "max": 0.65, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "two_pt_shot_rate_ewm": {
        "label": "2PT shot rate", "group": "2-Point Ratings",
        "help": "Fraction of shots attempted as 2-pointers (intent). Blended 60% into the 2PT projection when 2PT goal rate isn't set directly. League avg ~4% (A), ~12% (M).",
        "min": 0.0, "max": 0.80, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    # ── GOALIE RATINGS ────────────────────────────────────────────────────────
    "bayes_save_pct": {
        "label": "Save %", "group": "Goalie Ratings",
        "help": "Goalie's Bayesian save%. League avg ~0.537.",
        "min": 0.35, "max": 0.75, "step": 0.005, "fmt": "{:.3f}",
        "positions": ["G"],
    },
    "clean_save_rate_ewm": {
        "label": "Clean save rate", "group": "Goalie Ratings",
        "help": "Fraction of saves that are clean (controlled stops). High = consistent goalie, tighter sim variance. League avg ~0.34.",
        "min": 0.10, "max": 0.70, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["G"],
    },
    "var_index_saves": {
        "label": "Saves volatility (var/mean)", "group": "Goalie Ratings",
        "help": "Dispersion of the goalie's saves distribution: variance ÷ mean. 1.0 = steady; higher = more boom-or-bust, raising the deep saves X+ prices. Mean / O/U line held fixed.",
        "min": 1.00, "max": 3.00, "step": 0.05, "fmt": "{:.2f}",
        "positions": ["G"],
    },
    # ── FACEOFF RATINGS ───────────────────────────────────────────────────────
    "bayes_fo_pct": {
        "label": "FO win %", "group": "Faceoff Ratings",
        "help": "Faceoff win rate. 0.500 = league avg.",
        "min": 0.25, "max": 0.75, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["FO"],
    },
    "var_index_fo_wins": {
        "label": "FO wins volatility (var/mean)", "group": "Faceoff Ratings",
        "help": "Dispersion of the specialist's faceoff-wins distribution: variance ÷ mean. 1.0 = steady; higher = more boom-or-bust, raising the deep FO-win X+ prices. Mean / O/U line held fixed.",
        "min": 1.00, "max": 3.00, "step": 0.05, "fmt": "{:.2f}",
        "positions": ["FO"],
    },
}


# -- Formatting ------------------------------------------------------------
TEAM_COLORS = {
    "ATL": "#1d4ed8", "OUT": "#d97706", "CAN": "#dc2626", "RED": "#16a34a",
    "WAT": "#7c3aed", "WHP": "#0891b2", "CHA": "#334155", "ARC": "#b45309",
}
TEAM_NAMES = {
    "ATL": "Atlas",      "OUT": "Outlaws",    "CAN": "Cannons",
    "RED": "Redwoods",   "WAT": "Waterdogs",  "WHP": "Whipsnakes",
    "CHA": "Chaos",      "ARC": "Archers",
}

def team_color(tid: str) -> str:
    return TEAM_COLORS.get(str(tid).upper(), "#475569")

def team_name(tid: str) -> str:
    return TEAM_NAMES.get(str(tid).upper(), str(tid))

def fmt_prob(p: float) -> str:
    return f"{p * 100:.1f}%"

def fmt_goals(v: float) -> str:
    return f"{v:.1f}"

def fmt_odds(odds_str: str) -> str:
    try:
        v = int(str(odds_str).replace("+", ""))
        cls = "odds-fav" if v < 0 else ("odds-dog" if v > 0 else "odds-even")
    except Exception:
        cls = "odds-even"
    return f'<span class="{cls}">{odds_str}</span>'

def card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="pll-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="pll-card">'
        f'<div class="pll-card-label">{label}</div>'
        f'<div class="pll-card-value">{value}</div>'
        f'{sub_html}</div>'
    )

def pos_badge(pos: str) -> str:
    colors = {
        "A": "#1d4ed8", "M": "#059669", "D": "#dc2626",
        "FO": "#d97706", "SSDM": "#7c3aed", "LSM": "#0891b2", "G": "#475569",
    }
    c = colors.get(str(pos).upper(), "#475569")
    return (
        f'<span style="background:{c};color:#fff;border-radius:4px;'
        f'padding:1px 6px;font-size:0.75rem;font-weight:700;">{pos}</span>'
    )

SHARED_CSS = """
<style>
  .main .block-container { padding-top:1rem; max-width:1800px; }
  .pll-card {
    border:1px solid rgba(148,163,184,.20); border-radius:12px;
    padding:12px 16px;
    background:linear-gradient(160deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
    box-shadow:0 4px 16px rgba(0,0,0,.10); margin-bottom:8px;
  }
  .pll-card-label { color:#94a3b8; font-size:.78rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
  .pll-card-value { font-size:1.5rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
  .pll-card-sub   { color:#94a3b8; font-size:.78rem; margin-top:3px; }
  .odds-fav  { background:#16a34a; color:#fff; border-radius:6px;
    padding:2px 8px; font-weight:700; font-size:.85rem; }
  .odds-dog  { background:#2563eb; color:#fff; border-radius:6px;
    padding:2px 8px; font-weight:700; font-size:.85rem; }
  .odds-even { background:#475569; color:#fff; border-radius:6px;
    padding:2px 8px; font-weight:700; font-size:.85rem; }
  .note-text { color:#64748b; font-size:.80rem; font-style:italic; }
  .rating-changed { background:rgba(251,191,36,.12); border-left:3px solid #fbbf24;
    padding:2px 6px; border-radius:0 4px 4px 0; }
  .prop-summary { font-size:.88rem; color:#cbd5e1; }
  .prop-summary b { color:#f1f5f9; }
</style>
"""
