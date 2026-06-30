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
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM clean.team_game_stats").fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False


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
        result = subprocess.run(
            [sys.executable, str(bootstrap), "--force"],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        st.error(
            f"Database bootstrap failed.\n\n```\n{result.stderr[-2000:]}\n```\n\n"
            "Run the GitHub Action (Update PLL Data Warehouse) to populate data/."
        )
        st.stop()


_ensure_db()


# -- Engine cache ----------------------------------------------------------
@st.cache_resource(show_spinner="Loading projection engine…")
def get_engine() -> ProjectionEngine:
    engine = ProjectionEngine(db_path=DB_PATH)
    engine.load()
    # run_backtest=False keeps startup fast (~2-3s).
    # The calibrator is fitted lazily when the Model Performance page is visited.
    engine.fit(run_backtest=False)
    return engine


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
        "label": "Scoring rate (goals/game)",
        "help": "Team's recent avg goals/game. League avg ~11.2. Raise if offense is hot; lower if key scorer is out.",
        "min": 5.0, "max": 20.0, "step": 0.1, "fmt": "{:.1f}",
    },
    "shot_pct_ewm": {
        "label": "Shooting efficiency (goals/shot)",
        "help": "Fraction of shots that become goals. League avg ~0.274.",
        "min": 0.15, "max": 0.45, "step": 0.005, "fmt": "{:.3f}",
    },
    "shots_ewm": {
        "label": "Shot volume (shots/game)",
        "help": "Shots per game. League avg ~41.",
        "min": 25.0, "max": 60.0, "step": 0.5, "fmt": "{:.1f}",
    },
    "assists_ewm": {
        "label": "Assists/game",
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
        "label": "Goalie save% (saves / shots faced)",
        "help": "Starting goalie's Bayesian save%. League avg ~0.537.",
        "min": 0.35, "max": 0.75, "step": 0.005, "fmt": "{:.3f}",
    },
    "goals_against_ewm": {
        "label": "Goals allowed/game (defense)",
        "help": "Goals allowed per game. LOWER = better defense. League avg ~11.2.",
        "min": 5.0, "max": 20.0, "step": 0.1, "fmt": "{:.1f}",
    },
}

PLAYER_RATING_DEFS = {
    # ── Volume shares — what fraction of team totals does this player produce ──
    "share_goals_ewm": {
        "label": "Goal share",
        "help": "Player's share of team goals (0.20 = 20% of all team goals). Overriding this pins the player to the exact fraction you set, bypassing the credibility blend.",
        "min": 0.0, "max": 0.50, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    "share_assists_ewm": {
        "label": "Assist share",
        "help": "Player's share of team assists. Overriding disables the assist-conv nudge for this player.",
        "min": 0.0, "max": 0.50, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    "share_shots_ewm": {
        "label": "Shot share",
        "help": "Player's share of team shots (0.18 = 18% of all team shots). Overriding disables the shots-per-touch nudge for this player.",
        "min": 0.0, "max": 0.50, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "D", "FO", "SSDM", "LSM"],
    },
    # ── Efficiency ratings — quality of production per opportunity ──────────
    "shot_pct_ewm": {
        "label": "Shooting %",
        "help": "Goals per shot. League avg ~0.27. Raising this scales up proj goals relative to the baseline.",
        "min": 0.05, "max": 0.60, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "sog_rate_ewm": {
        "label": "Shots on goal %",
        "help": "Fraction of shots that are on goal. League avg ~0.63. Affects proj SOG without changing goal projection.",
        "min": 0.20, "max": 1.00, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "shots_per_touch_ewm": {
        "label": "Shots per touch",
        "help": "How often a player generates a shot from each touch. League avg ~0.21 (A/M). Nudges shot projection up/down vs position average (20% blend, ±30% cap).",
        "min": 0.02, "max": 0.60, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "assist_conv_ewm": {
        "label": "Assist conversion",
        "help": "Assists per assist opportunity — how often a pass in a scoring situation becomes a real assist. League avg ~0.31 (A), 0.25 (M). Nudges assist projection up/down (15% blend, ±20% cap).",
        "min": 0.00, "max": 1.00, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    "two_pt_rate_ewm": {
        "label": "2PT goal rate",
        "help": "Fraction of goals that are 2-pointers. League avg ~7%.",
        "min": 0.0, "max": 0.65, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["A", "M", "SSDM", "LSM"],
    },
    # ── Position-specific ratings ────────────────────────────────────────────
    "bayes_save_pct": {
        "label": "Save %",
        "help": "Goalie's Bayesian save%. League avg ~0.537.",
        "min": 0.35, "max": 0.75, "step": 0.005, "fmt": "{:.3f}",
        "positions": ["G"],
    },
    "bayes_fo_pct": {
        "label": "FO win %",
        "help": "Faceoff win rate. 0.500 = league avg.",
        "min": 0.25, "max": 0.75, "step": 0.01, "fmt": "{:.3f}",
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
