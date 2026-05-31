"""
Shared engine state for all projection app pages.
Loaded once per session via st.cache_resource.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ── Path setup — must resolve both root AND pages/ dir ───────────────────
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
)

# ── DB path ───────────────────────────────────────────────────────────────
DB_PATH = os.getenv(
    "PLL_DB_PATH",
    str(_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"),
)


# ── Bootstrap: rebuild DuckDB from parquets if the binary is missing ─────
def _ensure_db() -> None:
    """
    Called once at import time. If the .duckdb file is absent (fresh Streamlit
    Cloud deploy, new clone), runs bootstrap_db.py to rebuild it from the
    committed parquet files in data/curated_data/all_requested_seasons/.
    Takes ~5-10 seconds and only runs once per deployment.
    """
    if Path(DB_PATH).exists():
        return

    bootstrap = _ROOT / "scripts" / "bootstrap_db.py"
    if not bootstrap.exists():
        st.error(
            "Database not found and bootstrap script is missing. "
            "Make sure scripts/bootstrap_db.py is in the repository."
        )
        st.stop()

    with st.spinner("Building database from data files — takes about 10 seconds on first load…"):
        result = subprocess.run(
            [sys.executable, str(bootstrap)],
            capture_output=True, text=True,
        )

    if result.returncode != 0:
        st.error(
            f"Database bootstrap failed.\n\n"
            f"**Error:**\n```\n{result.stderr[-2000:]}\n```\n\n"
            "Check that the GitHub Action has run successfully and the "
            "data/curated_data/ parquet files exist in the repository."
        )
        st.stop()


_ensure_db()


# ── Engine cache ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading projection engine…")
def get_engine() -> ProjectionEngine:
    engine = ProjectionEngine(db_path=DB_PATH)
    engine.load()
    engine.fit(run_backtest=False)
    return engine


# ── Session-state helpers ─────────────────────────────────────────────────
def init_session() -> None:
    defaults = {
        "selected_game":    None,
        "last_result":      None,
        "depth_charts":     {},
        "team_adjustments": {},
        "line_overrides":   {},
        "hold_pct":         0.045,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_depth_chart(team_id: str) -> Dict:
    if team_id not in st.session_state.depth_charts:
        st.session_state.depth_charts[team_id] = {}
    return st.session_state.depth_charts[team_id]


def set_player_override(team_id: str, player_id: str, key: str, value) -> None:
    dc = get_depth_chart(team_id)
    if player_id not in dc:
        dc[player_id] = {}
    dc[player_id][key] = value


def build_overrides() -> Dict:
    merged: Dict = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            merged[pid] = settings
    return merged


def build_active_players() -> Dict:
    out: Dict = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            if "active" in settings:
                out[pid] = settings["active"]
    return out


# ── Formatting ────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "ATL": "#1d4ed8", "OUT": "#d97706", "CAN": "#dc2626", "RED": "#16a34a",
    "WAT": "#7c3aed", "WHP": "#0891b2", "CHA": "#334155", "ARC": "#b45309",
}
TEAM_NAMES = {
    "ATL": "Atlas",  "OUT": "Outlaws", "CAN": "Cannons",    "RED": "Redwoods",
    "WAT": "Waterdogs", "WHP": "Whipsnakes", "CHA": "Chaos", "ARC": "Archers",
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

# Shared CSS block injected by each page
SHARED_CSS = """
<style>
  .main .block-container { padding-top:1rem; max-width:1800px; }
  .pll-card {
    border:1px solid rgba(148,163,184,.20); border-radius:12px; padding:12px 16px;
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
</style>
"""
