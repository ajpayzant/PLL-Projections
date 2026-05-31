"""
Shared engine state for all projection app pages.
Uses st.cache_resource so the engine is loaded once per session.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Add repo root to path so projection_engine_v3 is importable from pages/
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projection_engine_v3 import (
    ProjectionEngine,
    ProjectionResult,
    TeamProjection,
    PlayerProjection,
    PricingEngine,
    _norm_pos,
)

DB_PATH = os.getenv(
    "PLL_DB_PATH",
    str(_ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"),
)

# ── Engine cache ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading projection engine…")
def get_engine() -> ProjectionEngine:
    engine = ProjectionEngine(db_path=DB_PATH)
    engine.load()
    engine.fit(run_backtest=False)
    return engine


# ── Session-state helpers ─────────────────────────────────────────────────

def init_session():
    """Initialise all session-state keys used across pages."""
    defaults = {
        "selected_game": None,          # dict from engine.upcoming_games()
        "last_result": None,            # ProjectionResult
        "depth_charts": {},             # {team_id: {player_id: {active, usage, is_starter}}}
        "team_adjustments": {},         # {team_id: {off_mult, def_mult_opp}}
        "line_overrides": {},           # {market_key: float}  e.g. {"total": 23.5}
        "hold_pct": 0.045,
        "n_sims": 20_000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_depth_chart(team_id: str) -> Dict:
    if team_id not in st.session_state.depth_charts:
        st.session_state.depth_charts[team_id] = {}
    return st.session_state.depth_charts[team_id]


def set_player_override(team_id: str, player_id: str, key: str, value):
    dc = get_depth_chart(team_id)
    if player_id not in dc:
        dc[player_id] = {}
    dc[player_id][key] = value


def build_overrides() -> Dict:
    """Merge depth charts into {player_id: {active, usage_multiplier, is_starter}} dict."""
    merged = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            merged[pid] = settings
    return merged


def build_active_players() -> Dict:
    out = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            if "active" in settings:
                out[pid] = settings["active"]
    return out


# ── Formatting helpers ────────────────────────────────────────────────────

TEAM_COLORS = {
    "ATL": "#1d4ed8",   # Atlas blue
    "OUT": "#d97706",   # Outlaws gold
    "CAN": "#dc2626",   # Cannons red
    "RED": "#16a34a",   # Redwoods green
    "WAT": "#7c3aed",   # Waterdogs purple
    "WHP": "#0891b2",   # Whipsnakes teal
    "CHA": "#0f172a",   # Chaos black
    "ARC": "#b45309",   # Archers brown
}

TEAM_NAMES = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
}


def team_color(team_id: str) -> str:
    return TEAM_COLORS.get(str(team_id).upper(), "#475569")


def team_name(team_id: str) -> str:
    return TEAM_NAMES.get(str(team_id).upper(), team_id)


def fmt_odds(odds_str: str) -> str:
    """Return coloured odds HTML."""
    try:
        v = int(odds_str.replace("+", ""))
        if v < 0:
            cls = "odds-fav"
        elif v > 0:
            cls = "odds-dog"
        else:
            cls = "odds-even"
    except Exception:
        cls = "odds-even"
    return f'<span class="{cls}">{odds_str}</span>'


def fmt_prob(p: float) -> str:
    return f"{p * 100:.1f}%"


def fmt_goals(v: float) -> str:
    return f"{v:.1f}"


def card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="pll-card-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="pll-card">
      <div class="pll-card-label">{label}</div>
      <div class="pll-card-value">{value}</div>
      {sub_html}
    </div>
    """


def pos_badge(pos: str) -> str:
    colors = {
        "A": "#1d4ed8", "M": "#059669", "D": "#dc2626",
        "FO": "#d97706", "SSDM": "#7c3aed", "LSM": "#0891b2", "G": "#475569",
    }
    c = colors.get(pos, "#475569")
    return f'<span style="background:{c};color:#fff;border-radius:4px;padding:1px 6px;font-size:0.75rem;font-weight:700;">{pos}</span>'
