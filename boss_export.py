"""
BOSS export — bridge between the PLL Projections model and the PLL BOSS Tool.

Produces a single JSON file per game containing, for every player and every
stat we offer:
  * the O/U line and its fair over/under American odds
  * the X+ milestone ladder (1+, 2+, 3+, ...) with fair American odds
  * the model's fair probability at each integer threshold (the "probability
    ladder"), which is what lets the BOSS Tool re-derive a consistent set of
    prices when the user adjusts an O/U or milestone by hand.

Design notes
------------
The BOSS Tool must be able to take a user's manual odds edit on one line and
flow it through to every other line for that player/stat so O/U and X+ never
disagree (Over 1.5 == 2+ is the *same* event). To do that without shipping
raw simulation arrays, we export the fair probability that the stat is >= k
for each integer k (`ge_probs`). From that ladder every O/U and X+ price is a
pure function, so the BOSS Tool can recompute the whole set after an edit by
shifting the ladder — see `boss_pricing.py` in the BOSS Tool.

This module is intentionally dependency-light (numpy + stdlib) so the exact
same file can be dropped into the BOSS Tool repo if desired.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Stats we offer as player props, mapped to BOSS-facing labels. Order matters
# only for display. Keys are the model's internal stat keys.
EXPORT_STATS: Dict[str, str] = {
    "goals":         "Goals",
    "assists":       "Assists",
    "points":        "Points",
    "shots_on_goal": "Shots on Goal",
    "two_pt_goals":  "2-Point Goals",
    "saves":         "Saves",
    "faceoff_wins":  "Faceoff Wins",
    "ground_balls":  "Ground Balls",
}

# How many integer thresholds of the X+ ladder to publish per stat. We publish
# a generous ladder; the BOSS Tool decides which to actually release.
MILESTONE_MAX: Dict[str, int] = {
    "goals":         6,
    "assists":       5,
    "points":        8,
    "shots_on_goal": 8,
    "two_pt_goals":  3,
    "saves":         22,
    "faceoff_wins":  26,
    "ground_balls":  10,
}

SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Odds math (kept identical to PricingEngine so exported odds match the app)
# ─────────────────────────────────────────────────────────────────────────────

def american_from_prob(prob: float) -> str:
    """Convert a *priced* probability (already includes hold) to American odds."""
    prob = min(max(prob, 0.001), 0.999)
    if prob >= 0.50:
        return str(int(-round((prob / (1.0 - prob)) * 100)))
    return "+" + str(int(round(((1.0 - prob) / prob) * 100)))


def apply_hold(p_over: float, p_under: float, hold_pct: float) -> tuple[float, float]:
    """Distribute a two-way market's hold proportionally (matches PricingEngine._hold)."""
    p_over = max(p_over, 1e-4)
    p_under = max(p_under, 1e-4)
    total = p_over + p_under
    if total <= 0:
        h = hold_pct / 2
        return 0.50 + h, 0.50 + h
    t = 1.0 + hold_pct
    return (p_over / total) * t, (p_under / total) * t


def _nearest_half(v: float) -> float:
    return float(np.floor(v) + 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Probability ladder
# ─────────────────────────────────────────────────────────────────────────────

def ge_probability_ladder(dist: np.ndarray, max_k: int) -> List[float]:
    """Return [P(X>=1), P(X>=2), ..., P(X>=max_k)] from a simulated distribution.

    This is the canonical object the BOSS Tool uses: every O/U and X+ price is
    derived from it, so recomputing after a manual edit keeps all lines mutually
    consistent (no arbitrage between an O/U over and its equivalent X+).
    """
    arr = np.asarray(dist, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0] * max_k
    n = arr.size
    return [float(np.count_nonzero(arr >= k) / n) for k in range(1, max_k + 1)]


def ou_from_ladder(ladder: Sequence[float], line: float) -> float:
    """Fair P(Over `line`) for a .5 line, read straight off the ge-ladder.

    Over k.5  ==  X >= k+1  ==  ladder[k]  (0-indexed: ge_probs[k] = P(X>=k+1)).
    """
    k = int(round(line - 0.5))  # Over k.5 -> need X >= k+1 -> ladder index k
    if k < 0:
        return 1.0
    if k >= len(ladder):
        return 0.0
    return float(ladder[k])


def best_ou_line(ladder: Sequence[float], max_k: int) -> float:
    """Pick the .5 line whose fair P(Over) is closest to 0.50 (matches _opt_line)."""
    best_line, best_d = 0.5, 2.0
    for k in range(0, max_k):
        line = k + 0.5
        p = ou_from_ladder(ladder, line)
        d = abs(p - 0.50)
        if d < best_d:
            best_d, best_line = d, line
    return best_line


# ─────────────────────────────────────────────────────────────────────────────
# Build export
# ─────────────────────────────────────────────────────────────────────────────

def _stat_block(dist: np.ndarray, stat_key: str, proj: float, hold_pct: float) -> Dict[str, Any]:
    max_k = MILESTONE_MAX.get(stat_key, 6)
    ladder = ge_probability_ladder(dist, max_k)

    # O/U at the model's balanced line
    line = best_ou_line(ladder, max_k)
    p_over = ou_from_ladder(ladder, line)
    p_under = 1.0 - p_over
    o_adj, u_adj = apply_hold(p_over, p_under, hold_pct)

    # X+ milestone ladder — each threshold priced two-way (Yes/No) with hold
    milestones = []
    for k in range(1, max_k + 1):
        p_yes = float(ladder[k - 1])
        p_no = 1.0 - p_yes
        y_adj, n_adj = apply_hold(p_yes, p_no, hold_pct)
        milestones.append({
            "threshold": k,
            "label": f"{k}+",
            "fair_prob": round(p_yes, 5),
            "yes_odds": american_from_prob(y_adj),
            "no_odds": american_from_prob(n_adj),
        })

    return {
        "proj": round(float(proj), 3),
        "ge_probs": [round(p, 5) for p in ladder],   # P(X>=1..max_k) — used for re-derivation
        "ou": {
            "line": line,
            "fair_over_prob": round(p_over, 5),
            "over_odds": american_from_prob(o_adj),
            "under_odds": american_from_prob(u_adj),
        },
        "milestones": milestones,
    }


def build_export(result: Any, hold_pct: float = 0.045,
                 game_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the full BOSS export dict for one game.

    `result` is a ProjectionResult (from run_game / engine.project). `game_meta`
    may carry fixture info the model doesn't know (BOSS fixture_id, date, etc.)
    which the user can also fill in on the BOSS Tool side.
    """
    sims_by_id = {ps.player_id: ps for ps in (result.home_player_sims + result.away_player_sims)}
    projs_by_id = {p.player_id: p for p in (result.home_players + result.away_players)}

    players_out: List[Dict[str, Any]] = []
    for pid, ps in sims_by_id.items():
        proj = projs_by_id.get(pid)
        if proj is None or not getattr(proj, "active", True):
            continue
        stats_out: Dict[str, Any] = {}
        for stat_key in EXPORT_STATS:
            dist = ps.stat_distributions.get(stat_key)
            if dist is None:
                continue
            proj_val = ps.proj_values.get(stat_key, 0.0)
            # Skip stats with no real projection (keeps the file lean & relevant)
            if proj_val is None or float(proj_val) < 0.05:
                continue
            stats_out[stat_key] = _stat_block(np.asarray(dist), stat_key, proj_val, hold_pct)
        if not stats_out:
            continue
        players_out.append({
            "player_id": pid,
            "full_name": ps.full_name,
            "team_id": proj.team_id,
            "position": proj.position,
            "stats": stats_out,
        })

    players_out.sort(key=lambda p: p["full_name"])

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": getattr(result, "generated_at", "")
                        or datetime.now(timezone.utc).isoformat(),
        "home_team": result.home_proj.team_id,
        "away_team": result.away_proj.team_id,
        "game_id": getattr(result, "game_id", ""),
        "hold_pct": hold_pct,
        "stat_labels": EXPORT_STATS,
    }
    if game_meta:
        meta.update(game_meta)

    return {"meta": meta, "players": players_out}


def export_json(result: Any, hold_pct: float = 0.045,
                game_meta: Optional[Dict[str, Any]] = None, indent: int = 2) -> str:
    """Return the export as a JSON string (for st.download_button)."""
    return json.dumps(build_export(result, hold_pct, game_meta), indent=indent)


def suggest_filename(result: Any, game_meta: Optional[Dict[str, Any]] = None) -> str:
    home = result.home_proj.team_id
    away = result.away_proj.team_id
    date = ""
    if game_meta and game_meta.get("game_date"):
        date = "_" + str(game_meta["game_date"])[:10]
    return f"boss_{away}_at_{home}{date}.json"
