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

# 2 adds per-market `offerable`/`suppress_reason`, plus `n_sims` on each stat
# block. Additive only, so a v1 reader still works, but a reader that ignores
# `offerable` will release markets we have flagged as unpriceable.
SCHEMA_VERSION = 2

# Publishing guards. Duplicated from projection_engine_v3 rather than imported so
# this module stays numpy+stdlib only (see the module docstring); the drift risk is
# covered by a test that asserts the two definitions agree.
#
# The ladders above are generous by design -- 22 thresholds for saves, 26 for
# faceoff wins -- and the top of a long ladder is estimated from a handful of the
# 20,000 sims. Publishing that turns sampling noise into a four-figure payout. In
# 2026 a goalie projected at 0.000 saves was offered Over 0.5 at +8,554 and
# recorded 11. Anything failing these checks is exported with offerable=false and
# a reason, so the BOSS Tool can still show the model's view but must not release
# the market.
MIN_SIM_SUPPORT = 50
MIN_PRICE_PROB = 0.02


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

    Deliberately UNCLAMPED: this is the model's raw view and the object the BOSS
    Tool re-derives from, so it must stay internally consistent and monotonic.
    The MIN_PRICE_PROB clamp is applied where prices are formed, not here.
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

def _resolve_hold(stat_key: str, hold_pct: float,
                  hold_by_stat: Optional[Dict[str, float]]) -> float:
    """Per-stat hold with fallback to the global hold_pct. Mirrors
    PricingEngine._hold_for so BOSS exports match the app exactly."""
    if hold_by_stat and stat_key in hold_by_stat:
        return hold_by_stat[stat_key]
    return hold_pct


def offerability(prob: float, n_sims: int) -> tuple[bool, str]:
    """Whether a fair probability rests on enough simulated support to publish.

    Checks both tails: ``prob * n_sims`` sims cleared the threshold and
    ``(1 - prob) * n_sims`` did not, and each side needs ``MIN_SIM_SUPPORT``
    before the estimate means anything. Returns the reason alongside the verdict
    so a suppressed market can be explained rather than silently dropped.
    """
    if n_sims <= 0:
        return False, "no simulated distribution"
    over = int(round(prob * n_sims))
    under = n_sims - over
    if over < MIN_SIM_SUPPORT:
        return False, f"insufficient over support ({over} sims)"
    if under < MIN_SIM_SUPPORT:
        return False, f"insufficient under support ({under} sims)"
    return True, ""


def _stat_block(dist: np.ndarray, stat_key: str, proj: float, hold_pct: float,
                hold_by_stat: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    hold_pct = _resolve_hold(stat_key, hold_pct, hold_by_stat)
    max_k = MILESTONE_MAX.get(stat_key, 6)
    ladder = ge_probability_ladder(dist, max_k)
    n_sims = int(np.count_nonzero(np.isfinite(np.asarray(dist, dtype=float))))
    # A projection of exactly zero means the model has no playing-time signal for
    # this player, not that the event is impossible; nothing built on it is
    # publishable at any price.
    no_signal = float(np.mean(np.asarray(dist, dtype=float))) <= 0.0 if n_sims else True

    # O/U at the model's balanced line
    line = best_ou_line(ladder, max_k)
    p_over = ou_from_ladder(ladder, line)
    ou_ok, ou_reason = offerability(p_over, n_sims)
    if no_signal:
        ou_ok, ou_reason = False, "no playing-time signal (projection is zero)"
    # Clamp into the band we can actually estimate, so a thin tail cannot produce
    # a five-figure price even on a market that clears the support checks.
    p_over = min(max(p_over, MIN_PRICE_PROB), 1.0 - MIN_PRICE_PROB)
    p_under = 1.0 - p_over
    o_adj, u_adj = apply_hold(p_over, p_under, hold_pct)

    # X+ milestone ladder — each threshold priced two-way (Yes/No) with hold
    milestones = []
    for k in range(1, max_k + 1):
        p_yes = float(ladder[k - 1])
        ok, reason = offerability(p_yes, n_sims)
        if no_signal:
            ok, reason = False, "no playing-time signal (projection is zero)"
        p_yes = min(max(p_yes, MIN_PRICE_PROB), 1.0 - MIN_PRICE_PROB)
        p_no = 1.0 - p_yes
        y_adj, n_adj = apply_hold(p_yes, p_no, hold_pct)
        milestones.append({
            "threshold": k,
            "label": f"{k}+",
            "fair_prob": round(p_yes, 5),
            "yes_odds": american_from_prob(y_adj),
            "no_odds": american_from_prob(n_adj),
            "offerable": ok,
            "suppress_reason": reason,
        })

    return {
        "proj": round(float(proj), 3),
        "n_sims": n_sims,
        "ge_probs": [round(p, 5) for p in ladder],   # P(X>=1..max_k) — used for re-derivation
        "ou": {
            "line": line,
            "fair_over_prob": round(p_over, 5),
            "over_odds": american_from_prob(o_adj),
            "under_odds": american_from_prob(u_adj),
            "offerable": ou_ok,
            "suppress_reason": ou_reason,
        },
        "milestones": milestones,
    }


def build_export(result: Any, hold_pct: float = 0.045,
                 game_meta: Optional[Dict[str, Any]] = None,
                 hold_by_stat: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
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
            stats_out[stat_key] = _stat_block(np.asarray(dist), stat_key, proj_val,
                                              hold_pct, hold_by_stat)
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
        "hold_by_stat": dict(hold_by_stat) if hold_by_stat else {},
        "stat_labels": EXPORT_STATS,
    }
    if game_meta:
        meta.update(game_meta)

    return {"meta": meta, "players": players_out}


def export_json(result: Any, hold_pct: float = 0.045,
                game_meta: Optional[Dict[str, Any]] = None, indent: int = 2,
                hold_by_stat: Optional[Dict[str, float]] = None) -> str:
    """Return the export as a JSON string (for st.download_button)."""
    return json.dumps(build_export(result, hold_pct, game_meta, hold_by_stat), indent=indent)


def suggest_filename(result: Any, game_meta: Optional[Dict[str, Any]] = None) -> str:
    home = result.home_proj.team_id
    away = result.away_proj.team_id
    date = ""
    if game_meta and game_meta.get("game_date"):
        date = "_" + str(game_meta["game_date"])[:10]
    return f"boss_{away}_at_{home}{date}.json"
