"""
live_pricing.py — price the live distribution and compute edge vs a book line.

Given a live full-game distribution for a stat (from live_model) and a book's
offered line + American odds (entered manually to start; BOSS-fed later), this:

  1. prices the model's FAIR two-way at that exact line (reusing the engine's
     PricingEngine so holds/format match the pregame app), and
  2. computes the EXPECTED VALUE of betting each side at the odds the book is
     offering, using the model's fair probability.

EV is the honest edge signal for live trading:

    EV_over  = p_model_over  * profit_per_unit(over_odds)  - (1 - p_model_over)

A positive EV side is +EV to bet; the magnitude is the edge per unit staked.
We also report the model's fair American odds and the no-vig book probability so
the trader can see the model line, the book line, and the gap at a glance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# American-odds helpers (self-contained; mirror PricingEngine._am inverse)
# --------------------------------------------------------------------------
def american_to_prob(odds) -> float:
    """American odds -> implied probability (includes the book's vig)."""
    o = float(str(odds).replace("+", ""))
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)


def american_to_profit(odds) -> float:
    """Profit per 1 unit staked if the bet wins (excludes returned stake)."""
    o = float(str(odds).replace("+", ""))
    if o < 0:
        return 100.0 / (-o)
    return o / 100.0


def prob_to_american(prob: float) -> str:
    """Fair probability -> American odds string (no vig)."""
    prob = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    if prob >= 0.5:
        return str(int(-round((prob / (1.0 - prob)) * 100)))
    return "+" + str(int(round(((1.0 - prob) / prob) * 100)))


def devig_two_way(over_odds, under_odds) -> tuple[float, float]:
    """Remove the vig from a two-way market -> (fair_over, fair_under)."""
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    tot = po + pu
    if tot <= 0:
        return 0.5, 0.5
    return po / tot, pu / tot


@dataclass
class EdgeQuote:
    """Model vs book comparison for one player-stat at one line."""
    player_id: str
    player_name: str
    stat: str
    line: float
    # model
    model_prob_over: float
    model_prob_under: float
    model_fair_over: str
    model_fair_under: str
    banked: float            # certain amount already accrued for this stat
    proj_full: float         # model mean full-game total
    is_settled: bool = False # True if the outcome at this line is already decided
                             # (no remaining variance — e.g. banked already clears
                             # the line, or an unprojected player with only banked)
    # book (manually entered; may be partial)
    book_over_odds: Optional[str] = None
    book_under_odds: Optional[str] = None
    book_fair_over: Optional[float] = None   # de-vigged book prob (over)
    # edge (EV per 1u staked; None if that side wasn't entered)
    ev_over: Optional[float] = None
    ev_under: Optional[float] = None

    @property
    def best_side(self) -> Optional[str]:
        evs = [(s, e) for s, e in (("Over", self.ev_over), ("Under", self.ev_under))
               if e is not None]
        if not evs:
            return None
        s, e = max(evs, key=lambda x: x[1])
        return s if e > 0 else None

    @property
    def best_ev(self) -> Optional[float]:
        evs = [e for e in (self.ev_over, self.ev_under) if e is not None]
        return max(evs) if evs else None


def prob_over(dist: np.ndarray, line: float) -> float:
    """Model fair probability the stat finishes strictly OVER the line."""
    arr = np.asarray(dist, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    return float(np.mean(arr > line))


def quote_edge(player_id: str, player_name: str, stat: str,
               dist: np.ndarray, line: float,
               banked: float = 0.0,
               book_over_odds: Optional[str] = None,
               book_under_odds: Optional[str] = None) -> EdgeQuote:
    """Build an EdgeQuote for one player-stat at a given line.

    dist         : the live full-game distribution (banked + rest-of-game).
    line         : the market line to price at (x.5). If the banked amount alone
                   already clears the line, the over is essentially locked and
                   model_prob_over -> ~1.0 automatically (the dist is banked+rest).
    book_*_odds  : the book's offered American odds. Pass whichever side(s) you're
                   quoting; EV is computed only for the side(s) provided.
    """
    arr = np.asarray(dist, dtype=float)
    arr = arr[np.isfinite(arr)]
    p_over = prob_over(dist, line)
    p_under = 1.0 - p_over
    proj_full = float(np.mean(arr)) if arr.size else 0.0
    # Settled = no remaining variance in the distribution (all sims identical),
    # i.e. the outcome at every line is already decided. Common for a banked-only
    # call-up, or any player whose banked total already sits far past the line.
    is_settled = bool(arr.size and float(np.std(arr)) < 1e-9)

    q = EdgeQuote(
        player_id=str(player_id), player_name=player_name, stat=stat, line=float(line),
        model_prob_over=round(p_over, 4), model_prob_under=round(p_under, 4),
        model_fair_over=("LOCKED" if is_settled else prob_to_american(p_over)),
        model_fair_under=("LOCKED" if is_settled else prob_to_american(p_under)),
        banked=float(banked), proj_full=round(proj_full, 3), is_settled=is_settled,
        book_over_odds=book_over_odds, book_under_odds=book_under_odds,
    )
    if book_over_odds is not None:
        q.ev_over = round(p_over * american_to_profit(book_over_odds) - (1.0 - p_over), 4)
    if book_under_odds is not None:
        q.ev_under = round(p_under * american_to_profit(book_under_odds) - (1.0 - p_under), 4)
    if book_over_odds is not None and book_under_odds is not None:
        fo, _ = devig_two_way(book_over_odds, book_under_odds)
        q.book_fair_over = round(fo, 4)
    return q


# --------------------------------------------------------------------------
# Convenience: price a whole LiveProjection for the traded markets
# --------------------------------------------------------------------------
# The stats the user trades. Points is the headline; goals/assists next.
TRADED_STATS = ("points", "goals", "assists", "shots_on_goal", "saves", "faceoff_wins")


def quote_player(ps, stat: str, banked_by_player: dict, line: Optional[float] = None,
                 book_over_odds: Optional[str] = None,
                 book_under_odds: Optional[str] = None) -> Optional[EdgeQuote]:
    """Quote one stat for one PlayerSimulation. Line defaults to the model's own
    balanced x.5 prop line when not supplied."""
    if stat not in ps.stat_distributions:
        return None
    dist = ps.stat_distributions[stat]
    if line is None:
        line = ps.prop_lines.get(stat, _default_line(dist))
    banked = float(banked_by_player.get(str(ps.player_id), {}).get(
        stat if stat != "points" else "points", 0.0))
    return quote_edge(ps.player_id, ps.full_name, stat, dist, float(line),
                      banked=banked, book_over_odds=book_over_odds,
                      book_under_odds=book_under_odds)


def _default_line(dist: np.ndarray) -> float:
    arr = np.asarray(dist, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    return float(np.floor(np.median(arr)) + 0.5)


if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from live_feed import LiveFeed
    from live_state import reconstruct
    from live_model import LiveModel
    from projection_engine_v3 import ProjectionEngine

    ap = argparse.ArgumentParser(description="Live model prices + fair odds for a game.")
    ap.add_argument("slug")
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--date")
    ap.add_argument("--stat", default="points", choices=list(TRADED_STATS))
    ap.add_argument("--pace-weight", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    st = LiveFeed(args.slug).poll()
    banked = reconstruct(st.events)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.getenv("PLL_DB_PATH",
                   os.path.join(root, "data", "analytics_database", "pll_warehouse.duckdb"))
    eng = ProjectionEngine(db_path=db); eng.load(); eng.fit(run_backtest=False)
    result = eng.project(home_team_id=args.home, away_team_id=args.away, game_date=args.date)
    live = LiveModel(eng).resimulate(
        result, banked.by_player, st.fraction_remaining, pace_weight=args.pace_weight,
        home_team_id=args.home, away_team_id=args.away,
        team_of=banked.team_of, events=st.events)

    quotes = []
    for ps in live.all_sims():
        q = quote_player(ps, args.stat, banked.by_player)
        if q and q.proj_full >= 0.3:
            quotes.append(q)
    quotes.sort(key=lambda q: q.proj_full, reverse=True)

    print(f"\n{args.slug}: P{st.period} {st.clock_minutes:02d}:{st.clock_seconds:02d} "
          f"| {st.away_score}-{st.home_score} | {st.fraction_remaining:.0%} remaining "
          f"| {args.stat.upper()}\n")
    print(f"{'player':22} {'line':>5} {'proj':>6} {'bank':>5} {'P(over)':>8} "
          f"{'fairO':>7} {'fairU':>7}")
    for q in quotes[:args.top]:
        print(f"{q.player_name[:22]:22} {q.line:5.1f} {q.proj_full:6.2f} {q.banked:5.0f} "
              f"{q.model_prob_over:8.3f} {q.model_fair_over:>7} {q.model_fair_under:>7}")
