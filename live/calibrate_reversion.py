"""
calibrate_reversion.py — sweep the in-game LEAD-PERSISTENCE curve.

The pace_weight sweep (calibrate_live.py) showed the pace blend adds early
odds-swing with no margin-accuracy gain. The remaining early-swing lever is how
much of the CURRENT banked lead we credit toward the final margin — the
`_lead_persistence(frac_rem) = max(1 - slope*frac_rem, floor)` curve in
live_model. This harness holds pace_weight fixed (low) and sweeps (slope, floor)
to find the curve that BEST predicts the final margin — and reports the early
win-prob move alongside, so we can see whether damping the early lead is free
(accuracy holds or improves) or costs accuracy.

Run:
    python live/calibrate_reversion.py
    python live/calibrate_reversion.py --games 40 --pace-weight 0.15
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import duckdb  # noqa: E402

import live_model  # noqa: E402  (patched per-candidate)
from live_state import reconstruct  # noqa: E402
from live_model import LiveModel  # noqa: E402
from projection_engine_v3 import ProjectionEngine  # noqa: E402
from calibrate_live import (  # noqa: E402
    REGULATION_SECONDS, SNAPSHOTS, _event_to_feed, load_games,
)

# (slope, floor) candidates. slope=0.4/floor=0.55 is the shipped curve.
# Higher slope + lower floor = credit LESS of an early lead (more reversion).
CANDIDATES: List[Tuple[float, float]] = [
    (0.0, 1.0),    # no reversion at all (banked lead fully persists) — baseline sanity
    (0.4, 0.55),   # SHIPPED
    (0.6, 0.45),
    (0.8, 0.35),
    (1.0, 0.25),
]


def replay(games: List[dict], engine: ProjectionEngine,
           pace_weight: float, candidates: List[Tuple[float, float]]) -> Dict:
    acc = defaultdict(lambda: {"margin_ae": [], "wp_move": [], "margin_move": [],
                               "n": 0})
    pregame_cache: Dict[tuple, object] = {}

    for gi, game in enumerate(games):
        key = (game["home"], game["away"], game["season"])
        if key not in pregame_cache:
            try:
                pregame_cache[key] = engine.project(
                    home_team_id=game["home"], away_team_id=game["away"],
                    game_date=None)
            except Exception:
                pregame_cache[key] = None
        pregame = pregame_cache[key]
        if pregame is None:
            continue

        try:
            pre_gm = engine.pricing.price_game(pregame.game_sim)
            pre_home_wp = float(pre_gm.home_win_prob)
            pre_margin = float(np.median(pregame.game_sim.home_scores
                                         - pregame.game_sim.away_scores))
        except Exception:
            pre_home_wp, pre_margin = 0.5, 0.0

        final_margin = game["final_home"] - game["final_away"]
        rows = game["rows"]
        model = LiveModel(engine, n_sims=4000)

        for frac_el in SNAPSHOTS:
            cutoff = frac_el * REGULATION_SECONDS
            upto = [r for r in rows if (r.get("elapsed_seconds") or 0) <= cutoff]
            if len(upto) < 3:
                continue
            feed_events = [_event_to_feed(r) for r in upto]
            banked = reconstruct(feed_events)
            frac_rem = max(0.0, 1.0 - frac_el)

            for (slope, floor) in candidates:
                # patch the module-level constants the live_game_sim reads
                live_model._REVERSION_SLOPE = slope
                live_model._REVERSION_FLOOR = floor
                try:
                    live = model.resimulate(
                        pregame, banked.by_player, frac_rem,
                        pace_weight=pace_weight,
                        home_team_id=game["home"], away_team_id=game["away"],
                        team_of=banked.team_of, events=feed_events)
                    gm = engine.pricing.price_game(live.game_sim)
                except Exception:
                    continue
                live_margin = float(np.median(live.game_sim.home_scores
                                              - live.game_sim.away_scores))
                live_home_wp = float(gm.home_win_prob)

                b = acc[(slope, floor, round(frac_el, 2))]
                b["margin_ae"].append(abs(live_margin - final_margin))
                b["wp_move"].append(abs(live_home_wp - pre_home_wp))
                b["margin_move"].append(abs(live_margin - pre_margin))
                b["n"] += 1

        if (gi + 1) % 20 == 0:
            print(f"  ...replayed {gi + 1}/{len(games)} games", flush=True)

    return acc


def summarize(acc: Dict, candidates: List[Tuple[float, float]]) -> None:
    print("\n" + "=" * 82)
    print("LEAD-PERSISTENCE (reversion) SWEEP  —  margin accuracy vs FINAL & early swing")
    print("shrink = max(1 - slope*frac_rem, floor).  Lower persistence = less early swing.")
    print("=" * 82)
    buckets = sorted({k[2] for k in acc})
    for frac_el in buckets:
        print(f"\n--- {frac_el*100:.0f}% elapsed ({(1-frac_el)*100:.0f}% remaining) ---")
        print(f"{'slope':>6} {'floor':>6} {'margAE':>8} {'WPmean':>8} "
              f"{'WPp95':>6} {'WPmax':>6} {'margMove':>9} {'n':>5}   note")
        for (slope, floor) in candidates:
            b = acc.get((slope, floor, frac_el))
            if not b or b["n"] == 0:
                continue
            note = "  <-- SHIPPED" if (slope, floor) == (0.4, 0.55) else ""
            wp = np.array(b["wp_move"])
            print(f"{slope:6.2f} {floor:6.2f} {np.mean(b['margin_ae']):8.3f} "
                  f"{np.mean(wp)*100:7.1f}% "
                  f"{np.percentile(wp,95)*100:6.1f}% {np.max(wp)*100:6.1f}% "
                  f"{np.mean(b['margin_move']):9.3f} {b['n']:5d}{note}")
    print("\nRead: if a higher slope / lower floor keeps margAE flat (or lowers it)")
    print("while shrinking WPmove early, damping the early lead is free accuracy-wise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--pace-weight", type=float, default=0.15)
    args = ap.parse_args()

    print("Loading engine …", flush=True)
    db = os.getenv("PLL_DB_PATH",
                   os.path.join(_ROOT, "data", "analytics_database",
                                "pll_warehouse.duckdb"))
    engine = ProjectionEngine(db_path=db)
    engine.load()
    engine.fit(run_backtest=False)

    print("Loading games from PBP …", flush=True)
    games = load_games(args.games)
    print(f"  {len(games)} games. Sweeping {len(CANDIDATES)} reversion curves "
          f"at pace_weight={args.pace_weight}.", flush=True)

    acc = replay(games, engine, args.pace_weight, CANDIDATES)
    summarize(acc, CANDIDATES)


if __name__ == "__main__":
    main()
