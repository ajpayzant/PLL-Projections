"""
calibrate_live.py — historical replay to calibrate live-projection stability.

Goal (per user): prevent MASSIVE / unrealistic odds swings early in a game
(e.g. a pregame +150 team becoming -200 after an early 2-0) WHILE keeping good
accuracy on the eventual game outcome. We settle the pace-blend aggressiveness
empirically instead of by feel (project guardrail: prove de-noising in backtest).

Method
------
Replay every historical game from pbp_events_clean.parquet. At a set of elapsed-
time snapshots (e.g. 15/25/50/75% of regulation) we:
  1. reconstruct BANKED per-player stats from only the events up to that time
     (reusing the exact attribution rules of live_state.reconstruct),
  2. run the live model (LiveModel.resimulate) at each candidate pace_weight,
  3. score the result two ways:
       ACCURACY  — abs error of the live full-game projection vs the ACTUAL final
                   (team goals for game markets; player points for props),
       STABILITY — how far the live projection has moved from the PREGAME
                   projection this early (a proxy for "odds swing"); combined with
                   the realized error this tells us if an early swing was EARNED
                   (predictive) or NOISE (overreaction).

We sweep pace_weight and report, per snapshot bucket, the accuracy and the
early-move magnitude, so we can pick the setting that damps early overreaction
without hurting outcome accuracy.

Run:
    python live/calibrate_live.py                # full sweep, all seasons
    python live/calibrate_live.py --games 40     # cap games for a quick pass
    python live/calibrate_live.py --weights 0.2 0.35 0.5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import duckdb  # noqa: E402

from live_state import reconstruct  # noqa: E402
from live_model import LiveModel  # noqa: E402
from projection_engine_v3 import ProjectionEngine  # noqa: E402

REGULATION_SECONDS = 4 * 12 * 60
PBP = os.path.join(_ROOT, "data", "curated_data",
                   "all_requested_seasons", "pbp_events_clean.parquet")

# Snapshot points as FRACTION ELAPSED. Early buckets are where the swing problem
# lives, so we sample them densely.
SNAPSHOTS = (0.10, 0.15, 0.25, 0.40, 0.55, 0.75)


def _event_to_feed(row: dict) -> dict:
    """Translate a pbp_events_clean row into the dict shape live_state.reconstruct
    expects (it mirrors the live feed's field names + nested details)."""
    def _s(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    return {
        "eventType": row.get("eventType"),
        "teamId": _s(row.get("teamId")),
        "shooterId": _s(row.get("shooterId")),
        "shotType": row.get("shotType") or None,
        "shotAssistId": _s(row.get("shotAssistId")),
        "goalieId": _s(row.get("goalieId")),
        "faceoffWinnerId": _s(row.get("faceoffWinnerId")),
        "gbPlayerId": _s(row.get("gbPlayerId")),
        "details": {
            "shotOnGoal": bool(row.get("detail_shotOnGoal")) if row.get("detail_shotOnGoal") is not None else None,
            "shotSaved": bool(row.get("detail_shotSaved")) if row.get("detail_shotSaved") is not None else None,
        },
        "description": row.get("description", ""),
    }


def load_games(limit: int | None) -> List[dict]:
    """Return per-game replay bundles: slug, teams, ordered events, final scores."""
    con = duckdb.connect()
    slugs = con.execute(f"""
        SELECT DISTINCT game_slug, season, home_team, away_team
        FROM read_parquet('{PBP}')
        WHERE home_team IS NOT NULL AND away_team IS NOT NULL
        ORDER BY season, game_slug
    """).df()
    if limit:
        slugs = slugs.head(limit)
    games = []
    for _, g in slugs.iterrows():
        ev = con.execute(f"""
            SELECT * FROM read_parquet('{PBP}')
            WHERE game_slug = ? ORDER BY event_index
        """, [g["game_slug"]]).df()
        if ev.empty:
            continue
        fh = int(ev["run_home_score"].max())
        fa = int(ev["run_away_score"].max())
        games.append({
            "slug": g["game_slug"], "season": int(g["season"]),
            "home": str(g["home_team"]), "away": str(g["away_team"]),
            "rows": ev.to_dict("records"),
            "final_home": fh, "final_away": fa,
        })
    con.close()
    return games


def replay(games: List[dict], engine: ProjectionEngine,
           weights: List[float]) -> Dict:
    """For each game, snapshot, and candidate weight, compute accuracy + move."""
    # accumulators keyed by (weight, snapshot_bucket)
    acc = defaultdict(lambda: {"margin_ae": [], "total_ae": [],
                               "wp_move": [], "margin_move": [],
                               "player_pts_ae": [], "n": 0})

    # pregame projections cached per matchup (independent of snapshot/weight)
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

        # pregame game margin/total baselines (for "how far did it move")
        try:
            pre_gm = engine.pricing.price_game(pregame.game_sim)
            pre_home_wp = float(pre_gm.home_win_prob)
            pre_margin = float(np.median(pregame.game_sim.home_scores
                                         - pregame.game_sim.away_scores))
        except Exception:
            pre_home_wp, pre_margin = 0.5, 0.0

        final_margin = game["final_home"] - game["final_away"]
        final_total = game["final_home"] + game["final_away"]

        rows = game["rows"]
        model = LiveModel(engine, n_sims=4000)  # smaller: means, not tails

        for frac_el in SNAPSHOTS:
            cutoff = frac_el * REGULATION_SECONDS
            upto = [r for r in rows if (r.get("elapsed_seconds") or 0) <= cutoff]
            if len(upto) < 3:
                continue
            feed_events = [_event_to_feed(r) for r in upto]
            banked = reconstruct(feed_events)
            frac_rem = max(0.0, 1.0 - frac_el)

            for w in weights:
                try:
                    live = model.resimulate(
                        pregame, banked.by_player, frac_rem, pace_weight=w,
                        home_team_id=game["home"], away_team_id=game["away"],
                        team_of=banked.team_of, events=feed_events)
                    gm = engine.pricing.price_game(live.game_sim)
                except Exception:
                    continue

                live_margin = float(np.median(live.game_sim.home_scores
                                              - live.game_sim.away_scores))
                live_total = float(np.median(live.game_sim.total_distribution))
                live_home_wp = float(gm.home_win_prob)

                b = acc[(w, round(frac_el, 2))]
                b["margin_ae"].append(abs(live_margin - final_margin))
                b["total_ae"].append(abs(live_total - final_total))
                b["wp_move"].append(abs(live_home_wp - pre_home_wp))
                b["margin_move"].append(abs(live_margin - pre_margin))
                b["n"] += 1

        if (gi + 1) % 20 == 0:
            print(f"  ...replayed {gi + 1}/{len(games)} games", flush=True)

    return acc


def summarize(acc: Dict, weights: List[float]) -> None:
    print("\n" + "=" * 78)
    print("LIVE STABILITY / ACCURACY SWEEP")
    print("Per snapshot (fraction elapsed): margin & total accuracy vs FINAL, and")
    print("how far the live line MOVED from pregame (WP move, margin move).")
    print("Goal: pick the pace_weight that keeps accuracy while shrinking the early")
    print("WP/margin move (the 'odds swing after an early goal').")
    print("=" * 78)
    buckets = sorted({k[1] for k in acc})
    for frac_el in buckets:
        print(f"\n--- {frac_el*100:.0f}% elapsed "
              f"({(1-frac_el)*100:.0f}% remaining) ---")
        print(f"{'pace_w':>7} {'margAE':>8} {'totAE':>8} "
              f"{'WPmove':>8} {'margMove':>9} {'n':>5}")
        for w in weights:
            b = acc.get((w, frac_el))
            if not b or b["n"] == 0:
                continue
            print(f"{w:7.2f} {np.mean(b['margin_ae']):8.3f} "
                  f"{np.mean(b['total_ae']):8.3f} "
                  f"{np.mean(b['wp_move'])*100:7.1f}% "
                  f"{np.mean(b['margin_move']):9.3f} {b['n']:5d}")
    print("\nRead: at a fixed snapshot, a LOWER pace_w should shrink WPmove/margMove")
    print("(less early swing). Check margAE/totAE don't rise materially — if they")
    print("hold, the extra early movement was NOISE and damping is free accuracy-wise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None, help="cap number of games")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.15, 0.25, 0.35, 0.5, 0.7])
    args = ap.parse_args()

    print(f"Loading engine …", flush=True)
    db = os.getenv("PLL_DB_PATH",
                   os.path.join(_ROOT, "data", "analytics_database",
                                "pll_warehouse.duckdb"))
    engine = ProjectionEngine(db_path=db)
    engine.load()
    engine.fit(run_backtest=False)

    print(f"Loading games from PBP …", flush=True)
    games = load_games(args.games)
    print(f"  {len(games)} games loaded. Sweeping weights {args.weights} "
          f"across {len(SNAPSHOTS)} snapshots each.", flush=True)

    acc = replay(games, engine, args.weights)
    summarize(acc, args.weights)


if __name__ == "__main__":
    main()
