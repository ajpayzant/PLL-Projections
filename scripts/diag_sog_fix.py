"""End-to-end check on the SOG fix, plus the residual projection bias.

Answers two separate questions in one engine pass, because building projections
dominates the runtime:

1. **Does the priced distribution now return the projection it was given?** The
   old ``min(independent NegBin, shots_draw)`` wiring lost 13-38% of the mean, so
   the number shown in the UI disagreed with the number being priced. Reported as
   ``stated`` (the deterministic PlayerProjection value the UI shows) against
   ``sim mean`` (the mean of the distribution the price comes from).

2. **Is proj_sog itself biased against realized SOG?** This is a PlayerModel
   question, independent of the draw, and it is measured here against actual
   player-game results for the same players.

Run: python scripts/diag_sog_fix.py [--games N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.fast_backtest as FB  # noqa: E402
from projection_engine_v3 import (  # noqa: E402
    GameSimulator,
    ProjectionEngine,
    RatingBuilder,
    TeamModel,
)

# Note the distribution key is "shots_on_goal", not "sog" -- "sog" is only the
# PHI_PLAYER key. Using the wrong one silently drops the stat from the report.
STATS = ("goals", "assists", "points", "shots", "shots_on_goal")
POSITIONS = ["A", "M", "FO", "G", "SSDM", "LSM", "D"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=8000)
    args = ap.parse_args()

    eng = ProjectionEngine()
    eng.load()
    tg, pg = eng.team_games, eng.player_games

    train_tg = tg[tg["season"].isin(FB.TRAIN_SEASONS)].copy()
    train_pg = pg[pg["season"].isin(FB.TRAIN_SEASONS)].copy()
    rb = RatingBuilder(train_tg, train_pg)
    rb.build_team_ratings()
    tm = TeamModel()
    tm.fit(rb._tr)
    project_game = FB.build_project_fn(
        tg, pg, tm._quality_model is not None, tm, args.sims)

    test_pg = pg[pg["season"] == FB.TEST_SEASON]
    rows = []

    for gid in list(test_pg["game_id"].unique())[:args.games]:
        game_pg = test_pg[test_pg["game_id"] == gid]
        game_tg = tg[tg["game_id"] == gid]
        if game_tg.empty or len(game_tg) != 2:
            continue
        gnum = int(game_pg["game_number"].iloc[0])
        actual_ids = {}
        for tid in game_tg["team_id"].unique():
            am = game_pg[(game_pg["team_id"] == tid)
                         & (game_pg["position"].isin(POSITIONS))]
            actual_ids[str(tid)] = set(am["player_id"].astype(str).tolist())

        try:
            res = project_game(gid, FB.TEST_SEASON, gnum,
                               str(game_tg.iloc[0]["team_id"]),
                               str(game_tg.iloc[1]["team_id"]),
                               actual_player_ids=actual_ids)
        except Exception as exc:                      # noqa: BLE001
            print(f"  skip {gid}: {type(exc).__name__}: {exc}")
            continue
        if res is None or "player_projs" not in res:
            continue

        sim = GameSimulator(n_sims=args.sims, seed=42)
        hp, ap_, gs = res["hp"], res["ap"], res["gs"]
        sides = [
            (str(game_tg.iloc[0]["team_id"]), gs.home_goals,
             hp.proj_goals, ap_.proj_save_pct),
            (str(game_tg.iloc[1]["team_id"]), gs.away_goals,
             ap_.proj_goals, hp.proj_save_pct),
        ]
        for tid, goal_draws, team_goals, opp_svp in sides:
            projs = res["player_projs"].get(tid)
            if not projs:
                continue
            actual = game_pg[game_pg["team_id"].astype(str) == tid]
            act_sog = dict(zip(actual["player_id"].astype(str),
                               actual["shots_on_goal"].fillna(0)))
            act_sh = dict(zip(actual["player_id"].astype(str),
                              actual["shots"].fillna(0)))
            for ps in sim.simulate_players(projs, goal_draws, team_goals,
                                           opp_save_pct=opp_svp):
                pid = str(ps.player_id)
                for stat in STATS:
                    dist = ps.stat_distributions.get(stat)
                    stated = ps.proj_values.get(stat, 0.0)
                    if dist is None or not stated or stated <= 0.05:
                        continue
                    rows.append({
                        "stat": stat,
                        "player_id": pid,
                        "stated": float(stated),
                        "sim_mean": float(np.mean(dist)),
                        "actual": (float(act_sog.get(pid, np.nan))
                                   if stat == "shots_on_goal"
                                   else float(act_sh.get(pid, np.nan))
                                   if stat == "shots" else np.nan),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows -- CHECK DID NOT RUN (treat as a failure, not a pass)")
        return

    print(f"\n{len(df)} player-stat rows from {args.games} games\n")
    print("1. STATED PROJECTION vs THE MEAN OF THE DISTRIBUTION ACTUALLY PRICED")
    print("   (these must agree; the old SOG clamp broke it by 13-38%)")
    print(f"   {'stat':8s} {'n':>5s} {'stated':>8s} {'priced':>8s} "
          f"{'drift':>8s} {'pct':>8s}")
    worst = 0.0
    for stat, g in df.groupby("stat"):
        st, sm = g.stated.mean(), g.sim_mean.mean()
        pct = (sm - st) / st * 100
        worst = max(worst, abs(pct))
        print(f"   {stat:8s} {len(g):5d} {st:8.3f} {sm:8.3f} "
              f"{sm - st:+8.3f} {pct:+7.2f}%")
    verdict = "PASS" if worst < 2.0 else "FAIL"
    print(f"\n   VERDICT: {verdict} (worst drift {worst:.2f}%, tolerance 2%)")

    print("\n2. PROJECTION vs ACTUAL — a PlayerModel question, not a draw question")
    for stat in ("shots", "shots_on_goal"):
        g = df[(df.stat == stat) & df.actual.notna()]
        if g.empty:
            continue
        print(f"   {stat:14s} n={len(g):4d}  proj={g.stated.mean():6.3f}  "
              f"actual={g.actual.mean():6.3f}  "
              f"bias={(g.actual - g.stated).mean():+6.3f}  "
              f"ratio={g.stated.mean() / max(g.actual.mean(), 1e-9):5.3f}")


if __name__ == "__main__":
    main()
