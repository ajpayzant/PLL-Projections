"""
fast_backtest.py
----------------
Fast iteration harness for validating engine changes on the feature branch.
Mirrors run_backtest.py's leakage-safe methodology but trims runtime so the
edit->test loop is minutes not tens-of-minutes:

  * team section can be limited to TEST + CURRENT seasons (skip re-evaluating
    train games we don't score anyway) via --team-seasons
  * player section runs on TEST 2025 with a configurable sim count
  * prints the SAME headline metrics as run_backtest.py so numbers are
    directly comparable to scripts/backtest_baseline.log

This is a DEV tool. Every change that looks good here is CONFIRMED on the full
run_backtest.py before it's kept.

Usage:
    python scripts/fast_backtest.py                      # team(test+current)+player, 3000 sims
    python scripts/fast_backtest.py --sims 2000 --players-only
    python scripts/fast_backtest.py --team-only
    python scripts/fast_backtest.py --tag fix1           # label the output block
"""
import argparse
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from projection_engine_v3 import (
    ProjectionEngine, RatingBuilder, TeamModel, GameSimulator,
    PlayerModel, _assign_goalie_saves_team,
    _assign_player_goalie_saves, _assign_faceoff_from_specialist,
)

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
EARLY_SEASON = 2026


def build_project_fn(tg, pg, quality_fitted, tm_train, n_sims):
    def project_game(gid, gsea, gnum, home_team, away_team, actual_player_ids=None):
        train = tg[(tg["season"] < gsea) |
                   ((tg["season"] == gsea) & (tg["game_number"] < gnum))].copy()
        if len(train) < 30:
            return None
        train_p = pg[(pg["season"] < gsea) |
                     ((pg["season"] == gsea) & (pg["game_number"] < gnum))].copy()
        rb = RatingBuilder(train, train_p)
        rb.build_team_ratings()
        if actual_player_ids is not None:
            rb.build_player_ratings()
        tm = TeamModel()
        tm.fit(rb._tr)
        if quality_fitted:
            tm._quality_model = tm_train._quality_model
            tm._quality_scaler = tm_train._quality_scaler
        hf = rb.get_team_rating(home_team)
        af = rb.get_team_rating(away_team)
        if not hf or not af:
            return None
        hp = tm.predict(hf, af)
        ap = tm.predict(af, hf)
        _assign_goalie_saves_team(hp, ap)
        _assign_goalie_saves_team(ap, hp)
        sim = GameSimulator(n_sims=n_sims, seed=42)
        gs = sim.simulate_game(hp, ap)
        if quality_fitted:
            q_home = tm.quality_win_prob(hf, af)
            if q_home is not None:
                bh = 0.65 * gs.home_win_prob + 0.35 * q_home
                gs.home_win_prob = bh / (bh + (1 - bh))
                gs.away_win_prob = 1 - gs.home_win_prob
        res = {"hp": hp, "ap": ap, "gs": gs, "rb": rb, "tm": tm}
        if actual_player_ids is not None and not rb._pr.empty:
            full_projs = {}  # unfiltered lists (needed for cross-team FO/saves steps)
            for tid, tp in [(home_team, hp), (away_team, ap)]:
                opp_tp = ap if tid == home_team else hp
                _assign_goalie_saves_team(tp, opp_tp)
                pm = PlayerModel(rb._pr)
                ids = actual_player_ids.get(str(tid), set())
                if ids:
                    pr_team = rb._pr[rb._pr["team_id"] == tid]
                    allp = set(pr_team["player_id"].astype(str).tolist())
                    overrides = {p: {"active": False, "usage_multiplier": 0.0}
                                 for p in allp if p not in ids}
                    gp = pm.project_roster(tid, tp, overrides=overrides,
                                           use_current_roster_filter=False)
                else:
                    gp = pm.project_roster(tid, tp, use_current_roster_filter=False)
                full_projs[tid] = gp
            # Mirror production post-reconcile steps (engine.project ~L3762-3769):
            # goalie saves use opponent SOG + 0.915 factor; faceoffs use the
            # specialist + log5 matchup. The harness previously skipped these,
            # scoring PlayerModel placeholder saves (factor 1.0) — a test-rig
            # artifact that inflated the measured saves bias.
            _assign_player_goalie_saves(full_projs[home_team], ap.proj_sog)
            _assign_player_goalie_saves(full_projs[away_team], hp.proj_sog)
            _assign_faceoff_from_specialist(full_projs[home_team], full_projs[away_team])
            _assign_faceoff_from_specialist(full_projs[away_team], full_projs[home_team])
            res["player_projs"] = {tid: [p for p in gp if p.active]
                                   for tid, gp in full_projs.items()}
        return res
    return project_game


def team_section(tg, project_game, seasons):
    rows = []
    for ssn in seasons:
        sgames = tg[tg["season"] == ssn].drop_duplicates("game_id")
        for _, grow in sgames.iterrows():
            gid, gnum = grow["game_id"], int(grow["game_number"])
            gr = tg[tg["game_id"] == gid]
            if len(gr) != 2:
                continue
            hr = gr[gr["is_home"] == 1]
            ar = gr[gr["is_home"] == 0]
            if hr.empty or ar.empty:
                hr, ar = gr.iloc[[0]], gr.iloc[[1]]
            ht, at = str(hr.iloc[0]["team_id"]), str(ar.iloc[0]["team_id"])
            try:
                res = project_game(gid, ssn, gnum, ht, at)
                if res is None:
                    continue
                hp, ap, gs = res["hp"], res["ap"], res["gs"]
                row = {
                    "season": ssn,
                    "pred_total": hp.proj_goals + ap.proj_goals,
                    "act_total": float(hr.iloc[0]["goals"]) + float(ar.iloc[0]["goals"]),
                    "pred_h": hp.proj_goals, "act_h": float(hr.iloc[0]["goals"]),
                    "pred_a": ap.proj_goals, "act_a": float(ar.iloc[0]["goals"]),
                    "pred_prob": gs.home_win_prob,
                    "actual_win": 1 if float(hr.iloc[0]["scores"]) > float(ar.iloc[0]["scores"]) else 0,
                }
                # ── Phase 0 instrumentation: points (scoreboard) + 2pt ────────
                # goal-count totals above are NOT the scoreboard; scores value a
                # 2pt goal at 2. Capture proj_scores vs actual scores so we can
                # see whether the 2pt handling introduces total-POINTS error the
                # goal-count MAE is blind to. Also capture per-team 2pt goals
                # (offense) and 2pt goals allowed (defense) to test opponent
                # 2pt signal in the diagnosis phase.
                if "scores" in hr.iloc[0] and "scores" in ar.iloc[0]:
                    row["pred_score_total"] = hp.proj_scores + ap.proj_scores
                    row["act_score_total"] = float(hr.iloc[0]["scores"]) + float(ar.iloc[0]["scores"])
                for side, pr, act in (("h", hp, hr.iloc[0]), ("a", ap, ar.iloc[0])):
                    if "two_point_goals" in act:
                        row[f"pred_2pt_{side}"] = float(getattr(pr, "proj_2pt_goals", 0.0))
                        row[f"act_2pt_{side}"] = float(act["two_point_goals"] or 0.0)
                    if "two_point_goals_against" in act:
                        row[f"act_2pt_allowed_{side}"] = float(act["two_point_goals_against"] or 0.0)
                # SOG + shots bias (root-cause diagnostic for saves over-projection)
                for side, pr, act in (("h", hp, hr.iloc[0]), ("a", ap, ar.iloc[0])):
                    if "shots_on_goal" in act:
                        row[f"pred_sog_{side}"] = pr.proj_sog
                        row[f"act_sog_{side}"] = float(act["shots_on_goal"])
                    if "shots" in act:
                        row[f"pred_shots_{side}"] = pr.proj_shots
                        row[f"act_shots_{side}"] = float(act["shots"])
                rows.append(row)
            except Exception:
                continue
    df = pd.DataFrame(rows)
    print("\n== TEAM ==")
    for ssn, g in df.groupby("season"):
        mae = np.mean(np.abs(g["pred_total"] - g["act_total"]))
        bias = np.mean(g["pred_total"] - g["act_total"])
        acc = np.mean((g["pred_prob"] > 0.5) == g["actual_win"].astype(bool))
        brier = np.mean((g["pred_prob"] - g["actual_win"]) ** 2)
        mae_side = np.mean(np.abs(np.r_[g["pred_h"] - g["act_h"], g["pred_a"] - g["act_a"]])) if len(g) else np.nan
        print(f"  {ssn}: n={len(g):3d}  totMAE={mae:.3f}  bias={bias:+.3f}  sideMAE={mae_side:.3f}  "
              f"acc={acc*100:.0f}%  Brier={brier:.4f}")
        # Points (scoreboard) total: goal-count MAE above ignores the 2pt
        # premium; this shows whether total-POINTS is biased/mis-scaled.
        if "pred_score_total" in g and g["pred_score_total"].notna().any():
            gsc = g.dropna(subset=["pred_score_total", "act_score_total"])
            sc_mae = np.mean(np.abs(gsc["pred_score_total"] - gsc["act_score_total"]))
            sc_bias = np.mean(gsc["pred_score_total"] - gsc["act_score_total"])
            print(f"        SCORE(pts): totMAE={sc_mae:.3f} bias={sc_bias:+.3f} "
                  f"pred={gsc['pred_score_total'].mean():.2f} act={gsc['act_score_total'].mean():.2f}")
        # 2pt goals per team: offense projection vs actual, and 2pt-ALLOWED
        # actuals (for the opponent-2pt signal test in diagnosis).
        if "pred_2pt_h" in g:
            p2 = np.r_[g["pred_2pt_h"], g["pred_2pt_a"]]
            a2 = np.r_[g["act_2pt_h"], g["act_2pt_a"]]
            print(f"        2PT: pred={np.nanmean(p2):.2f} act={np.nanmean(a2):.2f} "
                  f"bias={np.nanmean(p2-a2):+.2f} MAE={np.nanmean(np.abs(p2-a2)):.2f}")
        if "act_2pt_allowed_h" in g:
            aa = np.r_[g["act_2pt_allowed_h"], g["act_2pt_allowed_a"]]
            print(f"        2PT-ALLOWED(act): mean={np.nanmean(aa):.2f} sd={np.nanstd(aa):.2f}")
        # SOG / shots bias diagnostic (drives goalie shots-faced → saves)
        if "pred_sog_h" in g:
            ps = np.r_[g["pred_sog_h"], g["pred_sog_a"]]
            as_ = np.r_[g["act_sog_h"], g["act_sog_a"]]
            psh = np.r_[g["pred_shots_h"], g["pred_shots_a"]]
            ash = np.r_[g["act_shots_h"], g["act_shots_a"]]
            print(f"        SOG: pred={np.nanmean(ps):.2f} act={np.nanmean(as_):.2f} "
                  f"bias={np.nanmean(ps-as_):+.2f} MAE={np.nanmean(np.abs(ps-as_)):.2f}   "
                  f"SHOTS: pred={np.nanmean(psh):.2f} act={np.nanmean(ash):.2f} "
                  f"bias={np.nanmean(psh-ash):+.2f}")
    return df


def player_section(tg, pg, project_game):
    rows = []
    test_pg = pg[pg["season"] == TEST_SEASON].copy()
    for gid in test_pg["game_id"].unique():
        game_pg = test_pg[test_pg["game_id"] == gid]
        game_tg = tg[tg["game_id"] == gid]
        if game_tg.empty or len(game_tg) != 2:
            continue
        gnum = int(game_pg["game_number"].iloc[0])
        actual_ids = {}
        for tid in game_tg["team_id"].unique():
            am = game_pg[(game_pg["team_id"] == tid) &
                         (game_pg["position"].isin(["A", "M", "FO", "G", "SSDM", "LSM", "D"]))]
            actual_ids[str(tid)] = set(am["player_id"].astype(str).tolist())
        try:
            res = project_game(gid, TEST_SEASON, gnum,
                               str(game_tg.iloc[0]["team_id"]),
                               str(game_tg.iloc[1]["team_id"]),
                               actual_player_ids=actual_ids)
            if res is None or "player_projs" not in res:
                continue
            for tid, projs in res["player_projs"].items():
                actuals = game_pg[(game_pg["team_id"] == tid) &
                                  (game_pg["position"].isin(["A", "M"]))]
                for _, arow in actuals.iterrows():
                    pid = str(arow["player_id"])
                    proj = next((p for p in projs if p.player_id == pid), None)
                    if proj is None:
                        continue
                    for stat in ["goals", "assists", "shots"]:
                        av = float(arow.get(stat, 0) or 0)
                        pv = getattr(proj, f"proj_{stat}", 0.0)
                        rows.append({"stat": stat, "pred": pv, "actual": av,
                                     "abs_error": abs(pv - av), "error": pv - av})
                # Faceoff specialists: evaluate FO wins + FO%
                fo_actuals = game_pg[(game_pg["team_id"] == tid) &
                                     (game_pg["position"] == "FO")]
                for _, arow in fo_actuals.iterrows():
                    pid = str(arow["player_id"])
                    proj = next((p for p in projs if p.player_id == pid), None)
                    if proj is None:
                        continue
                    act_fw = float(arow.get("faceoffs_won", 0) or 0)
                    act_fl = float(arow.get("faceoffs_lost", 0) or 0)
                    if act_fw + act_fl < 5:  # not the primary specialist this game
                        continue
                    pv = getattr(proj, "proj_faceoff_wins", 0.0)
                    rows.append({"stat": "fo_wins", "pred": pv, "actual": act_fw,
                                 "abs_error": abs(pv - act_fw), "error": pv - act_fw})
                    act_fpct = act_fw / max(act_fw + act_fl, 1.0)
                    pv_fpct = getattr(proj, "proj_faceoff_pct", 0.0)
                    rows.append({"stat": "fo_pct", "pred": pv_fpct, "actual": act_fpct,
                                 "abs_error": abs(pv_fpct - act_fpct), "error": pv_fpct - act_fpct})
                # Goalies: evaluate saves (starter only — >=5 shots faced actual)
                goalie_actuals = game_pg[(game_pg["team_id"] == tid) &
                                         (game_pg["position"] == "G")]
                for _, arow in goalie_actuals.iterrows():
                    pid = str(arow["player_id"])
                    proj = next((p for p in projs if p.player_id == pid), None)
                    if proj is None:
                        continue
                    act_sv = float(arow.get("saves", 0) or 0)
                    act_ga = float(arow.get("goals_against", 0) or 0)
                    if act_sv + act_ga < 5:  # backup / garbage cameo
                        continue
                    pv = getattr(proj, "proj_saves", 0.0)
                    rows.append({"stat": "saves", "pred": pv, "actual": act_sv,
                                 "abs_error": abs(pv - act_sv), "error": pv - act_sv})
                    # implied shots-faced diagnostic: pred vs actual (saves+GA)
                    pv_svp = getattr(proj, "proj_save_pct", 0.0) or 0.01
                    impl_sf = pv / pv_svp
                    act_sf = act_sv + act_ga
                    rows.append({"stat": "sf_implied", "pred": impl_sf, "actual": act_sf,
                                 "abs_error": abs(impl_sf - act_sf), "error": impl_sf - act_sf})
                    # save% eval
                    act_svpct = act_sv / max(act_sv + act_ga, 1.0)
                    pv_svpct = getattr(proj, "proj_save_pct", 0.0)
                    rows.append({"stat": "save_pct", "pred": pv_svpct, "actual": act_svpct,
                                 "abs_error": abs(pv_svpct - act_svpct), "error": pv_svpct - act_svpct})
        except Exception:
            continue
    pf = pd.DataFrame(rows)
    print("\n== PLAYER (TEST 2025) ==")
    if pf.empty:
        print("  no rows")
        return pf
    for stat, g in pf.groupby("stat"):
        print(f"  {stat:8s}: MAE={g['abs_error'].mean():.3f}  bias={g['error'].mean():+.3f}  "
              f"corr={g['pred'].corr(g['actual']):.3f}  "
              f"avg_pred={g['pred'].mean():.2f}  avg_act={g['actual'].mean():.2f}")
    return pf


def poisson_floor_section(tg, pg, project_game, n_sims, max_games=20):
    """Assert every 0.5-line price is consistent with the projection behind it.

    On a 0.5 line, "Over" is just "at least one", so a Poisson at the same mean
    puts it at ``1 - exp(-mu)``. A count distribution may sit slightly below that
    -- real overdispersion does push mass toward zero -- but it cannot sit far
    below without contradicting its own mean.

    **The check applies to goals and assists only, and points is deliberately
    excluded from the verdict.** Points is a sum of correlated components, and
    the floor is not a valid bound on such a sum, for two separate reasons —
    both measured rather than assumed:

    1. Points is ``1*(1pt goals) + 2*(2pt goals) + assists``, so a two-point goal
       raises E[points] without adding a scoring event. ``1 - exp(-E[points])``
       therefore overstates P(at least one). On independent Poisson goals and
       assists with a 2-point rate, where true P(Over 0.5) is 0.7756, the naive
       floor claims 0.8128 — a spurious +3.7-point breach. Using the event rate
       (goals + assists) instead gives 0.7769, correct to 0.001.
    2. Even on the event rate, positive correlation between a player's goals and
       assists lowers P(at least one): a quiet game is quiet in both. This is
       real and the engine models it on purpose (the Cholesky layer and the team
       conditioning both induce it). At corr(g,a) = +0.086 the legitimate breach
       is already +0.029, and at +0.244 it is +0.080 — larger than this check's
       whole tolerance, on a distribution with nothing wrong with it.

    So a points breach is not evidence of a bug and must not fail the run. It is
    still printed, because a sudden jump in it is worth a look, but the verdict is
    taken on goals and assists — which are single count draws, where the floor is
    a genuine bound.

    This is the check the rest of this harness structurally cannot make. MAE,
    bias and correlation all read the mean, and the zero-inflation bug that cost
    the most money this season left the mean exactly right while moving P(Over)
    17 points. P10-P90 coverage misses it too, since a percentile band says
    nothing about where mass sits at a single integer. So this section is not a
    nice-to-have: it is the only assertion here that would have caught it.

    Runs on ``max_games`` test games rather than all of them because it has to
    build the full player simulation, which the sections above skip. The
    per-prop breach rate is a property of the distribution code, not of the
    slate, so a subset is sufficient to catch a regression -- exhaustive
    coverage lives in scripts/test_prop_distributions.py.
    """
    # 5 points of slack. Below the floor is the direction that underprices the
    # over, which is what lost money, so the tolerance is deliberately one-sided.
    TOL = 0.05
    rows = []
    test_pg = pg[pg["season"] == TEST_SEASON]
    game_ids = list(test_pg["game_id"].unique())[:max_games]

    for gid in game_ids:
        game_pg = test_pg[test_pg["game_id"] == gid]
        game_tg = tg[tg["game_id"] == gid]
        if game_tg.empty or len(game_tg) != 2:
            continue
        gnum = int(game_pg["game_number"].iloc[0])
        actual_ids = {}
        for tid in game_tg["team_id"].unique():
            am = game_pg[(game_pg["team_id"] == tid) &
                         (game_pg["position"].isin(["A", "M", "FO", "G", "SSDM", "LSM", "D"]))]
            actual_ids[str(tid)] = set(am["player_id"].astype(str).tolist())
        try:
            res = project_game(gid, TEST_SEASON, gnum,
                               str(game_tg.iloc[0]["team_id"]),
                               str(game_tg.iloc[1]["team_id"]),
                               actual_player_ids=actual_ids)
            if res is None or "player_projs" not in res:
                continue
            sim = GameSimulator(n_sims=n_sims, seed=42)
            hp, ap_, gsim = res["hp"], res["ap"], res["gs"]
            sides = [
                (str(game_tg.iloc[0]["team_id"]), gsim.home_goals, hp.proj_goals, ap_.proj_save_pct),
                (str(game_tg.iloc[1]["team_id"]), gsim.away_goals, ap_.proj_goals, hp.proj_save_pct),
            ]
            for tid, goal_draws, team_goals, opp_svp in sides:
                projs = res["player_projs"].get(tid)
                if not projs:
                    continue
                psims = sim.simulate_players(projs, goal_draws, team_goals,
                                             opp_save_pct=opp_svp)
                for ps in psims:
                    for stat in ("goals", "assists", "points"):
                        dist = ps.stat_distributions.get(stat)
                        mu = ps.proj_values.get(stat, 0.0)
                        if dist is None or not mu or mu <= 0.02:
                            continue
                        if stat == "points":
                            # Rate of scoring EVENTS, not of points. See docstring.
                            mu_events = (ps.proj_values.get("goals", 0.0)
                                         + ps.proj_values.get("assists", 0.0))
                        else:
                            mu_events = mu
                        if mu_events <= 0.02:
                            continue
                        rows.append({
                            "stat": stat, "mu": float(mu),
                            "mu_events": float(mu_events),
                            "p_over": float(np.mean(dist > 0.5)),
                            "floor": float(1.0 - np.exp(-mu_events)),
                        })
        except Exception:
            continue

    print(f"\n== POISSON FLOOR (0.5 lines, TEST 2025, {len(game_ids)} games) ==")
    pf = pd.DataFrame(rows)
    if pf.empty:
        print("  no rows -- CHECK DID NOT RUN (treat as a failure, not a pass)")
        return pf
    # Only these two are single count draws, so only these two are bindable.
    BINDING = ("goals", "assists")
    pf["breach"] = pf["floor"] - pf["p_over"]
    ok = True
    for stat, g in pf.groupby("stat"):
        worst = g["breach"].max()
        rate = (g["breach"] > TOL).mean()
        if stat in BINDING:
            status = "OK" if rate == 0 else "FAIL"
            ok = ok and rate == 0
        else:
            status = "info only - not a valid bound, see docstring"
        print(f"  {stat:8s}: n={len(g):5d}  avg mu={g['mu'].mean():.3f}  "
              f"avg mu_events={g['mu_events'].mean():.3f}  "
              f"avg P(Over)={g['p_over'].mean():.3f}  avg floor={g['floor'].mean():.3f}  "
              f"worst breach={worst:+.4f}  breaching>{TOL:.2f}={rate:.1%}  [{status}]")
    print(f"  VERDICT ({'+'.join(BINDING)}): "
          f"{'PASS' if ok else 'FAIL -- a price contradicts its own projection'}")
    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--team-only", action="store_true")
    ap.add_argument("--players-only", action="store_true")
    ap.add_argument("--team-seasons", type=int, nargs="*", default=[TEST_SEASON, EARLY_SEASON])
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--floor-games", type=int, default=20,
                    help="games for the Poisson-floor check; 0 skips it")
    ap.add_argument("--floor-only", action="store_true",
                    help="run only the Poisson-floor check (skips the mean-based "
                         "sections, which cannot see a shape bug anyway)")
    args = ap.parse_args()

    print("=" * 60)
    print(f"FAST BACKTEST  sims={args.sims}  tag={args.tag or '(none)'}")
    print("=" * 60)

    eng = ProjectionEngine()
    eng.load()
    tg, pg = eng.team_games, eng.player_games

    train_tg = tg[tg["season"].isin(TRAIN_SEASONS)].copy()
    train_pg = pg[pg["season"].isin(TRAIN_SEASONS)].copy()
    rb_train = RatingBuilder(train_tg, train_pg)
    rb_train.build_team_ratings()
    tm_train = TeamModel()
    tm_train.fit(rb_train._tr)
    quality_fitted = tm_train._quality_model is not None

    project_game = build_project_fn(tg, pg, quality_fitted, tm_train, args.sims)

    if not args.players_only and not args.floor_only:
        team_section(tg, project_game, args.team_seasons)
    if not args.team_only:
        if not args.floor_only:
            player_section(tg, pg, project_game)
        if args.floor_games:
            poisson_floor_section(tg, pg, project_game, args.sims, args.floor_games)

    print("\n" + "=" * 60)
    print(f"DONE  tag={args.tag or '(none)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
