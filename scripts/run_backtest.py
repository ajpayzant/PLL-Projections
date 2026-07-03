"""
PLL Projection Model Backtest
==============================
Proper methodology:
  - TRAIN seasons: 2022-2024  (fit ratings, quality model, calibrator)
  - TEST season:   2025        (held-out evaluation — never used in fitting)
  - 2026 shown separately as current-season early indicator

Team-level evaluation:
  - Leakage-safe: each game projected using only data from prior games
  - Quality composite model fitted once on full 2022-2024 then applied to 2025
  - Win probability evaluated via Brier score and accuracy

Player-level evaluation:
  - Projected roster limited to players who ACTUALLY PLAYED in that game
    (fixes the 40-player dilution problem from the old approach)
  - After reconcile, projected totals match actual team goals correctly
  - Correlation, MAE, bias, zero-rate all evaluated on realistic scale
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from projection_engine_v3 import (
    ProjectionEngine, RatingBuilder, TeamModel, GameSimulator,
    PlayerModel, _assign_goalie_saves_team, LG_GOALS, LG_SHOTS,
)

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON   = 2025
EARLY_SEASON  = 2026
N_SIMS        = 5_000

print("="*65)
print("PLL PROJECTION MODEL BACKTEST")
print(f"  Train: {TRAIN_SEASONS}  |  Test (hold-out): {TEST_SEASON}  |  Current: {EARLY_SEASON}")
print("="*65)

print("\nLoading data...")
eng = ProjectionEngine()
eng.load()
tg = eng.team_games
pg = eng.player_games

# ── Fit quality model on full training seasons ─────────────────────────────
# Train once on 2022-2024 so it has sufficient data (not per-game windows)
print("Fitting quality model on train seasons...")
train_tg_full = tg[tg["season"].isin(TRAIN_SEASONS)].copy()
train_pg_full = pg[pg["season"].isin(TRAIN_SEASONS)].copy()
rb_train = RatingBuilder(train_tg_full, train_pg_full)
rb_train.build_team_ratings()
tm_train = TeamModel()
tm_train.fit(rb_train._tr)
quality_fitted = tm_train._quality_model is not None
print(f"  Quality model fitted: {quality_fitted}")
if quality_fitted:
    print(f"  Quality model features: goal_diff, shot_diff, fo_pct, to_diff")


# ── Helper: project one game leakage-safe ─────────────────────────────────
def project_game(gid, gsea, gnum, home_team, away_team,
                 actual_player_ids=None):
    """
    Build ratings from all data strictly before this game,
    project both teams, and optionally project players.
    actual_player_ids: set of player_ids who played in this game
    """
    train = tg[
        (tg["season"] < gsea) |
        ((tg["season"] == gsea) & (tg["game_number"] < gnum))
    ].copy()
    if len(train) < 30:
        return None

    train_p = pg[
        (pg["season"] < gsea) |
        ((pg["season"] == gsea) & (pg["game_number"] < gnum))
    ].copy()

    rb = RatingBuilder(train, train_p)
    rb.build_team_ratings()
    if actual_player_ids is not None:
        rb.build_player_ratings()

    tm = TeamModel()
    tm.fit(rb._tr)

    # Transfer quality model from full-train fit so it has real power
    if quality_fitted:
        tm._quality_model  = tm_train._quality_model
        tm._quality_scaler = tm_train._quality_scaler

    hf = rb.get_team_rating(home_team)
    af = rb.get_team_rating(away_team)
    if not hf or not af:
        return None

    hp = tm.predict(hf, af)
    ap = tm.predict(af, hf)
    _assign_goalie_saves_team(hp, ap)
    _assign_goalie_saves_team(ap, hp)

    sim = GameSimulator(n_sims=N_SIMS, seed=42)
    gs  = sim.simulate_game(hp, ap)

    # Blend quality model win probability
    if quality_fitted:
        q_home = tm.quality_win_prob(hf, af)
        if q_home is not None:
            blended_h = 0.65 * gs.home_win_prob + 0.35 * q_home
            blended_a = 1.0 - blended_h
            total = blended_h + blended_a
            gs.home_win_prob = blended_h / total
            gs.away_win_prob = blended_a / total

    result = {
        "rb": rb, "tm": tm, "hp": hp, "ap": ap, "gs": gs,
        "hf": hf, "af": af,
    }

    if actual_player_ids is not None and not rb._pr.empty:
        player_projs = {}
        for tid, tp in [(home_team, hp), (away_team, ap)]:
            opp_tp = ap if tid == home_team else hp
            _assign_goalie_saves_team(tp, opp_tp)
            pm = PlayerModel(rb._pr)
            # Fix 1: pass actual participant IDs as active overrides BEFORE
            # project_roster runs reconcile. This means reconcile distributes
            # the team total only across the ~15 real game participants, giving
            # each player a projection at the correct scale — not diluted across 45.
            game_ids_set = actual_player_ids.get(str(tid), set())
            if game_ids_set:
                # Build overrides: non-participants are inactive
                overrides = {}
                pr_team = rb._pr[rb._pr["team_id"] == tid]
                all_pids = set(pr_team["player_id"].astype(str).tolist())
                for pid_o in all_pids:
                    if pid_o not in game_ids_set:
                        overrides[pid_o] = {"active": False, "usage_multiplier": 0.0}
                game_projs = pm.project_roster(tid, tp,
                                               overrides=overrides,
                                               use_current_roster_filter=False)
            else:
                game_projs = pm.project_roster(tid, tp, use_current_roster_filter=False)
            player_projs[tid] = [p for p in game_projs if p.active]
        result["player_projs"] = player_projs

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: TEAM-LEVEL BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SECTION 1: TEAM-LEVEL EVALUATION")
print("="*65)

team_rows = []
all_seasons = sorted(tg["season"].unique())

for ssn in all_seasons:
    season_games = tg[tg["season"] == ssn].drop_duplicates("game_id")
    for _, grow in season_games.iterrows():
        gid  = grow["game_id"]
        gnum = int(grow["game_number"])
        game_rows = tg[tg["game_id"] == gid]
        if len(game_rows) != 2:
            continue

        home_r = game_rows[game_rows["is_home"] == 1]
        away_r = game_rows[game_rows["is_home"] == 0]
        if home_r.empty or away_r.empty:
            home_r = game_rows.iloc[[0]]
            away_r = game_rows.iloc[[1]]

        home_team = str(home_r.iloc[0]["team_id"])
        away_team = str(away_r.iloc[0]["team_id"])
        act_hg = float(home_r.iloc[0]["goals"])
        act_ag = float(away_r.iloc[0]["goals"])
        act_hs = float(home_r.iloc[0]["scores"])
        act_as = float(away_r.iloc[0]["scores"])

        try:
            res = project_game(gid, ssn, gnum, home_team, away_team)
            if res is None:
                continue
            hp, ap, gs = res["hp"], res["ap"], res["gs"]
            team_rows.append({
                "season": ssn, "game_number": gnum,
                "home_team": home_team, "away_team": away_team,
                "pred_h": hp.proj_goals, "pred_a": ap.proj_goals,
                "pred_total": hp.proj_goals + ap.proj_goals,
                "pred_h_score": hp.proj_scores, "pred_a_score": ap.proj_scores,
                "act_h": act_hg, "act_a": act_ag,
                "act_total": act_hg + act_ag,
                "act_total_score": act_hs + act_as,
                "pred_prob": gs.home_win_prob,
                "actual_win": 1 if act_hs > act_as else 0,
                "split": "test" if ssn == TEST_SEASON else
                         "current" if ssn == EARLY_SEASON else "train",
            })
        except Exception:
            continue

df = pd.DataFrame(team_rows)
print(f"Total games evaluated: {len(df)}")

def _team_metrics(grp, label):
    if grp.empty:
        return
    mae   = np.mean(np.abs(grp["pred_total"] - grp["act_total"]))
    bias  = np.mean(grp["pred_total"] - grp["act_total"])
    rmse  = np.sqrt(np.mean((grp["pred_total"] - grp["act_total"])**2))
    brier = np.mean((grp["pred_prob"] - grp["actual_win"])**2)
    acc   = np.mean((grp["pred_prob"] > 0.5) == grp["actual_win"].astype(bool))
    mae_h = np.mean(np.abs(grp["pred_h"] - grp["act_h"]))
    mae_a = np.mean(np.abs(grp["pred_a"] - grp["act_a"]))
    mae_sc= np.mean(np.abs(grp["pred_h_score"] + grp["pred_a_score"] - grp["act_total_score"]))
    bias_sc = np.mean(grp["pred_h_score"] + grp["pred_a_score"] - grp["act_total_score"])
    print(f"\n  [{label}]  n={len(grp)}")
    print(f"    MAE total goals:   {mae:.3f}   (home {mae_h:.3f} / away {mae_a:.3f})")
    print(f"    RMSE total goals:  {rmse:.3f}")
    print(f"    Bias total goals:  {bias:+.3f}   {'HIGH' if bias > 0.3 else 'LOW' if bias < -0.3 else 'calibrated'}")
    print(f"    MAE total score:   {mae_sc:.3f}   bias {bias_sc:+.3f}")
    print(f"    Winner accuracy:   {acc*100:.1f}%   (Brier {brier:.4f} — {'GOOD' if brier < 0.25 else 'needs work'})")

print("\n-- By season --")
for ssn, grp in df.groupby("season"):
    tag = " [TRAIN]" if ssn in TRAIN_SEASONS else " [TEST hold-out]" if ssn == TEST_SEASON else " [CURRENT]"
    mae  = np.mean(np.abs(grp["pred_total"] - grp["act_total"]))
    bias = np.mean(grp["pred_total"] - grp["act_total"])
    acc  = np.mean((grp["pred_prob"] > 0.5) == grp["actual_win"].astype(bool))
    brier= np.mean((grp["pred_prob"] - grp["actual_win"])**2)
    print(f"  {ssn}{tag}: n={len(grp):3d}  MAE={mae:.2f}  bias={bias:+.2f}  winner_acc={acc*100:.0f}%  Brier={brier:.4f}")

_team_metrics(df[df["split"] == "train"],   "TRAIN 2022-2024")
_team_metrics(df[df["split"] == "test"],    "TEST 2025 (hold-out)")
_team_metrics(df[df["split"] == "current"], "CURRENT 2026 (early)")

print("\n-- Per-team goal bias (TEST season only) --")
test_df = df[df["split"] == "test"]
if not test_df.empty:
    hb = test_df.groupby("home_team").apply(lambda x: (x["pred_h"] - x["act_h"]).mean())
    ab = test_df.groupby("away_team").apply(lambda x: (x["pred_a"] - x["act_a"]).mean())
    tb = pd.DataFrame({"home_bias": hb, "away_bias": ab})
    tb["avg_bias"] = tb.mean(axis=1)
    print(tb.sort_values("avg_bias").round(3).to_string())

print("\n-- Win probability calibration (TEST season) --")
if not test_df.empty:
    test_df = test_df.copy()
    test_df["prob_bucket"] = pd.cut(test_df["pred_prob"],
                                     bins=[0.3,0.4,0.45,0.50,0.55,0.60,0.70,1.0],
                                     right=False)
    cal = test_df.groupby("prob_bucket", observed=True).agg(
        n=("actual_win","count"),
        pred_mean=("pred_prob","mean"),
        actual_rate=("actual_win","mean"),
    ).reset_index()
    print(cal.round(3).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: PLAYER-LEVEL BACKTEST (TEST season only)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SECTION 2: PLAYER-LEVEL EVALUATION (TEST season 2025)")
print("  Players projected using ONLY those who actually played each game")
print("="*65)

player_rows = []
test_pg = pg[pg["season"] == TEST_SEASON].copy()
game_ids = test_pg["game_id"].unique()
print(f"Games to evaluate: {len(game_ids)}")

for idx, gid in enumerate(game_ids):
    if idx % 5 == 0:
        print(f"  Processing game {idx+1}/{len(game_ids)}...")

    game_pg = test_pg[test_pg["game_id"] == gid]
    game_tg = tg[tg["game_id"] == gid]
    if game_tg.empty or len(game_tg) != 2:
        continue

    gnum = int(game_pg["game_number"].iloc[0])

    # Build actual player set per team for this game (A/M only for evaluation)
    actual_ids = {}
    for tid in game_tg["team_id"].unique():
        am_rows = game_pg[(game_pg["team_id"] == tid) &
                          (game_pg["position"].isin(["A","M","FO","G","SSDM","LSM","D"]))]
        actual_ids[str(tid)] = set(am_rows["player_id"].astype(str).tolist())

    try:
        res = project_game(gid, TEST_SEASON, gnum,
                           str(game_tg.iloc[0]["team_id"]),
                           str(game_tg.iloc[1]["team_id"]),
                           actual_player_ids=actual_ids)
        if res is None or "player_projs" not in res:
            continue

        for tid, projs in res["player_projs"].items():
            actuals = game_pg[
                (game_pg["team_id"] == tid) &
                (game_pg["position"].isin(["A", "M"]))
            ]
            for _, arow in actuals.iterrows():
                pid = str(arow["player_id"])
                proj = next((p for p in projs if p.player_id == pid), None)
                if proj is None:
                    continue
                for stat, act_col in [("goals","goals"),("assists","assists"),("shots","shots")]:
                    act_val = float(arow.get(act_col, 0) or 0)
                    pred_val = getattr(proj, f"proj_{stat}", 0.0)
                    player_rows.append({
                        "player":    arow["full_name"],
                        "position":  arow["position"],
                        "team":      tid,
                        "game_num":  gnum,
                        "stat":      stat,
                        "pred":      pred_val,
                        "actual":    act_val,
                        "error":     pred_val - act_val,
                        "abs_error": abs(pred_val - act_val),
                    })
    except Exception:
        continue

pf = pd.DataFrame(player_rows)
print(f"\nPlayer-game-stat rows: {len(pf)}")

if pf.empty:
    print("No rows collected — check data pipeline")
else:
    print("\n-- By stat (after proper roster filtering) --")
    for stat, grp in pf.groupby("stat"):
        mae  = grp["abs_error"].mean()
        bias = grp["error"].mean()
        avg_a = grp["actual"].mean()
        avg_p = grp["pred"].mean()
        corr  = grp["pred"].corr(grp["actual"])
        zero_a = (grp["actual"] == 0).mean()
        zero_p = (grp["pred"] < 0.5).mean()
        print(f"  {stat:8s}: MAE={mae:.3f}  bias={bias:+.3f}  "
              f"avg_actual={avg_a:.2f}  avg_pred={avg_p:.2f}  "
              f"corr={corr:.3f}  actual_0={zero_a:.0%}  pred_0={zero_p:.0%}")

    print("\n-- By position x stat --")
    for (pos, stat), grp in pf.groupby(["position", "stat"]):
        mae  = grp["abs_error"].mean()
        bias = grp["error"].mean()
        corr = grp["pred"].corr(grp["actual"])
        print(f"  {pos:6s} {stat:8s}: MAE={mae:.3f}  bias={bias:+.3f}  corr={corr:.3f}")

    print("\n-- Goal projection: players model underestimates most (10+ games) --")
    pg_g = pf[pf["stat"] == "goals"].groupby("player").agg(
        n=("actual","count"),
        mae=("abs_error","mean"),
        bias=("error","mean"),
        avg_actual=("actual","mean"),
        avg_pred=("pred","mean"),
    ).query("n >= 10").sort_values("bias")
    print("  Most underestimated:")
    print(pg_g.head(8).round(3).to_string())
    print("  Most overestimated:")
    print(pg_g.tail(8).round(3).to_string())

    print("\n-- Zero-inflation accuracy --")
    for stat in ["goals", "assists"]:
        grp = pf[pf["stat"] == stat]
        pred_zero   = grp["pred"] < 0.5
        actual_zero = grp["actual"] == 0
        pz  = (pred_zero & actual_zero).sum() / max(pred_zero.sum(), 1)
        pnz = (~pred_zero & actual_zero).sum() / max((~pred_zero).sum(), 1)
        print(f"  {stat:8s}: model_0={pred_zero.mean():.0%}  actual_0={actual_zero.mean():.0%}  "
              f"when_pred_0_is_0={pz:.0%}  when_pred_gt0_is_0={pnz:.0%}")

    print("\n-- Goal projection correlation by week of season --")
    goals_pf = pf[pf["stat"] == "goals"].copy()
    for wk, grp in goals_pf.groupby("game_num"):
        if len(grp) < 5:
            continue
        corr = grp["pred"].corr(grp["actual"])
        bias = grp["error"].mean()
        print(f"  Game {wk:2d}: n={len(grp):3d}  corr={corr:.3f}  bias={bias:+.3f}")

    print("\n-- Assist projection summary --")
    ag = pf[pf["stat"] == "assists"]
    print(f"  MAE={ag['abs_error'].mean():.3f}  bias={ag['error'].mean():+.3f}  "
          f"corr={ag['pred'].corr(ag['actual']):.3f}  "
          f"avg_pred={ag['pred'].mean():.2f}  avg_actual={ag['actual'].mean():.2f}")

    print("\n-- Shot projection summary --")
    sg = pf[pf["stat"] == "shots"]
    print(f"  MAE={sg['abs_error'].mean():.3f}  bias={sg['error'].mean():+.3f}  "
          f"corr={sg['pred'].corr(sg['actual']):.3f}  "
          f"avg_pred={sg['pred'].mean():.2f}  avg_actual={sg['actual'].mean():.2f}")

print("\n" + "="*65)
print("BACKTEST COMPLETE")
print("="*65)
