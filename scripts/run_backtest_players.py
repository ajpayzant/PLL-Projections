import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from projection_engine_v3 import (
    ProjectionEngine, RatingBuilder, TeamModel, GameSimulator,
    PlayerModel, _assign_goalie_saves_team,
)

print("Loading engine...")
eng = ProjectionEngine()
eng.load()
eng.fit(run_backtest=False)
tg = eng.team_games
pg = eng.player_games

print("\n" + "="*60)
print("PLAYER-LEVEL BACKTEST — 2024 & 2025 (A/M positions)")
print("="*60)

player_rows = []
val_pg = pg[pg["season"].isin([2024, 2025])].copy()
game_ids = val_pg["game_id"].unique()
print(f"Games to backtest: {len(game_ids)}")

for idx, gid in enumerate(game_ids):
    if idx % 10 == 0:
        print(f"  Processing game {idx+1}/{len(game_ids)}...")
    game_pg = val_pg[val_pg["game_id"] == gid]
    game_tg = tg[tg["game_id"] == gid]
    if game_tg.empty or len(game_tg) != 2:
        continue
    gsea = int(game_pg["season"].iloc[0])
    gnum = int(game_pg["game_number"].iloc[0])

    train_tg = tg[
        (tg["season"] < gsea) |
        ((tg["season"] == gsea) & (tg["game_number"] < gnum))
    ].copy()
    train_pg = pg[
        (pg["season"] < gsea) |
        ((pg["season"] == gsea) & (pg["game_number"] < gnum))
    ].copy()
    if len(train_tg) < 20 or len(train_pg) < 50:
        continue

    try:
        rb = RatingBuilder(train_tg, train_pg)
        rb.build_team_ratings()
        rb.build_player_ratings()
        tm = TeamModel()
        tm.fit(rb._tr)

        for tid in game_tg["team_id"].unique():
            opp_rows = game_tg[game_tg["team_id"] != tid]
            if opp_rows.empty:
                continue
            opp_tid = str(opp_rows.iloc[0]["team_id"])
            hf = rb.get_team_rating(str(tid))
            af = rb.get_team_rating(opp_tid)
            if not hf or not af:
                continue
            tp = tm.predict(hf, af)
            opp_tp = tm.predict(af, hf)
            _assign_goalie_saves_team(tp, opp_tp)

            pm = PlayerModel(rb._pr)
            projs = pm.project_roster(str(tid), tp, use_current_roster_filter=False)

            actuals = game_pg[
                (game_pg["team_id"] == tid) &
                (game_pg["position"].isin(["A", "M"]))
            ]
            for _, arow in actuals.iterrows():
                pid = str(arow["player_id"])
                proj = next((p for p in projs if p.player_id == pid), None)
                if proj is None or not proj.active:
                    continue
                for stat, act_col in [("goals","goals"),("assists","assists"),("shots","shots")]:
                    act_val = float(arow.get(act_col, 0) or 0)
                    pred_val = getattr(proj, f"proj_{stat}", 0.0)
                    player_rows.append({
                        "season": gsea, "player": arow["full_name"],
                        "position": arow["position"], "stat": stat,
                        "pred": pred_val, "actual": act_val,
                        "error": pred_val - act_val,
                        "abs_error": abs(pred_val - act_val),
                    })
    except Exception as e:
        continue

pf = pd.DataFrame(player_rows)
print(f"\nPlayer-game-stat rows collected: {len(pf)}")

if pf.empty:
    print("No rows — check player_id matching")
else:
    print("\n-- By stat --")
    for stat, grp in pf.groupby("stat"):
        mae = grp["abs_error"].mean()
        bias = grp["error"].mean()
        avg_a = grp["actual"].mean()
        avg_p = grp["pred"].mean()
        corr = grp["pred"].corr(grp["actual"])
        zero_a = (grp["actual"] == 0).mean()
        print(f"  {stat:8s}: MAE={mae:.3f}  bias={bias:+.3f}  avg_actual={avg_a:.2f}  avg_pred={avg_p:.2f}  corr={corr:.3f}  actual_zero_rate={zero_a:.0%}")

    print("\n-- By position x stat --")
    for (pos, stat), grp in pf.groupby(["position", "stat"]):
        mae = grp["abs_error"].mean()
        bias = grp["error"].mean()
        print(f"  {pos:6s} {stat:8s}: MAE={mae:.3f}  bias={bias:+.3f}")

    print("\n-- Most overestimated players (goals, 15+ games) --")
    pg_g = pf[pf["stat"]=="goals"].groupby("player").agg(
        n=("actual","count"), mae=("abs_error","mean"),
        bias=("error","mean"), avg_actual=("actual","mean"), avg_pred=("pred","mean")
    ).query("n >= 15").sort_values("bias", ascending=False)
    print(pg_g.head(8).round(3).to_string())

    print("\n-- Most underestimated players (goals, 15+ games) --")
    print(pg_g.tail(8).round(3).to_string())

    print("\n-- Zero-inflation accuracy --")
    for stat in ["goals", "assists"]:
        grp = pf[pf["stat"]==stat]
        pred_zero = grp["pred"] < 0.5
        actual_zero = grp["actual"] == 0
        n_pred_zero = pred_zero.sum()
        n_pred_nonzero = (~pred_zero).sum()
        pz_correct = (pred_zero & actual_zero).sum() / max(n_pred_zero, 1)
        pnz_zero = (~pred_zero & actual_zero).sum() / max(n_pred_nonzero, 1)
        print(f"  {stat}: model_zero_rate={pred_zero.mean():.0%}  actual_zero_rate={actual_zero.mean():.0%}  "
              f"when_pred_0_is_0={pz_correct:.0%}  when_pred_gt0_is_0={pnz_zero:.0%}")

    print("\n-- Shot projection --")
    sg = pf[pf["stat"]=="shots"]
    print(f"  MAE={sg['abs_error'].mean():.3f}  bias={sg['error'].mean():+.3f}  "
          f"corr={sg['pred'].corr(sg['actual']):.3f}  avg_pred={sg['pred'].mean():.2f}  avg_actual={sg['actual'].mean():.2f}")

    print("\n-- Assist projection --")
    ag = pf[pf["stat"]=="assists"]
    print(f"  MAE={ag['abs_error'].mean():.3f}  bias={ag['error'].mean():+.3f}  "
          f"corr={ag['pred'].corr(ag['actual']):.3f}  avg_pred={ag['pred'].mean():.2f}  avg_actual={ag['actual'].mean():.2f}")

print("\nDone.")
