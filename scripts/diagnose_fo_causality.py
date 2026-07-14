"""Diagnostic: is the PLAYER faceoff rating a better predictor of a team's
actual faceoff wins than the TEAM faceoff rating?

Since ~90% of a team's faceoffs are taken by one specialist, the player's own
leakage-safe bayes_fo_pct (built from player_game_stats) should predict the
team's faceoff outcome at least as well as the team-level bayes_fo_pct. If it
does, the engine's team->player FO flow is backwards and should be inverted.

Replicates the engine's bayes_fo_pct formula directly (vectorised, leakage-safe
via shift(1).cumsum()) — no per-game RatingBuilder, so this runs in seconds.
    team  bayes: _bayesian_rate over team faceoffs_won / (won+lost), a=b=2
    player bayes: same over player faceoffs_won / (won+lost), a=b=2
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import duckdb

LG_FO_PCT = 0.50
A = B = 2.0

con = duckdb.connect("data/analytics_database/pll_warehouse.duckdb", read_only=True)
tg = con.sql("""SELECT game_id, season, game_number, team_id,
                       COALESCE(faceoffs_won,0) AS fw, COALESCE(faceoffs_lost,0) AS fl
                FROM clean.team_game_stats""").df()
pg = con.sql("""SELECT game_id, season, game_number, team_id, player_id, full_name,
                       UPPER(position) AS position,
                       COALESCE(faceoffs_won,0) AS fw, COALESCE(faceoffs_lost,0) AS fl
                FROM clean.player_game_stats""").df()


def add_bayes(df, keys):
    """Add leakage-safe pre-game bayes_fo_pct within each group in `keys`."""
    df = df.sort_values(["season", "game_number"]).copy()
    df["denom"] = (df["fw"] + df["fl"]).clip(lower=0)
    g = df.groupby(keys, group_keys=False)
    # cumulative wins / denom BEFORE this game (shift 1)
    cum_w = g["fw"].apply(lambda s: s.shift(1).cumsum())
    cum_d = g["denom"].apply(lambda s: s.shift(1).cumsum())
    df["cum_w"] = cum_w.fillna(0.0)
    df["cum_d"] = cum_d.fillna(0.0)
    df["bayes_fo_pct"] = (df["cum_w"] + A) / (df["cum_d"] + A + B)
    return df


tg_b = add_bayes(tg, ["team_id"])
pg_b = add_bayes(pg, ["player_id"])

# team lookup of pre-game team bayes
team_lk = tg_b.set_index(["game_id", "team_id"])["bayes_fo_pct"].to_dict()

rows = []
for (gid, tid), grp in pg_b.groupby(["game_id", "team_id"]):
    sea = int(grp["season"].iloc[0]); gnum = int(grp["game_number"].iloc[0])
    if sea not in (2024, 2025, 2026):
        continue
    # actual team result this game
    trow = tg[(tg["game_id"] == gid) & (tg["team_id"] == tid)]
    if trow.empty:
        continue
    aw = float(trow.iloc[0]["fw"]); al = float(trow.iloc[0]["fl"])
    adenom = aw + al
    if adenom < 1:
        continue
    actual_fo_pct = aw / adenom

    # primary specialist = took most faceoffs THIS game
    grp2 = grp.copy(); grp2["gd"] = grp2["fw"] + grp2["fl"]
    grp2 = grp2[grp2["gd"] > 0]
    if grp2.empty:
        continue
    starter = grp2.sort_values("gd", ascending=False).iloc[0]
    player_fo_pct = float(starter["bayes_fo_pct"])
    starter_share = float(starter["gd"] / adenom)

    rows.append(dict(
        season=sea, game_number=gnum, team_id=str(tid),
        team_fo_pct=float(team_lk.get((gid, tid), LG_FO_PCT)),
        player_fo_pct=player_fo_pct,
        actual_fo_pct=actual_fo_pct, actual_denom=adenom, actual_wins=aw,
        starter_share=starter_share,
    ))

df = pd.DataFrame(rows)
print(f"\nCollected {len(df)} team-game rows")
print(f"Primary specialist's share of team faceoffs: mean={df['starter_share'].mean():.1%} "
      f"median={df['starter_share'].median():.1%}\n")


def summarize(pred_col, label, sub):
    x = sub.dropna(subset=[pred_col, "actual_fo_pct"])
    if len(x) < 5:
        print(f"  {label}: insufficient"); return
    err_pct = x[pred_col] - x["actual_fo_pct"]
    pred_wins = x[pred_col] * x["actual_denom"]
    err_w = pred_wins - x["actual_wins"]
    corr = x[pred_col].corr(x["actual_fo_pct"])
    print(f"  {label:16s} FO%: bias={err_pct.mean():+.3f} MAE={err_pct.abs().mean():.3f} "
          f"corr={corr:.3f}  | WINS(x actual denom): bias={err_w.mean():+.2f} "
          f"MAE={err_w.abs().mean():.2f} RMSE={np.sqrt((err_w**2).mean()):.2f}")


for sp, sea in [("TRAIN", 2024), ("TEST", 2025), ("CURRENT", 2026)]:
    sub = df[df["season"] == sea]
    print(f"── {sp} (n={len(sub)}) ──")
    summarize("team_fo_pct", "TEAM rating", sub)
    summarize("player_fo_pct", "PLAYER rating", sub)
    # blend
    sub2 = sub.copy(); sub2["blend"] = 0.5 * sub2["team_fo_pct"] + 0.5 * sub2["player_fo_pct"]
    summarize("blend", "50/50 blend", sub2)
    print()

d = df.dropna(subset=["team_fo_pct", "player_fo_pct"])
gap = (d["player_fo_pct"] - d["team_fo_pct"]).abs()
print(f"Team vs Player rating gap: mean={gap.mean():.3f}  "
      f">3pp: {(gap>0.03).mean()*100:.0f}%  >5pp: {(gap>0.05).mean()*100:.0f}%")
