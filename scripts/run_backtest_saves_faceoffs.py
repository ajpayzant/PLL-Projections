"""
Backtest for SAVE and FACEOFF projections (team-level + per-player).

Scope:
  - Backtest 2024-2026 games with leakage-safe ratings (all games strictly before
    the game being projected are used to build ratings).
  - Split labels: train (2024) / test (2025) / current (2026)
  - We DON'T sim field players here — we build NegBin distributions directly
    for saves (goalies) and fo_wins (FO specialists) using the exact same
    parameterisation as GameSimulator.simulate_players. This is ~20x faster than
    running the full player sim per game because we skip zi-NegBin field draws,
    Cholesky correlation, shot draws, etc.

Metrics reported:
  - Point estimate: MAE, RMSE, bias, correlation (per split, per season)
  - Distribution calibration: coverage inside [P25,P75] and [P10,P90]
    Empirical residual variance vs simulated variance (are we too tight/wide?)
  - Prop-line calibration: fair Over% (from sim CDF) vs realised Over hit rate
  - Empirical var/mean vs PHI_PLAYER['saves'|'fo_wins'] implied var/mean
  - Best/worst goalie & FO specialist bias
"""
import sys, warnings, logging, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
logging.getLogger("pll.v3").setLevel(logging.WARNING)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from projection_engine_v3 import (
    ProjectionEngine, RatingBuilder, TeamModel,
    PlayerModel, PricingEngine,
    _assign_goalie_saves_team, _assign_player_goalie_saves,
    _assign_faceoff_from_specialist,
    _negbinom_params,
    LG_SAVE_PCT, LG_FO_PCT, LG_FOS_PER_GAME, LG_SAVES, LG_CLEAN_SAVE_RATE,
    PHI_PLAYER,
)

BACKTEST_SEASONS = [2024, 2025, 2026]
TRAIN_LABEL = 2024
TEST_LABEL  = 2025
CURR_LABEL  = 2026
N_SIMS      = 4_000
SEED        = 17

print("=" * 72, flush=True)
print("PLL SAVE & FACEOFF BACKTEST", flush=True)
print(f"  Backtest seasons: {BACKTEST_SEASONS}  ({TRAIN_LABEL}=train, "
      f"{TEST_LABEL}=test, {CURR_LABEL}=current)", flush=True)
print(f"  Sims per player-game: {N_SIMS}", flush=True)
print("=" * 72, flush=True)

print("\nLoading data...", flush=True)
eng = ProjectionEngine()
eng.load()
tg = eng.team_games.copy()
pg = eng.player_games.copy()
pg["position"] = pg["position"].astype(str).str.strip().str.upper()

# Which games to score?
season_games = tg.drop_duplicates("game_id")[["game_id", "season", "game_number"]]
season_games = season_games[season_games["season"].isin(BACKTEST_SEASONS)]
games = season_games.sort_values(["season", "game_number"]).to_dict("records")
print(f"Total games to consider: {len(games)}", flush=True)

pricing = PricingEngine(hold_pct=0.045)
rng = np.random.default_rng(SEED)

goalie_rows    = []
fo_rows        = []
team_save_rows = []
team_fo_rows   = []


def _nb_draw(mu: float, phi: float, n: int) -> np.ndarray:
    nb_n, nb_p = _negbinom_params(max(mu, 0.01), phi)
    return rng.negative_binomial(nb_n, nb_p, n).astype(float)


def _split_of(season):
    if season == TRAIN_LABEL: return "train"
    if season == TEST_LABEL:  return "test"
    if season == CURR_LABEL:  return "current"
    return "other"


t0 = pd.Timestamp.utcnow()

for i, grow in enumerate(games):
    gid   = grow["game_id"]
    gsea  = int(grow["season"])
    gnum  = int(grow["game_number"])
    if (i + 1) % 10 == 0:
        elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
        eta = elapsed / (i + 1) * (len(games) - i - 1)
        print(f"  {i+1}/{len(games)}  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    game_pg = pg[pg["game_id"] == gid]
    game_tg = tg[tg["game_id"] == gid]
    if len(game_tg) != 2 or game_pg.empty:
        continue

    # Leakage-safe training set
    train_tg = tg[
        (tg["season"] < gsea) |
        ((tg["season"] == gsea) & (tg["game_number"] < gnum))
    ]
    train_pg = pg[
        (pg["season"] < gsea) |
        ((pg["season"] == gsea) & (pg["game_number"] < gnum))
    ]
    if len(train_tg) < 30 or len(train_pg) < 100:
        continue

    try:
        rb = RatingBuilder(train_tg, train_pg)
        rb.build_team_ratings()
        rb.build_player_ratings()
        if rb._pr.empty:
            continue
        tm = TeamModel()
        tm.fit(rb._tr)
    except Exception:
        continue

    home_r = game_tg[game_tg["is_home"] == 1]
    away_r = game_tg[game_tg["is_home"] == 0]
    if home_r.empty or away_r.empty:
        home_r = game_tg.iloc[[0]]
        away_r = game_tg.iloc[[1]]
    home_team = str(home_r.iloc[0]["team_id"])
    away_team = str(away_r.iloc[0]["team_id"])

    hf = rb.get_team_rating(home_team)
    af = rb.get_team_rating(away_team)
    if not hf or not af:
        continue

    try:
        hp = tm.predict(hf, af)
        ap = tm.predict(af, hf)
    except Exception:
        continue
    _assign_goalie_saves_team(hp, ap)
    _assign_goalie_saves_team(ap, hp)

    actual_ids_by_team = {}
    for tid_u in game_tg["team_id"].unique():
        actual_ids_by_team[str(tid_u)] = set(
            game_pg[game_pg["team_id"] == tid_u]["player_id"].astype(str).tolist()
        )

    # ── Team-level saves & fo_wins ──
    for tid_r in game_tg.itertuples():
        tid = str(tid_r.team_id)
        team_proj = hp if tid == home_team else ap
        team_save_rows.append({
            "season": gsea, "game_number": gnum, "team_id": tid,
            "pred":   float(team_proj.proj_saves),
            "actual": float(getattr(tid_r, "saves", 0) or 0),
            "split":  _split_of(gsea),
        })
        team_fo_rows.append({
            "season": gsea, "game_number": gnum, "team_id": tid,
            "pred":   float(team_proj.proj_faceoff_wins),
            "actual": float(getattr(tid_r, "faceoffs_won", 0) or 0),
            "split":  _split_of(gsea),
        })

    # ── Build player projections restricted to actual participants ──
    proj_by_team = {}
    for tid, tp in [(home_team, hp), (away_team, ap)]:
        try:
            pm = PlayerModel(rb._pr)
            active_ids = actual_ids_by_team.get(tid, set())
            overrides = {}
            pr_team = rb._pr[rb._pr["team_id"] == tid]
            for pid_o in pr_team["player_id"].astype(str):
                if pid_o not in active_ids:
                    overrides[pid_o] = {"active": False, "usage_multiplier": 0.0}
            projs = pm.project_roster(tid, tp, overrides=overrides,
                                       use_current_roster_filter=False)
            proj_by_team[tid] = [p for p in projs if p.active]
        except Exception:
            proj_by_team[tid] = []

    # Assign proper starter goalie saves
    try:
        _assign_player_goalie_saves(proj_by_team[home_team], ap.proj_sog, None)
        _assign_player_goalie_saves(proj_by_team[away_team], hp.proj_sog, None)
        # Player-driven faceoff wins with log5 opponent adj (mirrors ProjectionEngine).
        _assign_faceoff_from_specialist(proj_by_team[home_team], proj_by_team[away_team])
        _assign_faceoff_from_specialist(proj_by_team[away_team], proj_by_team[home_team])
    except Exception:
        pass

    # ── Goalie backtest (starter picked by actual shots_faced) ──
    for tid in [home_team, away_team]:
        team_actuals = game_pg[game_pg["team_id"] == tid]
        goalies_act = team_actuals[team_actuals["position"] == "G"].copy()
        if goalies_act.empty:
            continue
        goalies_act["shots_faced"] = (goalies_act["saves"].fillna(0)
                                      + goalies_act["goals_against"].fillna(0))
        starter_actual = goalies_act.sort_values("shots_faced", ascending=False).iloc[0]

        goalie_projs = [p for p in proj_by_team[tid] if p.position == "G"
                        and p.proj_saves > 0]
        if not goalie_projs:
            continue
        goalie_proj = goalie_projs[0]

        # NegBin sim distribution — mirrors simulate_players exactly
        csr = getattr(goalie_proj, "clean_save_rate", LG_CLEAN_SAVE_RATE) or LG_CLEAN_SAVE_RATE
        csr_ratio = csr / max(LG_CLEAN_SAVE_RATE, 0.01)
        goalie_phi = float(np.clip(PHI_PLAYER["saves"] * csr_ratio, 70.0, 220.0))
        dist = _nb_draw(max(goalie_proj.proj_saves, 0.01), goalie_phi, N_SIMS)

        act_saves = float(starter_actual.get("saves", 0) or 0)
        act_sf    = float(starter_actual["shots_faced"])
        act_svpct = act_saves / act_sf if act_sf > 0 else float("nan")
        pid_match = (goalie_proj.player_id == str(starter_actual["player_id"]))

        line = pricing._opt_line(dist)
        fair_over = float(np.mean(dist > line))
        over_hit  = int(act_saves > line)

        goalie_rows.append({
            "season": gsea, "game_number": gnum, "team_id": tid,
            "goalie": goalie_proj.full_name,
            "goalie_id_match": pid_match,
            "actual_starter": starter_actual["full_name"],
            "pred_saves":    float(goalie_proj.proj_saves),
            "actual_saves":  act_saves,
            "pred_sv_pct":   float(goalie_proj.proj_save_pct),
            "actual_sv_pct": act_svpct,
            "actual_shots_faced": act_sf,
            "clean_save_rate":    float(csr),
            "phi":  goalie_phi,
            "dist_mean":  float(dist.mean()),
            "dist_std":   float(dist.std()),
            "dist_p10":   float(np.percentile(dist, 10)),
            "dist_p25":   float(np.percentile(dist, 25)),
            "dist_p50":   float(np.percentile(dist, 50)),
            "dist_p75":   float(np.percentile(dist, 75)),
            "dist_p90":   float(np.percentile(dist, 90)),
            "line":       line,
            "fair_over":  fair_over,
            "over_hit":   over_hit,
            "split": _split_of(gsea),
        })

    # ── FO specialist backtest ──
    for tid in [home_team, away_team]:
        team_actuals = game_pg[game_pg["team_id"] == tid]
        fo_act = team_actuals[team_actuals["position"] == "FO"].copy()
        fo_act["fo_denom"] = (fo_act["faceoffs_won"].fillna(0)
                              + fo_act["faceoffs_lost"].fillna(0))
        fo_act = fo_act[fo_act["fo_denom"] > 0]
        if fo_act.empty:
            continue

        team_fo_projs = [p for p in proj_by_team[tid]
                         if p.position == "FO" and p.proj_faceoff_wins > 0]

        for _, arow in fo_act.iterrows():
            pid = str(arow["player_id"])
            proj = next((p for p in team_fo_projs if p.player_id == pid), None)
            if proj is None:
                continue
            dist = _nb_draw(max(proj.proj_faceoff_wins, 0.01),
                            PHI_PLAYER["fo_wins"], N_SIMS)
            act_wins  = float(arow.get("faceoffs_won", 0) or 0)
            act_denom = float(arow["fo_denom"])
            act_pct   = act_wins / act_denom if act_denom > 0 else float("nan")

            line = pricing._opt_line(dist)
            fair_over = float(np.mean(dist > line))
            over_hit  = int(act_wins > line)

            fo_rows.append({
                "season": gsea, "game_number": gnum, "team_id": tid,
                "player":  arow["full_name"],
                "pred_fo_wins":    float(proj.proj_faceoff_wins),
                "actual_fo_wins":  act_wins,
                "pred_fo_pct":     float(proj.proj_faceoff_pct),
                "actual_fo_pct":   act_pct,
                "actual_fo_denom": act_denom,
                "dist_mean":  float(dist.mean()),
                "dist_std":   float(dist.std()),
                "dist_p10":   float(np.percentile(dist, 10)),
                "dist_p25":   float(np.percentile(dist, 25)),
                "dist_p50":   float(np.percentile(dist, 50)),
                "dist_p75":   float(np.percentile(dist, 75)),
                "dist_p90":   float(np.percentile(dist, 90)),
                "line":       line,
                "fair_over":  fair_over,
                "over_hit":   over_hit,
                "split":      _split_of(gsea),
            })


print("\nCollected:", flush=True)
print(f"  Team-save rows: {len(team_save_rows)}", flush=True)
print(f"  Team-FO rows:   {len(team_fo_rows)}", flush=True)
print(f"  Goalie rows:    {len(goalie_rows)}", flush=True)
print(f"  FO player rows: {len(fo_rows)}", flush=True)

out_dir = "scripts/backtest_output"
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(team_save_rows).to_csv(f"{out_dir}/team_saves.csv", index=False)
pd.DataFrame(team_fo_rows).to_csv(f"{out_dir}/team_faceoffs.csv", index=False)
pd.DataFrame(goalie_rows).to_csv(f"{out_dir}/goalie_saves.csv", index=False)
pd.DataFrame(fo_rows).to_csv(f"{out_dir}/fo_wins.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════
def _metrics(df, pred_col, act_col, label):
    if df.empty:
        print(f"  [{label}] no rows")
        return
    err = df[pred_col] - df[act_col]
    mae  = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    bias = err.mean()
    corr = df[pred_col].corr(df[act_col])
    print(f"  [{label}] n={len(df)}  pred={df[pred_col].mean():.2f}  "
          f"actual={df[act_col].mean():.2f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
          f"bias={bias:+.3f}  corr={corr:.3f}")

def _coverage(df, act_col):
    if df.empty:
        return
    for lo, hi, label, target in [("dist_p25","dist_p75","50%", 0.50),
                                   ("dist_p10","dist_p90","80%", 0.80)]:
        inside = ((df[act_col] >= df[lo]) & (df[act_col] <= df[hi])).mean()
        flag = "OK" if abs(inside - target) < 0.06 else \
               "TOO_WIDE" if inside > target else "TOO_TIGHT"
        print(f"    Coverage {label}: {inside*100:.1f}%  (target {target*100:.0f}%)  [{flag}]")

def _variance_check(df, pred_col, act_col):
    if df.empty:
        return
    sim_var_avg   = (df["dist_std"] ** 2).mean()
    resid_var     = (df[act_col] - df[pred_col]).var()
    emp_marg_var  = df[act_col].var()
    print(f"    Avg simulated var    = {sim_var_avg:.3f}")
    print(f"    Empirical residual var = {resid_var:.3f} "
          f"(actual − pred)  → ratio sim/resid = {sim_var_avg/max(resid_var,0.01):.3f}")
    print(f"    Empirical marginal var = {emp_marg_var:.3f}")

def _prop_cal(df, act_col, label):
    if df.empty:
        return
    df = df.copy()
    df["actual_over"] = (df[act_col] > df["line"]).astype(int)
    df["bucket"] = pd.cut(df["fair_over"], bins=[0, .35, .45, .5, .55, .65, 1.0],
                          right=False)
    cal = df.groupby("bucket", observed=True).agg(
        n=("actual_over","count"),
        pred=("fair_over","mean"),
        actual=("actual_over","mean"),
    ).reset_index()
    print(f"    Prop-line calibration ({label}):")
    print(cal.round(3).to_string(index=False))
    over_rate = df["actual_over"].mean()
    pred_over = df["fair_over"].mean()
    print(f"    Overall  pred_over={pred_over:.3f}  actual_over={over_rate:.3f}")
    ls = np.where(df["actual_over"] == 1,
                  np.log(np.clip(df["fair_over"], 1e-4, 1)),
                  np.log(np.clip(1 - df["fair_over"], 1e-4, 1)))
    print(f"    Mean log-score = {ls.mean():.4f}  (higher is better; naive 50/50 = {np.log(0.5):.4f})")

def _section(title):
    print("\n" + "─" * 72)
    print(f"  {title}")
    print("─" * 72)


ts = pd.DataFrame(team_save_rows)
tf = pd.DataFrame(team_fo_rows)
gs = pd.DataFrame(goalie_rows)
fo = pd.DataFrame(fo_rows)

# ────────────────── TEAM SAVES ──────────────────
_section("TEAM SAVES  (opp SOG × Bayes save%)")
for split in ["train", "test", "current"]:
    _metrics(ts[ts["split"] == split], "pred", "actual", f"{split.upper()}")

# ────────────────── TEAM FO WINS ──────────────────
_section("TEAM FACEOFF WINS")
for split in ["train", "test", "current"]:
    _metrics(tf[tf["split"] == split], "pred", "actual", f"{split.upper()}")

# ────────────────── STARTER GOALIE SAVES ──────────────────
_section("STARTER GOALIE SAVES  (per game)")
for split in ["train", "test", "current"]:
    sub = gs[gs["split"] == split]
    _metrics(sub, "pred_saves", "actual_saves", f"{split.upper()}")
    if not sub.empty:
        _coverage(sub, "actual_saves")
        _variance_check(sub, "pred_saves", "actual_saves")

_section("STARTER GOALIE SAVES  — goalie ID match only (TEST)")
sub_m = gs[(gs["split"] == "test") & (gs["goalie_id_match"])]
_metrics(sub_m, "pred_saves", "actual_saves", "TEST-matched")
if not sub_m.empty:
    _coverage(sub_m, "actual_saves")
    _variance_check(sub_m, "pred_saves", "actual_saves")

_section("SAVES — prop-line calibration (TEST)")
_prop_cal(gs[gs["split"] == "test"], "actual_saves", "saves")

_section("SAVE% — pred vs actual  (per starter-game, TEST)")
sub_sp = gs[(gs["split"] == "test") & gs["actual_sv_pct"].notna()]
if not sub_sp.empty:
    err = sub_sp["pred_sv_pct"] - sub_sp["actual_sv_pct"]
    print(f"  n={len(sub_sp)}  pred_avg={sub_sp['pred_sv_pct'].mean():.3f}  "
          f"actual_avg={sub_sp['actual_sv_pct'].mean():.3f}  "
          f"MAE={err.abs().mean():.4f}  bias={err.mean():+.4f}  "
          f"corr={sub_sp['pred_sv_pct'].corr(sub_sp['actual_sv_pct']):.3f}")

_section("SAVES — best/worst by goalie (TEST, ≥5 games, matched IDs)")
tg_test = gs[(gs["split"] == "test") & (gs["goalie_id_match"])]
if not tg_test.empty:
    per = tg_test.groupby("goalie").agg(
        n=("actual_saves", "count"),
        avg_pred=("pred_saves", "mean"),
        avg_actual=("actual_saves", "mean"),
    )
    per["bias"] = per["avg_pred"] - per["avg_actual"]
    per["mae"] = tg_test.groupby("goalie").apply(
        lambda d: (d["pred_saves"] - d["actual_saves"]).abs().mean()
    )
    per = per[per["n"] >= 5].sort_values("bias")
    print(per.round(3).to_string())

# ────────────────── FO SPECIALIST WINS ──────────────────
_section("FACEOFF SPECIALIST WINS  (per game)")
for split in ["train", "test", "current"]:
    sub = fo[fo["split"] == split]
    _metrics(sub, "pred_fo_wins", "actual_fo_wins", f"{split.upper()}")
    if not sub.empty:
        _coverage(sub, "actual_fo_wins")
        _variance_check(sub, "pred_fo_wins", "actual_fo_wins")

_section("FO WINS — prop-line calibration (TEST)")
_prop_cal(fo[fo["split"] == "test"], "actual_fo_wins", "fo_wins")

_section("FO% — pred vs actual (per player-game, TEST, denom≥5)")
sub = fo[(fo["split"] == "test") & (fo["actual_fo_denom"] >= 5)]
if not sub.empty:
    err = sub["pred_fo_pct"] - sub["actual_fo_pct"]
    print(f"  n={len(sub)}  pred_avg={sub['pred_fo_pct'].mean():.3f}  "
          f"actual_avg={sub['actual_fo_pct'].mean():.3f}  "
          f"MAE={err.abs().mean():.4f}  bias={err.mean():+.4f}  "
          f"corr={sub['pred_fo_pct'].corr(sub['actual_fo_pct']):.3f}")

_section("FO — best/worst per player (TEST, ≥5 games)")
fp = fo[fo["split"] == "test"]
if not fp.empty:
    per = fp.groupby("player").agg(
        n=("actual_fo_wins", "count"),
        avg_pred=("pred_fo_wins", "mean"),
        avg_actual=("actual_fo_wins", "mean"),
    )
    per["bias"] = per["avg_pred"] - per["avg_actual"]
    per["mae"] = fp.groupby("player").apply(
        lambda d: (d["pred_fo_wins"] - d["actual_fo_wins"]).abs().mean()
    )
    per = per[per["n"] >= 5].sort_values("bias")
    print(per.round(3).to_string())

# ────────────────── EMPIRICAL VAR/MEAN vs PHI-implied ──────────────────
_section("Empirical var/mean vs PHI-implied var/mean")

def _var_mean(df, act_col):
    m = df[act_col].mean()
    v = df[act_col].var()
    return v / max(m, 0.01), m, v

def _phi_implied(mu, phi):
    n = max(int(round(phi)), 1)
    return 1.0 + mu / n

if not gs.empty:
    gs_test = gs[gs["split"] == "test"]
    vm, m, v = _var_mean(gs_test, "actual_saves")
    print(f"  Saves   (empirical, TEST):   n={len(gs_test)}  mean={m:.2f}  "
          f"var={v:.2f}  var/mean={vm:.3f}")
    phi_saves = PHI_PLAYER["saves"]
    print(f"    PHI_PLAYER['saves']={phi_saves} → NB var/mean at mu={m:.2f}: "
          f"{_phi_implied(m, phi_saves):.3f}")
    # Also residual var/mean
    res = gs_test["actual_saves"] - gs_test["pred_saves"]
    print(f"    Residual std = {res.std():.3f}  |  simulated std avg = "
          f"{gs_test['dist_std'].mean():.3f}")

if not fo.empty:
    fo_test = fo[fo["split"] == "test"]
    vm, m, v = _var_mean(fo_test, "actual_fo_wins")
    print(f"  FO wins (empirical, TEST): n={len(fo_test)}  mean={m:.2f}  "
          f"var={v:.2f}  var/mean={vm:.3f}")
    phi_fo = PHI_PLAYER["fo_wins"]
    print(f"    PHI_PLAYER['fo_wins']={phi_fo} → NB var/mean at mu={m:.2f}: "
          f"{_phi_implied(m, phi_fo):.3f}")
    res = fo_test["actual_fo_wins"] - fo_test["pred_fo_wins"]
    print(f"    Residual std = {res.std():.3f}  |  simulated std avg = "
          f"{fo_test['dist_std'].mean():.3f}")

print("\n" + "=" * 72)
print("BACKTEST COMPLETE")
print("=" * 72)
