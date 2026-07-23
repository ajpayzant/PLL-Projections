"""
Phase 1 signal test for opponent-2pt-allowed (leakage-safe, out-of-sample).

Question: a team projects its 2pt goals from its OWN offensive 2pt rate only —
the opponent's 2pt-ALLOWED tendency is collected but unused. Before wiring it
into the engine we must prove it carries real, exploitable signal (the same bar
that KILLED the Bradley-Terry faceoff idea and the possession-pace idea).

Method (mirrors the PBP head-to-head discipline):
- Build per-team-game rows with a leakage-safe EWM (only PRIOR games) of:
    own_2pt_rate       = two_point_goals / goals            (offense intent/skill)
    opp_2pt_allow_rate = two_point_goals_against / goals_against  (defense leak)
- Target = actual two_point_goals in the current game.
- TRAIN 2022-24, TEST 2025 holdout. Compare nested OLS models out-of-sample:
    M1: own prior 2pt rate  x  own goals        (what the engine does today)
    M2: M1 + opponent prior 2pt-allowed rate    (the proposed addition)
  If M2's out-of-sample R2 does not beat M1 by a meaningful margin, the
  opponent signal is noise -> DO NOT wire it (UI slider only, if anything).
- Also report autocorrelation of each rate (trait stability).
"""
import duckdb
import numpy as np
import pandas as pd

DB = r"data/analytics_database/pll_warehouse.duckdb"
TRAIN = [2022, 2023, 2024]
TEST = 2025
HL = 8  # EWM half-life in games (matches engine HL_GOALS-ish)


def ewm_prior(series: pd.Series, hl: int) -> pd.Series:
    """EWM using ONLY prior rows (shift 1) — leakage-safe."""
    return series.shift(1).ewm(halflife=hl, min_periods=1).mean()


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT season, game_id, game_number, team_id, opponent_team_id, is_home,
               goals, two_point_goals, goals_against, two_point_goals_against
        FROM clean.team_game_stats
        ORDER BY team_id, season, game_number
    """).df()

    df = df.dropna(subset=["goals", "two_point_goals"]).copy()
    df["goals"] = df["goals"].astype(float)
    df["two_point_goals"] = df["two_point_goals"].astype(float)
    df["goals_against"] = df["goals_against"].fillna(0).astype(float)
    df["two_point_goals_against"] = df["two_point_goals_against"].fillna(0).astype(float)

    # Per-game rates
    df["own_2pt_rate"] = df["two_point_goals"] / df["goals"].clip(lower=1)
    df["opp_2pt_allow_rate"] = (df["two_point_goals_against"] /
                                df["goals_against"].clip(lower=1))

    # Leakage-safe priors, computed within each team's chronological history.
    df = df.sort_values(["team_id", "season", "game_number"])
    df["own_2pt_rate_prior"] = (df.groupby("team_id")["own_2pt_rate"]
                                  .transform(lambda s: ewm_prior(s, HL)))
    df["own_2pt_allow_prior"] = (df.groupby("team_id")["opp_2pt_allow_rate"]
                                   .transform(lambda s: ewm_prior(s, HL)))

    # ── Autocorrelation (trait stability) ──────────────────────────────
    def autocorr(col, prior):
        sub = df.dropna(subset=[col, prior])
        return np.corrcoef(sub[col], sub[prior])[0, 1] if len(sub) > 10 else np.nan
    print("=" * 60)
    print("AUTOCORRELATION (own rate vs its own prior EWM):")
    print(f"  own_2pt_rate       : {autocorr('own_2pt_rate', 'own_2pt_rate_prior'):.3f}")
    print(f"  opp_2pt_allow_rate : {autocorr('opp_2pt_allow_rate', 'own_2pt_allow_prior'):.3f}")
    print("  (near 0 = noise/no stable trait; >0.3 = some signal)")

    # ── Bring OPPONENT's prior 2pt-allowed onto each team-game row ───────
    # For team T vs opponent O in game G, we want O's prior 2pt-ALLOWED rate.
    opp_lookup = df[["game_id", "team_id", "own_2pt_allow_prior"]].rename(
        columns={"team_id": "opponent_team_id",
                 "own_2pt_allow_prior": "opp_def_2pt_allow_prior"})
    df = df.merge(opp_lookup, on=["game_id", "opponent_team_id"], how="left")

    # ── Nested OLS, out-of-sample ───────────────────────────────────────
    feat_cols = ["own_2pt_rate_prior", "goals", "opp_def_2pt_allow_prior"]
    d = df.dropna(subset=feat_cols + ["two_point_goals"]).copy()
    tr = d[d["season"].isin(TRAIN)]
    te = d[d["season"] == TEST]
    print("=" * 60)
    print(f"OLS nested test  (train n={len(tr)}, test 2025 n={len(te)})")

    def fit_predict(train, test, cols):
        X = np.column_stack([np.ones(len(train))] + [train[c].values for c in cols])
        y = train["two_point_goals"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        Xt = np.column_stack([np.ones(len(test))] + [test[c].values for c in cols])
        pred = Xt @ beta
        ss_res = np.sum((test["two_point_goals"].values - pred) ** 2)
        ss_tot = np.sum((test["two_point_goals"].values - test["two_point_goals"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(test["two_point_goals"].values - pred))
        return r2, mae

    # M1: what the engine does today (own prior 2pt rate scaled by volume)
    r2_1, mae_1 = fit_predict(tr, te, ["own_2pt_rate_prior", "goals"])
    # M2: add opponent prior 2pt-allowed
    r2_2, mae_2 = fit_predict(tr, te, ["own_2pt_rate_prior", "goals", "opp_def_2pt_allow_prior"])

    print(f"  M1 own-only         : oos_R2={r2_1:+.4f}  MAE={mae_1:.3f}")
    print(f"  M2 +opp_2pt_allowed : oos_R2={r2_2:+.4f}  MAE={mae_2:.3f}")
    print(f"  delta (M2 - M1)     : dR2={r2_2 - r2_1:+.4f}  dMAE={mae_2 - mae_1:+.4f}")
    print("=" * 60)
    print("VERDICT GUIDE: dR2 <= ~0 or dMAE >= 0  -> opponent 2pt-allowed is")
    print("NOISE, do NOT wire into engine. dR2 >> 0 and dMAE < 0 -> real signal.")


if __name__ == "__main__":
    main()
