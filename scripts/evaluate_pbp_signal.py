"""
evaluate_pbp_signal.py
----------------------
Does play-by-play (PBP) data carry predictive signal BEYOND the box-score
features the engine already uses? This is the gate before wiring anything into
projection_engine_v3.py — per project discipline, only ship features proven to
help.

Method (mirrors the engine's leakage-safe EWM feature construction:
`series.shift(1).ewm(halflife=hl).mean()` — only prior games inform each row):

  For each candidate we build, per player/team and ordered by date:
    * a BASELINE prior feature from the box score (what the engine has today)
    * a CANDIDATE prior feature derived from PBP
  Then predict the realized NEXT-game outcome and compare:
    * corr(baseline, outcome)                 how good is today's feature
    * corr(candidate, outcome)                how good is the PBP feature alone
    * incremental R^2 of [baseline + candidate] over [baseline]   <-- the money metric
      (if PBP adds ~0 incremental R^2, it's redundant with the box score)

Read-only. No engine import, no warehouse writes.

Usage:
    python scripts/evaluate_pbp_signal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CUR = REPO_ROOT / "data" / "curated_data" / "all_requested_seasons"

HALFLIFE = 8  # matches engine HL for shot/rate stats


def _ewm_prior(s: pd.Series, hl: int = HALFLIFE) -> pd.Series:
    """Leakage-safe EWM: value at row i uses only rows < i (engine convention)."""
    return s.shift(1).ewm(halflife=hl, min_periods=1).mean()


def _order_key(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a game order key (date) for within-entity sequencing."""
    gm = pd.read_parquet(CUR / "game_manifest.parquet",
                         columns=["game_slug", "game_date_utc", "game_number", "season"])
    gm = gm.drop_duplicates("game_slug")
    out = df.merge(gm[["game_slug", "game_date_utc", "game_number"]], on="game_slug", how="left")
    return out


def _incremental_r2(y: np.ndarray, base: np.ndarray, extra: np.ndarray) -> dict:
    """
    R^2 of OLS(y ~ base) vs OLS(y ~ base + extra), on standardized inputs.
    Returns both R^2 and the incremental gain. Uses lstsq (no sklearn dep).
    """
    m = np.isfinite(y) & np.isfinite(base) & np.isfinite(extra)
    y, base, extra = y[m], base[m], extra[m]
    n = len(y)
    if n < 50:
        return {"n": n, "r2_base": np.nan, "r2_full": np.nan, "delta_r2": np.nan,
                "corr_base": np.nan, "corr_extra": np.nan}

    def r2(X):
        X = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    r2_base = r2(base)
    r2_full = r2(np.column_stack([base, extra]))
    return {
        "n": n,
        "r2_base": round(r2_base, 4),
        "r2_full": round(r2_full, 4),
        "delta_r2": round(r2_full - r2_base, 4),
        "corr_base": round(np.corrcoef(base, y)[0, 1], 3),
        "corr_extra": round(np.corrcoef(extra, y)[0, 1], 3),
    }


# ---------------------------------------------------------------------------
# Candidate 1: shot-quality mix predicts shooting efficiency
#   baseline = prior blended shot_pct (goals/shots) from box
#   candidate = prior 2pt-shot rate from PBP (harder shots -> lower conversion)
#   outcome  = next-game shot_pct
# ---------------------------------------------------------------------------
def eval_shot_quality():
    box = pd.read_parquet(CUR / "player_game_stats.parquet",
                          columns=["season", "game_slug", "player_id", "goals", "shots",
                                   "shots_on_goal", "two_point_shots", "position"])
    pbp = pd.read_parquet(CUR / "pbp_player_game.parquet",
                          columns=["season", "game_slug", "player_id",
                                   "pbp_shots", "pbp_goals", "pbp_two_pt_shots", "pbp_shots_garbage"])
    box["player_id"] = box["player_id"].astype(str)
    pbp["player_id"] = pbp["player_id"].astype(str)
    df = box.merge(pbp, on=["season", "game_slug", "player_id"], how="left").fillna(0)
    df = _order_key(df)
    # shooters only
    df = df[df["shots"] >= 1].copy()
    df = df.sort_values(["player_id", "game_date_utc", "game_number"])

    df["shot_pct"] = df["goals"] / df["shots"].clip(lower=1)
    df["two_pt_rate"] = df["pbp_two_pt_shots"] / df["pbp_shots"].clip(lower=1)
    df["garbage_shot_rate"] = df["pbp_shots_garbage"] / df["pbp_shots"].clip(lower=1)

    g = df.groupby("player_id", group_keys=False)
    df["prior_shot_pct"] = g["shot_pct"].apply(_ewm_prior)
    df["prior_two_pt_rate"] = g["two_pt_rate"].apply(_ewm_prior)
    df["prior_garbage_rate"] = g["garbage_shot_rate"].apply(_ewm_prior)
    # competitive-only shot_pct (exclude garbage-time shots) as an alt baseline
    df["comp_goals"] = df["pbp_goals"] - 0  # (goals in garbage not separated at player grain here)

    y = df["shot_pct"].to_numpy()
    print("  [shot efficiency] outcome = next-game shot_pct (shooters, n rows below)")
    print("   baseline feature = prior blended shot_pct")
    r = _incremental_r2(y, df["prior_shot_pct"].to_numpy(), df["prior_two_pt_rate"].to_numpy())
    print(f"   + PBP prior 2pt-rate      : {r}")
    r = _incremental_r2(y, df["prior_shot_pct"].to_numpy(), df["prior_garbage_rate"].to_numpy())
    print(f"   + PBP prior garbage-rate  : {r}")


# ---------------------------------------------------------------------------
# Candidate 2: clean/messy save mix predicts save rate
#   baseline = prior save_pct from box
#   candidate = prior clean-save share from PBP
#   outcome  = next-game save_pct
# ---------------------------------------------------------------------------
def eval_save_quality():
    box = pd.read_parquet(CUR / "player_game_stats.parquet",
                          columns=["season", "game_slug", "player_id", "saves",
                                   "goals_against", "clean_saves", "messy_saves", "position"])
    box["player_id"] = box["player_id"].astype(str)
    df = _order_key(box)
    df = df[(df["position"] == "G")].copy()
    df["shots_faced"] = df["saves"] + df["goals_against"]
    df = df[df["shots_faced"] >= 5].copy()
    df = df.sort_values(["player_id", "game_date_utc", "game_number"])

    df["save_pct"] = df["saves"] / df["shots_faced"].clip(lower=1)
    df["clean_share"] = df["clean_saves"] / df["saves"].clip(lower=1)

    g = df.groupby("player_id", group_keys=False)
    df["prior_save_pct"] = g["save_pct"].apply(_ewm_prior)
    df["prior_clean_share"] = g["clean_share"].apply(_ewm_prior)

    y = df["save_pct"].to_numpy()
    print("  [goalie save rate] outcome = next-game save_pct (goalies, >=5 shots faced)")
    print("   baseline feature = prior save_pct")
    r = _incremental_r2(y, df["prior_save_pct"].to_numpy(), df["prior_clean_share"].to_numpy())
    print(f"   + PBP prior clean-save share : {r}")


# ---------------------------------------------------------------------------
# Candidate 3: PBP possession pace predicts team scoring
#   baseline = prior team goals-per-game (box)
#   candidate = prior team shots-per-possession & sec-per-possession (PBP)
#   outcome  = next-game team goals
# ---------------------------------------------------------------------------
def eval_pace():
    box = pd.read_parquet(CUR / "team_game_stats.parquet",
                          columns=["season", "game_slug", "team_id", "goals", "shots"])
    pbp = pd.read_parquet(CUR / "pbp_team_game.parquet",
                          columns=["season", "game_slug", "team_id",
                                   "pbp_possessions", "pbp_shots_per_poss",
                                   "pbp_goals_per_poss", "pbp_sec_per_poss"])
    box["team_id"] = box["team_id"].astype(str)
    pbp["team_id"] = pbp["team_id"].astype(str)
    df = box.merge(pbp, on=["season", "game_slug", "team_id"], how="left")
    df = _order_key(df)
    df = df.sort_values(["team_id", "game_date_utc", "game_number"])

    g = df.groupby("team_id", group_keys=False)
    df["prior_goals"] = g["goals"].apply(_ewm_prior)
    df["prior_shots"] = g["shots"].apply(_ewm_prior)
    df["prior_shots_per_poss"] = g["pbp_shots_per_poss"].apply(_ewm_prior)
    df["prior_goals_per_poss"] = g["pbp_goals_per_poss"].apply(_ewm_prior)
    df["prior_sec_per_poss"] = g["pbp_sec_per_poss"].apply(_ewm_prior)
    df["prior_possessions"] = g["pbp_possessions"].apply(_ewm_prior)

    y = df["goals"].to_numpy()
    print("  [team scoring] outcome = next-game team goals")
    print("   baseline feature = prior team goals/game")
    for name, col in [("shots-per-poss", "prior_shots_per_poss"),
                      ("goals-per-poss", "prior_goals_per_poss"),
                      ("sec-per-poss", "prior_sec_per_poss"),
                      ("possessions/game", "prior_possessions")]:
        r = _incremental_r2(y, df["prior_goals"].to_numpy(), df[col].to_numpy())
        print(f"   + PBP prior {name:16s}: {r}")


# ---------------------------------------------------------------------------
# Candidate 4: FO matchup quality (opponent-adjusted) — sanity vs box FO%
#   baseline = prior own FO% (box)
#   candidate = prior own FO% from PBP head-to-head wins (should match) +
#               strength-of-FO-opponents faced (context box lacks)
#   outcome  = next-game FO%
# ---------------------------------------------------------------------------
def eval_faceoff():
    box = pd.read_parquet(CUR / "player_game_stats.parquet",
                          columns=["season", "game_slug", "player_id",
                                   "faceoffs_won", "faceoffs_lost", "faceoffs", "position"])
    box["player_id"] = box["player_id"].astype(str)
    df = _order_key(box)
    df = df[df["faceoffs"] >= 5].copy()
    df = df.sort_values(["player_id", "game_date_utc", "game_number"])
    df["fo_pct"] = df["faceoffs_won"] / df["faceoffs"].clip(lower=1)

    g = df.groupby("player_id", group_keys=False)
    df["prior_fo_pct"] = g["fo_pct"].apply(_ewm_prior)

    # PBP opponent strength: for each FO event, the loser's season FO%. Build
    # per-game avg opponent FO% faced by each winner-specialist.
    fo = pd.read_parquet(CUR / "pbp_faceoffs.parquet",
                         columns=["season", "game_slug", "winner_player_id", "loser_player_id"])
    fo = fo.dropna(subset=["winner_player_id", "loser_player_id"])
    # season FO% per player (both as winner and loser appearances)
    w = fo.groupby(["season", "winner_player_id"]).size().rename("w")
    l = fo.groupby(["season", "loser_player_id"]).size().rename("l")
    strength = pd.concat([w, l], axis=1).fillna(0)
    strength["season_fo_pct"] = strength["w"] / (strength["w"] + strength["l"]).clip(lower=1)
    season_fo = strength["season_fo_pct"].reset_index().rename(columns={"level_1": "player_id"})
    season_fo.columns = ["season", "player_id", "season_fo_pct"]

    # opponent (loser) strength faced per winner per game
    fo2 = fo.merge(season_fo.rename(columns={"player_id": "loser_player_id",
                                             "season_fo_pct": "opp_fo_pct"}),
                   on=["season", "loser_player_id"], how="left")
    opp = fo2.groupby(["season", "game_slug", "winner_player_id"])["opp_fo_pct"].mean() \
             .reset_index().rename(columns={"winner_player_id": "player_id",
                                            "opp_fo_pct": "opp_strength"})
    opp["player_id"] = opp["player_id"].astype(str)
    df = df.merge(opp, on=["season", "game_slug", "player_id"], how="left")
    df["prior_opp_strength"] = df.groupby("player_id", group_keys=False)["opp_strength"].apply(_ewm_prior)

    y = df["fo_pct"].to_numpy()
    print("  [faceoff %] outcome = next-game FO% (specialists, >=5 FOs)")
    print("   baseline feature = prior own FO%")
    r = _incremental_r2(y, df["prior_fo_pct"].to_numpy(), df["prior_opp_strength"].to_numpy())
    print(f"   + PBP prior opponent-FO-strength faced : {r}")


def main():
    print("=" * 78)
    print("PBP INCREMENTAL SIGNAL vs BOX-SCORE BASELINE")
    print("delta_r2 > 0 means PBP feature adds predictive info the box score lacks.")
    print("delta_r2 ~ 0 means the PBP feature is redundant with what the engine has.")
    print("=" * 78)
    for fn in [eval_shot_quality, eval_save_quality, eval_pace, eval_faceoff]:
        print("\n" + "-" * 78)
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"   ERROR in {fn.__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
