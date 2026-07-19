"""
evaluate_pbp_faceoff_h2h.py
---------------------------
Proper test of whether PBP head-to-head faceoff data improves FO projection.

The earlier eval only tried "opponent strength faced" as an add-on regressor —
a schedule correction, not a matchup model. This script instead asks the real
question:

    Given specialist A vs specialist B in a faceoff, can an OPPONENT-ADJUSTED
    skill rating (fit only from prior head-to-head results) predict the outcome
    better than the marginal-FO% (log5) approach the engine uses today?

Two rating systems, both fit leakage-safe (train on games strictly before the
test game's date):
  * MARGINAL  : each player's win rate ignoring opponent, combined via log5.
                This is essentially what the engine does now (fo_pct_ewm + log5).
  * BRADLEY-TERRY : latent skill r_i fit so P(i beats j) = sigma(r_i - r_j),
                iteratively from the full head-to-head win matrix. This is what
                the exact matchup data unlocks.

Evaluation grain = individual faceoff reps in the test season, predicted from a
model fit on all prior seasons + prior games. Metrics: log-loss and accuracy
(lower log-loss = better). Also aggregated to per-game FO% MAE.

Read-only. No engine import, no warehouse writes.

Usage:
    python scripts/evaluate_pbp_faceoff_h2h.py
    python scripts/evaluate_pbp_faceoff_h2h.py --test-season 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CUR = REPO_ROOT / "data" / "curated_data" / "all_requested_seasons"


def load_faceoffs() -> pd.DataFrame:
    fo = pd.read_parquet(CUR / "pbp_faceoffs.parquet",
                         columns=["season", "game_slug", "winner_player_id", "loser_player_id"])
    fo = fo.dropna(subset=["winner_player_id", "loser_player_id"]).copy()
    gm = pd.read_parquet(CUR / "game_manifest.parquet",
                         columns=["game_slug", "game_date_utc", "game_number"]).drop_duplicates("game_slug")
    fo = fo.merge(gm, on="game_slug", how="left")
    fo = fo.sort_values(["game_date_utc", "game_number", "game_slug"]).reset_index(drop=True)
    return fo


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_bradley_terry(reps: pd.DataFrame, n_iter: int = 200, lr: float = 0.5,
                      reg: float = 0.01) -> dict:
    """
    Fit latent skills r_i from head-to-head reps via gradient ascent on the
    Bradley-Terry log-likelihood. reps has columns winner_player_id, loser_player_id
    (one row per faceoff). Returns {player_id: rating}. Ratings centered at 0.
    """
    players = pd.Index(pd.unique(pd.concat([reps["winner_player_id"], reps["loser_player_id"]])))
    idx = {p: i for i, p in enumerate(players)}
    r = np.zeros(len(players))

    w = reps["winner_player_id"].map(idx).to_numpy()
    l = reps["loser_player_id"].map(idx).to_numpy()

    for _ in range(n_iter):
        diff = r[w] - r[l]
        p = _sigmoid(diff)          # P(winner beats loser) under current ratings
        g = (1.0 - p)               # gradient of loglik wrt (r_w - r_l)
        grad = np.zeros_like(r)
        np.add.at(grad, w, g)
        np.add.at(grad, l, -g)
        grad -= reg * r             # L2 shrink toward 0
        r += lr * grad / max(1, len(w))
        r -= r.mean()               # identifiability: center
    return {p: r[i] for p, i in idx.items()}


def marginal_rates(reps: pd.DataFrame, prior_strength: float = 5.0, lg: float = 0.5) -> dict:
    """Each player's opponent-blind FO win rate, Bayes-shrunk to league 0.5."""
    w = reps.groupby("winner_player_id").size()
    l = reps.groupby("loser_player_id").size()
    allp = pd.Index(pd.unique(pd.concat([reps["winner_player_id"], reps["loser_player_id"]])))
    out = {}
    for p in allp:
        wins = int(w.get(p, 0))
        losses = int(l.get(p, 0))
        n = wins + losses
        out[p] = (wins + prior_strength * lg) / (n + prior_strength)
    return out


def log5(a: float, b: float) -> float:
    """P(A beats B) from marginal rates a, b (Bill James log5)."""
    num = a * (1 - b)
    den = a * (1 - b) + b * (1 - a)
    return num / den if den > 0 else 0.5


def evaluate(test_season: int) -> None:
    fo = load_faceoffs()
    train = fo[fo["season"] < test_season]
    test = fo[fo["season"] == test_season]
    if len(train) < 500 or len(test) < 200:
        print(f"  season {test_season}: insufficient data (train={len(train)}, test={len(test)})")
        return

    # Fit both systems on TRAIN only (all prior seasons).
    bt = fit_bradley_terry(train)
    mg = marginal_rates(train)

    lg_rate = 0.5
    lg_bt = 0.0

    # Predict each test faceoff. Orient each rep as "does the winner win?" — but
    # that leaks orientation. Instead predict from a fixed ordering (player_id
    # sort) so the label is unbiased: for pair (X,Y) with X<Y, label = did X win.
    x = np.minimum(test["winner_player_id"].values, test["loser_player_id"].values)
    y = np.maximum(test["winner_player_id"].values, test["loser_player_id"].values)
    x_won = (test["winner_player_id"].values == x).astype(float)

    def rate_m(p):
        return mg.get(p, lg_rate)

    def rate_bt(p):
        return bt.get(p, lg_bt)

    p_marginal = np.array([log5(rate_m(xi), rate_m(yi)) for xi, yi in zip(x, y)])
    p_bt = np.array([_sigmoid(rate_bt(xi) - rate_bt(yi)) for xi, yi in zip(x, y)])

    eps = 1e-9
    def logloss(p, yv):
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    def acc(p, yv):
        return np.mean((p >= 0.5) == (yv >= 0.5))

    # coverage: how many test reps involve a player unseen in train?
    seen = set(mg.keys())
    unseen = np.array([(xi not in seen) or (yi not in seen) for xi, yi in zip(x, y)])

    print(f"  test season {test_season}: {len(test)} faceoff reps "
          f"({unseen.sum()} involve an unseen player -> fall back to league avg)")
    print(f"    {'model':16s} {'logloss':>9s} {'accuracy':>9s}")
    print(f"    {'MARGINAL(log5)':16s} {logloss(p_marginal, x_won):9.4f} {acc(p_marginal, x_won):9.4f}")
    print(f"    {'BRADLEY-TERRY':16s} {logloss(p_bt, x_won):9.4f} {acc(p_bt, x_won):9.4f}")

    # On reps where BOTH players are seen (the fair comparison, no fallback):
    m = ~unseen
    if m.sum() > 100:
        print(f"    -- restricted to {int(m.sum())} reps with both players seen --")
        print(f"    {'MARGINAL(log5)':16s} {logloss(p_marginal[m], x_won[m]):9.4f} {acc(p_marginal[m], x_won[m]):9.4f}")
        print(f"    {'BRADLEY-TERRY':16s} {logloss(p_bt[m], x_won[m]):9.4f} {acc(p_bt[m], x_won[m]):9.4f}")

    # Aggregate to per-game FO% for one specialist per team: compare predicted
    # game FO% to realized, MAE. Use the higher-volume player as "team FO man".
    _game_fo_mae(test, mg, bt, seen)


def _game_fo_mae(test, mg, bt, seen):
    """Per (game, specialist) realized FO% vs predicted; report MAE for each model."""
    rows = []
    # For each game, each player's wins/losses, and their per-rep opponents.
    reps = test.copy()
    # long format: each player appears as winner (won=1) or loser (won=0) with opp
    a = reps[["game_slug", "winner_player_id", "loser_player_id"]].rename(
        columns={"winner_player_id": "player", "loser_player_id": "opp"}); a["won"] = 1
    b = reps[["game_slug", "loser_player_id", "winner_player_id"]].rename(
        columns={"loser_player_id": "player", "winner_player_id": "opp"}); b["won"] = 0
    long = pd.concat([a, b], ignore_index=True)

    grp = long.groupby(["game_slug", "player"])
    for (g, pl), d in grp:
        if len(d) < 5:
            continue
        realized = d["won"].mean()
        # predicted FO% = mean over this game's reps of P(pl beats that opp)
        pm = np.mean([log5(mg.get(pl, 0.5), mg.get(o, 0.5)) for o in d["opp"]])
        pb = np.mean([_sigmoid(bt.get(pl, 0.0) - bt.get(o, 0.0)) for o in d["opp"]])
        rows.append((realized, pm, pb, pl in seen))
    r = pd.DataFrame(rows, columns=["realized", "pred_marginal", "pred_bt", "seen"])
    r = r[r["seen"]]
    if len(r) < 30:
        return
    mae_m = (r["pred_marginal"] - r["realized"]).abs().mean()
    mae_b = (r["pred_bt"] - r["realized"]).abs().mean()
    print(f"    per-game FO% MAE ({len(r)} specialist-games, both-seen): "
          f"marginal={mae_m:.4f}  bradley-terry={mae_b:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=None,
                    help="Season to test on (default: run 2024 and 2025)")
    args = ap.parse_args()

    print("=" * 78)
    print("FACEOFF HEAD-TO-HEAD: opponent-adjusted (Bradley-Terry) vs marginal log5")
    print("Lower log-loss / MAE = better. BT fit only on prior seasons (leakage-safe).")
    print("=" * 78)
    seasons = [args.test_season] if args.test_season else [2024, 2025, 2026]
    for s in seasons:
        print("-" * 78)
        evaluate(s)


if __name__ == "__main__":
    main()
