"""
check_pbp_parity.py
-------------------
Coverage / parity check: prove the play-by-play (PBP) derived data reconstructs
the current box-score data at least as completely, then quantify what PBP adds
that the box score cannot express.

Read-only. Touches no engine code and writes no warehouse tables. Prints a
report and (optionally) dumps mismatch detail to CSV for inspection.

Compares, at team-game and player-game grain:
    goals, shots, shots-on-goal, saves, clean/messy saves, faceoff wins,
    turnovers, ground balls, two-point shots.

Usage:
    python scripts/check_pbp_parity.py
    python scripts/check_pbp_parity.py --dump    # write mismatch CSVs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

REPO_ROOT = Path(__file__).resolve().parents[1]
CUR = REPO_ROOT / "data" / "curated_data" / "all_requested_seasons"
OUT_DIR = REPO_ROOT / "scripts" / "pbp_parity_output"


def load():
    box_team = pd.read_parquet(CUR / "team_game_stats.parquet")
    box_player = pd.read_parquet(CUR / "player_game_stats.parquet")
    pbp_team = pd.read_parquet(CUR / "pbp_team_game.parquet")
    pbp_player = pd.read_parquet(CUR / "pbp_player_game.parquet")
    return box_team, box_player, pbp_team, pbp_player


def compare_series(box: pd.Series, pbp: pd.Series, label: str) -> dict:
    """Row-aligned comparison of two integer stat series."""
    diff = (pbp.fillna(0) - box.fillna(0))
    n = len(diff)
    exact = int((diff == 0).sum())
    within1 = int((diff.abs() <= 1).sum())
    return {
        "stat": label,
        "rows": n,
        "exact_match": exact,
        "exact_pct": round(100 * exact / n, 1) if n else np.nan,
        "within_1": within1,
        "within_1_pct": round(100 * within1 / n, 1) if n else np.nan,
        "box_total": int(box.fillna(0).sum()),
        "pbp_total": int(pbp.fillna(0).sum()),
        "mean_diff": round(diff.mean(), 3),
        "mean_abs_diff": round(diff.abs().mean(), 3),
        "max_abs_diff": int(diff.abs().max()) if n else 0,
    }


def team_parity(box_team, pbp_team, dump=False):
    key = ["season", "game_slug", "team_id"]
    b = box_team.copy()
    b["team_id"] = b["team_id"].astype(str)
    p = pbp_team.copy()
    p["team_id"] = p["team_id"].astype(str)

    # box columns -> pbp columns
    pairs = [
        ("goals", "pbp_goals"),
        ("shots", "pbp_shots"),
        ("shots_on_goal", "pbp_sog"),
        ("two_point_shots", "pbp_two_pt_shots"),
        ("saves", None),           # saves are goalie-side; team save = opp shots saved (checked at player grain)
        ("faceoffs_won", "pbp_fo_won"),
        ("turnovers", "pbp_turnovers"),
        ("shot_clock_expirations", "pbp_shotclock_violations"),
    ]

    merged = b.merge(p, on=key, how="outer", suffixes=("_box", "_pbp"), indicator=True)
    join_report = merged["_merge"].value_counts().to_dict()

    rows = []
    for box_col, pbp_col in pairs:
        if pbp_col is None or box_col not in merged or pbp_col not in merged:
            continue
        rows.append(compare_series(merged[box_col], merged[pbp_col], f"team.{box_col}"))
    rep = pd.DataFrame(rows)

    if dump:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # dump worst goal mismatches for eyeballing
        m = merged.copy()
        m["goal_diff"] = m["pbp_goals"].fillna(0) - m["goals"].fillna(0)
        m.loc[m["goal_diff"].abs() > 0,
              key + ["goals", "pbp_goals", "shots", "pbp_shots", "goal_diff"]] \
            .sort_values("goal_diff").to_csv(OUT_DIR / "team_goal_mismatches.csv", index=False)
    return rep, join_report, merged


def player_parity(box_player, pbp_player, dump=False):
    key = ["season", "game_slug", "player_id"]
    b = box_player.copy()
    b["player_id"] = b["player_id"].astype(str)
    p = pbp_player.copy()
    p["player_id"] = p["player_id"].astype(str)

    pairs = [
        ("goals", "pbp_goals"),
        ("shots", "pbp_shots"),
        ("shots_on_goal", "pbp_sog"),
        ("assists", "pbp_assists"),
        ("two_point_shots", "pbp_two_pt_shots"),
        ("saves", "pbp_saves"),
        ("clean_saves", "pbp_clean_saves"),
        ("messy_saves", "pbp_messy_saves"),
        ("goals_against", "pbp_goals_against"),
        ("faceoffs_won", "pbp_fo_won"),
        ("faceoffs_lost", "pbp_fo_lost"),
        ("ground_balls", "pbp_ground_balls"),
        ("num_penalties", "pbp_penalties"),
    ]

    merged = b.merge(p, on=key, how="outer", suffixes=("_box", "_pbp"), indicator=True)
    join_report = merged["_merge"].value_counts().to_dict()

    rows = []
    for box_col, pbp_col in pairs:
        if box_col not in merged or pbp_col not in merged:
            continue
        rows.append(compare_series(merged[box_col], merged[pbp_col], f"player.{box_col}"))
    rep = pd.DataFrame(rows)

    if dump:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for box_col, pbp_col in [("goals", "pbp_goals"), ("saves", "pbp_saves"),
                                 ("faceoffs_won", "pbp_fo_won")]:
            m = merged.copy()
            d = m[pbp_col].fillna(0) - m[box_col].fillna(0)
            m["diff"] = d
            cols = key + ["full_name", box_col, pbp_col, "diff"]
            cols = [c for c in cols if c in m.columns]
            m.loc[d.abs() > 0, cols].sort_values("diff").to_csv(
                OUT_DIR / f"player_{box_col}_mismatches.csv", index=False)
    return rep, join_report, merged


def additive_summary():
    """Report data PBP has that the box score cannot express at all."""
    shots = pd.read_parquet(CUR / "pbp_shots.parquet")
    fo = pd.read_parquet(CUR / "pbp_faceoffs.parquet")
    poss = pd.read_parquet(CUR / "pbp_possessions.parquet")
    ev = pd.read_parquet(CUR / "pbp_events_clean.parquet")

    lines = []
    lines.append(f"Total shot events with quality attrs : {len(shots):,}")
    lines.append(f"  on-goal flag present               : {shots['is_on_goal'].notna().sum():,}")
    lines.append(f"  save-type (clean/messy) present    : {shots['save_type'].notna().sum():,}")
    lines.append(f"  garbage-time-flagged shots         : {int(shots['is_garbage_time'].sum()):,} ({100*shots['is_garbage_time'].mean():.1f}%)")
    lines.append(f"  man-up shots                       : {int(shots['is_man_up'].sum()):,}")
    lines.append(f"Faceoffs with head-to-head winner/loser: {fo['winner_player_id'].notna().sum():,}")
    uniq_matchups = fo.dropna(subset=['winner_player_id', 'loser_player_id'])
    uniq_matchups = uniq_matchups.assign(
        pair=uniq_matchups[['winner_player_id', 'loser_player_id']].apply(
            lambda r: tuple(sorted([r['winner_player_id'], r['loser_player_id']])), axis=1))
    lines.append(f"  distinct FO player matchups         : {uniq_matchups['pair'].nunique():,}")
    lines.append(f"Reconstructed possessions             : {len(poss):,}")
    lines.append(f"  with a measured duration            : {poss['duration_seconds'].notna().sum():,}")
    lines.append(f"  median possession length (s)        : {poss['duration_seconds'].median():.1f}")
    lines.append(f"Events carrying live win-probability  : {ev['wp_max'].notna().sum():,} / {len(ev):,}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true", help="Write mismatch CSVs")
    args = parser.parse_args()

    box_team, box_player, pbp_team, pbp_player = load()

    print("=" * 78)
    print("COVERAGE: games / rows")
    print("=" * 78)
    print(f"box  team-game rows : {len(box_team):,}  games={box_team['game_slug'].nunique()}")
    print(f"pbp  team-game rows : {len(pbp_team):,}  games={pbp_team['game_slug'].nunique()}")
    print(f"box  player-game rows : {len(box_player):,}  games={box_player['game_slug'].nunique()}")
    print(f"pbp  player-game rows : {len(pbp_player):,}  games={pbp_player['game_slug'].nunique()}")

    print("\n" + "=" * 78)
    print("TEAM-GAME PARITY  (pbp vs box)")
    print("=" * 78)
    trep, tjoin, _ = team_parity(box_team, pbp_team, dump=args.dump)
    print("join:", tjoin)
    print(trep.to_string(index=False))

    print("\n" + "=" * 78)
    print("PLAYER-GAME PARITY  (pbp vs box)")
    print("=" * 78)
    prep, pjoin, _ = player_parity(box_player, pbp_player, dump=args.dump)
    print("join:", pjoin)
    print(prep.to_string(index=False))

    print("\n" + "=" * 78)
    print("ADDITIVE VALUE  (present in PBP, absent from box score)")
    print("=" * 78)
    print(additive_summary())

    if args.dump:
        print(f"\nMismatch CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
