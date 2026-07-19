"""
build_pbp_tables.py
-------------------
Cleans and organizes the normalized play-by-play events (pbp_events.parquet)
into tidy, analysis-ready derived tables. Additive only — writes NEW parquet
files alongside the existing warehouse and never modifies box-score tables or
the projection engine.

Inputs (produced by scrape_play_by_play.py):
    data/curated_data/all_requested_seasons/pbp_events.parquet
    data/curated_data/all_requested_seasons/game_manifest.parquet   (for home/away teams)

Outputs (all new, prefixed `pbp_`):
    pbp_events_clean.parquet        one row/event, with derived running score,
                                    possession team, game-state, garbage-time flag
    pbp_shots.parquet               one row/shot-or-goal, shot quality attributes
    pbp_faceoffs.parquet            one row/faceoff, head-to-head winner/loser + team
    pbp_possessions.parquet         one row/possession (derived chain), pace features
    pbp_player_game.parquet         PBP-derived per-player-per-game aggregates
    pbp_team_game.parquet           PBP-derived per-team-per-game aggregates

Key data quirks handled here (verified against the raw feed):
  * homeScore/visitorScore are only trustworthy on `goal` events; on every other
    event they carry the FINAL score. Running score is re-derived from goals.
  * causedTurnoverId / commitedTurnoverId are always null in the feed; turnover
    ownership is taken from the event's `teamId` (canonical ABBR, e.g. RED/ARC).
  * `teamId` on every event is the acting team in canonical engine form; the
    text `description` uses city abbreviations (CAL/UTA) — we rely on teamId.
  * shotType domain: 1_PT, 2_PT, MU (man-up 1pt), MU_2_PT (man-up 2pt), '' / NaN.
  * detail_saveType: clean, messy, NaN (only present when a shot was saved).

Usage:
    python scripts/build_pbp_tables.py
    python scripts/build_pbp_tables.py --no-write     # compute + report only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pbp_tables")

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED_ALL_DIR = REPO_ROOT / "data" / "curated_data" / "all_requested_seasons"

EVENTS_PATH = CURATED_ALL_DIR / "pbp_events.parquet"
MANIFEST_PATH = CURATED_ALL_DIR / "game_manifest.parquet"

OUT_EVENTS_CLEAN = CURATED_ALL_DIR / "pbp_events_clean.parquet"
OUT_SHOTS = CURATED_ALL_DIR / "pbp_shots.parquet"
OUT_FACEOFFS = CURATED_ALL_DIR / "pbp_faceoffs.parquet"
OUT_POSSESSIONS = CURATED_ALL_DIR / "pbp_possessions.parquet"
OUT_PLAYER_GAME = CURATED_ALL_DIR / "pbp_player_game.parquet"
OUT_TEAM_GAME = CURATED_ALL_DIR / "pbp_team_game.parquet"

# A win-probability this lopsided => low-leverage "garbage time".
GARBAGE_WP_THRESHOLD = 0.95
REGULATION_SECONDS = 48 * 60  # 4 x 12:00

# Historical franchise remap so PBP team ids match the warehouse's canonical ids.
# Chrome (CHR) rolled into the Denver Outlaws (OUT); mirrors build_warehouse.py.
TEAM_ID_CANONICAL_MAP = {"CHR": "OUT"}


def canonical_team(team_id):
    if team_id is None:
        return team_id
    t = str(team_id).strip()
    return TEAM_ID_CANONICAL_MAP.get(t, t)


# ---------------------------------------------------------------------------
# Load + clean base events
# ---------------------------------------------------------------------------
def load_events() -> pd.DataFrame:
    ev = pd.read_parquet(EVENTS_PATH)
    # Canonicalize the acting-team id (CHR -> OUT) to match box-score tables.
    ev["teamId"] = ev["teamId"].map(canonical_team)
    # Stable ordering within each game.
    ev = ev.sort_values(["season", "game_slug", "event_index"]).reset_index(drop=True)
    return ev


def attach_home_away(ev: pd.DataFrame) -> pd.DataFrame:
    gm = pd.read_parquet(
        MANIFEST_PATH,
        columns=["game_slug", "home_team_id", "away_team_id", "home_score", "away_score"],
    ).drop_duplicates("game_slug")
    gm = gm.rename(columns={
        "home_team_id": "home_team",
        "away_team_id": "away_team",
        "home_score": "final_home_score",
        "away_score": "final_away_score",
    })
    return ev.merge(gm, on="game_slug", how="left")


def derive_running_score(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Re-derive the true running score from goal events.

    In the feed, home/visitor score are the running score ONLY on goal rows.
    We compute cumulative home/away goals from goal events (respecting 1 vs 2
    point shot types) and forward-fill across every event so each event knows
    the true score *as of* that moment.
    """
    ev = ev.copy()

    # Points scored on each goal event: MU_2_PT / 2_PT are worth 2, else 1.
    is_goal = ev["eventType"].eq("goal")
    pts = np.where(ev["shotType"].isin(["2_PT", "MU_2_PT"]), 2, 1)
    ev["goal_points"] = np.where(is_goal, pts, 0).astype(int)

    # Which side scored? teamId on a goal is the scoring team (canonical ABBR).
    scored_home = is_goal & ev["teamId"].eq(ev["home_team"])
    scored_away = is_goal & ev["teamId"].eq(ev["away_team"])

    ev["_home_pts_ev"] = np.where(scored_home, ev["goal_points"], 0)
    ev["_away_pts_ev"] = np.where(scored_away, ev["goal_points"], 0)

    grp = ev.groupby("game_slug", sort=False)
    # Running score AFTER this event.
    ev["run_home_score"] = grp["_home_pts_ev"].cumsum()
    ev["run_away_score"] = grp["_away_pts_ev"].cumsum()
    # Score BEFORE this event (state in which the event occurred).
    ev["pre_home_score"] = ev["run_home_score"] - ev["_home_pts_ev"]
    ev["pre_away_score"] = ev["run_away_score"] - ev["_away_pts_ev"]
    ev["pre_margin_home"] = ev["pre_home_score"] - ev["pre_away_score"]

    ev = ev.drop(columns=["_home_pts_ev", "_away_pts_ev"])
    return ev


def derive_game_state(ev: pd.DataFrame) -> pd.DataFrame:
    """Add possession team, elapsed clock, and a garbage-time flag."""
    ev = ev.copy()

    # secondsPassed is game-elapsed seconds; fall back to period/clock if absent.
    sp = pd.to_numeric(ev["secondsPassed"], errors="coerce")
    period = pd.to_numeric(ev["period"], errors="coerce")
    minutes = pd.to_numeric(ev["minutes"], errors="coerce")
    seconds = pd.to_numeric(ev["seconds"], errors="coerce")
    # Elapsed within a 12:00 quarter = 12:00 - clock; plus finished quarters.
    clock_elapsed = (period - 1).clip(lower=0) * 720 + (720 - (minutes * 60 + seconds))
    ev["elapsed_seconds"] = sp.fillna(clock_elapsed)

    # Garbage time: win prob is available on many events; when present and past
    # the threshold, flag low-leverage. WP is null on some events (e.g. GBs) —
    # forward-fill within game so state persists between updates.
    wp_home = pd.to_numeric(ev["homeTeamWinProbability"], errors="coerce")
    wp_away = pd.to_numeric(ev["awayTeamWinProbability"], errors="coerce")
    ev["wp_max"] = pd.concat([wp_home, wp_away], axis=1).max(axis=1)
    ev["wp_max"] = ev.groupby("game_slug", sort=False)["wp_max"].ffill()
    ev["is_garbage_time"] = ev["wp_max"].ge(GARBAGE_WP_THRESHOLD).fillna(False)

    return ev


# ---------------------------------------------------------------------------
# Derived surfaces
# ---------------------------------------------------------------------------
def build_shots(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per shot attempt (eventType in {shot, goal})."""
    shots = ev[ev["eventType"].isin(["shot", "goal"])].copy()

    shots["is_goal"] = shots["eventType"].eq("goal")
    shots["is_two_point"] = shots["shotType"].isin(["2_PT", "MU_2_PT"])
    shots["is_man_up"] = shots["shotType"].isin(["MU", "MU_2_PT"])
    shots["shot_points"] = np.where(shots["is_two_point"], 2, 1)

    # On-goal / saved: detail flags are only populated on `shot` rows; a goal is
    # by definition on-goal and not saved.
    on_goal = shots["detail_shotOnGoal"]
    shots["is_on_goal"] = np.where(shots["is_goal"], True, on_goal.fillna(False)).astype(bool)
    saved = shots["detail_shotSaved"]
    shots["is_saved"] = np.where(shots["is_goal"], False, saved.fillna(False)).astype(bool)
    shots["save_type"] = shots["detail_saveType"]  # clean/messy/NaN

    keep = [
        "season", "game_slug", "event_index", "elapsed_seconds", "period",
        "teamId", "shooterId", "goalieId", "shotAssistId", "closestDefenderId",
        "shotType", "shot_points", "is_goal", "is_two_point", "is_man_up",
        "is_on_goal", "is_saved", "save_type",
        "pre_margin_home", "is_garbage_time",
    ]
    return shots[keep].rename(columns={"teamId": "team_id"}).reset_index(drop=True)


def build_faceoffs(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per faceoff, with head-to-head winner/loser player ids + win team."""
    fo = ev[ev["eventType"].eq("faceoff")].copy()
    keep = [
        "season", "game_slug", "event_index", "elapsed_seconds", "period",
        "teamId", "faceoffWinnerId", "faceoffLoserId", "gbPlayerId",
        "pre_margin_home", "is_garbage_time",
    ]
    fo = fo[keep].rename(columns={
        "teamId": "win_team_id",
        "faceoffWinnerId": "winner_player_id",
        "faceoffLoserId": "loser_player_id",
        "gbPlayerId": "gb_player_id",
    })
    return fo.reset_index(drop=True)


def build_possessions(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Derive possessions by walking the event chain per game.

    A possession is a maximal run of events belonging to one offensive team,
    ended by a terminal event (goal, turnover, shotclockexpired) or a change in
    the acting team on a possession-defining event. This is a pragmatic
    reconstruction — the feed has no explicit possession id — good enough for
    pace/efficiency features (shots-per-possession, seconds-per-possession).
    """
    poss_events = {"faceoff", "groundball", "shot", "goal", "turnover", "shotclockexpired"}
    e = ev[ev["eventType"].isin(poss_events)].copy()
    e = e.sort_values(["game_slug", "event_index"]).reset_index(drop=True)

    rows = []
    for gslug, g in e.groupby("game_slug", sort=False):
        season = g["season"].iloc[0]
        cur_team = None
        start_elapsed = None
        n_shots = n_goals = 0
        pts = 0
        first_idx = None

        def flush(end_elapsed, end_type):
            nonlocal cur_team, start_elapsed, n_shots, n_goals, pts, first_idx
            if cur_team is None or first_idx is None:
                return
            dur = None
            if end_elapsed is not None and start_elapsed is not None:
                dur = max(0, end_elapsed - start_elapsed)
            rows.append({
                "season": season,
                "game_slug": gslug,
                "start_event_index": first_idx,
                "team_id": cur_team,
                "start_elapsed": start_elapsed,
                "duration_seconds": dur,
                "n_shots": n_shots,
                "n_goals": n_goals,
                "points": pts,
                "end_type": end_type,
            })
            cur_team = None
            start_elapsed = None
            n_shots = n_goals = 0
            pts = 0
            first_idx = None

        for _, r in g.iterrows():
            etype = r["eventType"]
            team = r["teamId"]
            elapsed = r["elapsed_seconds"]

            if cur_team is None and team not in (None, ""):
                cur_team = team
                start_elapsed = elapsed
                first_idx = int(r["event_index"])
            elif team not in (None, "") and team != cur_team:
                # possession flips to the other team
                flush(elapsed, "change")
                cur_team = team
                start_elapsed = elapsed
                first_idx = int(r["event_index"])

            if etype in ("shot", "goal"):
                n_shots += 1
                if etype == "goal":
                    n_goals += 1
                    pts += 2 if r["shotType"] in ("2_PT", "MU_2_PT") else 1

            if etype in ("goal", "turnover", "shotclockexpired"):
                flush(elapsed, etype)

        flush(None, "game_end")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-player / per-team aggregates from PBP
# ---------------------------------------------------------------------------
def build_player_game(shots: pd.DataFrame, faceoffs: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PBP to player-game. Mirrors box-score fields where possible,
    plus adds situational splits the box score can't express."""
    # --- shooting (attribute to shooterId) ---
    # Precompute per-shot helper columns so aggregation stays plain sums.
    s = shots.dropna(subset=["shooterId"]).copy()
    s["_goal"] = s["is_goal"].astype(int)
    s["_two_pt_goal"] = (s["is_two_point"] & s["is_goal"]).astype(int)
    s["_goal_points"] = np.where(s["is_goal"], s["shot_points"], 0)
    shoot = s.groupby(["season", "game_slug", "shooterId"]).agg(
        pbp_shots=("event_index", "size"),
        pbp_goals=("_goal", "sum"),
        pbp_goal_points=("_goal_points", "sum"),
        pbp_sog=("is_on_goal", "sum"),
        pbp_two_pt_shots=("is_two_point", "sum"),
        pbp_two_pt_goals=("_two_pt_goal", "sum"),
        pbp_manup_shots=("is_man_up", "sum"),
        pbp_shots_garbage=("is_garbage_time", "sum"),
    ).reset_index().rename(columns={"shooterId": "player_id"})

    # --- assists (attribute to shotAssistId on goals) ---
    a = shots[(shots["is_goal"]) & shots["shotAssistId"].notna()]
    assists = a.groupby(["season", "game_slug", "shotAssistId"]).size().rename("pbp_assists").reset_index()
    assists = assists.rename(columns={"shotAssistId": "player_id"})

    # --- goalie saves (attribute to goalieId on saved shots) ---
    sv = shots[shots["is_saved"] & shots["goalieId"].notna()].copy()
    saves = sv.groupby(["season", "game_slug", "goalieId"]).agg(
        pbp_saves=("event_index", "size"),
        pbp_clean_saves=("save_type", lambda x: int((x == "clean").sum())),
        pbp_messy_saves=("save_type", lambda x: int((x == "messy").sum())),
    ).reset_index().rename(columns={"goalieId": "player_id"})
    # goals against (attribute to goalieId on goals)
    ga = shots[shots["is_goal"] & shots["goalieId"].notna()]
    ga = ga.groupby(["season", "game_slug", "goalieId"]).size().rename("pbp_goals_against").reset_index()
    ga = ga.rename(columns={"goalieId": "player_id"})
    # shots faced = saves + goals against + (on-goal? here saves+GA is the SOG faced)
    goalie = saves.merge(ga, on=["season", "game_slug", "player_id"], how="outer")

    # --- faceoffs (winner / loser) ---
    fw = faceoffs.dropna(subset=["winner_player_id"]).groupby(
        ["season", "game_slug", "winner_player_id"]).size().rename("pbp_fo_won").reset_index()
    fw = fw.rename(columns={"winner_player_id": "player_id"})
    fl = faceoffs.dropna(subset=["loser_player_id"]).groupby(
        ["season", "game_slug", "loser_player_id"]).size().rename("pbp_fo_lost").reset_index()
    fl = fl.rename(columns={"loser_player_id": "player_id"})
    fo = fw.merge(fl, on=["season", "game_slug", "player_id"], how="outer")

    # --- groundballs ---
    gb = ev[ev["gbPlayerId"].notna()].groupby(
        ["season", "game_slug", "gbPlayerId"]).size().rename("pbp_ground_balls").reset_index()
    gb = gb.rename(columns={"gbPlayerId": "player_id"})

    # --- penalties ---
    pen = ev[ev["commitedPenaltyId"].notna()].groupby(
        ["season", "game_slug", "commitedPenaltyId"]).size().rename("pbp_penalties").reset_index()
    pen = pen.rename(columns={"commitedPenaltyId": "player_id"})

    # merge all
    out = shoot
    for df in [assists, goalie, fo, gb, pen]:
        out = out.merge(df, on=["season", "game_slug", "player_id"], how="outer")

    # tidy
    num_cols = [c for c in out.columns if c.startswith("pbp_")]
    out[num_cols] = out[num_cols].fillna(0)
    for c in num_cols:
        out[c] = out[c].astype(int) if out[c].dropna().mod(1).eq(0).all() else out[c]
    out = out.dropna(subset=["player_id"])
    return out.sort_values(["season", "game_slug", "player_id"]).reset_index(drop=True)


def build_team_game(shots: pd.DataFrame, faceoffs: pd.DataFrame, ev: pd.DataFrame,
                    possessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PBP to team-game with situational splits + pace."""
    s = shots.copy()
    s["_goal"] = s["is_goal"].astype(int)
    s["_goal_garbage"] = (s["is_goal"] & s["is_garbage_time"]).astype(int)
    s["_goal_points"] = np.where(s["is_goal"], s["shot_points"], 0)
    team_shots = s.groupby(["season", "game_slug", "team_id"]).agg(
        pbp_shots=("event_index", "size"),
        pbp_goals=("_goal", "sum"),
        pbp_points=("_goal_points", "sum"),
        pbp_sog=("is_on_goal", "sum"),
        pbp_two_pt_shots=("is_two_point", "sum"),
        pbp_manup_shots=("is_man_up", "sum"),
        pbp_shots_garbage=("is_garbage_time", "sum"),
        pbp_goals_garbage=("_goal_garbage", "sum"),
    ).reset_index()

    # faceoffs won by team
    fo = faceoffs.groupby(["season", "game_slug", "win_team_id"]).size().rename("pbp_fo_won").reset_index()
    fo = fo.rename(columns={"win_team_id": "team_id"})

    # turnovers / shotclock (teamId = team that committed)
    to = ev[ev["eventType"].eq("turnover")].groupby(
        ["season", "game_slug", "teamId"]).size().rename("pbp_turnovers").reset_index().rename(columns={"teamId": "team_id"})
    sc = ev[ev["eventType"].eq("shotclockexpired")].groupby(
        ["season", "game_slug", "teamId"]).size().rename("pbp_shotclock_violations").reset_index().rename(columns={"teamId": "team_id"})

    # possessions / pace
    p = possessions[possessions["team_id"].astype(str).str.len() > 0]
    pace = p.groupby(["season", "game_slug", "team_id"]).agg(
        pbp_possessions=("start_event_index", "size"),
        pbp_poss_seconds=("duration_seconds", "sum"),
        pbp_poss_seconds_med=("duration_seconds", "median"),
    ).reset_index()

    out = team_shots
    for df in [fo, to, sc, pace]:
        out = out.merge(df, on=["season", "game_slug", "team_id"], how="outer")

    # efficiency features
    out["pbp_shots_per_poss"] = out["pbp_shots"] / out["pbp_possessions"].replace(0, np.nan)
    out["pbp_goals_per_poss"] = out["pbp_goals"] / out["pbp_possessions"].replace(0, np.nan)
    out["pbp_sec_per_poss"] = out["pbp_poss_seconds"] / out["pbp_possessions"].replace(0, np.nan)

    num_cols = [c for c in out.columns if c.startswith("pbp_") and not c.endswith(("_per_poss", "_seconds", "_seconds_med", "_per_poss"))]
    out[num_cols] = out[num_cols].fillna(0)
    out = out[out["team_id"].astype(str).str.len() > 0]
    return out.sort_values(["season", "game_slug", "team_id"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean & organize PLL play-by-play into derived tables")
    parser.add_argument("--no-write", action="store_true", help="Compute + report only")
    args = parser.parse_args()

    if not EVENTS_PATH.exists():
        logger.error("Missing %s — run scrape_play_by_play.py first", EVENTS_PATH)
        return 1

    logger.info("Loading events…")
    ev = load_events()
    ev = attach_home_away(ev)
    ev = derive_running_score(ev)
    ev = derive_game_state(ev)
    logger.info("Base events: %d rows, %d games", len(ev), ev["game_slug"].nunique())

    shots = build_shots(ev)
    faceoffs = build_faceoffs(ev)
    possessions = build_possessions(ev)
    player_game = build_player_game(shots, faceoffs, ev)
    team_game = build_team_game(shots, faceoffs, ev, possessions)

    logger.info("shots=%d  faceoffs=%d  possessions=%d  player_game=%d  team_game=%d",
                len(shots), len(faceoffs), len(possessions), len(player_game), len(team_game))

    # quick sanity peek
    logger.info("Shots by type:\n%s", shots["shotType"].value_counts(dropna=False).to_string())
    logger.info("Save types:\n%s", shots.loc[shots["is_saved"], "save_type"].value_counts(dropna=False).to_string())
    logger.info("Garbage-time shots: %d / %d (%.1f%%)",
                int(shots["is_garbage_time"].sum()), len(shots),
                100 * shots["is_garbage_time"].mean())

    if args.no_write:
        logger.info("--no-write set; not writing parquet")
        return 0

    ev_clean_cols = [
        "season", "game_slug", "event_index", "eventType", "description",
        "period", "minutes", "seconds", "elapsed_seconds", "teamId",
        "home_team", "away_team", "run_home_score", "run_away_score",
        "pre_home_score", "pre_away_score", "pre_margin_home",
        "wp_max", "is_garbage_time",
        "shotType", "shooterId", "goalieId", "shotAssistId",
        "faceoffWinnerId", "faceoffLoserId", "gbPlayerId", "commitedPenaltyId",
        "penaltyLength", "penaltyDescription",
        "detail_shotOnGoal", "detail_shotSaved", "detail_saveType",
    ]
    ev[ev_clean_cols].to_parquet(OUT_EVENTS_CLEAN, index=False)
    shots.to_parquet(OUT_SHOTS, index=False)
    faceoffs.to_parquet(OUT_FACEOFFS, index=False)
    possessions.to_parquet(OUT_POSSESSIONS, index=False)
    player_game.to_parquet(OUT_PLAYER_GAME, index=False)
    team_game.to_parquet(OUT_TEAM_GAME, index=False)
    for p in [OUT_EVENTS_CLEAN, OUT_SHOTS, OUT_FACEOFFS, OUT_POSSESSIONS, OUT_PLAYER_GAME, OUT_TEAM_GAME]:
        logger.info("Wrote %s", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
