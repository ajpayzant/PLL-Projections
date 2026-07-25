"""
live_state.py — reconstruct per-player BANKED stats from live PBP events.

Mirrors the attribution rules in ``scripts/build_pbp_tables.py`` exactly, but
operates on the raw in-memory event list (from live_feed) instead of parquet,
so it can run every poll during a game.

Banked stats are what has ALREADY happened — they are certain. The live model
adds these to a fresh simulation of only the time remaining (see live_model.py):

    full_game_stat  =  banked_stat  +  rest_of_game_draw

Stat keys match the engine's PlayerSimulation.stat_distributions vocabulary
(goals, one_pt_goals, two_pt_goals, assists, points, shots, shots_on_goal,
saves, faceoff_wins) so banked and simulated line up 1:1. Points is derived the
engine's way: 1pt + 2*two_pt + assists.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Shot-type tags (from the PBP feed): 1_PT, 2_PT, MU (man-up 1pt), MU_2_PT.
_TWO_PT_TAGS = {"2_PT", "MU_2_PT"}

# The stat keys we bank (superset of what we currently trade; goals/assists/
# points are the priorities but shots/SOG/saves/FO are cheap to carry).
BANKED_KEYS = ("goals", "one_pt_goals", "two_pt_goals", "assists", "points",
               "shots", "shots_on_goal", "saves", "faceoff_wins", "ground_balls")


@dataclass
class BankedStats:
    """Per-player banked counting stats, keyed by warehouse player_id (str)."""
    by_player: dict[str, dict[str, float]]
    # convenience: player_id -> team_id seen acting for (last wins; only for display)
    team_of: dict[str, str] = field(default_factory=dict)

    def get(self, player_id: str, stat: str) -> float:
        return self.by_player.get(str(player_id), {}).get(stat, 0.0)

    def player_ids(self) -> set[str]:
        return set(self.by_player.keys())


def _blank() -> dict[str, float]:
    return {k: 0.0 for k in BANKED_KEYS}


def reconstruct(events: list[dict[str, Any]]) -> BankedStats:
    """Aggregate a full event list into per-player banked stats.

    Idempotent and cheap — recompute from scratch each poll (event lists are
    <300 rows even for a full game), which sidesteps any double-count risk from
    the feed re-ordering or correcting earlier events.
    """
    stats: dict[str, dict[str, float]] = defaultdict(_blank)
    team_of: dict[str, str] = {}

    for e in events:
        etype = e.get("eventType")
        team = e.get("teamId")

        if etype in ("shot", "goal"):
            shooter = e.get("shooterId")
            shot_type = e.get("shotType")
            is_two = shot_type in _TWO_PT_TAGS
            is_goal = etype == "goal"
            details = e.get("details") or {}
            # A goal is by definition on-goal; else read the detail flag.
            on_goal = True if is_goal else bool(details.get("shotOnGoal"))

            if shooter:
                shooter = str(shooter)
                if team:
                    team_of[shooter] = team
                st = stats[shooter]
                st["shots"] += 1
                if on_goal:
                    st["shots_on_goal"] += 1
                if is_goal:
                    st["goals"] += 1
                    if is_two:
                        st["two_pt_goals"] += 1
                    else:
                        st["one_pt_goals"] += 1

            # assist: credited to shotAssistId on GOALS only
            if is_goal:
                assister = e.get("shotAssistId")
                if assister:
                    assister = str(assister)
                    stats[assister]["assists"] += 1

            # goalie save: saved shot attributed to goalieId
            goalie = e.get("goalieId")
            if goalie and not is_goal and bool(details.get("shotSaved")):
                stats[str(goalie)]["saves"] += 1

        elif etype == "faceoff":
            winner = e.get("faceoffWinnerId")
            if winner:
                stats[str(winner)]["faceoff_wins"] += 1

        elif etype == "groundball":
            gb = e.get("gbPlayerId")
            if gb:
                stats[str(gb)]["ground_balls"] += 1

        # groundballs can also ride on faceoff events (gbPlayerId set); count those too
        if etype != "groundball":
            gb = e.get("gbPlayerId")
            if gb:
                stats[str(gb)]["ground_balls"] += 1

    # derive points the engine's way: 1pt + 2*2pt + assists
    for st in stats.values():
        st["points"] = st["one_pt_goals"] + 2.0 * st["two_pt_goals"] + st["assists"]

    return BankedStats(by_player=dict(stats), team_of=team_of)


if __name__ == "__main__":
    import argparse
    from live_feed import LiveFeed

    ap = argparse.ArgumentParser(description="Show banked per-player stats for a live game.")
    ap.add_argument("slug")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    state = LiveFeed(args.slug).poll()
    banked = reconstruct(state.events)
    print(f"{args.slug}: P{state.period} {state.clock_minutes:02d}:{state.clock_seconds:02d} "
          f"| {state.away_score}-{state.home_score} | {state.n_events} events\n")
    rows = []
    for pid, st in banked.by_player.items():
        if st["points"] or st["goals"] or st["assists"] or st["saves"] or st["shots"]:
            rows.append((pid, banked.team_of.get(pid, "?"), st))
    rows.sort(key=lambda r: (r[2]["points"], r[2]["goals"], r[2]["shots"]), reverse=True)
    print(f"{'player_id':10} {'tm':4} {'G':>4} {'A':>4} {'PTS':>4} {'SH':>4} {'SOG':>4} {'SV':>4} {'FO':>4}")
    for pid, tm, st in rows[:args.top]:
        print(f"{pid:10} {tm:4} {st['goals']:4.0f} {st['assists']:4.0f} {st['points']:4.0f} "
              f"{st['shots']:4.0f} {st['shots_on_goal']:4.0f} {st['saves']:4.0f} {st['faceoff_wins']:4.0f}")
