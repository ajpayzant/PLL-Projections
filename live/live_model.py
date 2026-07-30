"""
live_model.py — rest-of-game re-simulation for live PLL trading.

Full-game live distribution  =  BANKED (certain, already happened)
                              +  REST-OF-GAME (re-simulated for the time left)

The rest-of-game piece reuses the engine's OWN Monte Carlo (`GameSimulator`) so
the live model inherits every distributional assumption the pregame model was
built and validated on (team-goal conditioning, zero-inflation, per-player
volatility overrides, goalie matchup scaling, teammate correlation). We do NOT
reimplement any draw logic here — we only:

  1. scale each player's full-game projection down to the time remaining, and
  2. blend that pregame expectation with the pace actually observed so far, then
  3. add the banked (certain) counts onto the simulated remainder.

Scaling rationale — lacrosse scoring is well approximated as a rate process in
game time, so the expected production over the remaining fraction `f` of the
game is (full-game rate) × f. As the game unfolds we also learn something from
what's happened; a credibility blend nudges the remaining-game expectation
toward the observed pace, with the weight on observed pace GROWING as more of
the game elapses (a Q1 outlier barely moves it; a Q4 trend moves it a lot).

Priority stats are goals / assists / points (what the user trades); shots, SOG,
ground balls, saves and faceoff wins are carried along on the same machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

# The engine lives one directory up. The live app adds the repo root to sys.path
# (mirroring pages/_engine_state.py); when run as a script we do it here too.
try:
    from projection_engine_v3 import (
        PlayerProjection, PlayerSimulation, TeamProjection,
        ProjectionResult, GameSimulator, GameSimulation, LG_SAVE_PCT,
    )
except ModuleNotFoundError:  # pragma: no cover - script/CLI path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from projection_engine_v3 import (
        PlayerProjection, PlayerSimulation, TeamProjection,
        ProjectionResult, GameSimulator, GameSimulation, LG_SAVE_PCT,
    )

# PlayerProjection additive field  ->  banked-stat key it corresponds to.
# These are the counting fields that scale with time remaining AND have an
# observed banked count. proj_points is derived (recomputed from components);
# turnovers have no live banked feed, so both are scaled but not pace-blended.
_PROJ_TO_BANKED: Dict[str, str] = {
    "proj_goals":         "goals",
    "proj_1pt_goals":     "one_pt_goals",
    "proj_2pt_goals":     "two_pt_goals",
    "proj_assists":       "assists",
    "proj_shots":         "shots",
    "proj_sog":           "shots_on_goal",
    "proj_ground_balls":  "ground_balls",
    "proj_saves":         "saves",
    "proj_faceoff_wins":  "faceoff_wins",
}
# Additive fields with no banked counterpart — scaled by time only.
_PROJ_SCALE_ONLY = ("proj_points", "proj_turnovers", "proj_caused_turnovers")

# Distribution keys the simulator emits that get banked counts added back on.
_ADDITIVE_DIST_KEYS = (
    "goals", "one_pt_goals", "two_pt_goals", "assists",
    "shots", "shots_on_goal", "ground_balls", "saves", "faceoff_wins",
)

_EPS = 1e-6

# ---------------------------------------------------------------------------
# In-game lead mean-reversion.
#
# The rest-of-game re-simulation reproduces the REST-of-game margin variance
# correctly (verified empirically: sim SD matches historical SD*sqrt(frac_rem)
# to within ~5%). What a naive "banked lead + fresh remainder" does WRONG is
# treat the current lead as fully persistent (a random-walk / diffusion), so an
# early lead gets full credit toward the final margin. Real PLL games mean-revert:
# regressing final margin on the in-game lead over 187 games (2022-2026 PBP)
# gives a persistence factor beta ~= 0.70 at the end of Q1 rising to ~0.95 late.
#
# The reversion CURVE was then calibrated by full replay (live/calibrate_reversion.py):
# sweeping (slope, floor) over all 187 games at each elapsed snapshot showed
# slope=0.6 / floor=0.45 IMPROVES margin accuracy vs the final at every early
# snapshot (e.g. margAE 3.326->3.311 at 15% elapsed, 3.276->3.260 at 25%) while
# cutting the early win-prob swing ~25-30% (WPmove 9.2%->6.6% at 15%, 12.6%->10.9%
# at 25%) -- i.e. the extra early movement the old 0.4/0.55 curve produced was
# NOISE, so damping it is free accuracy-wise. Pushing further (0.8/1.0) starts to
# cost accuracy, so 0.6/0.45 is the empirical optimum, not the extreme. We shrink
# ONLY the margin (the totals are real, already-scored points); the lead's expected
# value toward the final is pulled toward zero by (1 - shrink)*lead. This is the
# primary lever against "unrealistic odds after an early goal". See
# project_pll_live_trading memory.
_REVERSION_SLOPE = 0.6      # shrink = 1 - slope*frac_rem  (calibrated 2026-07)
_REVERSION_FLOOR = 0.45     # never credit an early lead less than 45%


def _lead_persistence(frac_rem: float) -> float:
    """Fraction of the current in-game lead that persists to the final margin,
    as a function of the game fraction remaining. 1.0 at the buzzer (a lead is
    final), ~0.6 early (an early lead mean-reverts). Fit from historical PBP."""
    frac_rem = min(max(float(frac_rem), 0.0), 1.0)
    return max(1.0 - _REVERSION_SLOPE * frac_rem, _REVERSION_FLOOR)


@dataclass
class LiveProjection:
    """Full-game live projection: banked + re-simulated remainder, priced-ready."""
    home_team: str
    away_team: str
    home_player_sims: List[PlayerSimulation]
    away_player_sims: List[PlayerSimulation]
    fraction_remaining: float
    pace_weight: float
    banked_home_goals: float
    banked_away_goals: float
    # live full-game team-goal draw arrays (banked + remainder), for game markets
    home_goal_dist: np.ndarray
    away_goal_dist: np.ndarray
    # live full-game GameSimulation (banked scores + re-simulated remainder),
    # ready to hand to PricingEngine.price_game for live ML / spread / total.
    game_sim: Optional["GameSimulation"] = None

    def all_sims(self) -> List[PlayerSimulation]:
        return self.home_player_sims + self.away_player_sims


def _blend_rest(pregame_full: float, banked: float,
                frac_rem: float, weight: float) -> float:
    """Expected production over the REMAINING game.

    prior_rest   = pregame full-game expectation × fraction remaining
    pace_rest    = observed-so-far, projected forward at the same pace
    credibility  = weight × fraction ELAPSED  (trust the game more, later)

    Returns a blend of the two. At the opening whistle (frac_rem≈1) this is just
    the pregame expectation; late in a game it leans on what's actually happened.
    """
    pregame_full = max(float(pregame_full), 0.0)
    prior_rest = pregame_full * frac_rem
    frac_el = 1.0 - frac_rem
    if frac_el <= _EPS or banked <= 0.0:
        return prior_rest
    pace_rest = float(banked) * frac_rem / frac_el
    w = min(max(weight * frac_el, 0.0), 0.95)
    return (1.0 - w) * prior_rest + w * pace_rest


def _scale_players(players: List[PlayerProjection],
                   banked_by_player: Dict[str, Dict[str, float]],
                   frac_rem: float, weight: float) -> List[PlayerProjection]:
    """Return time-scaled, pace-blended copies of the pregame player projections."""
    out: List[PlayerProjection] = []
    for p in players:
        bk = banked_by_player.get(str(p.player_id), {})
        changes: Dict[str, float] = {}
        for proj_field, banked_key in _PROJ_TO_BANKED.items():
            full = getattr(p, proj_field, 0.0) or 0.0
            changes[proj_field] = _blend_rest(full, bk.get(banked_key, 0.0),
                                              frac_rem, weight)
        for proj_field in _PROJ_SCALE_ONLY:
            full = getattr(p, proj_field, 0.0) or 0.0
            changes[proj_field] = max(float(full), 0.0) * frac_rem
        out.append(replace(p, **changes))
    return out


def _team_banked(players: List[PlayerProjection],
                 banked_by_player: Dict[str, Dict[str, float]],
                 key: str) -> float:
    """Sum a banked stat over a team's roster (robust to missing players)."""
    return float(sum(banked_by_player.get(str(p.player_id), {}).get(key, 0.0)
                     for p in players))


def _team_banked_all(banked_by_player: Dict[str, Dict[str, float]],
                     team_of: Dict[str, str], team_id: Optional[str],
                     key: str) -> float:
    """Sum a banked stat over EVERY live player the feed attributes to team_id
    (including call-ups absent from the projection). Reconciles with the
    scoreboard where the projection-roster sum would fall short."""
    if not team_id:
        return 0.0
    tid = str(team_id).upper()
    return float(sum(bk.get(key, 0.0) for pid, bk in banked_by_player.items()
                     if str(team_of.get(str(pid), "")).upper() == tid))


def _add_banked_to_sims(rest_sims: List[PlayerSimulation],
                        banked_by_player: Dict[str, Dict[str, float]]
                        ) -> List[PlayerSimulation]:
    """Add the certain banked counts onto each re-simulated remainder array."""
    for ps in rest_sims:
        bk = banked_by_player.get(str(ps.player_id), {})
        d = ps.stat_distributions
        for k in _ADDITIVE_DIST_KEYS:
            if k in d and bk.get(k, 0.0):
                d[k] = d[k] + float(bk[k])
        # Points is DERIVED — recompute from (now full-game) components so the
        # banked one-pt / two-pt / assists flow through exactly once.
        if all(k in d for k in ("one_pt_goals", "two_pt_goals", "assists")):
            d["points"] = d["one_pt_goals"] + 2.0 * d["two_pt_goals"] + d["assists"]
        # Refresh summary stats to the full-game arrays.
        ps.proj_values = {k: float(np.mean(v)) for k, v in d.items()}
        ps.prop_lines = {k: _nearest_half(float(np.median(v))) for k, v in d.items()}
    return rest_sims


def _nearest_half(x: float) -> float:
    """Snap to the nearest x.5 (market lines never sit on an integer)."""
    return np.floor(x) + 0.5


def _banked_only_sim(player_id: str, name: str,
                     bk: Dict[str, float], n: int) -> PlayerSimulation:
    """A PlayerSimulation for a live player with NO pregame projection (a call-up
    / rookie absent from the warehouse). Their banked counts are certain, so each
    stat is a constant array; rest-of-game production is modeled as zero because
    we have no rate for them. This keeps their already-decided props (e.g. a goal
    already scored) visible for trading instead of silently dropping them."""
    dists: Dict[str, np.ndarray] = {}
    for k in _ADDITIVE_DIST_KEYS:
        v = float(bk.get(k, 0.0))
        if v:
            dists[k] = np.full(n, v)
    # points always present (derived), even if zero, so the props table can list it
    one, two, a = bk.get("one_pt_goals", 0.0), bk.get("two_pt_goals", 0.0), bk.get("assists", 0.0)
    dists["points"] = np.full(n, one + 2.0 * two + a)
    proj_vals = {k: float(np.mean(v)) for k, v in dists.items()}
    prop_lines = {k: _nearest_half(float(np.median(v))) for k, v in dists.items()}
    return PlayerSimulation(
        player_id=str(player_id),
        full_name=f"{name} (unprojected)",
        stat_distributions=dists, proj_values=proj_vals, prop_lines=prop_lines,
    )


def _name_from_events(events: list, player_id: str) -> str:
    """Best-effort display name for an unprojected player, pulled from the PBP
    'GOAL by X.' / 'Missed shot by Y.' free-text (the feed carries no name field)."""
    import re
    pid = str(player_id)
    for e in events:
        if str(e.get("shooterId")) == pid or str(e.get("faceoffWinnerId")) == pid \
                or str(e.get("gbPlayerId")) == pid:
            desc = str(e.get("description", ""))
            # "GOAL by Richie Connell." / "Missed shot by R. Connell." /
            # "Faceoff win M. Sisselberger (vs. ...)". Capture up to the period,
            # paren, or "Assist"/"Save" clause that follows the name.
            m = re.search(r"by ([A-Z][A-Za-z.'\-]*(?:\s+[A-Za-z.'\-]+)*)", desc)
            if m:
                nm = re.split(r"\s*(?:\.|\(|Assist|Save)", m.group(1))[0]
                return nm.strip().rstrip(".")
    return f"#{pid}"


class LiveModel:
    """Turns a pregame ProjectionResult + banked live stats into a full-game
    live projection by re-simulating only the time remaining."""

    def __init__(self, engine, seed: int = 123, n_sims: Optional[int] = None):
        # engine: a fitted ProjectionEngine (for player_games correlation input).
        # n_sims: override the sim count for the LIVE re-sim. Live pricing only
        # needs prop probabilities (not tail precision), so a smaller count than
        # the pregame default keeps the board responsive during a game. Defaults
        # to the engine's own count when not supplied.
        self.engine = engine
        n = int(n_sims) if n_sims else getattr(engine.simulator, "n_sims", 20000)
        self.simulator = GameSimulator(n_sims=n, seed=seed)

    def resimulate(self, result: ProjectionResult,
                   banked_by_player: Dict[str, Dict[str, float]],
                   fraction_remaining: float,
                   pace_weight: float = 0.5,
                   home_team_id: Optional[str] = None,
                   away_team_id: Optional[str] = None,
                   team_of: Optional[Dict[str, str]] = None,
                   events: Optional[list] = None) -> LiveProjection:
        frac_rem = float(min(max(fraction_remaining, 0.0), 1.0))

        # 1. Scale each team's PLAYER projections to the remaining game.
        h_players = _scale_players(result.home_players, banked_by_player,
                                   frac_rem, pace_weight)
        a_players = _scale_players(result.away_players, banked_by_player,
                                   frac_rem, pace_weight)

        # 2. Scale the TEAM projections (drives the team-goal draws that player
        #    goals are conditioned on). Team goals get the same pace blend.
        h_proj = self._scale_team(result.home_proj, result.home_players,
                                  banked_by_player, frac_rem, pace_weight)
        a_proj = self._scale_team(result.away_proj, result.away_players,
                                  banked_by_player, frac_rem, pace_weight)

        # 3. Re-simulate the remaining game with the engine's own Monte Carlo.
        game_sim = self.simulator.simulate_game(h_proj, a_proj)
        pg = getattr(self.engine, "player_games", None)
        h_rest = self.simulator.simulate_players(
            h_players, game_sim.home_goals, h_proj.proj_goals,
            opp_save_pct=a_proj.proj_save_pct, player_games=pg,
        )
        a_rest = self.simulator.simulate_players(
            a_players, game_sim.away_goals, a_proj.proj_goals,
            opp_save_pct=h_proj.proj_save_pct, player_games=pg,
        )

        # 4. Add banked (certain) counts onto the simulated remainder.
        h_sims = _add_banked_to_sims(h_rest, banked_by_player)
        a_sims = _add_banked_to_sims(a_rest, banked_by_player)

        # 4b. Surface live players who have NO pregame projection (call-ups /
        #     rookies absent from the warehouse). Their banked stats are certain
        #     but would otherwise be dropped — attach a banked-only sim to the
        #     correct side so their already-decided props stay tradeable.
        n = self.simulator.n_sims
        projected = {str(p.player_id) for p in result.home_players + result.away_players}
        team_of = team_of or {}
        for pid, bk in banked_by_player.items():
            if str(pid) in projected:
                continue
            if not any(bk.get(k, 0.0) for k in _ADDITIVE_DIST_KEYS):
                continue  # nothing banked worth showing
            side = str(team_of.get(str(pid), "")).upper()
            nm = _name_from_events(events or [], pid)
            sim = _banked_only_sim(pid, nm, bk, n)
            if home_team_id and side == str(home_team_id).upper():
                h_sims.append(sim)
            elif away_team_id and side == str(away_team_id).upper():
                a_sims.append(sim)
            else:
                # unknown orientation — default to home so it's not lost
                h_sims.append(sim)

        bh = _team_banked_all(banked_by_player, team_of, home_team_id, "goals")
        ba = _team_banked_all(banked_by_player, team_of, away_team_id, "goals")

        # Live full-game GameSimulation for ML / spread / total. Game markets
        # score 2-pt goals as 2, so we work in SCORES: banked scores (certain)
        # are added onto the re-simulated remainder's scores. The simulator's
        # game_sim already carries remainder home_scores/away_scores.
        bh_2 = _team_banked_all(banked_by_player, team_of, home_team_id, "two_pt_goals")
        ba_2 = _team_banked_all(banked_by_player, team_of, away_team_id, "two_pt_goals")
        banked_home_scores = bh + bh_2   # 1pt*1 + 2pt*2 = goals + (extra point per 2pt)
        banked_away_scores = ba + ba_2
        live_game_sim = self._live_game_sim(
            game_sim, bh, ba, banked_home_scores, banked_away_scores, frac_rem)

        return LiveProjection(
            home_team=result.home_team, away_team=result.away_team,
            home_player_sims=h_sims, away_player_sims=a_sims,
            fraction_remaining=frac_rem, pace_weight=pace_weight,
            banked_home_goals=bh, banked_away_goals=ba,
            home_goal_dist=game_sim.home_goals + bh,
            away_goal_dist=game_sim.away_goals + ba,
            game_sim=live_game_sim,
        )

    @staticmethod
    def _live_game_sim(rest: "GameSimulation", banked_home_goals: float,
                       banked_away_goals: float, banked_home_scores: float,
                       banked_away_scores: float,
                       frac_rem: float = 0.0) -> "GameSimulation":
        """Full-game live GameSimulation = banked (certain) + re-simulated
        remainder. Adds the banked scalars onto the remainder arrays, then
        recomputes win probs / spread / total exactly as simulate_game does.

        The banked TOTALS are certain (points already scored), but the banked
        LEAD mean-reverts: an early lead over-predicts the final margin. We pull
        the score margin toward even by (1 - persistence)*banked_lead — a mean
        shift on the margin distribution only, leaving each side's total scoring
        (and the game total) untouched. See _lead_persistence / project memory."""
        hs = rest.home_scores + banked_home_scores
        as_ = rest.away_scores + banked_away_scores
        banked_lead = float(banked_home_scores) - float(banked_away_scores)
        drift = (1.0 - _lead_persistence(frac_rem)) * banked_lead
        if abs(drift) > _EPS:
            # move each side halfway so the total (hs + as_) is preserved
            hs = hs - drift / 2.0
            as_ = as_ + drift / 2.0
        tied = hs == as_
        # deterministic 50/50 tie-break without RNG (Date/random unavailable in
        # some contexts): alternate by index parity, mean ~0.5 either way.
        even = (np.arange(hs.size) % 2 == 0)
        home_wins = (hs > as_) | (tied & even)
        away_wins = (as_ > hs) | (tied & ~even)
        return GameSimulation(
            n_sims=rest.n_sims,
            home_goals=rest.home_goals + banked_home_goals,
            away_goals=rest.away_goals + banked_away_goals,
            home_scores=hs, away_scores=as_,
            home_win_prob=float(np.mean(home_wins)),
            away_win_prob=float(np.mean(away_wins)),
            tie_prob=0.0,
            expected_total=float(np.median(hs + as_)),
            spread_home=float(np.median(hs - as_)),
            total_distribution=hs + as_,
            margin_distribution=hs - as_,
        )

    @staticmethod
    def _scale_team(tp: TeamProjection, players: List[PlayerProjection],
                    banked_by_player: Dict[str, Dict[str, float]],
                    frac_rem: float, weight: float) -> TeamProjection:
        """Scale a TeamProjection to the remaining game. Counts (goals, shots,
        SOG, 2pt, assists, faceoffs, saves) are time-scaled and pace-blended off
        the team's summed banked totals; rates (save%, FO%) are held flat."""
        bg = _team_banked(players, banked_by_player, "goals")
        b2 = _team_banked(players, banked_by_player, "two_pt_goals")
        bs = _team_banked(players, banked_by_player, "shots")
        bsog = _team_banked(players, banked_by_player, "shots_on_goal")
        ba = _team_banked(players, banked_by_player, "assists")
        bfo = _team_banked(players, banked_by_player, "faceoff_wins")
        bsv = _team_banked(players, banked_by_player, "saves")
        goals = _blend_rest(tp.proj_goals, bg, frac_rem, weight)
        two = _blend_rest(tp.proj_2pt_goals, b2, frac_rem, weight)
        return replace(
            tp,
            proj_goals=goals,
            proj_2pt_goals=min(two, goals),
            proj_1pt_goals=max(goals - min(two, goals), 0.0),
            proj_scores=goals + min(two, goals),  # 1pt*1 + 2pt adds one extra each
            proj_shots=_blend_rest(tp.proj_shots, bs, frac_rem, weight),
            proj_sog=_blend_rest(tp.proj_sog, bsog, frac_rem, weight),
            proj_assists=_blend_rest(tp.proj_assists, ba, frac_rem, weight),
            proj_faceoff_wins=_blend_rest(tp.proj_faceoff_wins, bfo, frac_rem, weight),
            proj_saves=_blend_rest(tp.proj_saves, bsv, frac_rem, weight),
            # rates unchanged
        )


if __name__ == "__main__":
    import argparse
    from live_feed import LiveFeed
    from live_state import reconstruct

    ap = argparse.ArgumentParser(
        description="Rest-of-game live re-sim test on a live PLL game.")
    ap.add_argument("slug", help="game slug, e.g. 2026-ev-36")
    ap.add_argument("--home", help="home team id (e.g. WHP); read from feed if omitted")
    ap.add_argument("--away", help="away team id (e.g. ARC)")
    ap.add_argument("--date", help="game date YYYY-MM-DD (for roster/context)")
    ap.add_argument("--pace-weight", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    # 1. Live state + banked.
    state = LiveFeed(args.slug).poll()
    banked = reconstruct(state.events)
    frac_rem = state.fraction_remaining

    # Team ids: prefer CLI, else infer home/away from the feed's score fields.
    # The feed labels events with teamId; home is whichever team owns homeScore.
    home_id, away_id = args.home, args.away
    if not (home_id and away_id):
        # crude inference: the two distinct teamIds present in events
        tids = [e.get("teamId") for e in state.events if e.get("teamId")]
        uniq = list(dict.fromkeys(tids))
        if len(uniq) >= 2 and not home_id:
            # cannot know orientation from feed reliably; require override
            print("Could not infer home/away orientation; pass --home/--away "
                  f"(teams seen: {uniq})")
        home_id = home_id or (uniq[0] if uniq else None)
        away_id = away_id or (uniq[1] if len(uniq) > 1 else None)

    # 2. Build engine + pregame projection.
    import os
    from projection_engine_v3 import ProjectionEngine
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.getenv("PLL_DB_PATH",
                   os.path.join(root, "data", "analytics_database", "pll_warehouse.duckdb"))
    print(f"Loading engine ({db}) …")
    engine = ProjectionEngine(db_path=db)
    engine.load(); engine.fit(run_backtest=False)
    print(f"Pregame project: {away_id} @ {home_id} ({args.date or 'latest'})")
    result = engine.project(home_team_id=home_id, away_team_id=away_id,
                            game_date=args.date)

    # 3. Live re-sim.
    lm = LiveModel(engine)
    live = lm.resimulate(result, banked.by_player, frac_rem,
                         pace_weight=args.pace_weight,
                         home_team_id=home_id, away_team_id=away_id,
                         team_of=banked.team_of, events=state.events)

    print(f"\n{args.slug}: P{state.period} {state.clock_minutes:02d}:{state.clock_seconds:02d} "
          f"| {state.away_score}-{state.home_score} | {frac_rem:.0%} remaining "
          f"| pace_weight={args.pace_weight}")
    print(f"banked goals: {live.away_team} {live.banked_away_goals:.0f}  "
          f"{live.home_team} {live.banked_home_goals:.0f}\n")
    rows = []
    for ps in live.all_sims():
        pv = ps.proj_values
        g, a, pts = pv.get("goals", 0), pv.get("assists", 0), pv.get("points", 0)
        if pts >= 0.5 or g >= 0.3:
            bk = banked.get(ps.player_id, "points")
            rows.append((ps.full_name, g, a, pts, bk))
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'player':22} {'projG':>6} {'projA':>6} {'projPTS':>8} {'bankedPTS':>10}")
    for name, g, a, pts, bk in rows[:args.top]:
        print(f"{name[:22]:22} {g:6.2f} {a:6.2f} {pts:8.2f} {bk:10.0f}")
