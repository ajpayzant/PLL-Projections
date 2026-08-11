"""Why does the full engine still breach the Poisson floor on goals?

``analysis/reprice_offered.py`` draws goals straight from ``_zinb`` and shows no
breach. ``scripts/fast_backtest.py --floor-only`` runs the real simulator and
shows 21% of goals props breaching. Both cannot be right about the same
distribution, so one of them is not looking at the distribution the engine ships.

The difference is the team-conditioning step in ``simulate_players``: after the
per-player draw, every field player's goals are rescaled so the team's total
matches the team goal draw, then rounded back to an integer.

    scale  = round(team_goal_draws) / sum(raw_goals)
    scaled = np.round(raw_goals[pid] * scale)

``np.round`` is the suspect. A player who drew exactly 1 goal keeps it only when
``scale >= 0.5``; below that the goal is rounded away. Since ``scale`` fluctuates
around 1.0 across sims, a nontrivial share of single-goal draws get erased —
which manufactures zeros AFTER ``_solve_excess_zero`` has already been careful
about them. Rounding down 1->0 and up 1->1 is not symmetric at the low end,
because there is nothing below zero to round up from.

This script measures the effect in isolation, with no engine run: take a NegBin
goal draw for a realistic 6-player field, apply the exact conditioning code, and
compare P(0) and the mean before and after.

Run: python scripts/diag_conditioning_zeros.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import projection_engine_v3 as E

N = 200_000
RNG = np.random.default_rng(4242)

# A realistic PLL field: one high-usage attacker down to a low-usage midfielder.
FIELD_MU = [1.9, 1.4, 1.1, 0.85, 0.6, 0.35]
TEAM_MU = sum(FIELD_MU)


def _draw_field(rng, mus):
    """Per-player zero-inflated draws, exactly as simulate_players builds them."""
    out = []
    for mu in mus:
        phi = E.PHI_PLAYER["goals"]
        # Use the engine's own prior for an attacker so the zero mass is realistic.
        prior = E._cap_zero_rate(E.ZERO_RATE["A_goals"], mu)
        z = E._solve_excess_zero(mu, phi, prior)
        nb_n, nb_p = E._negbinom_params(mu / max(1.0 - z, 0.01), phi)
        is_zero = rng.random(N) < z
        out.append(np.where(is_zero, 0.0,
                            rng.negative_binomial(nb_n, nb_p, N).astype(float)))
    return out


def _condition(raw, team_draws, stochastic):
    """The engine's conditioning step. stochastic=False is what ships today."""
    sum_raw = np.maximum(sum(raw), 0.01)
    scale = np.round(team_draws).clip(min=0) / sum_raw
    out = []
    for r in raw:
        scaled = r * scale
        if stochastic:
            scaled = np.floor(scaled + RNG.random(N))
        else:
            scaled = np.round(scaled)
        out.append(scaled.clip(min=0))
    return out


def _condition_transfer(raw, team_draws, mus, rng):
    """Candidate fix: reconcile by moving whole goals, never by multiplying.

    The multiplicative version cannot move a player out of zero -- ``0 * scale``
    is 0 for every scale -- while it readily rounds a 1 down to 0. Zero is an
    absorbing state, so P(0) can only ratchet upward.

    This instead computes the shortfall between the team draw and the sum of the
    per-player draws, then transfers that many whole goals:

      * surplus  -> remove goals from players who have them, weighted by how many
      * shortfall -> add goals to players, weighted by projected mean, which is
        the only branch that lets a zero become a one

    When the sums already agree it is the identity, so the per-player marginal is
    untouched in those sims rather than perturbed by a scale near 1.
    """
    n = len(raw[0])
    k = len(raw)
    goals = np.stack([r.copy() for r in raw])            # (k, n)
    target = np.round(team_draws).clip(min=0)
    diff = (target - goals.sum(axis=0)).astype(int)      # + means add, - means remove

    w_add = np.asarray(mus, dtype=float)
    w_add = w_add / w_add.sum()

    # Vectorising over sims is awkward because each sim transfers a different
    # count, so loop over the transfer budget instead: at most a handful of goals
    # separate the two sums, and each iteration handles every sim at once.
    for _ in range(int(np.abs(diff).max()) if n else 0):
        add = diff > 0
        rem = diff < 0
        if add.any():
            pick = rng.choice(k, size=n, p=w_add)
            sel = add & True
            goals[pick[sel], np.flatnonzero(sel)] += 1.0
            diff[sel] -= 1
        if rem.any():
            idx = np.flatnonzero(rem)
            have = goals[:, idx]
            tot = have.sum(axis=0)
            ok = tot > 0
            if ok.any():
                sub = idx[ok]
                p = have[:, ok] / tot[ok]
                # Weighted pick per sim via the inverse-CDF trick.
                u = rng.random(len(sub))
                pick = (np.cumsum(p, axis=0) < u).sum(axis=0).clip(0, k - 1)
                goals[pick, sub] -= 1.0
                diff[sub] += 1
            # A sim with no goals left to remove cannot reach a lower target;
            # leave it, which is the same floor the multiplicative version hits.
            diff[idx[~ok]] = 0
    return [goals[i] for i in range(k)]


def main() -> None:
    # Team goals: the engine draws these in simulate_game; a NegBin at the team
    # mean is the right stand-in and its own shape is not what is being tested.
    nb_n, nb_p = E._negbinom_params(TEAM_MU, 40.0)
    team_draws = RNG.negative_binomial(nb_n, nb_p, N).astype(float)

    raw = _draw_field(RNG, FIELD_MU)
    det = _condition(raw, team_draws, stochastic=False)
    sto = _condition(raw, team_draws, stochastic=True)
    trf = _condition_transfer(raw, team_draws, FIELD_MU, RNG)

    print(f"Field goal means: {FIELD_MU}   team mu={TEAM_MU:.2f}   draws={N:,}\n")
    print("P(0) after each conditioning method, and the breach of the Poisson")
    print("floor it implies on a 0.5 line (positive = we underprice the over).\n")
    print(f"{'mu':>6} {'P0 raw':>8} {'P0 det':>8} {'P0 sto':>8} {'P0 trf':>8} "
          f"{'floor':>7} {'br det':>8} {'br sto':>8} {'br trf':>8} "
          f"{'mn raw':>7} {'mn det':>7} {'mn trf':>7}")
    print("-" * 108)

    worst = {"det": 0.0, "sto": 0.0, "trf": 0.0}
    for mu, r, d, s, t in zip(FIELD_MU, raw, det, sto, trf):
        # On a 0.5 line, P(Over) is just 1 - P(0).
        p0 = {k: float((v == 0).mean()) for k, v in
              (("raw", r), ("det", d), ("sto", s), ("trf", t))}
        floor = 1.0 - np.exp(-mu)
        br = {k: floor - (1.0 - p0[k]) for k in ("det", "sto", "trf")}
        for k in worst:
            worst[k] = max(worst[k], br[k])
        print(f"{mu:6.2f} {p0['raw']:8.4f} {p0['det']:8.4f} {p0['sto']:8.4f} "
              f"{p0['trf']:8.4f} {floor:7.4f} "
              f"{br['det']:+8.4f} {br['sto']:+8.4f} {br['trf']:+8.4f} "
              f"{r.mean():7.3f} {d.mean():7.3f} {t.mean():7.3f}")

    print(f"\nWorst floor breach:")
    print(f"  deterministic round (SHIPPING TODAY): {worst['det']:+.4f}")
    print(f"  stochastic round:                     {worst['sto']:+.4f}")
    print(f"  integer transfer (candidate fix):     {worst['trf']:+.4f}")

    print(f"\nTeam total: raw={sum(x.mean() for x in raw):.3f}  "
          f"det={sum(x.mean() for x in det):.3f}  "
          f"sto={sum(x.mean() for x in sto):.3f}  "
          f"trf={sum(x.mean() for x in trf):.3f}  "
          f"team_draw={team_draws.mean():.3f}")
    # The point of the transfer method is that it reproduces the team draw
    # exactly per sim, not just on average -- that is the constraint the
    # conditioning step exists to enforce.
    exact = float((np.stack(trf).sum(axis=0) == np.round(team_draws).clip(min=0)).mean())
    exact_det = float((np.stack(det).sum(axis=0) == np.round(team_draws).clip(min=0)).mean())
    print(f"\nSims where the team total matches the team draw EXACTLY:")
    print(f"  deterministic: {exact_det:.1%}    integer transfer: {exact:.1%}")

    print("\nRead the P(0) columns, not the means: rounding can hold the team")
    print("total while still moving individual players' zero mass, which is the")
    print("quantity a 0.5-line price is made from.")


if __name__ == "__main__":
    main()
