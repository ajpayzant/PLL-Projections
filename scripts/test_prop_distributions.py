"""Regression tests for the prop distribution and pricing fixes.

Each test here corresponds to a defect measured on the 2026 book (weeks 8-12) and
documented in ``analysis/MODEL_FIX_PLAN.md``. The through-line is that all four
defects were invisible to the checks already in place, because every one of them
leaves the MEAN correct and only moves probability mass around:

* bias / MAE compare means, so they cannot see a shape error at all;
* P10-P90 coverage only asks whether the interval is wide enough, never whether
  the mass inside it sits in the right place. It read 95.8% while assists overs
  were priced 17 points light.

So these tests assert on probabilities and tails, never on means alone. The
"means must not move" tests are here too, but as a guard that a shape fix has not
leaked into the projection -- not as evidence the shape is right.

Run: python -m pytest scripts/test_prop_distributions.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boss_export as BX
import projection_engine_v3 as E

# Enough draws that a 1-point assertion on a probability is signal, not noise:
# the standard error on a proportion at N=200k is under 0.12 points.
N_DRAWS = 200_000
SEED = 20260810


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _nb_p0(mu: float, phi: float) -> float:
    """P(0) for the NegBin the engine would draw at this mean and dispersion."""
    n, p = E._negbinom_params(mu, phi)
    return p ** n


# ---------------------------------------------------------------------------
# Bug #1 -- zero inflation stacked on top of the negative binomial's own zeros
# ---------------------------------------------------------------------------

class TestExcessZeroSolver:
    """``_solve_excess_zero`` must invert P(0) = z + (1-z)*P_nb(0)."""

    @pytest.mark.parametrize("mu,phi,target", [
        (0.73, 4.0, 0.55),     # assists, the case that cost -25.4% hold
        (0.73, 20.0, 0.55),    # same but at the corrected dispersion
        (1.71, 40.0, 0.22),    # A goals
        (0.85, 20.0, 0.70),    # M assists, the highest prior in the table
        (3.00, 40.0, 0.05),    # a high-volume player with a small zero rate
        (0.20, 4.0, 0.90),     # a sparse player, near-saturated zero mass
    ])
    def test_total_zero_mass_hits_the_target(self, mu, phi, target):
        """The POINT of the fix: total P(0) equals the measured rate we asked for.

        Verified analytically rather than by sampling, so this test cannot pass
        by luck of the seed.
        """
        z = E._solve_excess_zero(mu, phi, target)
        if z == 0.0:
            # Legitimate outcome: the NB alone already meets or exceeds the
            # target, so no inflation belongs in the draw.
            assert _nb_p0(mu, phi) >= target - 1e-9
            return
        total_p0 = z + (1.0 - z) * _nb_p0(mu / (1.0 - z), phi)
        assert total_p0 == pytest.approx(target, abs=1e-6)

    def test_old_behaviour_overshot_the_target(self):
        """Documents the bug: passing the rate through directly overshoots.

        This is the regression guard. If someone reverts to using ``zero_prob``
        as the inflation parameter, total P(0) climbs back above the target and
        every over gets underpriced again.
        """
        mu, phi, target = 0.73, 4.0, 0.55
        buggy_total = target + (1.0 - target) * _nb_p0(mu / (1.0 - target), phi)
        assert buggy_total > target + 0.10          # ~0.66 vs the 0.55 intended
        fixed = E._solve_excess_zero(mu, phi, target)
        fixed_total = (fixed + (1.0 - fixed) * _nb_p0(mu / (1.0 - fixed), phi)
                       if fixed > 0 else _nb_p0(mu, phi))
        assert fixed_total <= buggy_total - 0.10

    def test_returns_zero_when_negbinom_already_covers_the_target(self):
        """At the offered population's numbers, no inflation is needed at all."""
        assert E._solve_excess_zero(0.73, 4.0, 0.48) == 0.0

    @pytest.mark.parametrize("target", [0.0, 1.0, 1.5, -0.2])
    def test_degenerate_targets_stay_in_range(self, target):
        z = E._solve_excess_zero(0.73, 4.0, target)
        assert 0.0 <= z <= 0.999


class TestZinbDraw:
    """The wired-up draw, sampled, must match the solver's promise."""

    @pytest.mark.parametrize("mu,zero_prob,phi_key", [
        (0.73, 0.55, "assists"),
        (1.71, 0.22, "goals"),
        (0.65, 0.70, "assists"),
    ])
    def test_mean_is_preserved_and_zero_mass_is_capped(self, rng, mu, zero_prob, phi_key):
        draws = _zinb_via_engine(rng, mu, phi_key, zero_prob, N_DRAWS)
        # Mean must survive the fix -- if it moves, the shape change has leaked
        # into the projection and the fix is wrong.
        assert draws.mean() == pytest.approx(mu, rel=0.03)
        # And P(0) must not exceed what the projection can support.
        ceiling = min(math.exp(-mu) + E.ZERO_RATE_POISSON_SLACK, 0.99)
        assert (draws == 0).mean() <= ceiling + 0.01

    def test_assists_over_half_is_no_longer_17_points_light(self, rng):
        """The headline number: 35.0% priced against a 52.3% realized rate."""
        draws = _zinb_via_engine(rng, 0.734, "assists", 0.55, N_DRAWS)
        p_over = float((draws > 0.5).mean())
        assert p_over > 0.48, f"still underpricing the over at {p_over:.1%}"
        assert p_over < 0.60, f"overcorrected to {p_over:.1%}"


def _zinb_via_engine(rng, mu, phi_key, zero_prob, n):
    """Reproduce the engine's nested ``_zinb`` exactly.

    ``_zinb`` is a closure over the simulator's ``rng`` and ``n``, so it cannot
    be imported. Rather than duplicate its body, this calls the same two module
    helpers in the same order the closure does; if that wiring changes, the
    ``test_engine_zinb_matches_this_helper`` check below fails loudly.
    """
    phi = E.PHI_PLAYER.get(phi_key, 2.0)
    z = E._solve_excess_zero(mu, phi, E._cap_zero_rate(zero_prob, mu))
    nb_n, nb_p = E._negbinom_params(mu / max(1.0 - z, 0.01), phi)
    is_zero = rng.random(n) < z
    return np.where(is_zero, 0.0, rng.negative_binomial(nb_n, nb_p, n).astype(float))


def test_engine_zinb_matches_this_helper():
    """Guard against the test helper drifting from the engine's real ``_zinb``.

    Compares source text of the closure to the call sequence asserted above.
    Cheap, and it catches the failure mode where the engine is changed and these
    tests keep passing against a stale copy of the logic.
    """
    import inspect
    src = inspect.getsource(E.MonteCarloSimulator) if hasattr(E, "MonteCarloSimulator") else \
        inspect.getsource(E)
    assert "_solve_excess_zero(mu, phi, _cap_zero_rate(zero_prob, mu))" in src, \
        "engine _zinb no longer wires _cap_zero_rate into _solve_excess_zero"


# ---------------------------------------------------------------------------
# Bug #2 -- pooled zero-rate priors applied to front-line players
# ---------------------------------------------------------------------------

class TestZeroRateCap:

    def test_cap_lowers_a_contradictory_rate(self):
        """A player projected for 1.71 goals cannot be blanked 40% of the time."""
        capped = E._cap_zero_rate(0.40, 1.71)
        assert capped < 0.40
        assert capped == pytest.approx(math.exp(-1.71) + E.ZERO_RATE_POISSON_SLACK, abs=1e-9)

    def test_cap_never_raises_a_rate(self):
        """Sparse and low-usage players must keep their priors untouched.

        The pooled priors are correct for the population they were measured on;
        the cap exists to stop them being applied where the projection
        contradicts them, not to lower them everywhere.
        """
        for zero_prob in (0.05, 0.22, 0.55, 0.95):
            for mu in (0.01, 0.1, 0.5, 1.0, 3.0):
                assert E._cap_zero_rate(zero_prob, mu) <= zero_prob + 1e-12

    def test_low_projection_keeps_a_high_prior(self):
        """The cap must stay nearly inert on sparse players.

        At mu=0.20 the Poisson floor is exp(-0.20) = 0.819, so a 0.90 prior is
        trimmed only to 0.849 -- the D/LSM/SSDM priors of 0.90-0.96 survive
        essentially intact, which is the point: they were measured on exactly
        those players and are correct there.
        """
        assert E._cap_zero_rate(0.90, 0.20) == pytest.approx(0.849, abs=0.002)
        assert E._cap_zero_rate(0.96, 0.05) > 0.94

    @pytest.mark.parametrize("mu", [0.0, -1.0])
    def test_zero_or_negative_mean_passes_through(self, mu):
        assert E._cap_zero_rate(0.55, mu) == 0.55

    def test_cap_is_bounded_below_one(self):
        assert E._cap_zero_rate(0.999, 0.0001) <= 0.99


class TestAssistsDispersion:
    """``PHI_PLAYER['assists']`` was 4.0, measured on the pooled population."""

    def test_assists_phi_is_near_poisson(self):
        """Offered players' assists run var/mean 0.99 (A) and 0.83 (M).

        The pooled 1.4 that justified phi=4.0 is a mixture artefact: combining
        low- and high-usage players inflates variance even when each player is
        individually Poisson. phi must reflect the population we price.
        """
        phi = E.PHI_PLAYER["assists"]
        var_over_mean = 1.0 + 0.85 / phi      # at the offered mean assists of 0.85
        assert var_over_mean < 1.10, (
            f"phi={phi} implies var/mean {var_over_mean:.2f}; offered assists "
            "are 0.96, i.e. Poisson or slightly underdispersed"
        )

    def test_faceoff_wins_phi_matches_within_player_dispersion(self):
        """Re-measured 2022-26 on the pre-game FO specialist (311 team-games).

        Within player-season, var/mean is 1.689 (implied phi 20.3). The pooled
        figure of 2.04 mixes in between-player skill, which the per-player
        projection already handles -- the same mixture artefact that made
        assists phi=4.0 wrong, in the opposite direction.
        """
        phi = E.PHI_PLAYER["fo_wins"]
        var_over_mean = 1.0 + 13.98 / phi        # at the measured specialist mean
        assert var_over_mean == pytest.approx(1.689, abs=0.06), (
            f"phi={phi} implies var/mean {var_over_mean:.3f}, measured 1.689"
        )

    def test_goals_and_saves_dispersion_left_alone(self):
        """Explicitly pinned: these were measured correctly and must not move.

        Goals: offered players run var/mean 1.06 against the 1.04 phi=40 implies.
        Saves: FULL appearances run 0.88 against the 1.10 phi=120 implies, so
        widening it to chase the pooled figure would corrupt the 91% of starts
        the model already fits. The low tail is fixed by the playing-time
        mixture instead -- see TestGoalieSaves.
        """
        assert E.PHI_PLAYER["goals"] == 40.0
        assert E.PHI_PLAYER["saves"] == 120.0


# ---------------------------------------------------------------------------
# Bug #3 -- goalie playing time was never simulated
# ---------------------------------------------------------------------------

class TestGoalieSaves:
    """Measured over 336 team-games (2022-26) with a predictable pre-game starter.

    Empirical targets, named starter's saves:
        mean 12.57   sd 3.67   var/mean 1.07
        P(<=0) 0.30%   P(<=2) 1.19%   P(<=6) 4.46%
    """

    MU = 12.568

    def test_projected_mean_is_unchanged_by_the_mixture(self, rng):
        """Non-negotiable: this fix must move shape only.

        ``LG_STARTER_SF_PER_OPP_SOG`` already embeds the mean backup share
        (0.915/0.942 = 0.971, against a measured E[share] of 0.972), so the draw
        divides it out before redrawing. Skip that and the projection silently
        drops ~3%.
        """
        draws = E._draw_goalie_saves(rng, self.MU, 120.0, N_DRAWS)
        assert draws.mean() == pytest.approx(self.MU, rel=0.01)

    def test_expected_share_matches_the_embedded_constant(self):
        """The two independent measurements must agree, or the mean will drift."""
        alpha, beta = E.GOALIE_PARTIAL_SHARE_BETA
        p = E.P_GOALIE_PARTIAL_GAME
        expected_share = (1.0 - p) + p * (alpha / (alpha + beta))
        embedded = E.LG_STARTER_SF_PER_OPP_SOG / E.LG_TEAM_SF_PER_OPP_SOG
        assert expected_share == pytest.approx(embedded, abs=0.005)

    def test_low_tail_is_reachable(self, rng):
        """The actual defect. A 2-save night happened 1.19% of the time; the
        old draw put 0.05% there, which is what made a saves MS longshot a
        guaranteed loss when it landed."""
        draws = E._draw_goalie_saves(rng, self.MU, 120.0, N_DRAWS)
        assert (draws <= 2).mean() > 0.004, "low tail still unreachable"
        assert (draws <= 2).mean() < 0.025, "low tail overcorrected"
        assert (draws <= 6).mean() == pytest.approx(0.0446, abs=0.025)

    def test_old_draw_could_not_reach_the_low_tail(self, rng):
        """Regression guard: the plain NegBin this replaced, at the same mean."""
        n, p = E._negbinom_params(self.MU, 120.0)
        old = rng.negative_binomial(n, p, N_DRAWS)
        assert (old <= 2).mean() < 0.002

    def test_central_quantiles_do_not_move(self, rng):
        """The mixture must not disturb the 91% of starts already fitted well."""
        n, p = E._negbinom_params(self.MU, 120.0)
        old = rng.negative_binomial(n, p, N_DRAWS).astype(float)
        new = E._draw_goalie_saves(rng, self.MU, 120.0, N_DRAWS)
        for q in (25, 50, 75):
            assert abs(np.percentile(new, q) - np.percentile(old, q)) <= 1.0

    def test_dispersion_override_disables_the_mixture(self, rng):
        """A user-set dispersion index is an explicit statement about spread.

        Honour it as given rather than compounding it with the league mixture.
        """
        draws = E._draw_goalie_saves(rng, self.MU, 120.0, N_DRAWS, partial_prob=0.0)
        assert (draws <= 2).mean() < 0.002
        assert draws.var() / draws.mean() == pytest.approx(1.0 + self.MU / 120.0, rel=0.05)

    def test_mean_preserved_across_the_projection_range(self, rng):
        for mu in (4.0, 8.0, 12.5, 18.0):
            draws = E._draw_goalie_saves(rng, mu, 120.0, 60_000)
            assert draws.mean() == pytest.approx(mu, rel=0.02), f"mean drift at mu={mu}"


# ---------------------------------------------------------------------------
# Bug #4 -- unpriceable tails reached the board
# ---------------------------------------------------------------------------

class _FakeSim:
    """Minimal stand-in for PlayerSimulation; PricingEngine only reads these."""

    def __init__(self, dist: np.ndarray, stat: str = "saves"):
        arr = np.asarray(dist, dtype=float)
        self.player_id = "test"
        self.full_name = "Test Player"
        self.stat_distributions = {stat: arr}
        self.proj_values = {stat: float(arr.mean())}
        self.prop_lines = {stat: 0.5}


class TestPricingGuards:

    def test_all_zero_projection_is_suppressed(self):
        """JC Higginbotham, Cannons, game 38: projected 0.000 saves, offered Over
        0.5 at +8,554, recorded 11. Zero means "no playing-time signal", not
        "impossible"."""
        pricing = E.PricingEngine()
        ml = pricing.price_prop(_FakeSim(np.zeros(E.N_SIMS)), "saves", line=0.5)
        assert ml.offerable is False
        assert "no playing-time signal" in ml.suppress_reason

    def test_thin_tail_is_suppressed(self):
        """Ten of 20,000 sims above the line is noise, not a probability."""
        dist = np.zeros(E.N_SIMS)
        dist[:10] = 5.0
        ml = E.PricingEngine().price_prop(_FakeSim(dist), "saves", line=0.5)
        assert ml.offerable is False
        assert "insufficient" in ml.suppress_reason

    def test_thin_under_side_is_suppressed(self):
        """Both tails matter: a market nobody can lose is also unpriceable."""
        dist = np.full(E.N_SIMS, 5.0)
        dist[:10] = 0.0
        ml = E.PricingEngine().price_prop(_FakeSim(dist), "saves", line=0.5)
        assert ml.offerable is False
        assert "under" in ml.suppress_reason

    def test_a_normal_market_stays_offerable(self):
        rng = np.random.default_rng(SEED)
        dist = rng.negative_binomial(*E._negbinom_params(12.5, 120.0), E.N_SIMS)
        ml = E.PricingEngine().price_prop(_FakeSim(dist.astype(float)), "saves", line=12.5)
        assert ml.offerable is True
        assert ml.suppress_reason == ""

    def test_price_is_clamped_out_of_the_five_figure_range(self):
        """Every 2026 prop priced above +8,000 was wrong by orders of magnitude."""
        dist = np.zeros(E.N_SIMS)
        dist[:60] = 5.0            # clears MIN_SIM_SUPPORT, so it is offerable
        ml = E.PricingEngine().price_prop(_FakeSim(dist), "saves", line=0.5)
        assert ml.offerable is True
        assert ml.fair_over_prob >= E.MIN_PRICE_PROB
        assert int(ml.over_odds.lstrip("+")) <= 5000

    def test_milestones_survive_suppression(self):
        """``price_milestones`` mutates ``ml.stat``, so a suppressed line must
        still be a MarketLine. This is why the fix is a flag, not ``None``."""
        mls = E.PricingEngine().price_milestones(
            _FakeSim(np.zeros(E.N_SIMS)), "saves", [1.0, 2.0])
        assert [m.stat for m in mls] == ["saves_1+", "saves_2+"]
        assert all(m.offerable is False for m in mls)

    def test_marketline_defaults_to_offerable(self):
        """Callers constructing a MarketLine directly must not be broken."""
        ml = E.MarketLine(stat="goals", line=0.5, fair_over_prob=0.5,
                          fair_under_prob=0.5, over_odds="-110", under_odds="-110",
                          juice=0.05)
        assert ml.offerable is True
        assert ml.suppress_reason == ""


class TestBossExportGuards:
    """The BOSS export prices from the distribution directly, bypassing
    PricingEngine, and publishes ladders up to 22+ saves / 26+ faceoff wins.
    Without its own guard the top of every ladder is sampling noise."""

    def test_guard_constants_match_the_engine(self):
        """boss_export duplicates these to stay numpy+stdlib only. If the engine
        moves and this does not, exported odds silently stop matching the app."""
        assert BX.MIN_SIM_SUPPORT == E.MIN_SIM_SUPPORT
        assert BX.MIN_PRICE_PROB == E.MIN_PRICE_PROB

    def test_near_certain_thresholds_are_flagged(self):
        """Both tails are guarded, and for saves it is the LOW thresholds that
        fail: off a 12.5 projection, 1+ saves is a certainty, so nobody can lose
        the No side and there is no market. 22+ still clears (272 of 20,000
        sims), so the guard is about support, not about distance from the mean."""
        rng = np.random.default_rng(SEED)
        dist = rng.negative_binomial(*E._negbinom_params(12.5, 120.0), E.N_SIMS)
        block = BX._stat_block(dist.astype(float), "saves", 12.5, 0.075)
        assert block["ou"]["offerable"] is True
        by_k = {m["threshold"]: m for m in block["milestones"]}
        assert by_k[1]["offerable"] is False
        assert "under support" in by_k[1]["suppress_reason"]
        assert by_k[12]["offerable"] is True

    def test_unsupported_high_threshold_is_flagged(self):
        """The over-support guard fires on a genuinely thin tail.

        Worth recording why this needs a hand-built distribution: at
        N_SIMS=20,000 a realistic ladder is better supported than it looks -- 6+
        goals off a 1.7-goal projection still draws 229 sims, comfortably past
        MIN_SIM_SUPPORT. So on normal field-player props the binding constraint
        is the UNDER side, and this guard is a backstop for degenerate
        distributions like the all-but-zero goalie case.
        """
        dist = np.zeros(E.N_SIMS)
        dist[:20] = 8.0                      # 20 of 20,000 clear any threshold
        block = BX._stat_block(dist, "goals", float(dist.mean()), 0.075)
        by_k = {m["threshold"]: m for m in block["milestones"]}
        assert by_k[1]["offerable"] is False
        assert "over support" in by_k[1]["suppress_reason"]

    def test_zero_projection_suppresses_every_market(self):
        block = BX._stat_block(np.zeros(E.N_SIMS), "saves", 0.0, 0.075)
        assert block["ou"]["offerable"] is False
        assert all(not m["offerable"] for m in block["milestones"])

    def test_ge_ladder_stays_unclamped_and_monotonic(self):
        """The ladder is what the BOSS Tool re-derives from, so clamping it would
        break monotonicity and create arbitrage between an O/U and its X+."""
        rng = np.random.default_rng(SEED)
        dist = rng.negative_binomial(*E._negbinom_params(12.5, 120.0), E.N_SIMS)
        ladder = BX.ge_probability_ladder(dist.astype(float), 22)
        assert all(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1))
        assert min(ladder) < E.MIN_PRICE_PROB       # genuinely unclamped

    def test_exported_odds_never_exceed_the_ceiling(self):
        rng = np.random.default_rng(SEED)
        dist = rng.negative_binomial(*E._negbinom_params(12.5, 120.0), E.N_SIMS)
        block = BX._stat_block(dist.astype(float), "saves", 12.5, 0.075)
        for m in block["milestones"]:
            odds = m["yes_odds"]
            if odds.startswith("+"):
                assert int(odds[1:]) <= 5000, f"{m['label']} priced at {odds}"


# ---------------------------------------------------------------------------
# The cross-cutting test that would have caught all of this on day one
# ---------------------------------------------------------------------------

class TestPoissonFloor:
    """On a 0.5 line, fair P(Over) is P(at least one), and a count distribution
    with the same mean cannot fall far below ``1 - exp(-mu)``.

    This is the check that mean-based accuracy metrics structurally cannot make.
    Assists failed it by 15 points all season while bias read -0.04 and P10-P90
    coverage read 95.8%.
    """

    TOLERANCE = 0.05

    @pytest.mark.parametrize("mu,zero_key,phi_key", [
        (0.73, "A_assists", "assists"),
        (0.85, "M_assists", "assists"),
        (1.71, "A_goals", "goals"),
        (1.20, "M_goals", "goals"),
        (0.50, "A_assists", "assists"),
        (2.50, "A_goals", "goals"),
    ])
    def test_p_over_half_respects_the_poisson_floor(self, rng, mu, zero_key, phi_key):
        draws = _zinb_via_engine(rng, mu, phi_key, E.ZERO_RATE[zero_key], N_DRAWS)
        p_over = float((draws > 0.5).mean())
        floor = 1.0 - math.exp(-mu)
        assert p_over >= floor - self.TOLERANCE, (
            f"{zero_key} at mu={mu}: priced {p_over:.1%} against a Poisson floor "
            f"of {floor:.1%} -- the shape is pushing mass to zero again"
        )

    def test_the_old_wiring_fails_this_test(self, rng):
        """Proves the test has teeth: it must reject the pre-fix behaviour.

        A test that passes both before and after a fix proves nothing.
        """
        mu, target = 0.73, E.ZERO_RATE["A_assists"]
        phi = E.PHI_PLAYER["assists"]
        nb_n, nb_p = E._negbinom_params(mu / (1.0 - target), phi)
        buggy = np.where(rng.random(N_DRAWS) < target, 0.0,
                         rng.negative_binomial(nb_n, nb_p, N_DRAWS))
        p_over = float((buggy > 0.5).mean())
        assert p_over < (1.0 - math.exp(-mu)) - self.TOLERANCE


# Realistic PLL field: a lead attacker down to a low-usage midfielder. Used by
# the conditioning tests below, where the SPREAD of means matters -- the defect
# being guarded against hit low-mean players hardest.
_FIELD_MU = [1.9, 1.4, 1.1, 0.85, 0.6, 0.35]


def _field_draws(rng, mus, n=N_DRAWS):
    """Per-player zero-inflated goal draws, as simulate_players builds them."""
    out = []
    for mu in mus:
        out.append(_zinb_via_engine(rng, mu, "goals", E.ZERO_RATE["A_goals"], n))
    return out


def _team_draws(rng, mus, n=N_DRAWS):
    nb_n, nb_p = E._negbinom_params(sum(mus), 40.0)
    return rng.negative_binomial(nb_n, nb_p, n).astype(float)


class TestTeamTotalConditioning:
    """Reconciling player goals to the team total must not manufacture zeros.

    This defect was found by the Poisson-floor check in scripts/fast_backtest.py
    AFTER the zero-inflation fix had landed, which is the point of keeping that
    check: it is the same "mean right, shape wrong" failure mode arriving by a
    second route, and every mean-based metric was clean because the team total
    was conserved throughout.

    The old code did ``round(draw * team_total / sum_of_draws)``. Zero is an
    absorbing state under multiplication -- ``0 * scale == 0`` always, while a
    1 falls to 0 whenever ``scale < 0.5`` -- so mass could only ratchet into
    zero, never out. Measured cost: 4-7 points of extra zero mass per player,
    pushing fair P(Over) on a 0.5 line ~7 points below the Poisson floor.
    """

    TOLERANCE = 0.05

    def test_conditioning_hits_the_team_total_exactly(self, rng):
        """The constraint this step exists to enforce. The old code hit it ~50%."""
        raw = _field_draws(rng, _FIELD_MU)
        team = _team_draws(rng, _FIELD_MU)
        out = E._condition_to_team_total(rng, raw, team, _FIELD_MU)
        target = np.round(team).clip(min=0)
        # A sim whose team draw is below the field's own floor of zero cannot be
        # matched downward; that is the one documented exception.
        assert float((np.stack(out).sum(axis=0) == target).mean()) > 0.999

    def test_conditioning_does_not_add_zero_mass(self, rng):
        """P(0) per player must survive reconciliation, because P(0) IS the price
        on a 0.5 line. The mean surviving is not sufficient and never was."""
        raw = _field_draws(rng, _FIELD_MU)
        team = _team_draws(rng, _FIELD_MU)
        out = E._condition_to_team_total(rng, raw, team, _FIELD_MU)
        for mu, before, after in zip(_FIELD_MU, raw, out):
            p0_before = float((before == 0).mean())
            p0_after = float((after == 0).mean())
            assert p0_after - p0_before < 0.02, (
                f"mu={mu}: zero mass grew {p0_before:.4f} -> {p0_after:.4f} "
                f"during conditioning; a 0.5-line over is priced off exactly this"
            )

    def test_conditioning_respects_the_poisson_floor(self, rng):
        raw = _field_draws(rng, _FIELD_MU)
        team = _team_draws(rng, _FIELD_MU)
        out = E._condition_to_team_total(rng, raw, team, _FIELD_MU)
        for mu, arr in zip(_FIELD_MU, out):
            p_over = float((arr > 0.5).mean())
            floor = 1.0 - math.exp(-mu)
            assert p_over >= floor - self.TOLERANCE, (
                f"mu={mu}: conditioned draw prices {p_over:.1%} against a floor "
                f"of {floor:.1%}"
            )

    def test_conditioning_preserves_each_players_mean(self, rng):
        """Reconciliation redistributes goals; it must not re-forecast anyone."""
        raw = _field_draws(rng, _FIELD_MU)
        team = _team_draws(rng, _FIELD_MU)
        out = E._condition_to_team_total(rng, raw, team, _FIELD_MU)
        for mu, before, after in zip(_FIELD_MU, raw, out):
            assert abs(float(after.mean()) - float(before.mean())) < 0.05, (
                f"mu={mu}: mean moved {before.mean():.3f} -> {after.mean():.3f}"
            )

    def test_the_old_multiplicative_wiring_fails_this(self, rng):
        """Teeth check: the rescale-and-round version must be rejected."""
        raw = _field_draws(rng, _FIELD_MU)
        team = _team_draws(rng, _FIELD_MU)
        sum_raw = np.maximum(sum(raw), 0.01)
        scale = np.round(team).clip(min=0) / sum_raw
        breached = 0
        for mu, arr in zip(_FIELD_MU, raw):
            old = np.round(arr * scale).clip(min=0)
            if float((old > 0.5).mean()) < (1.0 - math.exp(-mu)) - self.TOLERANCE:
                breached += 1
        assert breached >= len(_FIELD_MU) - 1, (
            "the old conditioning should breach the floor for nearly every "
            "player; if it no longer does, this test has stopped testing anything"
        )

    def test_identity_when_the_sums_already_agree(self, rng):
        """No transfer needed means no perturbation. The multiplicative version
        applied a scale near 1.0 and rounded anyway, which is where the damage
        came from."""
        raw = _field_draws(rng, _FIELD_MU)
        team = sum(raw)  # exact agreement by construction
        out = E._condition_to_team_total(rng, raw, team, _FIELD_MU)
        for before, after in zip(raw, out):
            assert np.array_equal(before, after)

    def test_weights_fall_back_to_uniform_when_degenerate(self, rng):
        """A team of all-zero projections must not raise or produce NaN."""
        raw = _field_draws(rng, _FIELD_MU)
        team = _team_draws(rng, _FIELD_MU)
        out = E._condition_to_team_total(rng, raw, team, [0.0] * len(_FIELD_MU))
        assert all(np.all(np.isfinite(a)) for a in out)
        assert all(np.all(a >= 0) for a in out)

    def test_empty_field_returns_empty(self, rng):
        assert E._condition_to_team_total(rng, [], np.zeros(10), []) == []


# ---------------------------------------------------------------------------
# SOG thinning (_draw_thinned)
# ---------------------------------------------------------------------------

# (proj_shots, sog_rate) pairs spanning the volume range the old clamp damaged.
# The loss scaled inversely with volume -- -13% at 6 shots, -38% at 1.5 -- so the
# low end matters most and is deliberately over-represented.
_SOG_CASES = [
    (6.00, 0.63),
    (4.50, 0.63),
    (3.50, 0.63),
    (2.50, 0.63),
    (1.50, 0.63),
    (0.80, 0.60),
    (4.50, 0.85),   # high-rate shooter
]


def _shots_draw(rng, mu, n=N_DRAWS):
    """A shots draw as simulate_players builds it (plain NegBin, no inflation)."""
    nb_n, nb_p = E._negbinom_params(mu, E.PHI_PLAYER["shots"])
    return rng.negative_binomial(nb_n, nb_p, n).astype(float)


class TestSOGThinning:
    """SOG is a subset of shots, so it must be thinned out of the shots draw.

    Drawing it independently and clipping with ``np.minimum`` cost 13-38% of the
    projected mean, because the minimum of two independent draws sits below both.
    On the book that showed up as SOG projections running 0.788x actual and a
    +16.3 point calibration gap.
    """

    # 1.5% of the mean. Sampling error at N_DRAWS is well under this; the bug
    # being guarded against was 10-40x larger.
    MEAN_TOL = 0.015

    @pytest.mark.parametrize("shots,rate", _SOG_CASES)
    def test_thinning_preserves_the_projected_mean(self, rng, shots, rate):
        """E[Binomial(N, r)] = r*E[N], so the mean comes out for free."""
        sh = _shots_draw(rng, shots)
        sog = E._draw_thinned(rng, sh, rate)
        target = rate * float(sh.mean())
        assert abs(float(sog.mean()) - target) <= self.MEAN_TOL * target, (
            f"proj_shots={shots} rate={rate}: got {sog.mean():.4f}, "
            f"want {target:.4f}"
        )

    @pytest.mark.parametrize("shots,rate", _SOG_CASES)
    def test_the_old_clamp_wiring_fails_this(self, rng, shots, rate):
        """Teeth check. If min(independent NegBin, shots) ever stops losing mass,
        this test has stopped testing anything."""
        sh = _shots_draw(rng, shots)
        mu_sog = shots * rate
        nb_n, nb_p = E._negbinom_params(mu_sog, E.PHI_PLAYER["sog"])
        indep = rng.negative_binomial(nb_n, nb_p, len(sh)).astype(float)
        old = np.minimum(indep, sh)
        loss = (mu_sog - float(old.mean())) / mu_sog
        assert loss > 0.10, (
            f"the old clamp should lose >10% of the mean at proj_shots={shots}; "
            f"measured {loss:.1%}"
        )

    @pytest.mark.parametrize("shots,rate", _SOG_CASES)
    def test_sog_never_exceeds_shots_in_any_sim(self, rng, shots, rate):
        """The invariant the clamp existed to enforce, now true by construction
        rather than imposed after the fact."""
        sh = _shots_draw(rng, shots)
        sog = E._draw_thinned(rng, sh, rate)
        assert np.all(sog <= sh)
        assert np.all(sog >= 0)

    def test_thinned_sog_dispersion_matches_measurement(self, rng):
        """Thinning inherits the parent's width, so this is really a check on
        PHI_PLAYER["shots"].

        At the matched population mean (mu_shots=5.211, rate=0.6375, over 2,179
        full-workload player-games) actual SOG is var/mean 1.095 and thinning a
        phi=15 shots draw returns 1.213 -- close, and much closer than the phi=5
        this replaced, which gave 1.663. The band below would fail if phi["shots"]
        drifted back toward 5."""
        sh = _shots_draw(rng, 5.211)
        sog = E._draw_thinned(rng, sh, 0.6375)
        vm = float(sog.var()) / float(sog.mean())
        assert 0.95 <= vm <= 1.35, (
            f"thinned SOG var/mean {vm:.3f} outside the measured band; actual is "
            f"1.095 and phi['shots']={E.PHI_PLAYER['shots']}"
        )

    def test_shots_dispersion_stays_near_the_measured_value(self, rng):
        """The parent draw's own shape, checked directly at the matched mean.
        Actual is var/mean 1.211, sd 2.512. phi=5 gave 2.038/3.256 -- far too
        wide -- and that width propagated into SOG."""
        sh = _shots_draw(rng, 5.211)
        vm = float(sh.var()) / float(sh.mean())
        assert 1.05 <= vm <= 1.55, f"shots var/mean {vm:.3f}; actual is 1.211"

    @pytest.mark.parametrize("vi", [1.05, 1.3, 1.6, 2.0])
    def test_var_index_override_moves_dispersion_not_the_mean(self, rng, vi):
        """The user's per-player dispersion slider must still work, and must
        reshape only the tails. A plain binomial cannot widen on request, so the
        per-shot probability is drawn from a Beta with the same mean."""
        sh = _shots_draw(rng, 4.0)
        rate = 0.63
        sog = E._draw_thinned(rng, sh, rate, var_index=vi)
        target = rate * float(sh.mean())
        assert abs(float(sog.mean()) - target) <= self.MEAN_TOL * target
        got = float(sog.var()) / float(sog.mean())
        # The thinned draw has a natural floor set by the parent's own width (see
        # test_thinned_sog_dispersion_matches_measurement), so a request below it
        # clamps rather than failing: a subset count cannot be steadier than one
        # draw per parent event. Only requests above the floor are held to target.
        floor = float(E._draw_thinned(rng, sh, rate).var()) / float(
            E._draw_thinned(rng, sh, rate).mean())
        if vi > floor + 0.15:
            assert abs(got - vi) <= 0.10, f"asked for {vi}, got {got:.3f}"
        else:
            assert got >= floor - 0.10

    def test_var_index_is_monotone(self, rng):
        """Higher requested dispersion must not produce a narrower result."""
        sh = _shots_draw(rng, 4.0)
        seen = [float(E._draw_thinned(rng, sh, 0.63, var_index=v).var())
                for v in (1.2, 1.5, 1.8, 2.2)]
        assert seen == sorted(seen), seen

    def test_degenerate_rates(self, rng):
        sh = _shots_draw(rng, 3.0)
        assert np.all(E._draw_thinned(rng, sh, 0.0) == 0.0)
        assert np.array_equal(E._draw_thinned(rng, sh, 1.0), sh)
        # Out-of-range rates clamp rather than raising or producing sog > shots.
        assert np.array_equal(E._draw_thinned(rng, sh, 1.4), sh)
        assert np.all(E._draw_thinned(rng, sh, -0.2) == 0.0)

    def test_all_zero_parent_is_safe(self, rng):
        """A player projected for no shots must yield no SOG, not a NaN."""
        out = E._draw_thinned(rng, np.zeros(1000), 0.63, var_index=1.5)
        assert np.all(out == 0.0)
        assert np.all(np.isfinite(out))
