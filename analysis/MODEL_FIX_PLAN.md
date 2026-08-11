# PLL Projection Model — Distribution Fix Plan

What to change in `projection_engine_v3.py` so prop prices match reality. Every
diagnosis below was reproduced against the engine's own math and validated against
the 741 offered props from weeks 8–12.

**Headline:** the distribution machinery is well built. The negative-binomial
sampler is correct, the dispersion values were empirically derived, and the
mean projections are good. The mispricing comes from **four specific, local bugs**
— not from the model's architecture.

| # | Bug | Location | Effect | Effort |
|---|---|---|---|---|
| **1** | **Zero-inflation double-counts zeros** | `_zinb`, line 3074 | **−17 pts on Assists P(Over)** | ~15 lines |
| **2** | Zero rates use roster-wide priors for star players | `ZERO_RATE`, line 174 | Compounds #1 | ~10 lines |
| **3** | No goalie pull/relief modelling | `_assign_player_goalie_saves`, line 4297 | 56% of goalie error | ~30 lines |
| **4** | No odds ceiling; 0.000 projections priced | `_am` / `price_prop`, line 3451 | +8,554 on a "0-save" starter | ~8 lines |

Fix #1 alone closes most of the −25.4% hold on Assists O/U.

---

## Bug #1 — Zero-inflation stacks on top of the negative binomial

**This is the big one and it is a genuine bug, not a tuning choice.**

`_zinb()` at line 3074 does this:

```python
nb_n, nb_p = _negbinom_params(mu / max(1.0 - zero_prob, 0.01), phi)
is_zero = rng.random(n) < zero_prob
counts  = rng.negative_binomial(nb_n, nb_p, n).astype(float)
return np.where(is_zero, 0.0, counts)
```

The intent is right: inflate the mean by `1/(1-z)` so that after zeroing out a
fraction `z`, the overall mean returns to `mu`. **The mean does come back correct
— that's why the projections look good.** But the *zero mass* does not.

A negative binomial already produces zeros on its own. So the total probability of
zero becomes:

```
P(0) = z  +  (1 - z) × P_negbinom(0)
       ↑            ↑
   intended     unintended extra
```

Worked through with the engine's real numbers for assists (`mu=0.73`, `phi=4.0`,
`z=0.55`):

| | |
|---|---|
| Zero-inflation contributes | 0.550 |
| The NB adds on top | (1 − 0.55) × 0.256 = **0.115** |
| **Total P(0)** | **0.665** — we wanted 0.550 |
| **So P(Over 0.5)** | **33.5%** — should be 45.0% |

**The engine actually printed 35.0%. Reality was 52.3%.** This reproduces the
observed mispricing almost exactly, which is what confirms the diagnosis.

Note the mean stayed correct at 0.731 in the same simulation. That is precisely why
this went unnoticed: **every mean-based accuracy check passes while the probability
that sets the price is 17 points wrong.**

### The fix — solve for the excess zero mass instead of assuming it

`zero_prob` should be treated as the **target total** P(0), then the inflation
component back-solved so the NB's own zeros are accounted for:

```python
def _solve_excess_zero(mu: float, phi: float, target_p0: float) -> float:
    """Excess zero mass such that TOTAL P(0) == target_p0.

    A NegBin already places mass at zero, so passing target_p0 straight into a
    ZINB double-counts it: P(0) = z + (1-z)*P_nb(0) > target_p0. This inverts
    that relation by bisection, which is robust because P(0) is monotonic in z.
    Returns 0.0 when the NB alone already meets or exceeds the target — in that
    case no inflation is needed and the raw NB is the right distribution.
    """
    mu = max(mu, 0.01)
    nb_n, nb_p = _negbinom_params(mu, phi)
    if nb_p ** nb_n >= target_p0:
        return 0.0
    lo, hi = 0.0, 0.999
    for _ in range(60):          # 60 iterations = machine precision
        mid = 0.5 * (lo + hi)
        n_i, p_i = _negbinom_params(mu / max(1.0 - mid, 0.01), phi)
        if mid + (1.0 - mid) * (p_i ** n_i) < target_p0:
            lo = mid
        else:
            hi = mid
    return lo
```

Then in `_zinb`, replace the direct use of `zero_prob`:

```python
z = _solve_excess_zero(mu, phi, zero_prob)
nb_n, nb_p = _negbinom_params(mu / max(1.0 - z, 0.01), phi)
is_zero = rng.random(n) < z
```

### Validated result

Simulated at 400,000 draws, comparing the current engine, the fix, and what
actually happened on line-0.5 props:

| Stat | mu | Current | **Fixed** | **Actual** |
|---|---:|---:|---:|---:|
| Assists (A) | 1.02 | 37.6% | **59.6%** | **59.5%** |
| Assists (M) | 0.65 | 24.7% | **45.2%** | 51.0% |
| Assists (all, line 0.5) | 0.73 | 30.1% | **49.0%** | 52.3% |
| Goals (all, line 0.5) | 1.20 | 57.0% | **69.2%** | 75.9% |

Assists for attackers lands at 59.6% against an actual 59.5%. The mean is
preserved throughout (1.022 vs a 1.020 target), so **projections don't move —
only the prices do.**

A residual gap remains (49.0% vs 52.3% on assists, 69.2% vs 75.9% on goals). That
is Bug #2 — plus one refinement specific to assists, below.

### Refinement: `PHI_PLAYER["assists"] = 4.0` is itself too wide

Once the double-count is removed, assists still price ~3 points light, and the
reason is separate. At `mu=0.734`, the intrinsic zero mass of the negative binomial
alone already exceeds the observed 0.477:

| `phi` | NB-alone P(0) | Implied P(Over) | var/mean |
|---:|---:|---:|---:|
| **4 (current)** | 0.510 | 49.0% | 1.18 |
| 8 | 0.495 | 50.5% | 1.09 |
| 20 | 0.486 | 51.4% | 1.04 |
| ∞ (Poisson) | 0.480 | 52.0% | 1.00 |

So for offered players, `_solve_excess_zero` correctly returns **zero** excess
inflation — the NB is already at or past the target. To reach 52.3% the dispersion
itself has to come down.

Measuring variance/mean on the actual results of offered players settles it:

| Stat | Pos | n | Mean | **var/mean** | |
|---|---|---:|---:|---:|---|
| Assists | A | 116 | 0.96 | **0.99** | underdispersed |
| Assists | M | 102 | 0.64 | **0.83** | underdispersed |
| Assists | all | 218 | 0.81 | **0.96** | underdispersed |
| Goals | all | 220 | 1.60 | 1.06 | ≈Poisson |
| Points | all | 220 | 2.40 | 1.10 | ≈Poisson |

**Offered players' assists are essentially Poisson, even slightly
*under*dispersed** — the opposite of the `var/mean ≈ 1.4` the `phi=4.0` comment
cites. Both can be true: pooled across all players, mixing high- and low-usage
roles inflates variance (a mixture of different means is overdispersed even when
each component is Poisson). The engine's value was measured on the pooled
population; we price only the top of it.

**So raise `PHI_PLAYER["assists"]` from 4.0 to ~20 for the offered population.**
This is the one place a dispersion constant genuinely needs changing, and note the
direction: **narrower, not wider.** My earlier framing of "widen the distributions"
was wrong for assists specifically — the correct fix is to stop pushing mass to
zero, which the double-count and the low `phi` were both doing.

If you'd rather not change a global constant on n=218, the cleaner version is
position- and usage-aware dispersion (`phi` as a function of projected volume),
since the pooled 1.4 is real for the roster as a whole and only wrong at the top.

---

## Bug #2 — Star players get roster-wide zero rates

`ZERO_RATE` at line 174 holds position-level priors:

```python
"A_goals": 0.22, "M_goals": 0.40,
"A_assists": 0.55, "M_assists": 0.70,
```

The comment says these are set *"slightly above empirical so cap pulls them down
for established players."* The per-player override at line 1673 does compute an
EWM zero rate from history, capped at `career_zero * 1.1`.

The problem is who we offer props on. These priors are measured across **all**
attackers and midfielders — including bench players who rarely score. **We only
offer props on the top ~7 players per team**, whose real zero rates are much lower:

| Stat / Pos | Engine prior | **Actual, offered players** | Overstated by |
|---|---:|---:|---:|
| A assists | 0.55 | **0.405** | +14.5 pts |
| M assists | 0.70 | **0.490** | +21.0 pts |
| A goals | 0.22 | 0.195 | +2.5 pts |
| M goals | 0.40 | **0.275** | +12.5 pts |

Every overstated point of zero mass is a point taken off P(Over), and the
direction is one-way: **we always underprice the over on exactly the players we
offer.**

### The fix

1. **Lower the priors toward the offered population** — `A_assists ≈ 0.42`,
   `M_assists ≈ 0.50`, `M_goals ≈ 0.30`. Keep the existing higher values for
   low-usage positions, which are accurate there and never offered anyway.
2. **Cap zero rate by projection size.** A player projected for 1.5 goals cannot
   plausibly have a 40% chance of being blanked. Add a consistency check — for a
   NegBin at that mean, `P(0)` is around 22%, so the prior should never exceed it
   by much. Anything beyond ~1.15× the NB-implied zero rate is a modelling
   contradiction, not a signal.
3. **Prefer the empirical rate sooner.** The `n_games >= 12` gate at line 1683 is
   conservative for a 10–12-game PLL season; a player is on the position prior for
   more than a full year. Lower it to ~6 games with Bayesian shrinkage toward the
   prior rather than a hard switch.

Fixes #1 and #2 together should land assists and goals within a couple of points
of realized rates.

---

> ## ⚠️ Bug #3 as written below was WRONG. See `MODEL_FIX_RESULTS.md`.
>
> This section proposed `P_HOOK = 0.14`, inferred from 6 low-save props out of 43.
> Checking it against the warehouse (336 team-games, 2022-26) before implementing
> showed the real rate is **0.086**, and that the mechanism is a *share* of shots
> faced, not a binary hook. Two further claims here are also wrong:
>
> * **"The mean comes down naturally (~10%)" — no.** `LG_STARTER_SF_PER_OPP_SOG`
>   already embeds the mean backup share (0.915/0.942 = 0.971 against a measured
>   E[share] of 0.972), so `proj_saves` is *already* a mixture mean. Applying a
>   share on top would have cut every goalie projection ~3% for no reason. The
>   shipped fix divides `E[share]` out first.
> * **"56% of the goalie error" — no.** The saves calibration gap is a *mean*
>   error, and the fix did not close it (−20.1 → −21.1 pts). The engine's saves
>   constants verify as correct on 5 seasons; the offered window's save% was
>   0.485 against a league 0.536 (z = −2.22). Kept because the low tail was
>   genuinely unreachable — P(≤2 saves) was 0.05% against 1.19% observed — which
>   is what a saves MS payout is priced off.
>
> Retained below as the original reasoning. **Do not implement it as written.**

## Bug #3 — Goalies are never pulled

`_assign_player_goalie_saves()` at line 4297 gives the starter a fixed share of
shots and zeroes everyone else:

```python
shots_faced = max(opp_sog * LG_STARTER_SF_PER_OPP_SOG, 1.0)   # always 91.5%
starter.proj_saves = shots_faced * sv_pct
...
g.proj_saves = 0.0        # every non-starter, unconditionally
g.usage_multiplier = 0.0
```

**Every goalie is assumed to play the whole game.** There is no path in the code
for an early hook, an injury, or a blowout substitution. This is the single
clearest cause of the goalie losses.

The data shows exactly how much it costs. Sorting all 43 offered goalie props:

```
0, 1, 2, 3, 3, 6, | 7, 7, 8, 8, ... 18, 18, 20
```

| | n | Projected | Actual | Bias | Share of error |
|---|---:|---:|---:|---:|---:|
| Pulled / limited | 6 (14%) | 12.78 | **2.50** | −10.28 | **56%** |
| Normal appearances | 37 (86%) | 13.27 | 11.97 | −1.30 | 44% |

Now the decisive check — **is the dispersion wrong, or is the scenario missing?**

| Population | Variance ÷ mean | Engine assumes (`phi=120`) |
|---|---:|---:|
| Normal appearances only | **0.95** | 1.11 |
| All 43 including pulls | **2.00** | 1.11 |

**In normal games `phi=120` is correct** — 0.95 against an assumed 1.11 is close,
slightly conservative. The variance only explodes when pulled games are mixed in.

**So do not widen the saves dispersion.** That comment block at line 200 is right,
and re-tuning `phi` would corrupt the 86% of games the model already handles well
in order to accommodate the 14% it doesn't model at all. **Add the missing
scenario instead.**

### The fix — a two-component mixture

Draw playing time first, then saves conditional on it:

```python
# Fraction of a game the starter actually plays. Roughly 14% of starts ended
# early this season (6 of 43 offered props at <=6 saves), so a small hook
# probability with a wide conditional share reproduces the observed spread
# without touching the phi that already fits full appearances.
P_HOOK: float = 0.14
HOOK_SHARE_RANGE: Tuple[float, float] = (0.05, 0.45)

hooked = rng.random(n) < P_HOOK
share  = np.where(hooked, rng.uniform(*HOOK_SHARE_RANGE, n), 1.0)
saves  = rng.negative_binomial(*_negbinom_params(mu * share, PHI_PLAYER["saves"]))
```

Two consequences, both wanted:

1. **The mean comes down naturally.** `0.14 × ~0.25 + 0.86 × 1.0 ≈ 0.90`, so
   projections drop ~10% — very close to the 11% overshoot measured in normal
   games. It falls out of the mechanism instead of needing a fudge factor.
2. **The backup's distribution stops being degenerate.** When the starter is
   hooked, the backup faces the rest. That directly fixes the "0.000 projection"
   pathology at its source rather than patching it at pricing time.

`P_HOOK` should be fit from play-by-play (`scripts/build_pbp_tables.py` has the
data) rather than left at 0.14 from a 43-prop sample — 6 events is a thin basis for
a production constant. Ideally condition it on projected game state, since blowouts
drive hooks.

---

## Bug #4 — No odds ceiling, and 0.000 projections reach the board

Two gaps in `PricingEngine`. First, `_am()` at line 3451:

```python
prob = min(max(prob, 0.001), 0.999)
```

A floor of 0.001 permits **+99,900**. There is no cap anywhere in the file.

Second, `price_prop()` computes `ov = float(np.mean(dist > line))` with no check
that the distribution is meaningful. With `N_SIMS = 20_000`, the smallest
non-zero probability resolvable is 1/20,000 — and a projection of exactly 0.000
yields an all-zeros array, so `ov` is 0.0, and after hold is applied the price
comes out around +8,500.

That is exactly what happened: **JC Higginbotham, Cannons, game 38, projected
0.000 saves, offered at +8,554, recorded 11 saves.** Across all goalie rows, 10
were projected at exactly 0.000 and **6 went over.**

### The fix

```python
MAX_PRICE_PROB: float = 0.02    # ≈ +4900 ceiling before hold
MIN_SIM_SUPPORT: int = 50       # need 50+ of 20,000 sims above the line

def price_prop(self, ps, stat, line=None):
    ...
    support = int(np.sum(dist > line))
    if support < MIN_SIM_SUPPORT or float(np.mean(dist)) <= 0.0:
        # Not enough simulated support to price a tail this thin, or no
        # playing-time signal at all. A projection of exactly zero means the
        # model knows nothing about this player — not that the event is
        # impossible. Suppress rather than guess.
        return None
    ov = max(float(support) / len(dist), MAX_PRICE_PROB)
```

Returning `None` requires callers to skip the market; `_price_players` at line
4342 and `boss_export.py` both need a guard. **Suppressing a market we cannot
price is strictly better than offering it at 85-to-1.**

This is a safety net, not the real fix — Bug #3 removes most 0.000 projections at
the source. Keep both: the mixture fixes the modelling, the guard catches whatever
still slips through.

---

## What NOT to change

Worth stating explicitly, because these look like plausible targets and are not:

- **`PHI_PLAYER["saves"] = 120.0` is correct.** Var/mean is 0.95 in normal
  appearances. Widening it would break the 86% case to paper over the 14% case.
- **`PHI_PLAYER["goals"] = 40.0` (near-Poisson) is right.** Offered players' goals
  show var/mean **1.06**, so `phi=40` (implying ~1.04) is nearly exact. The goals
  mispricing is zero-inflation, not dispersion. Note the actual over-rate (75.9%)
  exceeded even a plain Poisson (69.4%) — so goals are, if anything, *less*
  zero-heavy than a standard count model, the opposite of needing fatter tails.
- **The mean projections.** Points is off 0.26 on 2.66 with a calibration gap
  under 2 points. Assists is off **0.04**. These are good.
- **`_negbinom_params`.** The `n = round(phi)` mean-preservation comment shows this
  was already debugged carefully; the math is right.
- **The Cholesky correlation layer** (line 3155). It preserves marginals by
  construction, so it cannot cause a calibration gap.

**Faceoff Wins is the one genuine dispersion miss:** actual var/mean is **1.81**
against `phi=30`'s assumed 1.42. The implied correct value is **phi ≈ 17**. But
n=40 is a thin basis for retuning a constant whose current value cites a full
2024–26 backtest, so re-measure on the full history before changing it. Either way
that market should not return as MS.

---

## Order of work

| Step | Change | Why first |
|---|---|---|
| **1** | Bug #1 — `_solve_excess_zero` | Biggest calibration win, ~15 lines, no retuning |
| **2** | Bug #4 — odds cap + support guard | Pure downside protection, independent of everything |
| **3** | Bug #2 — zero-rate priors, and `phi["assists"]` 4 → ~20 | Closes the residual gap after #1 |
| **4** | Bug #3 — goalie hook mixture | Highest value on saves, but needs a PBP fit |
| **5** | Re-measure faceoff dispersion on full history | Market suspended meanwhile |

Steps 1–3 are all in the "mass piled at zero" family and are best done together,
then validated as one change. Step 1 is the true bug; steps 2–3 are calibration of
constants that were measured on the wrong population.

### How to verify each step

A concrete, cheap regression test that would have caught Bug #1 on day one:

> For every prop with a 0.5 line, the engine's fair P(Over) must be **within a few
> points of `1 − exp(−mu)`** (the Poisson probability of at least one event) and
> must never fall **below** it by more than ~5 points. Assert this across a slate.

Today assists fail it by 15 points. This test is worth adding to
`scripts/fast_backtest.py` regardless of the fixes, since it catches the entire
class of "mean right, shape wrong" bugs that mean-based checks cannot see.

Then re-run the existing backtest harness and confirm two things at once: the
`bias`/`MAE` columns should be **essentially unchanged** (means aren't moving),
while calibration gap by stat should collapse toward zero. If bias moves much,
something has leaked into the mean and the change is wrong.

---

## Expected effect on P&L

Against the −0.92 correlation between calibration gap and realized hold:

| Market | Gap now | Gap after #1+#2 | Hold now | Plausible after |
|---|---:|---:|---:|---:|
| Assists O/U | +16.0 pts | ~2–4 pts | −25.4% | positive |
| Goals O/U | +5.9 pts | ~1–3 pts | +4.0% | ~+7% |
| Points O/U | +1.8 pts | unchanged | +5.4% | unchanged |
| Saves MS | −16.9 pts | ~3–5 pts (after #3) | −9.7% | positive |

Points is already right and should not move — a useful control. If Points
calibration shifts materially after these changes, something regressed.

Faceoff Wins MS stays suspended regardless; that market's problem is structural
(only ~1.8 specialists per game makes MS payouts extreme) and no distribution fix
addresses it.

**The through-line:** every losing market traces to a distribution that puts too
much mass in the wrong place — assists and goals from double-counted zeros, saves
from a scenario that was never simulated. None of it requires rebuilding the
model. The hard part, getting the averages right, is already done.
