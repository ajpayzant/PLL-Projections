# Distribution Fixes — What Changed, and the Evidence It Worked

Companion to `MODEL_FIX_PLAN.md`. The plan proposed five fixes from a 741-prop
sample. Six shipped. This records what was actually shipped after each was checked
against the warehouse, **including the places the plan was wrong**, and the
before/after numbers on the real book.

Branch: `fix/prop-distribution-calibration`. Tests: `scripts/test_prop_distributions.py`
(61 passing). Re-pricing harness: `analysis/reprice_offered.py`. Standing
regression gate: `python scripts/fast_backtest.py --floor-only`.

---

## The one-line summary

The overall calibration gap on the 740 graded offered props went from
**+9.3 points to +2.2 points**, and the share of 0.5-line props whose stated
probability contradicted their own projection went from **99% to 0%** — with the
projected means held fixed, so nothing about the forecasts moved.

---

## What was shipped

| # | Fix | Verdict on the plan |
|---|---|---|
| 1 | `_solve_excess_zero` — stop double-counting zeros | ✅ **As planned.** The one true bug. |
| 2 | `phi["assists"]` 4.0 → 20.0; `_cap_zero_rate`; `ZERO_RATE_POISSON_SLACK` | ⚠️ Direction as planned, **constants refit** — the plan's slack of 0.10 never engaged |
| 3 | Goalie playing-time mixture | ❌ **Plan's mechanism and rate were both wrong.** Rebuilt from 336 team-games. |
| 4 | `MIN_SIM_SUPPORT` / `MIN_PRICE_PROB` + `offerable` flag | ⚠️ As planned, but the plan **missed `boss_export.py` entirely** |
| 5 | `phi["fo_wins"]` 30.0 → 20.0 | ⚠️ Right call, **wrong number** — plan said ≈17 from n=40 |
| **6** | **`_condition_to_team_total` — a SECOND zero-inflation bug** | 🆕 **Not in the plan at all.** Found by the regression check the plan asked for. |

---

## Fix #1 — zero-inflation double-count (the real bug)

`_zinb` passed the target zero rate straight into a zero-inflated draw, so total
`P(0) = z + (1−z)·P_nb(0)` — the negative binomial's own zeros landed *on top of*
the intended inflation. The mean stayed correct, which is exactly why five
seasons of bias/MAE checks never flagged it.

`_solve_excess_zero` inverts the relation by bisection: given a target total
`P(0)`, solve for the inflation component that produces it. Returns 0.0 when the
NegBin already meets the target, in which case the raw NegBin is the right
distribution.

**Result on 218 offered assists props: calibration gap +23.2 → +4.5 points.**

---

## Fix #2 — the assists dispersion, and a refitted slack

`phi=4.0` (var/mean 1.4) was measured on the pooled roster. Pooling players of
different usage inflates var/mean even when every individual is Poisson — a
mixture of different means is overdispersed by construction. Offered players'
assists are essentially Poisson (var/mean 0.96). `phi=20.0`.

**This is the counterintuitive part and worth stating plainly: the fix makes
assists narrower, not wider.** The earlier "widen the distributions"
recommendation was the wrong knob for this stat.

`ZERO_RATE_POISSON_SLACK` was set at 0.10 in the plan. **It never engaged**, and
four of six Poisson-floor test cells failed with it in place. Refitting on 1,149
player-seasons with ≥6 games showed the mean excess over `exp(−mu)` is *negative*
in every bucket:

| population | n | mean excess | median |
|---|---:|---:|---:|
| front-line (mu ≥ 0.5) | 487 | −0.018 | −0.018 |
| sparse (mu < 0.25) | 662 | −0.004 | 0.000 |

So the players we price are Poisson or very slightly *less* zero-heavy than
Poisson. The wide per-season scatter (p90 ≈ +0.10) is sampling noise — at 10
games the standard error on a zero rate is ~0.15 — not real excess zeros.
**0.03**, which still sits above anything the data supports while being tight
enough to bind.

---

## Fix #3 — the plan was wrong twice; here is what is actually true

The plan proposed `P_HOOK = 0.14` with a uniform 5–45% share, inferred from
6 low-save props out of 43, and claimed the mixture would also correct a ~10%
mean overshoot. Checking it first against 336 team-games (2022–26) overturned
both halves.

**Wrong claim 1 — the rate.** 0.14 came from 6 events. Building a *pre-game*
named-starter proxy from strictly prior-game cumulative shots faced (mirroring
the engine's own `(confidence, proj_save_pct)` selection rule) gives:

```
P(starter faced <99% of team shots) = 0.086
    per season: .014  .101  .099  .029  .105
share | partial, Beta method-of-moments = Beta(1.59, 0.77), mean 0.674
```

A note on method, because my first measurement here was wrong in an instructive
way: I initially found only 0.26% by defining the "primary" goalie as whoever
faced the most shots. That is circular — by construction the primary cannot be
the short outing. The pre-game proxy is the only honest way to ask the question.

**Wrong claim 2 — the mean.** `LG_STARTER_SF_PER_OPP_SOG / LG_TEAM_SF_PER_OPP_SOG`
= 0.915/0.942 = **0.971**, which *already is* the average share of team shots the
starter faces. Measured E[share] from the mixture above: 0.972. They agree to
0.001. So `proj_saves` was **already a mixture mean**, and applying a share on
top would have cut every goalie projection ~3% for no reason. The shipped code
divides `E[share]` out before redrawing:

```python
expected_share = (1 - partial_prob) + partial_prob * mean_partial_share
mu_full = mu / max(expected_share, 0.01)
```

**So why keep the fix at all?** Because the aggregate sd was already right
(3.67 actual vs 3.73 implied by `phi=120`) — the defect was *where the mass sat*.
Simulating every start at the average share made a short outing impossible:

| | model before | model after | actual |
|---|---:|---:|---:|
| P(saves ≤ 2) | **0.05%** | 0.78% | **1.19%** |
| mean | 12.57 | 12.57 | — |
| P25–P75 | unchanged | unchanged | — |

A 0.05%-vs-1.19% error is a 24× mispricing on precisely the longshot a saves MS
market pays out on. This is a **reallocation of tail mass, not a widening**, and
`phi=120` is deliberately left alone: full appearances run var/mean 0.88 against
the 1.10 `phi=120` implies, so widening it would corrupt the 91% of starts the
model already fits to reach the 9% it never simulated.

### The saves calibration gap did NOT close — and that is the correct outcome

Saves went **−20.1 → −21.1 points**. Before concluding the fix failed, note that
a *shape* fix cannot close a *mean* error, and the evidence says the residual is
a mean error that isn't ours:

- Starter selection is correct **19 of 20** times, so it isn't a wrong-goalie problem.
- Engine constants verify on five seasons: SF/oppSOG **0.9088** vs coded 0.915;
  save% **0.5364** vs coded 0.537 → 12.67 implied vs 12.57 actual.
- The offered window's save% was **0.4848** against a league 0.5364 — **z = −2.22**
  on 460 shots — while 2026's full season finished at **0.5362**, right on norm.
- Decomposing: volume (−4.2%) plus rate (−9.6%) takes 13.52 → 11.71 against an
  observed 10.64.

So most of the gap is a 10-game fluctuation. **The rate trim the plan recommended
was therefore dropped as unjustified** — trimming a verified constant to chase a
2-sigma stretch would bake the fluctuation in.

---

## Fix #4 — the guards, and the export path the plan missed

`price_prop` now checks simulated support on **both** tails and flags rather than
returns `None`:

```python
MIN_SIM_SUPPORT: int = 50     # need 50 of 20,000 sims on the thin side
MIN_PRICE_PROB: float = 0.02  # ~+4900 ceiling before hold
```

Flagging beats returning `None`: `MarketLine` flows through `asdict()` into the
UI and the export, and `price_milestones` mutates `ml.stat`, so a `None` in that
path would have meant null-guards in four callers. `offerable` / `suppress_reason`
carry the decision instead, and `pages/3_Player_Props.py` surfaces it in an
`Offer?` column that stays blank unless something is wrong.

**The plan missed the path that actually mattered.** `boss_export.py` bypasses
`PricingEngine` entirely and prices straight off the distribution, publishing
ladders to 22+ saves and 26+ FO wins. None of the engine's guards protected it.
It now carries the same two constants (duplicated to keep the module
numpy+stdlib-only per its own docstring, with a test pinning them against the
engine so they cannot drift), plus `SCHEMA_VERSION` 1 → 2 and an explicit note
that a reader ignoring `offerable` will release markets flagged as unpriceable.

`ge_probability_ladder` is left deliberately **unclamped** — it is the model's raw
view and the object the BOSS Tool re-derives from, so it must stay internally
consistent and monotonic.

---

## Fix #5 — faceoff dispersion, and a control that proves the method

The plan suggested `phi ≈ 17` from n=40 and recommended re-measuring first. Doing
that on 311 team-games of the pre-game FO specialist (2022–26) produces two
different answers depending on the method — and choosing between them is the
whole question:

| estimator | var/mean | implied phi |
|---|---:|---:|
| pooled across players | 2.039 | 12.9 |
| **within player-season** | **1.689** | **20.3** |

Pooling replicates the exact mixture artefact that made `phi=4` wrong for
assists. A per-player projection already captures between-player skill
differences, so the dispersion constant must describe variation *within* a
player. **`phi = 20.0`.**

To check that this estimator isn't simply biased toward wider tails, I ran it on
saves as a control. It returns var/mean **0.869** for full appearances against
`phi=120`'s 1.109 — i.e. it says saves must *not* widen. It answers in both
directions, which is what makes the FO answer credible.

| | model before | model after | actual |
|---|---:|---:|---:|
| P(FO wins ≤ 6) | 4.20% | **5.43%** | **5.47%** |
| sd | 4.41 | 4.74 | 5.23 |

Faceoff volume is genuinely lumpy: a specialist's win rate has sd 0.169 across
games, driven by matchup and by how many faceoffs the game produces at all.

**Note the FO O/U gap barely moves (0.1009 → 0.1005), and that is expected.** The
O/U line sits at the median, where a symmetric tail change does almost nothing.
The −81% loss was on **MS**, which is priced entirely off the tails this change
fixes. This is the clearest case in the whole exercise that O/U calibration and
MS calibration are different questions.

---

## Fix #6 — a second zero-inflation bug, found by the regression check

Worth recording how this surfaced, because it is the argument for the check
itself. The plan's closing recommendation was to add a standing Poisson-floor
assertion. Adding it to `scripts/fast_backtest.py` — **after** fixes #1–#5 were
already validated — immediately failed on goals, in the full engine path, at
**20.9% of props**. The offered-book reprice showed no such breach, so one of the
two was not looking at the distribution the engine actually ships.

The reprice harness draws goals straight from `_zinb`. The real simulator adds a
step it skips: after the per-player draw, every field player's goals are rescaled
so the team's total matches the team goal draw.

```python
scale  = round(team_goal_draws) / sum(raw_goals)
scaled = np.round(raw_goals[pid] * scale)          # <-- the bug
```

**Zero is an absorbing state under multiplication.** `0 * scale` is zero for
every scale, but a player who drew one goal loses it whenever `scale < 0.5`. Mass
flows into zero and can never flow back out. Measured on a 200k-draw six-player
field:

| player mu | P(0) before conditioning | P(0) after | floor breach it causes |
|---:|---:|---:|---:|
| 1.90 | 0.1793 | 0.2226 | +0.073 |
| 1.40 | 0.2540 | 0.3142 | +0.068 |
| 1.10 | 0.3401 | 0.4066 | +0.074 |
| 0.85 | 0.4330 | 0.5012 | +0.074 |
| 0.60 | 0.5513 | 0.6154 | +0.067 |
| 0.35 | 0.7056 | 0.7556 | +0.051 |

This is the same defect as the zero-inflation double-count arriving by a
different route, and equally invisible to every mean-based metric — **the team
total is conserved, so bias and MAE both read clean.** It had been there the whole
time, underneath fix #1.

A second flaw showed up in the same measurement: rounding each player
independently landed on the exact team total in only **49.5%** of sims, so the
constraint the step exists to enforce was half-enforced.

`_condition_to_team_total` reconciles additively instead — compute the shortfall,
then transfer that many whole goals, removing from players who have them and
adding in proportion to projected mean. Adding is the branch that lets a zero
become a one, which multiplication could never do. When the sums already agree it
is the identity, so a player's marginal is left alone rather than perturbed by a
scale near 1.0.

| | before | after |
|---|---:|---:|
| Worst floor breach | +0.0738 | **+0.0182** |
| Sims hitting the team total exactly | 49.5% | **100%** |
| Goals props breaching by >5 pts (full engine) | 20.9% | **0%** |

Cost: 11.6 ms per team at `N_SIMS = 20,000`.

### Two measurement artifacts ruled out along the way

The floor check also flagged **points**, and that one was my test's error, not the
model's. Points is `1×(1pt) + 2×(2pt) + assists`, so it is not a unit-step count
and `1 − exp(−E[points])` is not a valid bound on it — for two independent
reasons, both measured rather than assumed:

1. A two-point goal raises `E[points]` without adding a scoring event. On
   independent Poisson goals and assists where true P(Over 0.5) is 0.7756, the
   naive floor claims 0.8128 — a spurious +3.7-point breach. Using the event rate
   (goals + assists) gives 0.7769, right to 0.001.
2. Even on the event rate, positive correlation between a player's goals and
   assists genuinely lowers P(at least one) — a quiet game is quiet in both. At
   corr(g,a) = +0.086 the legitimate breach is +0.029; at +0.244 it is +0.080.
   The engine models this correlation on purpose.

So the check now takes its verdict on goals and assists only, and prints points
as information. **Points was never broken.**

---

## Validation on the real book — 740 graded offered props

`analysis/reprice_offered.py` re-draws every graded offered prop through the new
code at 60k draws, **holding the engine's original projected mean fixed**, and
reads fair P(Over) at the line that was actually offered. Same input, different
shape, so the change is isolated.

### Means must not move

Largest mean drift across all five stats: **0.0030**. Nothing leaked into the
projections; every number below is a pure shape effect.

### Calibration gap (realized over-rate − stated fair P(Over))

| Stat | before | after | |
|---|---:|---:|---|
| **Assists** | +0.2318 | **+0.0449** | ✅ |
| — attackers | +0.1920 | **+0.0119** | ✅ |
| — midfielders | +0.2771 | **+0.0824** | ✅ |
| **Goals** | +0.0585 | **+0.0067** | ✅ |
| **Points** | +0.0448 | +0.0445 | ✅ control, correctly flat |
| **FO Wins** | +0.1009 | +0.1005 | ➖ O/U at the median; MS is what moved |
| **Saves** | −0.2008 | −0.2107 | ➖ mean error, not shape (see Fix #3) |
| **Overall** | **+0.0927** | **+0.0216** | ✅ |

Points is the control: it was already well calibrated and it must not move. It
moved 0.0003. That is the single most reassuring number here — it says the fixes
are local to the defects and did not perturb what already worked.

### Poisson floor (0.5 lines)

Share of props whose stated P(Over) sat more than 5 points below `1 − exp(−mu)` —
i.e. contradicted its own projection:

| Stat | before | after |
|---|---:|---:|
| Assists | **98.97%** | **0%** |
| Goals | **100%** | **0%** |

This check is now wired into `scripts/fast_backtest.py` as a standing section,
not just the pytest suite, because it is the only assertion in that harness that
can see a mean-right/shape-wrong bug.

---

## The full saves/FO backtest, and why two of its verdicts are not readable

`scripts/run_backtest_saves_faceoffs.py` was rewired to draw both the old and the
new shape in one pass, on identical projections and the same seed, so a single
40-minute run answers "did this regress?" without comparing across two RNG
streams. On 222 goalie starts and 211 FO specialist games:

| | before | after | actual |
|---|---:|---:|---:|
| Goalie saves — sd | 3.780 | 4.093 | 3.576 |
| Goalie saves — mean drift | | **0.0044** | |
| FO wins — sd | 4.289 | **4.603** | 4.504 |
| FO wins — mean drift | | **0.0045** | |

Mean drift under 0.005 on both confirms the projections did not move. FO sd moves
almost exactly onto the observed value.

The run also printed WORSE on coverage and on both tails. **Both are flaws in the
measuring instrument, and I checked rather than accepting them:**

- **Coverage is not a pass/fail signal on a discrete distribution.** Feeding a
  known-correct NegBin its own draws returns coverage 0.559/0.845 against a
  nominal 0.50/0.80 at saves. A percentile of an integer distribution lands *on*
  an integer, so the closed interval `[p25, p75]` holds more than half the mass.
  Over-covering is the expected reading for a correct distribution.
- **The goalie low tail is unmeasurable in this harness.** It picks the starter as
  whoever *actually* faced the most shots — the same circular selection I had
  already caught and corrected once. By construction the player it selects cannot
  be the one who left early, so its sample has a minimum of 4 saves and **zero**
  games at ≤2 in 222 rows, against a real 1.19%. Its "actual" low tail is an
  artifact of the selection rule. Both facts are now documented in the script so
  the next reader does not take a verdict off them.

### The FO tail disagreement resolved: it is projection spread, not dispersion

The harness shows model P(≤6) = 9.5% against an observed 3.8%, which looks like
the widening overshot. It did not. The model is internally consistent there —
0.0947 predicted versus 0.0948 if its own draws were the truth — so the gap is in
the projected means, not the shape. Broken out by projected bucket:

| projected FO wins | n | projected | actual | bias |
|---|---:|---:|---:|---:|
| ≤10 | 27 | 8.96 | 9.67 | **−0.71** |
| 10–12 | 59 | 11.05 | 12.39 | **−1.35** |
| 12–14 | 49 | 12.94 | 12.98 | −0.04 |
| 14–16 | 46 | 15.07 | 14.35 | **+0.72** |
| 16+ | 30 | 16.90 | 15.73 | **+1.16** |

**Projections are spread too wide across players** — low ones too low, high ones
too high — while the overall mean is fine (bias −0.15). Shrinking projected spread
20–35% toward the mean improves both that tail and MAE. That is a `PlayerModel`
finding and is tracked separately; no dispersion constant can fix a mean that is
spread too wide.

Judged the only fair way — at the matched population mean (mu = 13.693 over 283
full-workload team-games), with no projection error mixed in:

| | sd | P(≤8) | P(≤6) |
|---|---:|---:|---:|
| phi = 30 (old) | 4.461 | 0.1138 | 0.0377 |
| **phi = 20 (shipped)** | **4.788** | **0.1331** | 0.0489 |
| actual | 5.146 | 0.1590 | 0.0424 |

`phi=20` is closer on sd and on P(≤8), and both remain short of actual — the real
distribution is skewed (+0.74 against a NegBin's +0.50) in a way no `phi`
reproduces. At the very deep P(≤6) tail `phi=20` overshoots and `phi=30` happens
to sit closer, but that is one crossing point on a shape that is too narrow
everywhere else, and erring wide is the safe side for an MS longshot.

---

## What was deliberately NOT changed

- **`phi["saves"] = 120.0`** — full appearances are var/mean 0.88. Widening
  breaks 91% of starts to reach 9%.
- **`phi["goals"] = 40.0`** — offered goals run var/mean 1.06 against the 1.04
  it implies. Nearly exact.
- **The goalie save rate** — verified on five seasons; the offered window was a
  2.2-sigma stretch, not a bias.
- **The mean projections generally** — assists is off 0.04 on 0.85. The hard part
  was already done.
- **`_negbinom_params`, and the Cholesky correlation layer** — the mean-preservation
  algebra is correct, and Cholesky preserves marginals by construction, so it
  cannot produce a calibration gap.
