# PLL Player Props — Season Review (2026, through Week 12)

Analysis of `PLL Projections 2026.xlsx` against `PLL Player Props Results.xlsx`.

**Scope, stated up front, because it changes the conclusions:**

| | |
|---|---|
| Weeks props were offered | **4, 5, 6, 8, 9, 10, 11, 12** (weeks 1–3 not ready, week 7 All-Star, week 13+ unplayed) |
| Weeks covered by the projections workbook | **8–12** (games 24–45) = **74.7%** of season handle |
| Projections in the workbook | 1,825 graded — **every rostered player** |
| Props plausibly *offered* | **741** (40.6%) — ~6.9 players per team per game |
| Stats with a real market | Points, Goals, Assists, Saves, Faceoff Wins. **SOG is projected but never offered.** |

The offered set is a **proxy**, since the workbook does not flag which props went
live: top 5 A/M by projected Points per team, plus each team's highest-projected
goalie and faceoff specialist. `analyze_offered.py` re-runs the headline numbers
at depths 3–7; the conclusions below hold across that whole range (bias −0.16 to
−0.27, calibration gap +0.056 to +0.090). Nothing here rests on where the cut
falls. An authoritative offered list should replace the proxy.

---

## 1. The single most important correction

Judged on **all** players, the model looks like it *under*-projects (bias +0.086).
Judged on the players **you actually offered**, it *over*-projects (bias −0.232).

| | All 1,825 projections | The 741 offered |
|---|---:|---:|
| Mean projection | 1.82 | 3.02 |
| Bias (actual − projected) | **+0.086** (too low) | **−0.232** (too high) |
| MAE | 1.048 | 1.358 |
| Over-rate | 48.3% | 52.6% |
| Fair P(Over) | 36.2% | 46.1% |
| **Calibration gap** | **+12.1 pts** | **+6.6 pts** |
| P10–P90 coverage | 90.5% | 95.8% |

These are two different models' worth of behaviour, and the offered column is the
only one that touched money. My earlier read that "defensive-position props are
24.3% of the book" was wrong as a book-exposure statement — SSDM/D/LSM props were
never offered. That remains a real model-quality observation (SSDM projects 0.21
and delivers 0.42) but it is **not** where the season's loss came from.

**What survives, and matters more:** the calibration gap is still **positive** on
the offered book (+6.6 pts). Overs cash more often than you price them, even
though the projections themselves are slightly *too high*. That combination is
the central finding of this review, and §3 explains it.

---

## 2. Where the money went (active weeks only)

| Metric | Value |
|---|---|
| Handle | **$282,014** |
| GGR | **−$5,735** |
| Realized hold | **−2.03%** |
| Target hold | +7.50% |
| Gap | **−9.53 pts ≈ −$26,900** |
| Losing weeks | **3 of 8** |
| Weekly hold σ | **21.1 pts** (−31.7% to +24.3%) |

| Market | Handle | Share | GGR | Hold |
|---|---:|---:|---:|---:|
| Points O/U | $95,300 | 33.8% | +$5,133 | +5.4% |
| Goals O/U | $66,600 | 23.6% | +$2,659 | +4.0% |
| Points MS | $28,900 | 10.2% | +$1,743 | +6.0% |
| Saves O/U | $8,239 | 2.9% | +$1,306 | **+15.9%** |
| Assists MS | $13,400 | 4.8% | −$430 | −3.2% |
| 2 Point Goal MS | $3,487 | 1.2% | −$551 | −15.8% |
| Goals MS | $26,339 | 9.3% | −$568 | −2.2% |
| Saves MS | $13,696 | 4.9% | −$1,332 | −9.7% |
| **Assists O/U** | $13,353 | 4.7% | **−$3,395** | **−25.4%** |
| **Faceoff Wins MS** | $12,700 | 4.5% | **−$10,300** | **−81.1%** |

**Faceoff Wins MS is 4.5% of handle and 180% of the season's loss.** Without it
the book holds **+1.70%**. Without any MS market it holds **+3.11%**.

| Structure | Handle | Share | GGR | Hold |
|---|---:|---:|---:|---:|
| O/U | $183,492 | 65.1% | +$5,703 | **+3.1%** |
| MS | $98,522 | 34.9% | −$11,438 | **−11.6%** |

Every dollar of loss is in market-selection. In an MS market a correct longshot
pays multiples of stake, so a tail-probability error is punished far harder than
the same error in a two-way O/U.

### Calibration predicts hold almost perfectly

Across the four O/U markets, calibration gap vs realized hold correlates
**−0.92**:

| Market | Calibration gap | Realized hold |
|---|---:|---:|
| Points O/U | +1.8 pts | +12.2% |
| Goals O/U | +5.9 pts | +6.1% |
| Assists O/U | **+16.0 pts** | **−1.7%** |
| Saves O/U | −16.9 pts | +15.9% |

*(Weeks 8–12 only, so these hold figures differ from the season table above.)*
This is the strongest result in the analysis: hold is not luck. It is a direct,
near-linear readout of how far the priced probability sits from reality. Fix
calibration and hold follows.

---

## 3. What the projections say about the model, in plain terms

> **Status: the model defects diagnosed in this section have been fixed.** See
> `MODEL_FIX_RESULTS.md` for what shipped and the before/after numbers. Two
> conclusions in this section did not survive implementation:
>
> * **"The spread is too narrow" is the wrong description for assists and goals.**
>   The real defect was a zero-inflation double-count that piled probability on
>   exactly zero. Removing it raised P(Over) without widening anything — and the
>   correct assists dispersion is *narrower* than what was shipped, not wider.
>   The diagnostic in this section (comparing to a plain Poisson on the model's
>   own mean) was the right test and pointed at the right place; only the
>   prescription was wrong.
> * **Saves is not "over-projected and far too confident."** The save-rate and
>   shot-volume constants verify against five seasons. The offered window was a
>   2.2-sigma bad stretch for goalies (save% 0.485 vs a league 0.536), and 2026
>   as a whole finished on norm. The real saves defect was narrower: the low tail
>   was unreachable, P(≤2 saves) = 0.05% against an actual 1.19%.

### The one-sentence verdict

**Your projections get the average right and the spread wrong.** The model knows
roughly how many points a player will score. It does not know how much that
number bounces around, and it consistently thinks players are more predictable
than they are. Since odds are made from the spread — not the average — the
mispricing is in the odds even when the projection looks good.

### Why "bias is negative but overs still cash" is not a contradiction

Every prop is priced against a `.5` line, so what matters is not the mean but
`P(player clears the line)`. Take the props with a **0.5 line**, where fair
P(Over) is exactly `P(at least 1)`. Compare the model's stated probability to a
plain textbook Poisson on the model's *own* projected mean:

| Stat | n | Projected mean | Model's fair P(Over) | Plain Poisson on same mean | **Actual over-rate** |
|---|---:|---:|---:|---:|---:|
| Assists | 195 | 0.73 | **35.0%** | 49.7% | **52.3%** |
| Goals | 54 | 1.20 | **52.2%** | 69.4% | **75.9%** |
| Points | 9 | 1.52 | 55.9% | 77.8% | 77.8% |

The model prices these overs *lower than the simplest possible count
distribution* fitted to its own mean — and reality lands close to the Poisson,
not to the model. That is conclusive: **the problem is distribution shape, not
the projection.** The simulation is generating too many players who land exactly
on their average and too few who have a 2-point night or a blank. Too much
probability mass piled on the middle means not enough left for the over.

This is also why P10–P90 coverage of 95.8% looks *reassuring but isn't*. The
intervals are wide enough to contain outcomes; the mass **inside** them is
distributed wrong. Coverage is the wrong test and it gave a false pass.

### Stat-by-stat scorecard (offered players only)

| Stat | n | Proj | Actual | Bias | MAE | Calib gap | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **Points** | 220 | 2.66 | 2.40 | −0.26 | 1.30 | **+1.8 pts** | ✅ **Solid** — your best market, correctly |
| **Goals** | 220 | 1.71 | 1.60 | −0.11 | 1.05 | +5.9 pts | 🟡 **Good, slightly short on the over** |
| **Assists** | 218 | 0.85 | 0.81 | −0.04 | 0.69 | **+16.0 pts** | 🔴 **Reevaluate — best mean, worst pricing** |
| **Saves** | 43 | 13.21 | 10.65 | **−2.55** | 4.48 | −16.9 pts | 🔴 **Reevaluate — over-projected and far too confident** |
| **FO Wins** | 40 | 13.12 | 13.85 | +0.73 | 3.69 | +9.6 pts | 🟡 **Mean fine, spread badly too narrow** |
| *SOG* | *430* | *1.86* | *2.37* | *+0.50* | — | — | *No market — model quality only* |

**Points is genuinely solid.** Bias −0.26 on a mean of 2.66 (10% high), and a
calibration gap of under 2 points. Points O/U is a third of your handle and holds
+5.4%. This is the part of the model to protect, not touch.

**Assists is the clearest fix available.** The mean is nearly perfect (−0.04, the
best of any stat) and it still lost 25.4% — the entire failure is the +16-point
calibration gap. Assists live at ~0.85 per game, right where the "everyone lands
on their average" flaw bites hardest: an assist is a lumpy, discrete event, and a
distribution too tight around 0.85 badly understates the chance of getting one.
Look at where the gap sits by projection size:

| Projected assists | n | Fair P(Over) | Actual over-rate | Gap |
|---|---:|---:|---:|---:|
| 0.00–0.50 | 56 | 20.3% | 41.1% | **+20.8 pts** |
| 0.50–0.75 | 61 | 31.9% | 50.8% | **+18.9 pts** |
| 0.75–1.00 | 33 | 43.0% | 54.5% | +11.6 pts |
| 1.00–1.50 | 49 | 50.4% | 63.3% | +12.9 pts |
| 1.50+ | 19 | 49.0% | 57.9% | +8.9 pts |

The gap is worst at low projections and shrinks as the projection grows — exactly
the signature of a too-narrow count distribution, and exactly what a
negative-binomial or zero-inflated fit is designed to correct.

**Saves is the opposite failure and the more dangerous one:**

| Metric | Value |
|---|---|
| n (starting goalies) | 43 |
| Bias | **−2.55** (projects 13.2, actual 10.7 — 24% too high) |
| **Projected σ** | **2.29** |
| **Actual σ** | **4.61** |
| Projected range | 0.0 – 16.1 |
| Actual range | **0 – 20** |
| Over-rate | **32.6%** against a ~49% price |
| P10–P90 coverage | **76.7%** |

**These aggregates conflate two different failures, and separating them changes
the fix.** Sorting the 43 actual results shows a clean break: six games at ≤6
saves (goalies pulled or barely used), 37 normal games.

| | n | Projected | Actual | Bias | Share of total error |
|---|---:|---:|---:|---:|---:|
| Pulled / limited | 6 (14%) | 12.78 | **2.50** | −10.28 | **56%** |
| Normal appearances | 37 (86%) | 13.27 | 11.97 | **−1.30** | 44% |

So the save *rate* is broadly acceptable — 11% high in normal games — while the
real defect is that **relief and early hooks are not modelled at all.** Every
goalie is projected to play a full game, making a 2-save night effectively
impossible in the simulation; it occurred 6 times in 43. Restricted to normal
appearances, projected σ is 2.42 against an actual 3.37 — still over-confident,
but far from the 2.29-vs-4.61 the pooled figure suggests. Fixing playing-time
modelling is both cheaper and higher-value than re-fitting the save rate.

Saves O/U still held +15.9% — the bias happened to favour the house on a two-way
market — but Saves MS lost 9.7%, because there the same error pays out at long
odds. **Do not read +15.9% as validation.** That is an uncontrolled error that
landed favourably.

One starting-goalie prop was projected at **exactly 0.000 saves** and priced Over
0.5 at **+8,554** (JC Higginbotham, Cannons, game 38). He recorded **11 saves.**
Across all goalie rows, 10 were projected at exactly 0.000 and **6 went over**
(11, 11, 7, 6, 4, 2 saves). Most were backups and probably never reached the
board — so treat this as a **latent pricing-logic hazard** rather than a
quantified loss. But at least one made the starter cut, so the hazard is live. A
projection of exactly zero means *"no playing-time signal,"* not *"impossible."*

**Faceoff Wins: the projection is fine, the market structure is not.** Bias +0.73
on a mean of 13.1 is respectable. But projected σ is **1.35** against an actual σ
of **5.00** — the model puts nearly every FO specialist in a narrow band around
13 while real outcomes swing far wider. With only ~1.8 FO players per game, MS has
very few real outcomes, so one correct longshot is ruinous. Hence −81% hold on an
essentially reasonable projection.

**SOG is projected 27% low (1.86 → 2.37) but has no market**, so it cost nothing.
For attackers and midfielders specifically the ratio is **1.23×** (2.38 → 2.93);
the 1.27× all-position figure is inflated by SSDM's 2.62× and must not be applied
to attackers.
Worth fixing before you ever offer it, and it is a useful independent
confirmation that shot generation is under-modelled — which is plausibly the same
root cause as the goalie saves bias.

---

## 4. What is working

- **Points and Goals O/U** — 57.4% of handle at **+4.8%** combined (+$7,792).
- **Points projections** — calibration gap under 2 points. The core engine is
  sound for its primary use case.
- **Attackers and Midfielders** — bias −0.13 and −0.15, coverage 98%.
- **Stability** — bias by week ran −0.09 to −0.38 with no drift, and the
  calibration gap was positive in all five weeks. This is a **systematic,
  fixable** error, not a cold streak. It was also visible from the first week.
- **Pricing mechanics** — the ~6.1% average overround is built correctly. The
  machinery works; the probability inputs are wrong.

---

## 5. Recommendations

### Immediate (before the next slate)

1. **Suspend Faceoff Wins MS.** 4.5% of handle, 180% of the loss, −81% over four
   weeks. The projection is reasonable; the MS structure on a ~1.8-player market
   is not. Reintroduce as O/U only, with limits, after fix #4.
2. **Hard-block any prop priced off a 0.000 projection.** A one-line validation
   rule. At least one reached the board at +8,554 and lost.
3. **Cap maximum offered odds** (suggest +2000 MS / +1500 O/U). Every prop priced
   above +8,000 this season was wrong by orders of magnitude.
4. **Cut the MS share.** O/U +3.1%, MS −11.6%. Until #5 lands, MS should be a
   small, limited slice of the book.

### Model fixes, in priority order

> **Items 5–7 are shipped** on `fix/prop-distribution-calibration`; item 8's
> Poisson-floor gate is in `scripts/fast_backtest.py` and
> `scripts/test_prop_distributions.py`. Where the shipped fix differs from what
> is written below, the reason is in `MODEL_FIX_RESULTS.md`:
> assists needed *narrower* dispersion, not wider (5); the save-rate trim was
> dropped as unjustified (6); the FO σ target of 5.0 overshot the correctly
> measured value of ~4.7 (7).

5. **Widen the count distribution — the highest-value fix.** For Assists, Goals,
   and Points, replace the current too-tight distribution with a properly
   dispersed discrete one (negative binomial, or zero-inflated where a scoreless
   game is common), fit per position. Success test: on 0.5-line props, stated fair
   P(Over) should land at or above a Poisson on the same mean, not 15 points
   below. This alone addresses the +16-pt Assists gap and the +5.9-pt Goals gap.
6. **Model goalie playing time, then trim the rate.** Explicit early-hook/relief
   modelling is the priority: 14% of goalie props produced 56% of the error, and
   the simulation currently treats a 2-save night as near-impossible. Separately
   trim the save rate ~11% in normal appearances and widen σ from 2.4 toward the
   observed 3.4, ideally by driving saves off simulated opponent shot volume
   rather than a per-goalie average.
7. **Widen the Faceoff Wins distribution** (σ 1.35 → ~5.0) before reintroducing
   the market in any form.
8. **Add a weekly calibration gate.** Track calibration gap by stat, with an
   alert threshold. The gap was positive every week from week 8 and never
   corrected. Given the −0.92 correlation with hold, this metric is an early
   warning system for P&L. This converts a season-long leak into a one-week leak.
9. **Fix SOG (+25%) before offering it.** No exposure today, and it likely shares
   a root cause with the goalie bias.

### Pricing and offering strategy

10. **Tier the hold by demonstrated calibration** instead of a flat 7.5%: keep
    ~6% on Points O/U (gap +1.8 pts, proven), ~8% on Goals O/U, and **12–15% on
    Assists, Saves MS, and any market whose calibration gap exceeds 10 points**
    until the model earns its way back down.
11. **Set market-aware limits.** Assists and Saves MS need materially lower max
    stakes than Points and Goals O/U.
12. **Expand what is working.** Points and Goals O/U for A/M are 57% of handle
    and profitable. Growing that base is worth more than adding exotic markets.

### Reporting hygiene

13. The Overview tab marks 9 games "Pending" actuals sync, but **all 22 games
    have actuals present.** The flag is stale and would cause any dashboard
    reading it to under-report accuracy history.
14. **Archive per-game projections for weeks 4–6.** The workbook only covers
    weeks 8–12, so 25% of season handle cannot be tied back to the projections
    that priced it. Retaining every priced snapshot — and a flag for which props
    actually went live — would remove the need for the proxy used here.

---

## 6. Expected impact

| Scenario | Handle | GGR | Hold |
|---|---:|---:|---:|
| As offered | $282,014 | −$5,735 | −2.03% |
| Suspend Faceoff Wins MS | $269,314 | +$4,565 | **+1.70%** |
| O/U only | $183,492 | +$5,703 | **+3.11%** |

Suspending one market that is 4.5% of handle turns the season profitable. Closing
the Assists calibration gap and repairing goalie saves addresses the rest of the
loss, and tiered hold on the corrected book makes 7.5% reachable — because it
would no longer be funded out of tail losses.

**The core engine does not need rebuilding.** Points is well projected, mid-range
projections are near-unbiased, and pricing mechanics are correct. The work is
concentrated in one place: **the model's distributions are too narrow.** Every
losing market traces to that — Assists overs cashing far above price, goalie
outcomes escaping a 2-σ band, faceoff longshots landing at 90-to-1. Getting the
average right was the hard part and it is done. Getting the spread right is what
turns accurate projections into a profitable book.

---

### Reproducing this analysis

```bash
python analysis/extract_pll_data.py    # parse both workbooks → analysis/data/*.parquet
python analysis/analyze_accuracy.py    # all projections: accuracy, line quality, calibration
python analysis/analyze_offered.py     # offered subset only, with proxy sensitivity
```

`extract_pll_data.py` locates each block by marker text rather than fixed row
offsets and reports any non-matching sheet rather than skipping it silently; all
22 sheets parsed cleanly. `analyze_offered.py` isolates the offered-set proxy in
`offered_mask()` so an authoritative list can be dropped in, and prints the
depth-3-to-7 sensitivity table so no conclusion rests on the cutoff.
