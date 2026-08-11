# PLL Player Props — Plain English Summary

A companion to `PLL_PROPS_SEASON_REVIEW.md`, written without jargon. Every term is
defined where it first appears, and every claim is tied to a specific number from
the two workbooks.

---

## The vocabulary, once, so the rest reads cleanly

| Term | What it actually means |
|---|---|
| **Handle** | Total money customers bet. $282,014 this season. |
| **GGR** | What we kept. Negative means customers beat us. |
| **Hold %** | GGR ÷ Handle. Our profit margin. Target was 7.5%; we got −2.03%. |
| **The line** | The number the customer bets over or under. Always ends in .5, e.g. "0.5 assists," so there are no ties. |
| **Projection** | Our forecast of a player's *average* result. |
| **Fair P(Over)** | Our stated probability that the player goes over the line. **This is what the odds are built from.** |
| **Over-rate** | How often the over *actually* won. |
| **Calibration gap** | Over-rate minus Fair P(Over). If we say 35% and it happens 52% of the time, the gap is +17 points. **This is the number that decides whether we win or lose.** |
| **O/U (Over/Under)** | Two-way market. Customer picks over or under. Roughly even odds. |
| **MS (Market Selection)** | Longshot-style market where a correct pick pays several times the stake. Small errors get expensive fast. |
| **Bias** | Actual minus projected. Negative = we projected too high. |
| **Sigma (σ)** | Spread. How much results bounce around the average. Small σ = "confident, results cluster tightly." Large σ = "results are all over the place." |

---

## 1. The one thing to understand

**Our projections are good at guessing the average. They are bad at knowing how
much results bounce around. We build our odds from the bounce, not the average.
That's why accurate-looking projections lost money.**

Here is that sentence as an actual example from the data.

**Assists, line 0.5** — 195 props on offered players. "Will this player get at
least one assist?"

| | |
|---|---|
| Our average projection | 0.73 assists |
| We said the chance of 1+ assist was | **35.0%** |
| So we priced the Over at about | **+186** |
| It actually happened | **52.3% of the time** |
| A fair price would have been | **−110** |

We sold a coin flip at 2-to-1 odds. Working the money through: for every $100 a
customer bet on the over at +186, they won 52% of the time, which means **we lost
about $50 for every $100 bet.** Not a bad beat — that is the mathematically
expected result of that pricing.

And here's the part that matters: **our average projection of 0.73 assists was
almost perfectly correct.** Actual average was 0.81. We were off by 0.04 assists.
The forecast was right and we still lost a quarter of the money in that market.

---

## 2. Proof this is the distribution, not the projection

If our projection is right but our probability is wrong, the error has to be in
the *shape* — how we spread outcomes around the average. Here's a clean test.

There's a standard textbook formula for "how often does an event happen at least
once, given its average rate" (the Poisson distribution). It's the simplest
possible answer, taught in any stats class. I fed it **our own projected averages**
and compared:

| Stat | Props | Our projected average | **We said** | **Textbook formula says** | **What actually happened** |
|---|---:|---:|---:|---:|---:|
| Assists | 195 | 0.73 | **35.0%** | 49.7% | **52.3%** |
| Goals | 54 | 1.20 | **52.2%** | 69.4% | **75.9%** |
| Points | 9 | 1.52 | 55.9% | 77.8% | 77.8% |

Read the Assists row: using our own average, a first-year stats formula would have
said 49.7%. Reality was 52.3% — the textbook was nearly right. **We said 35%.**

Our simulation is *less* accurate than the simplest formula applied to our own
numbers. That rules out the projections as the culprit. Something in how we
generate outcomes around the average is squeezing them too tightly toward the
middle — too many simulated players landing on exactly their average, not enough
having a 2-assist night or a blank. When too much probability is packed in the
middle, there isn't enough left for the over, so we underprice it.

**Why nobody caught this:** we track whether actual results land inside our
P10–P90 range, and they do 95.8% of the time, which looks excellent. But that test
only asks whether the range is *wide enough* — it never checks whether the
probability is spread correctly *inside* the range. It's like verifying a
dartboard is big enough to contain every throw without checking where the darts
landed. The test we were relying on cannot detect this failure.

---

## 3. Stat-by-stat: what's solid, what needs work

Only players we actually offered — roughly 7 per team per game (top offensive
players, starting goalie, faceoff specialist).

| Stat | Props | We project | Actually | Off by | **Calibration gap** | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **Points** | 220 | 2.66 | 2.40 | −0.26 | **+1.8 pts** | ✅ **Solid — leave it alone** |
| **Goals** | 220 | 1.71 | 1.60 | −0.11 | +5.9 pts | 🟡 **Good, slightly underpriced overs** |
| **Assists** | 218 | 0.85 | 0.81 | −0.04 | **+16.0 pts** | 🔴 **Best forecast, worst pricing** |
| **Saves** | 43 | 13.21 | 10.65 | **−2.55** | −16.9 pts | 🔴 **Needs real work** |
| **Faceoff Wins** | 40 | 13.12 | 13.85 | +0.73 | +9.6 pts | 🟡 **Forecast fine, market broken** |
| *Shots on Goal* | *430* | *1.86* | *2.37* | *+0.50* | *no market* | *Never offered — costs nothing today* |

**Points is genuinely good.** Calibration gap under 2 points means our stated
probabilities were nearly exactly right. It's a third of all money bet and it held
+5.4%. Don't touch it.

**Assists is the best fix available** — the forecast is already the most accurate
of any stat (off by 0.04) and it lost the most as a percentage (−25.4%). Nothing
needs re-forecasting; we only need to fix the shape. Note how the gap shrinks
as the projection grows, which is the fingerprint of a too-tight distribution:

> **Correction, after building the fix.** "Widen the spread" was the wrong
> instruction for assists. The real defect was that our simulation piled up too
> much probability on *exactly zero*, from a double-counting bug. The correct
> repair makes the assists distribution **narrower**, not wider, and removes the
> extra zeros. Same symptom, opposite knob. Result on these 218 props: the gap
> went from **+23.2 points to +4.5**. See `MODEL_FIX_RESULTS.md`.

| Projected assists | Props | We said | Actually happened | Gap |
|---|---:|---:|---:|---:|
| 0.00–0.50 | 56 | 20.3% | 41.1% | **+20.8 pts** — worst |
| 0.50–0.75 | 61 | 31.9% | 50.8% | **+18.9 pts** |
| 0.75–1.00 | 33 | 43.0% | 54.5% | +11.6 pts |
| 1.00–1.50 | 49 | 50.4% | 63.3% | +12.9 pts |
| 1.50+ | 19 | 49.0% | 57.9% | +8.9 pts — best |

The lower the projection, the worse we price it. The pattern is that consistent.

---

## 4. Goalie saves is really two separate problems

> **Correction, after checking this against five seasons of games.** The story
> below is built on 6 low-save nights out of 43, and it over-reads them.
>
> * **"14% of starts end early" is too high.** Across 336 team-games the real rate
>   a starter faces less than the full workload is **8.6%**. Six events out of 43
>   was too thin to set a number from.
> * **"The save rate is ~11% too high" does not hold up.** Our save-rate and
>   shot-volume settings check out against five seasons almost exactly. What
>   actually happened is that the weeks we took bets on were a genuinely bad
>   stretch for goalies: shooters converted at a rate 2.2 standard deviations
>   better than normal on those 460 shots, and 2026 as a whole finished right on
>   the league average. So this is mostly **bad luck in a 10-game window, not a
>   broken model** — and we should not trim the rate to chase it.
> * **The pulled-goalie point is real and worth fixing, for a different reason.**
>   Not because it corrects the average, but because our simulation made a 2-save
>   night practically impossible: it gave that outcome a **0.05%** chance against
>   a real **1.19%**. That is exactly the kind of longshot an MS market pays out
>   on. Fixed; it now reads 0.78%.

This is the one place my earlier read was too blunt. I said the model projects
goalies 24% too high. The truth is more specific and more actionable.

Every offered goalie prop, actual saves, sorted:

```
0, 1, 2, 3, 3, 6, | 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11,
                  | 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 15, 15, 15, 15, 15,
                  | 17, 17, 18, 18, 20
```

There's an obvious break. Six games sit at 6 saves or fewer — those are goalies
who got pulled or barely played. The other 37 are normal games.

| | Props | We projected | Actually | Off by |
|---|---:|---:|---:|---:|
| **Pulled / limited** | 6 (14%) | 12.78 | **2.50** | −10.28 |
| **Normal games** | 37 (86%) | 13.27 | 11.97 | **−1.30** |

**Those 6 games are 14% of the props and 56% of the entire error.**

So the fix splits cleanly:

1. **The save rate is roughly fine.** In normal games we're off by 1.3 saves on
   ~12 — about 11% high. Worth trimming, not rebuilding.
2. **We don't model goalies getting pulled at all.** This is the real problem. We
   project every goalie to play a full game, so a 2-save night is something our
   simulation treats as essentially impossible. It happened 6 times in 43.

That is a much easier fix than rebuilding the save model, and it's the one that
matters. Separately, in normal games our spread is 2.42 against an actual 3.37 —
still too confident, but nothing like the 2.29-vs-4.61 figure I quoted before,
which was inflated by the pulled-goalie games.

**One more thing here.** A starting goalie was projected at **exactly 0.000
saves** and we still put him on the board at **+8,554** — roughly 85-to-1. JC
Higginbotham, Cannons, game 38. He made **11 saves.** A projection of exactly zero
doesn't mean "impossible," it means "our model has no playing-time information for
this guy." We treated missing information as a near-certainty and sold it at 85-to-1.

Across all goalies, 10 were projected at exactly 0.000 and **6 of them went over**
(11, 11, 7, 6, 4, 2 saves). Most were backups who probably never reached the board,
but at least one did, so this is a live hazard. **A one-line rule blocking any prop
built off a 0.000 projection prevents it entirely.**

---

## 5. Faceoff Wins: the forecast is fine, the market is not

Faceoff Wins MS was **4.5% of the money bet and 180% of the season's loss.** It
lost 81% of everything staked. To be clear about how unusual that is: normally the
worst case is losing roughly your stake. Losing 81% of handle across a whole market
means the payouts were enormous relative to what came in.

But the projection is *reasonable* — off by +0.73 on an average of 13.1. So the
failure isn't the forecast:

- **Our spread is far too tight.** σ of 1.35 against an actual σ of 5.00. We place
  nearly every faceoff specialist in a narrow band around 13 wins; real results
  swing much wider.
- **The market structure amplifies it.** There are only about 1.8 faceoff
  specialists per game, so MS has very few possible outcomes and pays long odds on
  each. One correct longshot wipes out many weeks of margin.

Combine "too confident" with "pays 10-to-1 when wrong" and you get −81%.

---

## 6. The finding that ties it all together

Across the four Over/Under markets, calibration gap and hold move together almost
perfectly — correlation **−0.92**, which for four data points is about as tight as
real data gets:

| Market | Calibration gap | What we held |
|---|---:|---:|
| Points O/U | +1.8 pts | **+12.2%** |
| Goals O/U | +5.9 pts | +6.1% |
| Assists O/U | **+16.0 pts** | **−1.7%** |
| Saves O/U | −16.9 pts | +15.9% |

*(Weeks 8–12, the weeks where we have both projections and results.)*

**Our results were not luck.** The bigger the gap between our stated probability
and reality, the worse we did — in an almost straight line. That's encouraging: it
means this is an engineering problem with a measurable target, not a run of bad
variance. Close the gap and the hold follows.

It also gives us a single number to monitor weekly. **The calibration gap was
positive every single week from week 8 onward and never corrected.** Anyone
watching it would have seen this in week 1 instead of at season's end.

One caution on that table: Saves O/U shows a *negative* gap and held +15.9%,
meaning our error happened to fall in our favor. That is not the model working —
it's an uncontrolled error that landed on the right side. The same error cost us
9.7% on Saves MS. Don't read +15.9% as validation.

---

## 7. Over/Under vs Market Selection

| Type | Money bet | Share | Result | Hold |
|---|---:|---:|---:|---:|
| **O/U** | $183,492 | 65% | +$5,703 | **+3.1%** |
| **MS** | $98,522 | 35% | −$11,438 | **−11.6%** |

**Every dollar of the season's loss came from MS markets.** O/U was profitable.

The reason is structural, not bad luck. In O/U, if we misjudge a probability by 15
points, we lose roughly 15 cents on the dollar. In MS, that same 15-point error
sits on a bet paying 10-to-1, so when it hits we pay out ten times the stake. **The
identical modeling error costs an order of magnitude more in MS.** Until the
distributions are widened, MS markets magnify exactly the flaw we have.

---

## 8. What to do, in order

**This week, before the next slate:**

1. **Turn off Faceoff Wins MS.** 4.5% of the money, 180% of the loss. Removing
   just this one market turns the season from −2.03% to **+1.70%** — profitable.
2. **Block any prop built off a 0.000 projection.** One line of code. Zero means
   "no information," not "impossible."
3. **Cap the longest odds we'll offer** (around +2000 MS, +1500 O/U). Every prop
   we priced above +8,000 this season was wrong by a wide margin.
4. **Shrink the MS share of the board** until fix #5 is done.

**Model work, highest value first:**

> **Items 5–7 are now DONE, and two of the three were done differently than
> written here.** Full before/after numbers in `MODEL_FIX_RESULTS.md`. Short
> version: the success test in item 5 is the right test and it now passes on
> 100% of props (it failed 99% of them before), but reaching it meant making
> assists *narrower*, not wider. Item 6's rate trim was dropped as unjustified.
> Item 7's σ target of 5.0 was overshooting; the correct value was measured
> properly and is closer to 4.7.

5. **Widen the spread on Assists, Goals, and Points.** The single highest-value
   fix. Success test is concrete and checkable: on 0.5-line props, our stated
   probability should land at or above the textbook Poisson number on the same
   average — not 15 points below it, as today.
   *→ Done, via the opposite knob. The extra zeros were the problem; removing them
   raised our stated probabilities without widening anything. Assists gap
   +23.2 → +4.5 points, goals +5.9 → +0.7.*
6. **Model goalies getting pulled.** Bigger win than adjusting the save rate: 14%
   of goalie props caused 56% of the error. Separately trim the save rate ~11%.
   *→ Playing time is now simulated, at the measured 8.6% rate rather than 14%.
   The rate trim was NOT applied: our save rate verifies as correct on five
   seasons, and the bad stretch was luck. This fixes the longshot pricing, which
   is where the money was lost, not the average.*
7. **Widen the Faceoff Wins spread** (σ 1.35 → ~5.0) before that market comes back
   in any form.
   *→ Done. σ 4.41 → 4.74 against an actual 5.23, and the longshot end now lines
   up almost exactly: we say a specialist wins 6 or fewer 5.43% of the time,
   against a real 5.47%. The 1.35 figure in this document was measuring something
   else and is not comparable.*
8. **Track the calibration gap weekly, by stat, with an alert.** Given the −0.92
   relationship to hold, this is an early-warning system for P&L.
9. **Fix Shots on Goal (+27%) before ever offering it.** No exposure today. For
   attackers and midfielders the correction is +23%; the +27% figure is inflated by
   defensive players and shouldn't be applied to attackers.

**Pricing:**

10. **Stop using one flat 7.5% target.** Charge based on how well we actually
    price each market: ~6% on Points O/U (proven, gap +1.8), ~8% on Goals O/U, and
    **12–15% on Assists, Saves MS, and anything with a gap over 10 points** until
    the model earns its way back down.
11. **Lower the maximum bet** on Assists and Saves MS relative to Points/Goals O/U.
12. **Grow what works.** Points and Goals O/U are 57% of the money and profitable.
    Expanding that base beats adding exotic markets.

**Data housekeeping:**

13. The Overview tab says 9 games are still "Pending" actuals, but **all 22 games
    have results.** Stale flag — any dashboard reading it under-reports our
    accuracy history.
14. **Save the projections for weeks 4–6.** The workbook only covers weeks 8–12, so
    25% of the season's money can't be traced back to the numbers that priced it.
15. **Record which props actually went live.** See the caveat below — this is the
    one thing that would materially sharpen this analysis.

---

## 9. Bottom line

| Scenario | Money bet | Result | Hold |
|---|---:|---:|---:|
| What we did | $282,014 | −$5,735 | −2.03% |
| Without Faceoff Wins MS | $269,314 | +$4,565 | **+1.70%** |
| O/U markets only | $183,492 | +$5,703 | **+3.11%** |

**The projection engine does not need to be rebuilt.** Points is well projected,
mid-range projections are close to unbiased, and the odds-building machinery is
correct. Nearly everything traces to one root cause: **our distributions are too
narrow — we're more confident than we should be.** Assists overs cashing 52% of
the time at a 35% price, goalies posting 2 saves when we called it impossible,
faceoff longshots landing at 90-to-1: all the same flaw wearing different clothes.

Getting the averages right was the hard part, and that's done. Getting the spread
right is what turns good projections into a profitable book.

---

## One caveat you should know about

The results file tells us what we *made*, but the projections file doesn't record
which props actually went live. So I reconstructed the offered list: each team's
top 5 attackers/midfielders by projected points, plus its top goalie and top
faceoff specialist — about 7 players per team per game, matching your "3 to 7"
description.

I re-ran everything at 3, 4, 5, 6, and 7 players per team, and **every conclusion
above holds across that whole range.** Nothing depends on where I drew the line.

But it is still a reconstruction. **If you can export the actual list of offered
props, I'd rerun against it** — that would also let weeks 4–6 into the analysis,
which is another 25% of the season's money currently outside it.
