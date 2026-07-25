# PLL Live Trading

A **standalone** Streamlit app for live in-game PLL trading. It runs separately
from the main PLL Projections app (so constant play-by-play polling never slows
the projections UI) but stays **connected** to it through the shared
`data/session_autosave.json` file — the live app inherits the game selection,
per-market margins (holds), team-rating overrides and depth-chart overrides you
set in the Projections app, then re-projects and re-simulates the game *in play*.

## Run

```bash
streamlit run live/app.py
```

The sidebar auto-detects the currently-live game (with a confirm step), lets you
pick any game from the schedule, or accept a manual play-by-play slug. The board
auto-refreshes every few seconds.

## How it works

Full-game live projection = **banked** (what's already happened, certain)
**+ rest-of-game** (re-simulated for only the time remaining):

```
live_schedule → detect the live game (eventStatus==1), map slug + team ids
live_feed     → poll  stats.premierlacrosseleague.com/api/v4/games/{slug}/play-by-plays
live_state    → reconstruct banked per-player stats (goals/assists/points/…)
live_model    → scale the pregame projection to the time left, blend toward the
                pace observed so far, re-run the engine's Monte Carlo for the
                remainder, then add the banked (certain) counts back on
live_pricing  → fair prices + EV edge vs a manually-entered book line
```

The rest-of-game re-simulation reuses the **projection engine's own**
`GameSimulator` (team-goal conditioning, zero-inflation, goalie matchup scaling,
teammate correlation, per-player volatility overrides) — no distribution logic
is re-implemented, so the live model inherits everything the pregame model was
validated on.

### Pace weight

`pace_weight` (sidebar) controls how much the remaining-game projection leans on
the pace observed **so far** vs the pregame model. The blend weight grows with
the fraction of the game elapsed, so an early-game outlier barely moves the
projection while a sustained late-game trend moves it a lot. `0` = ignore
in-game pace entirely; `1` = fully trust observed pace once the game is late.

### Edge / EV

Enter the book's American odds for a side and the board shows the **expected
value per $1 staked**, using the model's fair probability at that exact line:

```
EV_over = P_model(over) × profit(over_odds) − (1 − P_model(over))
```

A positive-EV side is worth betting; the magnitude is the edge. Manual odds
entry to start; a BOSS feed can replace it later.

### Unprojected players

A live scorer with no warehouse history (a call-up / rookie absent from the
pregame roster) is surfaced with their **banked** stats as a settled ("LOCK")
line, so their already-decided props stay visible instead of being dropped.

## Files

| File | Role |
|---|---|
| `live_feed.py` | play-by-play poller → `GameState` (score, clock, events, time-remaining) |
| `live_state.py` | banked per-player stat reconstruction from raw events |
| `live_model.py` | rest-of-game re-simulation + banked; `LiveModel.resimulate()` |
| `live_pricing.py` | fair odds, de-vig, EV/edge vs a book line; `EdgeQuote` |
| `live_schedule.py` | game discovery + live auto-detect (`/api/v4/games?year=`) |
| `app.py` | the Streamlit live-trading board |

Each module has a CLI for standalone testing, e.g.:

```bash
python live/live_schedule.py --year 2026        # detect the live game
python live/live_feed.py 2026-ev-36 --polls 5   # tail the feed
python live/live_state.py 2026-ev-36            # banked stats
python live/live_model.py 2026-ev-36 --home ARC --away WHP     # rest-of-game
python live/live_pricing.py 2026-ev-36 --home ARC --away WHP --stat points
```
