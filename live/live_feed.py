"""
live_feed.py — live play-by-play poller for in-game PLL trading.

Polls the SAME endpoint the historical scraper uses
(``stats.premierlacrosseleague.com/api/v4/games/{slug}/play-by-plays``) on a
short interval and exposes the current game state (score, period, clock,
seconds remaining) plus the raw event list.

The feed was verified live to:
  * update within seconds of each play (goals/shots/turnovers/faceoffs appended),
  * respond in ~0.15s (server-cached) so 5-10s polling is safe, and
  * carry real player ids (shooterId / shotAssistId / goalieId / faceoffWinnerId)
    in the SAME format as the warehouse (zero-padded strings like "003174").

This module is deliberately dependency-light (requests only) so it can be used
from a Streamlit app, a CLI monitor, or a background thread.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

# Regulation is four 12-minute quarters. `secondsPassed` on each event is the
# seconds ELAPSED within its own period (verified: a P1 9:56 goal => 124s = 720-596).
PBP_HOST = "https://stats.premierlacrosseleague.com"
QUARTER_SECONDS = 12 * 60
REGULATION_SECONDS = 4 * QUARTER_SECONDS
TIME_ZONE = "America/Los_Angeles"


def build_session() -> requests.Session:
    """Session with the exact headers the box/PBP host requires (a bare GET 403s)."""
    s = requests.Session()
    s.headers.update({
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "origin": PBP_HOST,
        "pragma": "no-cache",
        "referer": f"{PBP_HOST}/",
        "time-zone": TIME_ZONE,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
    })
    return s


def pbp_url(game_slug: str) -> str:
    return f"{PBP_HOST}/api/v4/games/{game_slug}/play-by-plays"


@dataclass
class GameState:
    """A single snapshot of a live (or finished) game."""
    game_slug: str
    events: list[dict[str, Any]]
    home_score: int
    away_score: int
    period: int
    clock_minutes: int
    clock_seconds: int
    fetched_at: float
    n_events: int = 0
    ok: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        self.n_events = len(self.events)

    # ----- derived game-clock helpers -----
    @property
    def is_overtime(self) -> bool:
        """True once play is past the four regulation quarters (period >= 5).
        PLL overtime is sudden-death, so from a rest-of-game standpoint there is
        no meaningful 'time remaining' to simulate — the game is effectively all
        banked until it reads final."""
        return self.period >= 5

    @property
    def seconds_elapsed(self) -> int:
        """Total REGULATION seconds elapsed (period-aware). In overtime this pins
        to a full regulation (2880s) — OT is sudden-death bonus time that the
        rest-of-game re-sim must not treat as more regulation to play."""
        if self.period <= 0:
            return 0
        if self.is_overtime:
            return REGULATION_SECONDS
        completed = (min(self.period, 4) - 1) * QUARTER_SECONDS
        # clock counts DOWN within a quarter, so elapsed-in-quarter = QUARTER - remaining
        in_q = QUARTER_SECONDS - (self.clock_minutes * 60 + self.clock_seconds)
        in_q = max(0, min(QUARTER_SECONDS, in_q))
        return completed + in_q

    @property
    def seconds_remaining(self) -> int:
        """Regulation seconds left. 0 in OT / final (nothing left to re-simulate:
        every stat is banked and the outcome is decided by sudden death)."""
        if self.is_overtime:
            return 0
        return max(0, REGULATION_SECONDS - self.seconds_elapsed)

    @property
    def fraction_remaining(self) -> float:
        """Share of regulation still to play, in [0, 1]. Drives the rest-of-game sim."""
        return self.seconds_remaining / REGULATION_SECONDS

    @property
    def is_pregame(self) -> bool:
        return self.period <= 0 or (self.period == 1 and self.seconds_elapsed == 0
                                    and not any(e.get("eventType") not in (None, "pregame")
                                                for e in self.events))

    @property
    def is_final(self) -> bool:
        """Best-effort final detection: regulation clock exhausted with events present.
        (The schedule endpoint's event_status is the authoritative check; this is a
        feed-only fallback.) NOTE: this cannot distinguish 'end of regulation tied,
        OT about to start' from 'game over' — a tied game at 0:00 of Q4 may go to
        OT. The app prefers the schedule's eventStatus == FINAL over this."""
        return self.period >= 4 and self.seconds_remaining == 0 and self.n_events > 0


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_state(game_slug: str, payload: dict, fetched_at: float) -> GameState:
    """Turn a raw endpoint payload into a GameState. The events live at
    ``data.items``; each carries running score only on goal events, so we read the
    LATEST event's score fields (which the feed keeps current across event types)."""
    items = (((payload or {}).get("data") or {}).get("items")) or []
    if not items:
        return GameState(game_slug, [], 0, 0, 0, 0, 0, fetched_at, ok=True)
    last = items[-1]
    return GameState(
        game_slug=game_slug,
        events=items,
        home_score=_coerce_int(last.get("homeScore")),
        away_score=_coerce_int(last.get("visitorScore")),
        period=_coerce_int(last.get("period")),
        clock_minutes=_coerce_int(last.get("minutes")),
        clock_seconds=_coerce_int(last.get("seconds")),
        fetched_at=fetched_at,
    )


def fetch_state(game_slug: str, session: Optional[requests.Session] = None,
                timeout: int = 15) -> GameState:
    """Fetch and parse a single snapshot. Never raises — returns ok=False on error."""
    s = session or build_session()
    try:
        r = s.get(pbp_url(game_slug), timeout=timeout)
        r.raise_for_status()
        return parse_state(game_slug, r.json(), time.time())
    except Exception as exc:  # network / json / http — surface, don't crash the loop
        return GameState(game_slug, [], 0, 0, 0, 0, 0, time.time(),
                         ok=False, error=str(exc))


class LiveFeed:
    """Stateful poller that only reports when the event count changes.

    Usage (blocking loop / CLI):
        feed = LiveFeed("2026-ev-36")
        for state in feed.stream(interval=8):
            ...  # called each poll; state.new_events holds just-arrived events

    Usage (pull, e.g. from Streamlit auto-refresh):
        feed = LiveFeed(slug); state = feed.poll()
    """

    def __init__(self, game_slug: str, session: Optional[requests.Session] = None):
        self.game_slug = game_slug
        self.session = session or build_session()
        self._last_n = 0
        self.last_state: Optional[GameState] = None

    def poll(self) -> GameState:
        state = fetch_state(self.game_slug, self.session)
        if state.ok:
            # attach only the events that are new since the previous successful poll
            new = state.events[self._last_n:] if state.n_events >= self._last_n else state.events
            setattr(state, "new_events", new)
            self._last_n = state.n_events
            self.last_state = state
        else:
            setattr(state, "new_events", [])
        return state

    def stream(self, interval: float = 8.0, max_polls: Optional[int] = None):
        """Yield a GameState each poll. Stops after max_polls (None = forever) or
        once the game reads final."""
        i = 0
        while max_polls is None or i < max_polls:
            state = self.poll()
            yield state
            if state.ok and state.is_final:
                break
            i += 1
            time.sleep(interval)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tail a live PLL game's play-by-play.")
    ap.add_argument("slug", help="game slug, e.g. 2026-ev-36")
    ap.add_argument("--interval", type=float, default=8.0)
    ap.add_argument("--polls", type=int, default=None, help="stop after N polls")
    args = ap.parse_args()

    feed = LiveFeed(args.slug)
    for st in feed.stream(interval=args.interval, max_polls=args.polls):
        if not st.ok:
            print(f"[error] {st.error}")
            continue
        tag = f"  <<< +{len(st.new_events)} new" if getattr(st, "new_events", None) else ""
        print(f"P{st.period} {st.clock_minutes:02d}:{st.clock_seconds:02d} "
              f"| {st.away_score}-{st.home_score} | {st.n_events} ev "
              f"| {st.seconds_remaining}s left ({st.fraction_remaining:.0%}){tag}")
