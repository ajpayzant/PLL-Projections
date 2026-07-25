"""
live_schedule.py — discover games and auto-detect the currently-live one.

Uses the PLL games list endpoint:
    https://stats.premierlacrosseleague.com/api/v4/games?year=YYYY
(list at data.items). Each item carries:
    slugname   -> the play-by-play slug (e.g. "2026-ev-36"), used by live_feed
    eventStatus-> 0 = scheduled, 1 = LIVE, 3 = final
    homeTeam.officialId / awayTeam.officialId -> team ids for engine.project()
    homeScore / visitorScore, period, clockMinutes/Seconds, startTime (unix)

Auto-detect returns any status==1 game; the app confirms it and allows a manual
slug override, per the user's "auto-detect + confirm, with manual override" pick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from live_feed import build_session, PBP_HOST

STATUS_SCHEDULED = 0
STATUS_LIVE = 1
STATUS_FINAL = 3


def games_url(year: int) -> str:
    return f"{PBP_HOST}/api/v4/games?year={year}"


@dataclass
class GameInfo:
    slug: str
    game_number: Optional[int]
    home_team_id: str
    away_team_id: str
    home_name: str
    away_name: str
    status: int
    home_score: int
    visitor_score: int
    period: int
    clock_minutes: int
    clock_seconds: int
    start_time: Optional[int]  # unix seconds
    year: int

    @property
    def is_live(self) -> bool:
        return self.status == STATUS_LIVE

    @property
    def is_final(self) -> bool:
        return self.status == STATUS_FINAL

    @property
    def is_scheduled(self) -> bool:
        return self.status == STATUS_SCHEDULED

    @property
    def game_date(self) -> Optional[str]:
        """YYYY-MM-DD from the unix start time (used for roster/context)."""
        if not self.start_time:
            return None
        import datetime as dt
        try:
            return dt.datetime.utcfromtimestamp(int(self.start_time)).strftime("%Y-%m-%d")
        except Exception:
            return None

    @property
    def label(self) -> str:
        tag = {STATUS_LIVE: "LIVE", STATUS_FINAL: "FINAL",
               STATUS_SCHEDULED: "upcoming"}.get(self.status, "?")
        score = f" {self.visitor_score}-{self.home_score}" if self.status != STATUS_SCHEDULED else ""
        return f"[{tag}] {self.away_name} @ {self.home_name}{score} ({self.slug})"


def _coerce_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _parse_game(g: dict, year: int) -> Optional[GameInfo]:
    slug = g.get("slugname")
    ht, at = g.get("homeTeam") or {}, g.get("awayTeam") or {}
    hid = ht.get("officialId") or ht.get("team_id")
    aid = at.get("officialId") or at.get("team_id")
    if not slug or not hid or not aid:
        return None
    return GameInfo(
        slug=str(slug),
        game_number=_coerce_int(g.get("gameNumber"), None) if g.get("gameNumber") is not None else None,
        home_team_id=str(hid), away_team_id=str(aid),
        home_name=str(ht.get("fullName", hid)), away_name=str(at.get("fullName", aid)),
        status=_coerce_int(g.get("eventStatus"), 0),
        home_score=_coerce_int(g.get("homeScore")),
        visitor_score=_coerce_int(g.get("visitorScore")),
        period=_coerce_int(g.get("period")),
        clock_minutes=_coerce_int(g.get("clockMinutes")),
        clock_seconds=_coerce_int(g.get("clockSeconds")),
        start_time=_coerce_int(g.get("startTime"), None) if g.get("startTime") else None,
        year=year,
    )


def list_games(year: int, session=None, timeout: int = 15) -> List[GameInfo]:
    """All games for a season, parsed. Empty list on any error (never raises)."""
    s = session or build_session()
    try:
        r = s.get(games_url(year), timeout=timeout)
        r.raise_for_status()
        items = (((r.json() or {}).get("data") or {}).get("items")) or []
    except Exception:
        return []
    out = [_parse_game(g, year) for g in items]
    return [g for g in out if g is not None]


def detect_live(year: int, session=None) -> List[GameInfo]:
    """Return every game currently in-progress (eventStatus == 1)."""
    return [g for g in list_games(year, session=session) if g.is_live]


def find_game(year: int, slug: str, session=None) -> Optional[GameInfo]:
    """Look up a specific game by slug (for the manual-override path)."""
    slug = str(slug).strip()
    for g in list_games(year, session=session):
        if g.slug == slug:
            return g
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="List PLL games / detect the live one.")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--all", action="store_true", help="list every game, not just live")
    args = ap.parse_args()

    games = list_games(args.year)
    print(f"{len(games)} games for {args.year}")
    live = [g for g in games if g.is_live]
    print(f"\n{len(live)} LIVE now:")
    for g in live:
        print(f"  {g.label}  P{g.period} {g.clock_minutes:02d}:{g.clock_seconds:02d}")
    if args.all:
        print("\nall games:")
        for g in games:
            print(" ", g.label)
