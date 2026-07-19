"""
scrape_play_by_play.py
----------------------
Scrapes PLL play-by-play (PBP) event feeds and normalizes them into a single
tidy parquet table (one row per event) for evaluation / modeling.

The main warehouse builder (build_warehouse.py, Block 4) deliberately skips
play-by-play. This script fills that gap as a *separate, additive* surface so
the existing box-score pipeline and the projection engine stay untouched until
any PBP-derived feature proves itself on the backtest.

Endpoint (verified for every season 2022-2026):
    https://stats.premierlacrosseleague.com/api/v4/games/{game_slug}/play-by-plays

Note the host differs from the box-score API:
    box score : api.stats.premierlacrosseleague.com/api/v4/events/{slug}/...
    play-by-play : stats.premierlacrosseleague.com/api/v4/games/{slug}/play-by-plays

The universal game identifier is the `game_slug` column in game_manifest.parquet
(NOT game_id / event_numeric_id, which 404 on this endpoint).

Outputs:
    data/source_data/api_responses/season_{yr}/game_{slug}/play_by_plays.json.gz   (raw cache, gitignored)
    data/curated_data/all_requested_seasons/pbp_events.parquet                     (normalized, committed)
    data/curated_data/all_requested_seasons/pbp_scrape_log.parquet                 (per-game scrape status)

Usage:
    python scripts/scrape_play_by_play.py                       # scrape missing, all seasons
    python scripts/scrape_play_by_play.py --seasons 2025 2026   # subset of seasons
    python scripts/scrape_play_by_play.py --force               # re-download everything
    python scripts/scrape_play_by_play.py --limit 5             # first 5 games (smoke test)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pbp_scrape")

# -----------------------------
# Paths (mirror build_warehouse.py conventions)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
API_RESPONSES_DIR = DATA_ROOT / "source_data" / "api_responses"
CURATED_ALL_DIR = DATA_ROOT / "curated_data" / "all_requested_seasons"

GAME_MANIFEST_PATH = CURATED_ALL_DIR / "game_manifest.parquet"
PBP_EVENTS_PATH = CURATED_ALL_DIR / "pbp_events.parquet"
PBP_SCRAPE_LOG_PATH = CURATED_ALL_DIR / "pbp_scrape_log.parquet"

# -----------------------------
# HTTP config — the play-by-play host differs from the box-score API host
# -----------------------------
PBP_HOST = "https://stats.premierlacrosseleague.com"
TIME_ZONE = "America/Los_Angeles"


def build_session() -> requests.Session:
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


# -----------------------------
# Cache helpers
# -----------------------------
def write_gzip_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _get(url: str, session: requests.Session, timeout: int = 30) -> requests.Response:
    return session.get(url, timeout=timeout)


def fetch_pbp_with_cache(
    game_slug: str,
    cache_path: Path,
    session: requests.Session,
    force: bool = False,
) -> tuple[Optional[dict], int, str]:
    """Return (payload, http_status, fetch_mode). fetch_mode in {cached, downloaded, error}."""
    if cache_path.exists() and not force:
        try:
            return read_gzip_json(cache_path), 200, "cached"
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass

    r = _get(pbp_url(game_slug), session=session)
    try:
        payload = r.json()
    except Exception:
        payload = None

    if r.status_code == 200 and payload is not None:
        write_gzip_json(cache_path, payload)

    return payload, r.status_code, "downloaded"


# -----------------------------
# Normalization
# -----------------------------
# Scalar fields we keep verbatim from each event item.
_SCALAR_FIELDS = [
    "markerId",
    "eventType",
    "description",
    "period",
    "minutes",
    "seconds",
    "secondsPassed",
    "teamId",
    "shotType",
    "penaltyLength",
    "penaltyDescription",
    "homeScore",
    "visitorScore",
    "awayTeamWinProbability",
    "homeTeamWinProbability",
    "faceoffWinnerId",
    "faceoffLoserId",
    "gbPlayerId",
    "commitedTurnoverId",   # (sic) — spelled this way in the API
    "causedTurnoverId",
    "shooterId",
    "goalieId",
    "offenseGoalieId",
    "commitedPenaltyId",    # (sic)
    "shotAssistId",
    "assistOpportunityPlayerId",
    "closestDefenderId",
]

# Nested details.* fields (present mainly on shot/goal events).
_DETAIL_FIELDS = [
    "shotOnGoal",
    "shotSaved",
    "saveType",
]


def _marker_core(marker_id: Any) -> str:
    """Numeric core of a markerId, e.g. 'faceoff-100' -> '100', '100' -> '100'."""
    m = re.search(r"(\d+)$", str(marker_id) if marker_id is not None else "")
    return m.group(1) if m else ""


def _dedupe_marker_twins(items: list[dict]) -> list[dict]:
    """
    Drop malformed duplicate events seen in a handful of 2023/24 games where a
    faceoff is logged twice: once with a canonical markerId ('faceoff-100') and
    once with a bare-digit twin ('100') sharing the same numeric core.

    Rule: within an (eventType, core) group that contains BOTH a named marker
    and a bare-digit marker, keep only the named one. If a core has only the
    bare-digit form (true for whole 2023 games), keep it — it's the only copy.
    """
    # Index bare-digit vs named occurrences per (eventType, core).
    named_keys: set[tuple] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("markerId", ""))
        if not mid.isdigit():  # named marker (has a non-digit prefix)
            named_keys.add((it.get("eventType"), _marker_core(mid)))

    kept = []
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("markerId", ""))
        if mid.isdigit() and (it.get("eventType"), _marker_core(mid)) in named_keys:
            continue  # bare-digit twin of a named marker -> drop
        kept.append(it)
    return kept


def normalize_game_pbp(payload: dict, season: int, game_slug: str) -> list[dict]:
    """Flatten one game's PBP payload into a list of per-event row dicts."""
    items = (payload or {}).get("data", {}).get("items", []) or []
    items = _dedupe_marker_twins(items)
    rows: list[dict] = []

    for idx, ev in enumerate(items):
        if not isinstance(ev, dict):
            continue

        row: dict[str, Any] = {
            "season": season,
            "game_slug": game_slug,
            "event_index": idx,
        }
        for f in _SCALAR_FIELDS:
            row[f] = ev.get(f)

        details = ev.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        for f in _DETAIL_FIELDS:
            row[f"detail_{f}"] = details.get(f)

        rows.append(row)

    return rows


def load_game_slugs(seasons: Optional[list[int]]) -> pd.DataFrame:
    if not GAME_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Game manifest not found at {GAME_MANIFEST_PATH}. "
            "Run build_warehouse.py (or restore committed parquet) first."
        )
    gm = pd.read_parquet(GAME_MANIFEST_PATH, columns=["season", "game_slug"])
    gm = gm.dropna(subset=["game_slug"]).drop_duplicates("game_slug")
    if seasons:
        gm = gm[gm["season"].isin(seasons)]
    return gm.sort_values(["season", "game_slug"]).reset_index(drop=True)


def scrape(
    seasons: Optional[list[int]] = None,
    force: bool = False,
    limit: Optional[int] = None,
    sleep: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape PBP for the requested games. Returns (events_df, log_df)."""
    games = load_game_slugs(seasons)
    if limit:
        games = games.head(limit)

    logger.info("Scraping play-by-play for %d games", len(games))
    session = build_session()

    all_event_rows: list[dict] = []
    log_rows: list[dict] = []

    for _, g in tqdm(games.iterrows(), total=len(games), desc="PBP"):
        season = int(g["season"])
        slug = str(g["game_slug"])
        cache_path = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}" / "play_by_plays.json.gz"

        try:
            payload, status, mode = fetch_pbp_with_cache(slug, cache_path, session, force=force)
            event_rows = normalize_game_pbp(payload, season, slug) if payload else []
            all_event_rows.extend(event_rows)
            log_rows.append({
                "season": season,
                "game_slug": slug,
                "http_status": status,
                "fetch_mode": mode,
                "n_events": len(event_rows),
                "cache_path": str(cache_path) if status == 200 else None,
                "error": None,
            })
            if mode == "downloaded":
                time.sleep(sleep)
        except Exception as e:
            log_rows.append({
                "season": season,
                "game_slug": slug,
                "http_status": None,
                "fetch_mode": "error",
                "n_events": 0,
                "cache_path": None,
                "error": str(e)[:500],
            })

    events_df = pd.DataFrame(all_event_rows)
    log_df = pd.DataFrame(log_rows)
    return events_df, log_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape & normalize PLL play-by-play feeds")
    parser.add_argument("--seasons", type=int, nargs="*", default=None,
                        help="Seasons to scrape (default: all in manifest)")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--limit", type=int, default=None, help="Only first N games (smoke test)")
    parser.add_argument("--no-write", action="store_true",
                        help="Scrape/normalize but do not write curated parquet")
    args = parser.parse_args()

    events_df, log_df = scrape(seasons=args.seasons, force=args.force, limit=args.limit)

    # Report
    logger.info("Scraped %d events across %d games", len(events_df), len(log_df))
    if not log_df.empty:
        by_status = log_df.groupby(["fetch_mode", "http_status"], dropna=False).size()
        logger.info("Fetch modes:\n%s", by_status.to_string())
        failed = log_df[log_df["http_status"].ne(200) | log_df["error"].notna()]
        if not failed.empty:
            logger.warning("%d games failed:\n%s", len(failed),
                           failed[["season", "game_slug", "http_status", "error"]].to_string(index=False))
    if not events_df.empty:
        logger.info("Event type distribution:\n%s",
                    events_df["eventType"].value_counts().to_string())

    if args.no_write:
        logger.info("--no-write set; skipping parquet write")
        return 0

    # Merge with any existing events not re-scraped this run (partial-season runs)
    if args.seasons and PBP_EVENTS_PATH.exists() and not events_df.empty:
        existing = pd.read_parquet(PBP_EVENTS_PATH)
        existing = existing[~existing["season"].isin(args.seasons)]
        events_df = pd.concat([existing, events_df], ignore_index=True)

    CURATED_ALL_DIR.mkdir(parents=True, exist_ok=True)
    if not events_df.empty:
        events_df = events_df.sort_values(["season", "game_slug", "event_index"]).reset_index(drop=True)
        events_df.to_parquet(PBP_EVENTS_PATH, index=False)
        logger.info("Wrote %s (%d rows)", PBP_EVENTS_PATH, len(events_df))
    log_df.to_parquet(PBP_SCRAPE_LOG_PATH, index=False)
    logger.info("Wrote %s (%d rows)", PBP_SCRAPE_LOG_PATH, len(log_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
