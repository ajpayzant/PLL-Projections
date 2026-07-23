"""
PLL Current Roster Cache
========================
Repo-friendly official roster scraper for the Streamlit projection app.

Purpose:
- Fetch official current PLL roster pages over plain HTTP (no browser) and
  parse the Next.js RSC JSON payload (`self.__next_f.push([1, "..."])`) for the
  structured `roster` array. Keys on JSON field names, not hashed CSS classes,
  so it survives site reskins (the old Playwright + div.css-fps5zs scrape broke
  when the site regenerated its style hashes).
- Write data/reference_tables/current_rosters.csv.
- Avoid Google Sheets/gspread dependencies inside the Streamlit projection app.

Usage:
    python pll_current_roster_cache.py

Deploy notes:
    pip install pandas   # no Playwright/Chromium required
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# HTTP fetch timeout (ms kept for back-compat with existing env var name).
PAGE_TIMEOUT_MS = int(os.getenv("PLL_ROSTER_PAGE_TIMEOUT_MS", "60000"))

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "reference_tables" / "current_rosters.csv"
DEFAULT_DIAGNOSTICS_PATH = REPO_ROOT / "data" / "reference_tables" / "current_rosters_diagnostics.csv"

# Public roster page team codes differ from the projection warehouse/stat IDs.
# Team_ID is what projection_engine_v3.py expects.
PLL_TEAMS: List[Dict[str, object]] = [
    {
        "Team_ID": "CAN",
        "Team_Code": "BOS",
        "Team": "Boston Cannons",
        "Division": "Eastern",
        "URLs": [
            "https://premierlacrosseleague.com/teams/boston-cannons/roster",
            "https://premierlacrosseleague.com/teams/Cannons/roster",
        ],
    },
    {
        "Team_ID": "RED",
        "Team_Code": "CAL",
        "Team": "California Redwoods",
        "Division": "Western",
        "URLs": [
            "https://premierlacrosseleague.com/teams/california-redwoods/roster",
            "https://premierlacrosseleague.com/teams/Redwoods/roster",
            "https://premierlacrosseleague.com/teams/redwoods/roster",
        ],
    },
    {
        "Team_ID": "CHA",
        "Team_Code": "CAR",
        "Team": "Carolina Chaos",
        "Division": "Western",
        "URLs": [
            "https://premierlacrosseleague.com/teams/carolina-chaos/roster",
            "https://premierlacrosseleague.com/teams/Chaos/roster",
        ],
    },
    {
        "Team_ID": "OUT",
        "Team_Code": "DEN",
        "Team": "Denver Outlaws",
        "Division": "Western",
        "URLs": [
            "https://premierlacrosseleague.com/teams/denver-outlaws/roster",
            "https://premierlacrosseleague.com/teams/Outlaws/roster",
        ],
    },
    {
        "Team_ID": "WHP",
        "Team_Code": "MD",
        "Team": "Maryland Whipsnakes",
        "Division": "Eastern",
        "URLs": [
            "https://premierlacrosseleague.com/teams/maryland-whipsnakes/roster",
            "https://premierlacrosseleague.com/teams/Whipsnakes/roster",
        ],
    },
    {
        "Team_ID": "ATL",
        "Team_Code": "NY",
        "Team": "New York Atlas",
        "Division": "Eastern",
        "URLs": [
            "https://premierlacrosseleague.com/teams/new-york-atlas/roster",
            "https://premierlacrosseleague.com/teams/Atlas/roster",
        ],
    },
    {
        "Team_ID": "WAT",
        "Team_Code": "PHI",
        "Team": "Philadelphia Waterdogs",
        "Division": "Eastern",
        "URLs": [
            "https://premierlacrosseleague.com/teams/philadelphia-waterdogs/roster",
            "https://premierlacrosseleague.com/teams/Waterdogs/roster",
        ],
    },
    {
        "Team_ID": "ARC",
        "Team_Code": "UTA",
        "Team": "Utah Archers",
        "Division": "Western",
        "URLs": [
            "https://premierlacrosseleague.com/teams/utah-archers/roster",
            "https://premierlacrosseleague.com/teams/Archers/roster",
        ],
    },
]

POSITIONS = ["A", "M", "SSDM", "LSM", "D", "FO", "G"]
POSITION_ORDER = {"A": 1, "M": 2, "SSDM": 3, "LSM": 4, "D": 5, "FO": 6, "G": 7, "UNK": 99}
POSITION_GROUP = {
    "A": "Attack",
    "M": "Midfield",
    "SSDM": "Short Stick Defensive Midfield",
    "LSM": "Long Stick Midfield",
    "D": "Defense",
    "FO": "Faceoff",
    "G": "Goalie",
    "UNK": "Unknown",
}

SCRAPE_COLUMNS = [
    "Player",
    "First_Name",
    "Last_Name",
    "Team",
    "Team_ID",
    "Team_Code",
    "Division",
    "Position",
    "Position_Group",
    "Jersey",
    "Handedness",
    "Height",
    "Age",
    "College",
    "Country",
    "Image_Slug",
    "Image_URL",
    "Page_URL",
    "Page_Title",
    "Extracted_At",
]


def clean_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def normalize_key(value) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def normalize_position(pos) -> str:
    pos = clean_text(pos).upper()
    aliases = {
        "ATTACK": "A",
        "MIDFIELD": "M",
        "DEFENSE": "D",
        "FACEOFF": "FO",
        "FACE-OFF": "FO",
        "GOALIE": "G",
        "GOALTENDER": "G",
        "LONG STICK MIDFIELD": "LSM",
        "SHORT STICK DEFENSIVE MIDFIELD": "SSDM",
    }
    return aliases.get(pos, pos if pos else "UNK")


def clean_age(value) -> str:
    value = clean_text(value)
    match = re.search(r"\b(\d{1,2})\b", value)
    return match.group(1) if match else ""


def clean_height(value) -> str:
    return clean_text(value).replace("`", "").replace('"', "")


# ---------------------------------------------------------------------------
# RSC JSON extraction
# ---------------------------------------------------------------------------
# The PLL site is a Next.js App Router (React Server Components) app. Player
# data is streamed in `self.__next_f.push([1, "<json-chunk>"])` calls and is
# NOT reliably present as hashed CSS classes (the old div.css-fps5zs / .points
# selectors churn on every rebuild). We fetch the server-rendered HTML with
# plain HTTP (no browser), reassemble the RSC payload, and parse the
# `"roster":[...]` array of structured player objects. This survives CSS
# reskins because it keys on JSON field names, not style classes.

RSC_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def _decode_rsc_blob(html: str) -> str:
    """Reassemble the RSC streaming payload from the __next_f push chunks."""
    blob = []
    for chunk in RSC_PUSH_RE.findall(html):
        try:
            blob.append(json.loads(chunk))  # each chunk is a JSON-encoded string
        except Exception:
            continue
    return "".join(blob)


def _extract_roster_array(blob: str) -> List[dict]:
    """Bracket-match the `"roster":[ ... ]` array and json.loads it.

    The roster objects carry positionName/officialId/jerseyNum/firstName/
    lastName/position/height/age/college/country/handedness/injuryStatus/
    profileUrl. We take the first well-formed `"roster"` array on the page.
    """
    key = '"roster":'
    search_from = 0
    while True:
        i = blob.find(key, search_from)
        if i < 0:
            return []
        j = blob.find("[", i)
        if j < 0:
            return []
        depth = 0
        end = -1
        for k in range(j, len(blob)):
            c = blob[k]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end < 0:
            search_from = i + len(key)
            continue
        try:
            arr = json.loads(blob[j:end + 1])
        except Exception:
            search_from = i + len(key)
            continue
        # A real roster array is a list of dicts with player fields.
        if isinstance(arr, list) and arr and isinstance(arr[0], dict) and "officialId" in arr[0]:
            return arr
        search_from = end + 1


def _image_slug_from_url(url: str) -> str:
    """Derive a stable player slug from the profile image URL."""
    m = re.search(r"/Players/\d+/([a-z0-9\-]+)\.(?:webp|png|jpg)", str(url), re.IGNORECASE)
    return m.group(1) if m else ""


def _roster_obj_to_row(obj: dict, team: Dict[str, object]) -> dict:
    """Map one RSC roster object to the SCRAPE_COLUMNS schema."""
    first = clean_text(obj.get("firstName"))
    last = clean_text(obj.get("lastName"))
    suffix = clean_text(obj.get("lastNameSuffix"))
    if suffix:
        last = f"{last} {suffix}".strip()
    image_url = clean_text(obj.get("profileUrl"))
    return {
        "Player": clean_text(f"{first} {last}"),
        "First_Name": first,
        "Last_Name": last,
        "Team": team["Team"],
        "Team_ID": team["Team_ID"],
        "Team_Code": team["Team_Code"],
        "Division": team["Division"],
        "Position": normalize_position(obj.get("position") or obj.get("positionName")),
        "Jersey": clean_text(obj.get("jerseyNum")),
        "Handedness": clean_text(obj.get("handedness")),
        "Height": clean_height(obj.get("height")),
        "Age": clean_age(obj.get("age")),
        "College": clean_text(obj.get("college")),
        "Country": clean_text(obj.get("country")),
        "Image_Slug": _image_slug_from_url(image_url) or clean_text(obj.get("slug")),
        "Image_URL": image_url,
        "Page_URL": "",
        "Page_Title": "",
        "Extracted_At": datetime.now(timezone.utc).isoformat(),
    }


def dedupe_team_rows(rows: List[dict]) -> List[dict]:
    best: Dict[str, Tuple[int, dict]] = {}

    for row in rows:
        player = clean_text(row.get("Player"))
        image_slug = clean_text(row.get("Image_Slug"))
        key = f"{player}|{image_slug}"

        if not player:
            continue

        score = sum(1 for v in row.values() if clean_text(v))
        if key not in best or score > best[key][0]:
            best[key] = (score, row)

    return [x[1] for x in best.values()]


def scrape_team_roster(team: Dict[str, object]) -> Tuple[List[dict], List[dict]]:
    """Fetch one team's roster page and parse the RSC `roster` JSON array.

    Tries each candidate URL until one yields a well-formed roster array.
    Returns (rows, diagnostics). No browser required.
    """
    print(f"\nSCRAPING {team['Team_Code']} / {team['Team_ID']} — {team['Team']}")
    diagnostics: List[dict] = []
    best_rows: List[dict] = []

    for url in team["URLs"]:  # type: ignore[index]
        print(f"Trying URL: {url}")
        try:
            html = _http_get(url, timeout=PAGE_TIMEOUT_MS // 1000)
            blob = _decode_rsc_blob(html)
            roster = _extract_roster_array(blob)
            rows = [_roster_obj_to_row(o, team) for o in roster
                    if clean_text(o.get("firstName")) or clean_text(o.get("lastName"))]
            rows = [r for r in rows if r["Player"] and r["Position"] != "UNK"]
            deduped_rows = dedupe_team_rows(rows)

            print(f"  RSC chunks decoded: {len(RSC_PUSH_RE.findall(html))}")
            print(f"  Roster objects: {len(roster)}")
            print(f"  Valid players: {len(deduped_rows)}")

            diagnostics.append({
                "Team_ID": team["Team_ID"],
                "Team_Code": team["Team_Code"],
                "Team": team["Team"],
                "URL_Tried": url,
                "Final_URL": url,
                "Raw_Card_Containers": len(roster),
                "Raw_Valid_Rows": len(rows),
                "Deduped_Players": len(deduped_rows),
                "Status": "OK" if len(deduped_rows) else "NO_PLAYERS",
                "Error": "",
            })

            if len(deduped_rows) > len(best_rows):
                best_rows = deduped_rows
            if len(deduped_rows) >= 15:
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            diagnostics.append({
                "Team_ID": team["Team_ID"],
                "Team_Code": team["Team_Code"],
                "Team": team["Team"],
                "URL_Tried": url,
                "Final_URL": "",
                "Raw_Card_Containers": 0,
                "Raw_Valid_Rows": 0,
                "Deduped_Players": 0,
                "Status": "ERROR",
                "Error": str(e),
            })

    print(f"FINAL {team['Team_Code']} / {team['Team_ID']} PLAYERS: {len(best_rows)}")
    return best_rows, diagnostics


def sort_master_roster(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    team_rank = {str(team["Team_ID"]): i for i, team in enumerate(PLL_TEAMS)}
    out = df.copy()
    out["_team_rank"] = out["Team_ID"].map(team_rank)
    out["_pos_rank"] = out["Position"].map(lambda x: POSITION_ORDER.get(x, 99))
    out["_last_name"] = out["Last_Name"].astype(str).str.lower()

    out = (
        out.sort_values(["_team_rank", "_pos_rank", "_last_name", "Player"])
        .drop(columns=["_team_rank", "_pos_rank", "_last_name"], errors="ignore")
        .reset_index(drop=True)
    )
    return out


def scrape_all_pll_rosters() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape every PLL team roster via HTTP + RSC JSON (no browser)."""
    all_rows: List[dict] = []
    all_diagnostics: List[dict] = []
    for team in PLL_TEAMS:
        rows, diagnostics = scrape_team_roster(team)
        all_rows.extend(rows)
        all_diagnostics.extend(diagnostics)

    roster_df = pd.DataFrame(all_rows)
    if roster_df.empty:
        roster_df = pd.DataFrame(columns=SCRAPE_COLUMNS)
    else:
        for col in SCRAPE_COLUMNS:
            if col not in roster_df.columns:
                roster_df[col] = ""
        roster_df = roster_df[SCRAPE_COLUMNS].copy()
        for col in roster_df.columns:
            roster_df[col] = roster_df[col].map(clean_text)
        roster_df["Position"] = roster_df["Position"].map(normalize_position)
        roster_df["Position_Group"] = roster_df["Position"].map(lambda x: POSITION_GROUP.get(x, "Unknown"))
        roster_df["Height"] = roster_df["Height"].map(clean_height)
        roster_df["Age"] = roster_df["Age"].map(clean_age)
        roster_df = roster_df[SCRAPE_COLUMNS].copy()
        roster_df = roster_df.drop_duplicates(subset=["Team_ID", "Team_Code", "Player", "Image_Slug"])
        roster_df = sort_master_roster(roster_df)

    diagnostics_df = pd.DataFrame(all_diagnostics)
    return roster_df, diagnostics_df


# Back-compat alias: the scraper no longer needs a browser or an event loop,
# but callers (write_current_rosters_csv, any Streamlit refresh path) still
# import scrape_all_pll_rosters_sync. Keep the name pointing at the HTTP path.
def scrape_all_pll_rosters_sync() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return scrape_all_pll_rosters()


def validate_scrape(roster_df: pd.DataFrame, min_total_players: int = 120, min_teams: int = 8, min_players_per_team: int = 15) -> List[str]:
    issues: List[str] = []
    if roster_df.empty:
        return ["NO_PLAYERS_SCRAPED"]

    total_players = len(roster_df)
    teams = roster_df["Team_ID"].nunique()
    if total_players < min_total_players:
        issues.append(f"LOW_TOTAL_PLAYERS_{total_players}")
    if teams < min_teams:
        issues.append(f"MISSING_TEAMS_{teams}")

    counts = roster_df.groupby("Team_ID").size().to_dict()
    for team in PLL_TEAMS:
        team_id = str(team["Team_ID"])
        count = counts.get(team_id, 0)
        if count < min_players_per_team:
            issues.append(f"{team_id}_LOW_ROSTER_COUNT_{count}")

    return issues


def write_current_rosters_csv(output_path: Path = DEFAULT_OUTPUT_PATH, diagnostics_path: Path = DEFAULT_DIAGNOSTICS_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    roster_df, diagnostics_df = scrape_all_pll_rosters_sync()
    issues = validate_scrape(roster_df)

    output_path = Path(output_path)
    diagnostics_path = Path(diagnostics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    # Always write diagnostics first, even on failure, so a broken CI run is
    # diagnosable (which team/URL failed) instead of exiting with no artifact.
    diagnostics_df.to_csv(diagnostics_path, index=False)
    print(f"Wrote diagnostics to {diagnostics_path}")

    # Hard-fail only on a genuinely broken scrape (nothing scraped, or multiple
    # teams missing entirely). A single team coming up short (e.g. the site was
    # slow for one roster page) should NOT discard the other 7 teams' fresh data
    # and leave every projection on a stale roster — write what we have and warn.
    teams_scraped = int(roster_df["Team_ID"].nunique()) if not roster_df.empty else 0
    fatal = roster_df.empty or teams_scraped < 7 or len(roster_df) < 120
    if fatal:
        raise RuntimeError(
            "Roster scrape validation FAILED (not writing rosters): "
            + ", ".join(issues)
            + f" | teams={teams_scraped} players={len(roster_df)}"
        )
    if issues:
        print("WARNING: roster scrape completed with issues (writing anyway): "
              + ", ".join(issues))

    roster_df.to_csv(output_path, index=False)
    print(f"Wrote {len(roster_df)} roster rows to {output_path}")
    print(roster_df.groupby(["Team_ID", "Team_Code", "Team"]).size().reset_index(name="Roster_Count").to_string(index=False))

    return roster_df, diagnostics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape official PLL rosters and cache them for the projection app.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path for current_rosters.csv")
    parser.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS_PATH), help="Path for diagnostics CSV")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    print("PLL current roster cache started:", started)
    write_current_rosters_csv(Path(args.output), Path(args.diagnostics))
    print("PLL current roster cache complete:", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
