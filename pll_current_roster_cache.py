"""
PLL Current Roster Cache
========================
Repo-friendly version of the working Google Sheets roster scraper.

Purpose:
- Scrape official current PLL roster pages with Playwright/Chromium.
- Write data/reference_tables/current_rosters.csv.
- Avoid Google Sheets/gspread dependencies inside the Streamlit projection app.

Usage:
    python pll_current_roster_cache.py

Deploy notes:
    pip install playwright pandas
    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

HEADLESS = str(os.getenv("PLL_ROSTER_HEADLESS", "1")).strip().lower() not in {"0", "false", "no", "off"}
PAGE_TIMEOUT_MS = int(os.getenv("PLL_ROSTER_PAGE_TIMEOUT_MS", "60000"))
CARD_WAIT_TIMEOUT_MS = int(os.getenv("PLL_ROSTER_CARD_WAIT_TIMEOUT_MS", "30000"))
SCROLL_PASSES = int(os.getenv("PLL_ROSTER_SCROLL_PASSES", "8"))
SCROLL_PAUSE_MS = int(os.getenv("PLL_ROSTER_SCROLL_PAUSE_MS", "600"))

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


ROSTER_EXTRACTOR_JS = """
(teamInfo) => {
  function cleanText(x) {
    return (x || "").replace(/\s+/g, " ").trim();
  }

  function normalizePosition(pos) {
    pos = cleanText(pos).toUpperCase();

    const aliases = {
      "ATTACK": "A",
      "MIDFIELD": "M",
      "DEFENSE": "D",
      "FACEOFF": "FO",
      "FACE-OFF": "FO",
      "GOALIE": "G",
      "GOALTENDER": "G",
      "LONG STICK MIDFIELD": "LSM",
      "SHORT STICK DEFENSIVE MIDFIELD": "SSDM"
    };

    return aliases[pos] || pos || "UNK";
  }

  function extractDetails(card) {
    const details = {};

    card.querySelectorAll("div").forEach(div => {
      const spans = Array.from(div.children || []).filter(el => el.tagName === "SPAN");

      if (spans.length >= 2) {
        const label = cleanText(spans[0].innerText);
        const value = cleanText(spans[1].innerText);

        if (label && value) {
          details[label] = value;
        }
      }
    });

    return details;
  }

  const cards = Array.from(document.querySelectorAll("div.css-fps5zs"));
  const rows = [];

  cards.forEach(card => {
    const firstName = cleanText(card.querySelector("p.firstName")?.innerText);
    const lastName = cleanText(card.querySelector("p.lastName")?.innerText);
    const player = cleanText(`${firstName} ${lastName}`);

    const jersey = cleanText(card.querySelector(".points")?.innerText);

    const playerImg = card.querySelector(".playerImg img");
    const imageSlug = cleanText(playerImg?.getAttribute("alt"));
    const imageURL = cleanText(playerImg?.getAttribute("src"));

    let country = "";

    card.querySelectorAll("img").forEach(img => {
      const alt = cleanText(img.getAttribute("alt"));

      if (alt.toLowerCase().startsWith("country")) {
        country = cleanText(alt.replace(/^Country:\s*/i, ""));
      }
    });

    const details = extractDetails(card);

    const row = {
      Player: player,
      First_Name: firstName,
      Last_Name: lastName,
      Team: teamInfo.Team,
      Team_ID: teamInfo.Team_ID,
      Team_Code: teamInfo.Team_Code,
      Division: teamInfo.Division,
      Position: normalizePosition(details["Position"]),
      Jersey: jersey,
      Handedness: cleanText(details["Hand"]),
      Height: cleanText(details["Height"]),
      Age: cleanText(details["Age"]),
      College: cleanText(details["College"]),
      Country: country,
      Image_Slug: imageSlug,
      Image_URL: imageURL,
      Page_URL: window.location.href,
      Page_Title: document.title,
      Extracted_At: new Date().toISOString()
    };

    if (row.Player && row.Position && row.Position !== "UNK") {
      rows.push(row);
    }
  });

  return {
    raw_card_count: cards.length,
    raw_valid_rows: rows.length,
    rows: rows
  };
}
"""


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


async def launch_browser(playwright):
    args = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-setuid-sandbox",
        "--disable-software-rasterizer",
    ]
    return await playwright.chromium.launch(headless=HEADLESS, args=args)


async def scrape_team_roster(page, team: Dict[str, object]) -> Tuple[List[dict], List[dict]]:
    print(f"\nSCRAPING {team['Team_Code']} / {team['Team_ID']} — {team['Team']}")

    diagnostics: List[dict] = []
    best_rows: List[dict] = []

    for url in team["URLs"]:  # type: ignore[index]
        print(f"Trying URL: {url}")

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

            try:
                await page.wait_for_selector("p.firstName", timeout=CARD_WAIT_TIMEOUT_MS)
            except Exception:
                print("  Warning: p.firstName selector did not appear before timeout.")

            for _ in range(SCROLL_PASSES):
                await page.mouse.wheel(0, 2500)
                await page.wait_for_timeout(SCROLL_PAUSE_MS)

            await page.mouse.wheel(0, -10000)
            await page.wait_for_timeout(1000)

            result = await page.evaluate(
                ROSTER_EXTRACTOR_JS,
                {
                    "Team": team["Team"],
                    "Team_ID": team["Team_ID"],
                    "Team_Code": team["Team_Code"],
                    "Division": team["Division"],
                },
            )

            raw_card_count = result.get("raw_card_count", 0)
            raw_valid_rows = result.get("raw_valid_rows", 0)
            rows = result.get("rows", [])
            deduped_rows = dedupe_team_rows(rows)

            print(f"  Raw card containers: {raw_card_count}")
            print(f"  Raw valid rows: {raw_valid_rows}")
            print(f"  Deduped players: {len(deduped_rows)}")

            diagnostics.append({
                "Team_ID": team["Team_ID"],
                "Team_Code": team["Team_Code"],
                "Team": team["Team"],
                "URL_Tried": url,
                "Final_URL": page.url,
                "Raw_Card_Containers": raw_card_count,
                "Raw_Valid_Rows": raw_valid_rows,
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


async def scrape_all_pll_rosters_async() -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright && python -m playwright install chromium"
        ) from exc

    all_rows: List[dict] = []
    all_diagnostics: List[dict] = []

    async with async_playwright() as p:
        browser = await launch_browser(p)
        context = await browser.new_context(
            viewport={"width": 1600, "height": 2400},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            for team in PLL_TEAMS:
                rows, diagnostics = await scrape_team_roster(page, team)
                all_rows.extend(rows)
                all_diagnostics.extend(diagnostics)
                await page.wait_for_timeout(1000)
        finally:
            await context.close()
            await browser.close()

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


def scrape_all_pll_rosters_sync() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the async scraper from normal sync code, including Streamlit contexts."""
    try:
        loop = asyncio.get_running_loop()
        running = loop.is_running()
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(scrape_all_pll_rosters_async())

    result: dict = {}

    def _runner():
        try:
            result["value"] = asyncio.run(scrape_all_pll_rosters_async())
        except BaseException as exc:  # pass exception to caller thread
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result["value"]


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
    if issues:
        raise RuntimeError("Roster scrape validation failed: " + ", ".join(issues))

    output_path = Path(output_path)
    diagnostics_path = Path(diagnostics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    roster_df.to_csv(output_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)

    print(f"Wrote {len(roster_df)} roster rows to {output_path}")
    print(f"Wrote diagnostics to {diagnostics_path}")
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
