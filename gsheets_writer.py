"""
Google Sheets writer for PLL projection snapshots.

Architecture:
- One master spreadsheet (PLL Projections 2026) lives in the shared Drive folder.
- Each saved game gets its own tab named: Away@Home_GameN_YYYY-MM-DD
- Within each tab, sections are stacked vertically separated by blank rows:
    Row 1:    Section header "METADATA"
    Rows 2+:  Metadata key/value pairs
    Gap
    Header "TEAM PROJECTIONS"
    Team rows
    Gap
    Header "GAME LINES"
    Lines rows
    Gap
    Header "PLAYER PROPS"
    Props rows (includes Actual Result and Hit/Miss columns for later fill-in)

The actuals sync reads completed game stats from the warehouse and writes
actual values back into the Actual Result column, then auto-computes Hit/Miss.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pll.sheets")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

STAT_LABELS = {
    "goals": "Goals", "assists": "Assists", "points": "Points",
    "shots_on_goal": "SOG", "saves": "Saves", "faceoff_wins": "FO Wins",
}


def _get_credentials():
    """Build Google credentials from Streamlit secrets."""
    import streamlit as st
    from google.oauth2.service_account import Credentials
    info = dict(st.secrets["gcp_service_account"])
    return Credentials.from_service_account_info(info, scopes=SCOPES)


def _get_client():
    """Return an authenticated gspread client."""
    import gspread
    return gspread.authorize(_get_credentials())


def _get_sheet_id() -> str:
    import streamlit as st
    return str(st.secrets["google_drive"]["projections_sheet_id"])


def _tab_name(game: Dict) -> str:
    """Generate a unique, readable tab name for a game."""
    from pages._engine_state import team_name
    away = team_name(str(game.get("away_team_id", game.get("away_team", "Away"))))
    home = team_name(str(game.get("home_team_id", game.get("home_team", "Home"))))
    gn   = game.get("game_number", "?")
    date = str(game.get("game_date", ""))[:10]
    return f"{away}@{home}_G{gn}_{date}"


def _build_sections(result, game: Dict, hold_pct: float, engine) -> List[List[Any]]:
    """
    Build all worksheet rows as a single flat list-of-lists.
    Sections separated by blank rows with bold headers.
    """
    from pages._engine_state import team_name
    from projection_engine_v3 import PricingEngine as _PE

    rows: List[List[Any]] = []

    def _header(text: str):
        rows.append([text])
        return rows

    def _blank():
        rows.append([])

    # ── METADATA ──────────────────────────────────────────────────────────────
    _header("METADATA")
    pm = engine.player_model
    fd = getattr(pm, "last_roster_filter_details", {}) or {}
    h_src = fd.get(result.home_proj.team_id, {}).get("reason", "unknown")
    a_src = fd.get(result.away_proj.team_id, {}).get("reason", "unknown")
    for field, val in [
        ("Game",         f"{team_name(result.away_proj.team_id)} @ {team_name(result.home_proj.team_id)}"),
        ("Game Number",  game.get("game_number", "")),
        ("Game Date",    str(game.get("game_date", ""))[:10]),
        ("Home Team",    team_name(result.home_proj.team_id)),
        ("Away Team",    team_name(result.away_proj.team_id)),
        ("Saved At",     _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
        ("Hold %",       f"{hold_pct*100:.1f}%"),
        ("Sims",         result.game_sim.n_sims),
        ("Home Roster",  h_src),
        ("Away Roster",  a_src),
        ("Model",        result.home_proj.model_used),
    ]:
        rows.append([field, str(val)])

    _blank()
    _blank()

    # ── TEAM PROJECTIONS ──────────────────────────────────────────────────────
    _header("TEAM PROJECTIONS")
    rows.append(["Team", "Goals", "Score", "Shots", "SOG", "FO%", "FO Wins",
                 "Assists", "Saves", "Save%", "2PT Goals", "TOs", "GBs",
                 "Actual Goals", "Actual Score"])
    for proj in [result.home_proj, result.away_proj]:
        rows.append([
            team_name(proj.team_id),
            round(proj.proj_goals, 2), round(proj.proj_scores, 2),
            round(proj.proj_shots, 1), round(proj.proj_sog, 1),
            round(proj.proj_faceoff_pct, 3), round(proj.proj_faceoff_wins, 1),
            round(proj.proj_assists, 1), round(proj.proj_saves, 1),
            round(proj.proj_save_pct, 3), round(proj.proj_2pt_goals, 2),
            round(proj.proj_turnovers, 1), round(proj.proj_ground_balls, 1),
            "", "",  # Actual Goals, Actual Score — filled in by sync
        ])

    _blank()
    _blank()

    # ── GAME LINES ────────────────────────────────────────────────────────────
    _header("GAME LINES")
    rows.append(["Market", "Line", "Odds", "Fair Prob"])
    gs = result.game_sim
    gm = result.game_market
    home_tt = _PE._force_half_only(float(np.median(gs.home_scores)))
    away_tt = _PE._force_half_only(float(np.median(gs.away_scores)))
    for market, line, odds, fair in [
        (f"{team_name(result.away_proj.team_id)} ML", "--", gm.away_ml, f"{gm.away_win_prob*100:.1f}%"),
        (f"{team_name(result.home_proj.team_id)} ML", "--", gm.home_ml, f"{gm.home_win_prob*100:.1f}%"),
        (f"{team_name(result.away_proj.team_id)} Spread", f"{gm.spread_home:+.1f}", gm.spread_away_odds, "--"),
        (f"{team_name(result.home_proj.team_id)} Spread", f"{-gm.spread_home:+.1f}", gm.spread_home_odds, "--"),
        ("Total Over",  f"{gm.total_line:.1f}", gm.over_odds,  f"{float(np.mean(gs.total_distribution > gm.total_line))*100:.1f}%"),
        ("Total Under", f"{gm.total_line:.1f}", gm.under_odds, f"{float(np.mean(gs.total_distribution <= gm.total_line))*100:.1f}%"),
        (f"{team_name(result.home_proj.team_id)} Team Total O", f"{home_tt:.1f}", "--", f"{float(np.mean(gs.home_scores > home_tt))*100:.1f}%"),
        (f"{team_name(result.home_proj.team_id)} Team Total U", f"{home_tt:.1f}", "--", f"{float(np.mean(gs.home_scores <= home_tt))*100:.1f}%"),
        (f"{team_name(result.away_proj.team_id)} Team Total O", f"{away_tt:.1f}", "--", f"{float(np.mean(gs.away_scores > away_tt))*100:.1f}%"),
        (f"{team_name(result.away_proj.team_id)} Team Total U", f"{away_tt:.1f}", "--", f"{float(np.mean(gs.away_scores <= away_tt))*100:.1f}%"),
    ]:
        rows.append([market, line, str(odds), fair])

    _blank()
    _blank()

    # ── PLAYER PROPS ──────────────────────────────────────────────────────────
    _header("PLAYER PROPS")
    rows.append([
        "Player", "Team", "Pos", "Stat", "Projection",
        "Main Line", "Over Odds", "Under Odds", "Fair P(Over)",
        "P10", "P50", "P90",
        "Actual Result", "Hit/Miss",   # filled in by actuals sync
    ])

    all_players = {p.player_id: p for p in result.home_players + result.away_players}
    markets     = result.player_markets
    sims_all    = result.home_player_sims + result.away_player_sims

    prop_rows = []
    for ps in sims_all:
        proj = all_players.get(ps.player_id)
        if proj is None or not proj.active:
            continue
        pm_data = markets.get(ps.player_id, {})
        pv  = pm_data.get("proj_values", {})
        ms  = pm_data.get("markets", {})
        stats = (["saves"] if proj.position == "G"
                 else ["faceoff_wins"] if proj.position == "FO"
                 else ["goals", "assists", "points", "shots_on_goal"])
        for stat in stats:
            if stat not in ps.stat_distributions:
                continue
            mkt = ms.get(stat, {})
            proj_val = round(float(pv.get(stat, 0)), 3)
            if proj_val < 0.05 and proj.position not in ("G", "FO"):
                continue
            dist = ps.stat_distributions[stat]
            prop_rows.append([
                proj.full_name or proj.player_id,
                team_name(proj.team_id),
                proj.position,
                STAT_LABELS.get(stat, stat),
                proj_val,
                mkt.get("line", ""),
                str(mkt.get("over_odds", "")),
                str(mkt.get("under_odds", "")),
                round(float(mkt.get("fair_over_prob", 0)), 3),
                round(float(np.percentile(dist, 10)), 2),
                round(float(np.percentile(dist, 50)), 2),
                round(float(np.percentile(dist, 90)), 2),
                "",  # Actual Result
                "",  # Hit/Miss
            ])

    prop_rows.sort(key=lambda r: (r[1], r[2], r[0], r[3]))
    rows.extend(prop_rows)

    return rows


def _rgb(r: int, g: int, b: int) -> Dict:
    return {"red": r / 255, "green": g / 255, "blue": b / 255}


def _range(sid: int, r0: int, r1: int, c0: int, c1: int) -> Dict:
    return {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1,
            "startColumnIndex": c0, "endColumnIndex": c1}


def _repeat(sid: int, r0: int, r1: int, c0: int, c1: int,
            fmt: Dict, fields: str) -> Dict:
    return {"repeatCell": {
        "range": _range(sid, r0, r1, c0, c1),
        "cell": {"userEnteredFormat": fmt},
        "fields": fields,
    }}


def save_snapshot(result, game: Dict, hold_pct: float, engine) -> str:
    """
    Write projection snapshot to a new tab in the master Google Sheet.
    Returns the tab name on success. Raises on failure.
    """
    gc  = _get_client()
    sh  = gc.open_by_key(_get_sheet_id())
    tab = _tab_name(game)

    # Delete existing tab with same name if re-saving
    existing = next((ws for ws in sh.worksheets() if ws.title == tab), None)
    if existing:
        sh.del_worksheet(existing)

    rows   = _build_sections(result, game, hold_pct, engine)
    n_rows = max(len(rows) + 5, 50)
    n_cols = 15   # A–O: 14 data cols + Actual Score in col O

    ws = sh.add_worksheet(title=tab, rows=n_rows, cols=n_cols)
    sid = ws.id

    # Write all data in one batch call
    ws.update(values=rows, range_name="A1")

    # ── Identify key row indices (0-indexed) ─────────────────────────────────
    section_rows: Dict[str, int] = {}   # section name → row index of header row
    col_header_rows: List[int]   = []   # row indices of column header rows (bold)
    data_bands: List[tuple]      = []   # (start_row, end_row) of data bands for zebra

    SECTIONS = {"METADATA", "TEAM PROJECTIONS", "GAME LINES", "PLAYER PROPS"}
    i = 0
    while i < len(rows):
        r = rows[i]
        cell0 = r[0].strip().upper() if r else ""
        if cell0 in SECTIONS and len(r) == 1:
            section_rows[cell0] = i
            # Next non-empty row is the column header
            j = i + 1
            while j < len(rows) and not any(rows[j]):
                j += 1
            if j < len(rows) and any(rows[j]):
                col_header_rows.append(j)
                # Collect data rows after the column header
                k = j + 1
                band_start = k
                while k < len(rows):
                    if not any(rows[k]):
                        break
                    k += 1
                if k > band_start:
                    data_bands.append((band_start, k))
        i += 1

    reqs = []

    # ── 1. Default cell style for entire sheet ───────────────────────────────
    reqs.append(_repeat(sid, 0, n_rows, 0, n_cols, {
        "textFormat": {"fontFamily": "Arial", "fontSize": 10},
        "verticalAlignment": "MIDDLE",
        "wrapStrategy": "CLIP",
    }, "userEnteredFormat(textFormat,verticalAlignment,wrapStrategy)"))

    # ── 2. Section header rows — dark navy background, white bold text ───────
    NAVY   = _rgb(23, 37, 63)
    WHITE  = _rgb(255, 255, 255)
    for ri in section_rows.values():
        reqs.append(_repeat(sid, ri, ri + 1, 0, n_cols, {
            "backgroundColor": NAVY,
            "textFormat": {"bold": True, "fontSize": 11,
                           "foregroundColor": WHITE, "fontFamily": "Arial"},
        }, "userEnteredFormat(backgroundColor,textFormat)"))

    # ── 3. Column header rows — medium blue, white bold text ─────────────────
    MED_BLUE = _rgb(37, 77, 130)
    for ri in col_header_rows:
        reqs.append(_repeat(sid, ri, ri + 1, 0, n_cols, {
            "backgroundColor": MED_BLUE,
            "textFormat": {"bold": True, "foregroundColor": WHITE,
                           "fontSize": 10, "fontFamily": "Arial"},
            "horizontalAlignment": "CENTER",
        }, "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"))

    # ── 4. Alternating row shading for data bands ────────────────────────────
    LIGHT_BLUE = _rgb(235, 242, 252)
    WHITE_BG   = _rgb(255, 255, 255)
    for band_start, band_end in data_bands:
        for ri in range(band_start, band_end):
            bg = LIGHT_BLUE if (ri - band_start) % 2 == 1 else WHITE_BG
            reqs.append(_repeat(sid, ri, ri + 1, 0, n_cols, {
                "backgroundColor": bg,
            }, "userEnteredFormat(backgroundColor)"))

    # ── 5. Freeze first row and set row height for section headers ───────────
    reqs.append({"updateSheetProperties": {
        "properties": {
            "sheetId": sid,
            "gridProperties": {"frozenRowCount": 0},
        },
        "fields": "gridProperties.frozenRowCount",
    }})

    # Set section header rows to height 28px, col headers to 24px
    for ri in section_rows.values():
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": sid, "dimension": "ROWS",
                      "startIndex": ri, "endIndex": ri + 1},
            "properties": {"pixelSize": 28},
            "fields": "pixelSize",
        }})
    for ri in col_header_rows:
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": sid, "dimension": "ROWS",
                      "startIndex": ri, "endIndex": ri + 1},
            "properties": {"pixelSize": 24},
            "fields": "pixelSize",
        }})

    # ── 6. Column widths — fixed for A and B, autofit deferred for C–O ─────────
    # Col A: fixed 180px for long player names
    # Col B: fixed 120px for team names / value labels
    # Cols C–O: autofit fired in a second batch after data is committed
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1},
        "properties": {"pixelSize": 180}, "fields": "pixelSize",
    }})
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2},
        "properties": {"pixelSize": 120}, "fields": "pixelSize",
    }})

    # ── 7. Number formatting for numeric data columns ─────────────────────────
    # Find Player Props data band and apply number formats
    if "PLAYER PROPS" in section_rows and len(data_bands) >= 1:
        pp_section_row = section_rows["PLAYER PROPS"]
        # Find the data band that starts after the Player Props section
        pp_band = next(
            ((s, e) for s, e in data_bands if s > pp_section_row), None
        )
        if pp_band:
            bs, be = pp_band
            # Projection, P10, P50, P90 cols (4, 9, 10, 11 — 0-indexed)
            for ci in [4, 9, 10, 11]:
                reqs.append(_repeat(sid, bs, be, ci, ci + 1, {
                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"},
                }, "userEnteredFormat.numberFormat"))
            # Fair P(Over) col (8) — percentage
            reqs.append(_repeat(sid, bs, be, 8, 9, {
                "numberFormat": {"type": "NUMBER", "pattern": "0.0%"},
            }, "userEnteredFormat.numberFormat"))

    # ── 8. Merge B2:D2 — matchup title spans three columns ───────────────────
    # Only merge — do NOT touch textFormat or alignment so it keeps the same
    # font/size/color as the rest of the metadata and stays left-aligned.
    reqs.append({"mergeCells": {
        "range": _range(sid, 1, 2, 1, 4),
        "mergeType": "MERGE_ALL",
    }})

    # ── 9. Centering — ALL columns in data bands, ALL columns in headers ─────
    # Every data cell (A through O) is centered. This covers:
    #   Team Projections: col A (Atlas/Waterdogs), cols B-O (all numbers)
    #   Game Lines: col A (Market), col B (Line), col C (Odds), col D (Fair Prob)
    #   Player Props: col A (Player name), col B (Team), col C+ (all values)
    # Col A player names are intentionally centered — looks cleaner in a table.
    for band_start, band_end in data_bands:
        reqs.append(_repeat(sid, band_start, band_end, 0, n_cols, {
            "horizontalAlignment": "CENTER",
        }, "userEnteredFormat(horizontalAlignment)"))
        # Override col A in Player Props to left-align long player names only
        # Player Props is the last data band — detect it by being the largest band
    # Column header rows — center all columns
    for ri in col_header_rows:
        reqs.append(_repeat(sid, ri, ri + 1, 0, n_cols, {
            "horizontalAlignment": "CENTER",
        }, "userEnteredFormat(horizontalAlignment)"))

    # Player names (col A in the largest/last data band) left-aligned for readability
    if data_bands:
        pp_start, pp_end = max(data_bands, key=lambda b: b[1] - b[0])
        reqs.append(_repeat(sid, pp_start, pp_end, 0, 1, {
            "horizontalAlignment": "LEFT",
        }, "userEnteredFormat(horizontalAlignment)"))

    # Metadata value column (col B, first/smallest data band) — left-aligned.
    # These are text labels like team names, dates, roster sources, not numbers.
    if data_bands:
        meta_start, meta_end = min(data_bands, key=lambda b: b[0])
        reqs.append(_repeat(sid, meta_start, meta_end, 1, 2, {
            "horizontalAlignment": "LEFT",
        }, "userEnteredFormat(horizontalAlignment)"))

    # ── 10. Conditional formatting: Hit/Miss column ──────────────────────────
    # Hit = green, Miss = red — applied to col 13 (N) across the whole sheet
    GREEN_BG  = _rgb(198, 239, 206)
    GREEN_FG  = _rgb(0, 97, 0)
    RED_BG    = _rgb(255, 199, 206)
    RED_FG    = _rgb(156, 0, 6)

    for band_start, band_end in data_bands:
        reqs.append({"addConditionalFormatRule": {
            "rule": {
                "ranges": [_range(sid, band_start, band_end, 13, 14)],
                "booleanRule": {
                    "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": "Hit"}]},
                    "format": {"backgroundColor": GREEN_BG,
                               "textFormat": {"foregroundColor": GREEN_FG, "bold": True}},
                },
            },
            "index": 0,
        }})
        reqs.append({"addConditionalFormatRule": {
            "rule": {
                "ranges": [_range(sid, band_start, band_end, 13, 14)],
                "booleanRule": {
                    "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": "Miss"}]},
                    "format": {"backgroundColor": RED_BG,
                               "textFormat": {"foregroundColor": RED_FG, "bold": True}},
                },
            },
            "index": 1,
        }})

    # ── 9. Conditional formatting: Fair P(Over) — green if >55%, red if <45% ─
    for band_start, band_end in data_bands:
        reqs.append({"addConditionalFormatRule": {
            "rule": {
                "ranges": [_range(sid, band_start, band_end, 8, 9)],
                "booleanRule": {
                    "condition": {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "0.55"}]},
                    "format": {"backgroundColor": GREEN_BG},
                },
            },
            "index": 2,
        }})
        reqs.append({"addConditionalFormatRule": {
            "rule": {
                "ranges": [_range(sid, band_start, band_end, 8, 9)],
                "booleanRule": {
                    "condition": {"type": "NUMBER_LESS", "values": [{"userEnteredValue": "0.45"}]},
                    "format": {"backgroundColor": RED_BG},
                },
            },
            "index": 3,
        }})

    # ── 10. Tab colour — dark blue to match the navy headers ─────────────────
    reqs.append({"updateSheetProperties": {
        "properties": {
            "sheetId": sid,
            "tabColorStyle": {"rgbColor": _rgb(23, 37, 63)},
        },
        "fields": "tabColorStyle",
    }})

    # ── Fire all formatting requests ─────────────────────────────────────────
    if reqs:
        sh.batch_update({"requests": reqs})

    # ── Add filter on Player Props column headers ─────────────────────────────
    # Must come BEFORE autofit so the dropdown arrows are present when column
    # widths are measured — otherwise the arrow clips the header text.
    if "PLAYER PROPS" in section_rows:
        pp_hdr_idx = next(
            (r for r in col_header_rows if r > section_rows["PLAYER PROPS"]), None
        )
        if pp_hdr_idx is not None and data_bands:
            pp_band = next(
                ((s, e) for s, e in data_bands if s > section_rows["PLAYER PROPS"]), None
            )
            if pp_band:
                pp_start, pp_end = pp_band
                sh.batch_update({"requests": [{
                    "setBasicFilter": {
                        "filter": {
                            "range": _range(sid, pp_hdr_idx, pp_end, 0, n_cols),
                        }
                    }
                }]})

    # ── Autofit cols C–O — runs last so filter arrows are already rendered ────
    sh.batch_update({"requests": [{
        "autoResizeDimensions": {
            "dimensions": {"sheetId": sid, "dimension": "COLUMNS",
                           "startIndex": 2, "endIndex": n_cols},
        }
    }]})

    # ── Update Sheet1 dashboard ───────────────────────────────────────────────
    _update_dashboard(sh, tab)

    logger.info("Saved snapshot to tab: %s", tab)
    return tab


def _update_dashboard(sh, latest_tab: str) -> None:
    """
    Rewrite Sheet1 as a dashboard showing all saved games and instructions.
    Called every time a snapshot is saved so the index stays current.
    """
    try:
        ws = next((w for w in sh.worksheets()
                   if w.title in ("Sheet1", "Overview")), None)
        if ws is None:
            ws = sh.add_worksheet(title="Overview", rows=100, cols=10)
        elif ws.title == "Sheet1":
            ws.update_title("Overview")
        sid = ws.id

        # ── Collect all game tabs ─────────────────────────────────────────────
        games = []
        for w in sh.worksheets():
            t = w.title
            if "@" not in t or "_G" not in t:
                continue
            try:
                matchup, rest = t.split("_G", 1)
                gn, date = rest.split("_", 1)
                away, home = matchup.split("@", 1)
                # Check if actuals have been synced by scanning col N (Actual Result).
                # Col N is index 14 (1-indexed) in the sheet. Skip the header value
                # itself and only count non-empty numeric-looking values.
                vals = w.col_values(14)   # col 14 = N = Actual Result (1-indexed)
                has_actuals = any(
                    v.strip() not in ("", "Actual Result")
                    and v.strip().lstrip("-").replace(".", "", 1).isdigit()
                    for v in vals
                )
                games.append({
                    "tab": t, "away": away, "home": home,
                    "gn": gn, "date": date, "actuals": has_actuals,
                })
            except Exception:
                continue
        games.sort(key=lambda g: g["date"], reverse=True)

        # ── Build dashboard rows ──────────────────────────────────────────────
        rows: List[List[Any]] = []

        # Title block
        rows.append(["PLL PROJECTION SNAPSHOTS"])
        rows.append(["Master tracking sheet for all game projections and actuals"])
        rows.append([f"Last updated: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"])
        rows.append(["Service account:", "pll-projections-writer@pll-projections.iam.gserviceaccount.com"])
        rows.append([])

        # Game index
        rows.append(["SAVED GAMES"])
        rows.append(["Game Date", "Matchup", "Game #", "Tab Name", "Actuals Synced"])
        for g in games:
            rows.append([
                g["date"],
                f"{g['away']} @ {g['home']}",
                g["gn"],
                g["tab"],
                "✓ Yes" if g["actuals"] else "Pending",
            ])
        if not games:
            rows.append(["No games saved yet.", "", "", "", ""])
        rows.append([])
        rows.append([])

        # Instructions
        rows.append(["HOW TO USE"])
        rows.append(["Step", "Action", "Details"])
        rows.append(["1", "Run projection", "Go to the PLL app, select a game, run the projection"])
        rows.append(["2", "Save snapshot", "Click ☁️ Save to Google Sheets — creates a new tab here"])
        rows.append(["3", "Sync actuals", "After the game, click 🔄 Sync Actuals to fill in real results"])
        rows.append(["4", "Review history", "Go to Page 5 (Projection History) in the app for model accuracy stats"])
        rows.append([])
        rows.append(["COLUMN GUIDE — PLAYER PROPS TAB"])
        rows.append(["Column", "Description"])
        rows.append(["Projection", "Model's expected value for this stat"])
        rows.append(["Main Line", "Balanced prop line (x.5)"])
        rows.append(["Fair P(Over)", "Model's true probability of going over — green >55%, red <45%"])
        rows.append(["P10 / P50 / P90", "10th / 50th / 90th percentile from 20,000 simulations"])
        rows.append(["Actual Result", "Filled in automatically by Sync Actuals after the game"])
        rows.append(["Hit/Miss", "Green = Hit (actual ≥ line), Red = Miss"])

        # Write all rows — clear values AND all existing formatting first so
        # stale formatting from prior versions doesn't bleed through.
        ws.clear()
        sh.batch_update({"requests": [{
            "updateCells": {
                "range": {"sheetId": sid,
                          "startRowIndex": 0, "endRowIndex": 100,
                          "startColumnIndex": 0, "endColumnIndex": 10},
                "fields": "userEnteredFormat",
            }
        }]})
        ws.update(values=rows, range_name="A1")

        # ── Format dashboard ──────────────────────────────────────────────────
        NAVY     = _rgb(23, 37, 63)
        MED_BLUE = _rgb(37, 77, 130)
        WHITE    = _rgb(255, 255, 255)
        GREEN_BG = _rgb(198, 239, 206)
        GREEN_FG = _rgb(0, 97, 0)
        AMBER_BG = _rgb(255, 243, 205)
        AMBER_FG = _rgb(133, 77, 14)
        LIGHT    = _rgb(235, 242, 252)

        # Find row indices for section headers
        title_row   = 0
        games_hdr   = next((i for i, r in enumerate(rows) if r and r[0] == "SAVED GAMES"), None)
        games_col_hdr = games_hdr + 1 if games_hdr is not None else None
        games_data_start = games_col_hdr + 1 if games_col_hdr is not None else None
        games_data_end   = games_data_start + len(games) if games_data_start is not None else None
        how_hdr     = next((i for i, r in enumerate(rows) if r and r[0] == "HOW TO USE"), None)
        how_col_hdr = how_hdr + 1 if how_hdr is not None else None
        col_guide   = next((i for i, r in enumerate(rows) if r and r[0] == "COLUMN GUIDE — PLAYER PROPS TAB"), None)
        col_guide_hdr = col_guide + 1 if col_guide is not None else None

        reqs = []

        BLACK = _rgb(0, 0, 0)

        # Step 1 — font/alignment baseline for entire sheet (no color set here)
        reqs.append(_repeat(sid, 0, len(rows) + 5, 0, 10, {
            "textFormat": {"fontFamily": "Arial", "fontSize": 10},
            "verticalAlignment": "MIDDLE",
        }, "userEnteredFormat(textFormat,verticalAlignment)"))

        # Step 2 — explicitly black text on all rows so data rows are readable
        reqs.append(_repeat(sid, 0, len(rows) + 5, 0, 10, {
            "textFormat": {"foregroundColor": BLACK},
        }, "userEnteredFormat.textFormat.foregroundColor"))

        # Step 3 — coloured header rows override foreground to white AFTER step 2
        # Title row — large navy, white text
        reqs.append(_repeat(sid, title_row, title_row + 1, 0, 10, {
            "backgroundColor": NAVY,
            "textFormat": {"bold": True, "fontSize": 14,
                           "foregroundColor": WHITE, "fontFamily": "Arial"},
        }, "userEnteredFormat(backgroundColor,textFormat)"))

        # Subtitle rows (rows 1–3) — medium blue, white text
        reqs.append(_repeat(sid, 1, 4, 0, 10, {
            "backgroundColor": MED_BLUE,
            "textFormat": {"foregroundColor": WHITE, "fontSize": 10,
                           "fontFamily": "Arial", "bold": False},
        }, "userEnteredFormat(backgroundColor,textFormat)"))

        # Section headers (SAVED GAMES, HOW TO USE, COLUMN GUIDE) — navy, white text
        for ri in [r for r in [games_hdr, how_hdr, col_guide] if r is not None]:
            reqs.append(_repeat(sid, ri, ri + 1, 0, 10, {
                "backgroundColor": NAVY,
                "textFormat": {"bold": True, "fontSize": 11,
                               "foregroundColor": WHITE, "fontFamily": "Arial"},
            }, "userEnteredFormat(backgroundColor,textFormat)"))

        # Column header rows — medium blue, white text, centered
        for ri in [r for r in [games_col_hdr, how_col_hdr, col_guide_hdr] if r is not None]:
            reqs.append(_repeat(sid, ri, ri + 1, 0, 10, {
                "backgroundColor": MED_BLUE,
                "textFormat": {"bold": True, "foregroundColor": WHITE,
                               "fontSize": 10, "fontFamily": "Arial"},
                "horizontalAlignment": "CENTER",
            }, "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"))

        # Game index data rows — zebra + conditional for actuals column
        if games_data_start is not None and games_data_end is not None:
            for ri in range(games_data_start, games_data_end):
                bg = LIGHT if (ri - games_data_start) % 2 == 1 else WHITE
                reqs.append(_repeat(sid, ri, ri + 1, 0, 10, {
                    "backgroundColor": bg,
                }, "userEnteredFormat(backgroundColor)"))
            # Conditional: "✓ Yes" in col E (index 4) → green
            reqs.append({"addConditionalFormatRule": {
                "rule": {
                    "ranges": [_range(sid, games_data_start, games_data_end, 4, 5)],
                    "booleanRule": {
                        "condition": {"type": "TEXT_CONTAINS",
                                      "values": [{"userEnteredValue": "Yes"}]},
                        "format": {"backgroundColor": GREEN_BG,
                                   "textFormat": {"foregroundColor": GREEN_FG, "bold": True}},
                    },
                },
                "index": 0,
            }})
            reqs.append({"addConditionalFormatRule": {
                "rule": {
                    "ranges": [_range(sid, games_data_start, games_data_end, 4, 5)],
                    "booleanRule": {
                        "condition": {"type": "TEXT_CONTAINS",
                                      "values": [{"userEnteredValue": "Pending"}]},
                        "format": {"backgroundColor": AMBER_BG,
                                   "textFormat": {"foregroundColor": AMBER_FG, "bold": True}},
                    },
                },
                "index": 1,
            }})

        # Tab colour — match game tabs
        reqs.append({"updateSheetProperties": {
            "properties": {"sheetId": sid,
                           "tabColorStyle": {"rgbColor": _rgb(23, 37, 63)}},
            "fields": "tabColorStyle",
        }})

        sh.batch_update({"requests": reqs})

        # Autofit all columns
        sh.batch_update({"requests": [{
            "autoResizeDimensions": {
                "dimensions": {"sheetId": sid, "dimension": "COLUMNS",
                               "startIndex": 0, "endIndex": 10},
            }
        }]})

        logger.info("Dashboard updated on Sheet1")
    except Exception as e:
        logger.warning("Dashboard update failed (non-fatal): %s", e)


def list_saved_games() -> List[Dict]:
    """
    Return a list of saved game tabs as dicts with keys:
    tab_name, away, home, game_number, game_date
    """
    try:
        gc = _get_client()
        sh = gc.open_by_key(_get_sheet_id())
        games = []
        for ws in sh.worksheets():
            t = ws.title
            # Skip the default Sheet1 or any non-game tabs
            if "@" not in t or "_G" not in t:
                continue
            try:
                matchup, rest = t.split("_G", 1)
                gn, date = rest.split("_", 1)
                away, home = matchup.split("@", 1)
                games.append({
                    "tab_name":    t,
                    "away":        away,
                    "home":        home,
                    "game_number": gn,
                    "game_date":   date,
                    "sheet_id":    ws.id,
                })
            except Exception:
                continue
        return sorted(games, key=lambda g: g["game_date"], reverse=True)
    except Exception as e:
        logger.warning("list_saved_games failed: %s", e)
        return []


def read_game_tab(tab_name: str) -> Dict[str, pd.DataFrame]:
    """
    Read a saved game tab and return a dict of DataFrames:
    {metadata, team_projections, game_lines, player_props}
    """
    gc = _get_client()
    sh = gc.open_by_key(_get_sheet_id())
    ws = sh.worksheet(tab_name)
    all_vals = ws.get_all_values()

    sections: Dict[str, List[List]] = {}
    current_section = None
    current_rows: List[List] = []

    SECTION_HEADERS = {"METADATA", "TEAM PROJECTIONS", "GAME LINES", "PLAYER PROPS"}

    for row in all_vals:
        if not any(c.strip() for c in row):
            if current_section and current_rows:
                sections[current_section] = current_rows
                current_rows = []
                current_section = None
            continue
        cell0 = row[0].strip().upper() if row else ""
        if cell0 in SECTION_HEADERS and not any(row[1:]):
            if current_section and current_rows:
                sections[current_section] = current_rows
            current_section = cell0
            current_rows = []
        elif current_section:
            current_rows.append(row)

    if current_section and current_rows:
        sections[current_section] = current_rows

    result = {}
    for key, raw in sections.items():
        if not raw:
            continue
        header = raw[0]
        data   = raw[1:]
        df = pd.DataFrame(data, columns=header)
        df.columns = [str(c).strip() for c in df.columns]
        result[key.lower().replace(" ", "_")] = df

    return result


def sync_actuals(tab_name: str, db_path: str) -> Dict[str, int]:
    """
    For a completed game, pull actual stats from the warehouse and
    write them into the Actual Result column of the Player Props section.
    Also fills Actual Goals / Actual Score for team rows.

    Returns {"players_updated": N, "teams_updated": N}
    """
    import duckdb

    gc = _get_client()
    sh = gc.open_by_key(_get_sheet_id())
    ws = sh.worksheet(tab_name)
    all_vals = ws.get_all_values()

    # Parse game date and team names from tab name to find the game_id
    try:
        matchup, rest = tab_name.split("_G", 1)
        _, date_str = rest.split("_", 1)
        away_name, home_name = matchup.split("@", 1)
    except Exception as e:
        raise ValueError(f"Cannot parse tab name '{tab_name}': {e}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        # Find the game_id from the schedule
        game_row = con.execute("""
            SELECT game_id, home_team_id, away_team_id
            FROM clean.game_schedule_all
            WHERE CAST(game_date AS VARCHAR) LIKE ?
              AND LOWER(COALESCE(event_status_label,'')) IN ('final','completed')
            LIMIT 1
        """, [f"{date_str}%"]).fetchone()

        if not game_row:
            raise ValueError(f"No completed game found for date {date_str}. "
                             "Run the data pipeline first to ingest actuals.")

        game_id = game_row[0]

        # Pull player actuals
        player_actuals = con.execute("""
            SELECT full_name, position,
                   goals, assists, goals+assists AS points,
                   shots_on_goal, saves, faceoffs_won
            FROM clean.player_game_stats
            WHERE game_id = ?
        """, [game_id]).df()

        # Pull team actuals
        team_actuals = con.execute("""
            SELECT team_id, goals, scores
            FROM clean.team_game_stats
            WHERE game_id = ?
        """, [game_id]).df()

    finally:
        con.close()

    # Build lookup: player_name → {stat: actual}
    player_lookup: Dict[str, Dict[str, float]] = {}
    stat_col_map = {
        "Goals": "goals", "Assists": "assists", "Points": "points",
        "SOG": "shots_on_goal", "Saves": "saves", "FO Wins": "faceoffs_won",
    }
    for _, row in player_actuals.iterrows():
        name = str(row["full_name"]).strip()
        player_lookup[name] = {
            "Goals":    float(row.get("goals", 0) or 0),
            "Assists":  float(row.get("assists", 0) or 0),
            "Points":   float(row.get("points", 0) or 0),
            "SOG":      float(row.get("shots_on_goal", 0) or 0),
            "Saves":    float(row.get("saves", 0) or 0),
            "FO Wins":  float(row.get("faceoffs_won", 0) or 0),
        }

    # Find column indices in spreadsheet
    PROP_HEADER    = ["Player", "Team", "Pos", "Stat", "Projection",
                      "Main Line", "Over Odds", "Under Odds", "Fair P(Over)",
                      "P10", "P50", "P90", "Actual Result", "Hit/Miss"]
    TEAM_HEADER    = ["Team", "Goals", "Score", "Shots", "SOG", "FO%", "FO Wins",
                      "Assists", "Saves", "Save%", "2PT Goals", "TOs", "GBs",
                      "Actual Goals", "Actual Score"]

    updates = []  # list of (row_1indexed, col_1indexed, value)
    players_updated = 0
    teams_updated   = 0

    in_props_section  = False
    in_teams_section  = False
    props_header_row  = None
    teams_header_row  = None

    for i, row in enumerate(all_vals):
        cell0 = row[0].strip().upper() if row else ""

        if cell0 == "PLAYER PROPS" and not any(c for c in row[1:] if c):
            in_props_section = True
            in_teams_section = False
            props_header_row = None
            continue

        if cell0 == "TEAM PROJECTIONS" and not any(c for c in row[1:] if c):
            in_teams_section = True
            in_props_section = False
            teams_header_row = None
            continue

        if not any(c.strip() for c in row):
            in_props_section = False
            in_teams_section = False
            continue

        if in_props_section:
            if props_header_row is None:
                props_header_row = i
                continue
            # Data row — cols: Player(0) Stat(3) Projection(4) MainLine(5) Actual(12) Hit(13)
            if len(row) < 13:
                continue
            player_name = row[0].strip()
            stat_label  = row[3].strip()
            line_val    = row[5].strip()
            actual_col  = PROP_HEADER.index("Actual Result") + 1   # 1-indexed
            hit_col     = PROP_HEADER.index("Hit/Miss") + 1

            actuals_for_player = player_lookup.get(player_name, {})
            actual = actuals_for_player.get(stat_label)
            if actual is not None:
                updates.append((i + 1, actual_col, actual))
                # Hit/Miss: actual >= line
                try:
                    line_f = float(line_val)
                    hit = "Hit" if actual >= line_f else "Miss"
                except (ValueError, TypeError):
                    hit = ""
                updates.append((i + 1, hit_col, hit))
                players_updated += 1

        if in_teams_section:
            if teams_header_row is None:
                teams_header_row = i
                continue
            if len(row) < 2:
                continue
            team_name_cell = row[0].strip()
            actual_g_col   = TEAM_HEADER.index("Actual Goals") + 1
            actual_s_col   = TEAM_HEADER.index("Actual Score") + 1
            for _, trow in team_actuals.iterrows():
                from pages._engine_state import team_name as _tn
                if _tn(str(trow["team_id"])).lower() == team_name_cell.lower():
                    updates.append((i + 1, actual_g_col, float(trow.get("goals", 0) or 0)))
                    updates.append((i + 1, actual_s_col, float(trow.get("scores", 0) or 0)))
                    teams_updated += 1
                    break

    # Batch write all updates
    if updates:
        cell_list = []
        for r, c, v in updates:
            cell = ws.cell(r, c)
            cell.value = v
            cell_list.append(cell)
        ws.update_cells(cell_list)

    logger.info("Synced actuals for %s: %d player rows, %d team rows",
                tab_name, players_updated, teams_updated)
    return {"players_updated": players_updated, "teams_updated": teams_updated}
