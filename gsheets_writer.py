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


def save_snapshot(result, game: Dict, hold_pct: float, engine) -> str:
    """
    Write projection snapshot to a new tab in the master Google Sheet.
    Returns the tab name on success. Raises on failure.
    """
    gc       = _get_client()
    sh       = gc.open_by_key(_get_sheet_id())
    tab      = _tab_name(game)

    # Delete existing tab with same name if re-saving
    existing = next((ws for ws in sh.worksheets() if ws.title == tab), None)
    if existing:
        sh.del_worksheet(existing)

    rows = _build_sections(result, game, hold_pct, engine)
    n_rows = max(len(rows) + 5, 50)
    n_cols = 14

    ws = sh.add_worksheet(title=tab, rows=n_rows, cols=n_cols)

    # Write all data in one batch call
    ws.update(values=rows, range_name="A1")

    # Bold the section headers (rows where col B is empty and col A has text)
    header_rows = [i + 1 for i, r in enumerate(rows)
                   if len(r) == 1 and r[0]]
    if header_rows:
        fmt_reqs = []
        for hr in header_rows:
            fmt_reqs.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": hr - 1,
                        "endRowIndex": hr,
                        "startColumnIndex": 0,
                        "endColumnIndex": n_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"bold": True},
                            "backgroundColor": {"red": 0.13, "green": 0.23, "blue": 0.37},
                        }
                    },
                    "fields": "userEnteredFormat(textFormat,backgroundColor)",
                }
            })
        sh.batch_update({"requests": fmt_reqs})

    logger.info("Saved snapshot to tab: %s", tab)
    return tab


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
