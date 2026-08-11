"""Extract PLL projection snapshots and player-prop P&L into tidy frames.

The projections workbook stores one sheet per game, each a *report* rather than a
table: stacked METADATA / TEAM PROJECTIONS / GAME LINES / PLAYER PROPS blocks with
their own header rows. This locates each block by its marker text instead of by
fixed row offsets, so a sheet with an extra metadata row or a different number of
players still parses — and any sheet that does not match is reported rather than
silently skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
PROJECTIONS_FILE = DOWNLOADS / "PLL Projections 2026.xlsx"
RESULTS_FILE = DOWNLOADS / "PLL Player Props Results.xlsx"
OUT_DIR = Path(__file__).parent / "data"

PLAYER_PROP_MARKER = "PLAYER PROPS"
TEAM_MARKER = "TEAM PROJECTIONS"
LINES_MARKER = "GAME LINES"


def _find_marker(frame: pd.DataFrame, marker: str) -> int | None:
    """Row index of the block header whose first cell equals ``marker``."""
    first = frame[0].astype(str).str.strip()
    hits = frame.index[first == marker].tolist()
    return int(hits[0]) if hits else None


def _block(
    frame: pd.DataFrame, marker: str, *, expected: list[str] | None = None
) -> pd.DataFrame | None:
    """The table under ``marker``: its header row, then rows until a blank line.

    Returns ``None`` when the marker is absent, so the caller can report the sheet
    rather than treating a layout change as an empty result.
    """
    start = _find_marker(frame, marker)
    if start is None:
        return None
    header_row = start + 1
    header = [str(v).strip() for v in frame.iloc[header_row].tolist()]
    rows: list[list] = []
    for i in range(header_row + 1, len(frame)):
        values = frame.iloc[i].tolist()
        # A fully blank first cell ends the block; trailing NaN columns are normal.
        if pd.isna(values[0]) or str(values[0]).strip() == "":
            break
        rows.append(values)
    if not rows:
        return None
    width = len([h for h in header if h and h != "nan"])
    out = pd.DataFrame(rows).iloc[:, :width]
    out.columns = header[:width]
    if expected is not None and list(out.columns) != expected:
        out.attrs["header_mismatch"] = list(out.columns)
    return out


def _metadata(frame: pd.DataFrame) -> dict[str, str]:
    """The key/value METADATA block at the top of a game sheet."""
    meta: dict[str, str] = {}
    for i in range(len(frame)):
        key = frame.iloc[i, 0]
        if pd.isna(key):
            continue
        key = str(key).strip()
        if key in {TEAM_MARKER, LINES_MARKER, PLAYER_PROP_MARKER}:
            break
        if key == "METADATA":
            continue
        value = frame.iloc[i, 1]
        if not pd.isna(value):
            meta[key] = str(value).strip()
    return meta


def _game_number(sheet_name: str, meta: dict[str, str]) -> int | None:
    if "Game Number" in meta:
        try:
            return int(float(meta["Game Number"]))
        except ValueError:
            pass
    match = re.search(r"_G(\d+)_", sheet_name)
    return int(match.group(1)) if match else None


def load_projections() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Tidy player props and team projections across every game sheet.

    Returns ``(props, teams, problems)``; ``problems`` names sheets whose layout
    did not match, so a parsing gap is visible instead of silent.
    """
    book = pd.read_excel(PROJECTIONS_FILE, sheet_name=None, header=None)
    prop_frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []
    problems: list[str] = []

    for sheet, frame in book.items():
        if sheet == "Overview":
            continue
        meta = _metadata(frame)
        game_number = _game_number(sheet, meta)
        context = {
            "sheet": sheet,
            "game_number": game_number,
            "game_date": meta.get("Game Date"),
            "home_team": meta.get("Home Team"),
            "away_team": meta.get("Away Team"),
            "hold_pct": meta.get("Hold %"),
            "model": meta.get("Model"),
            "sims": meta.get("Sims"),
        }

        props = _block(frame, PLAYER_PROP_MARKER)
        if props is None:
            problems.append(f"{sheet}: no PLAYER PROPS block")
        else:
            for key, value in context.items():
                props[key] = value
            prop_frames.append(props)

        teams = _block(frame, TEAM_MARKER)
        if teams is None:
            problems.append(f"{sheet}: no TEAM PROJECTIONS block")
        else:
            for key, value in context.items():
                teams[key] = value
            team_frames.append(teams)

    props_all = pd.concat(prop_frames, ignore_index=True) if prop_frames else pd.DataFrame()
    teams_all = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()
    return props_all, teams_all, problems


def load_results() -> pd.DataFrame:
    """Per-week, per-market handle / GGR / hold from the results workbook."""
    book = pd.read_excel(RESULTS_FILE, sheet_name=None)
    frames = []
    for sheet, frame in book.items():
        clean = frame.copy()
        clean.columns = [str(c).strip() for c in clean.columns]
        clean["week_label"] = sheet
        match = re.search(r"(\d+)", sheet)
        clean["week"] = int(match.group(1)) if match else None
        frames.append(clean)
    out = pd.concat(frames, ignore_index=True)
    out["Market"] = out["Market"].astype(str).str.strip()
    return out


def _to_number(series: pd.Series) -> pd.Series:
    """Coerce a column that may hold numbers, blanks, or stray text."""
    return pd.to_numeric(series, errors="coerce")


def clean_props(props: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce and derive the columns the analysis needs."""
    out = props.copy()
    numeric = [
        "Projection", "Main Line", "Over Odds", "Under Odds", "Fair P(Over)",
        "P10", "P50", "P90", "Actual Result",
    ]
    for column in numeric:
        if column in out.columns:
            out[column] = _to_number(out[column])

    out["hold_pct"] = _to_number(
        out["hold_pct"].astype(str).str.rstrip("%")
    ) / 100.0
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    for column in ("Player", "Team", "Pos", "Stat", "Hit/Miss"):
        if column in out.columns:
            out[column] = out[column].astype(str).str.strip()

    # A prop is only gradeable when it was actually offered (a line exists) and
    # the game has been played (an actual was synced).
    out["has_line"] = out["Main Line"].notna()
    out["has_actual"] = out["Actual Result"].notna()
    out["graded"] = out["has_line"] & out["has_actual"]

    out["error"] = out["Actual Result"] - out["Projection"]
    out["abs_error"] = out["error"].abs()
    # Did the over cash? Ties are pushes on a .5 line, which PLL lines always are,
    # but guard anyway rather than silently scoring a push as a loss.
    out["over_won"] = pd.NA
    live = out["graded"]
    out.loc[live, "over_won"] = (
        out.loc[live, "Actual Result"] > out.loc[live, "Main Line"]
    )
    out["push"] = False
    out.loc[live, "push"] = (
        out.loc[live, "Actual Result"] == out.loc[live, "Main Line"]
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    props, teams, problems = load_projections()
    props = clean_props(props)
    results = load_results()

    props.to_parquet(OUT_DIR / "props.parquet", index=False)
    teams.to_parquet(OUT_DIR / "teams.parquet", index=False)
    results.to_parquet(OUT_DIR / "results.parquet", index=False)

    print(f"props rows      : {len(props):,}")
    print(f"  with a line   : {int(props['has_line'].sum()):,}")
    print(f"  graded        : {int(props['graded'].sum()):,}")
    print(f"team rows       : {len(teams):,}")
    print(f"results rows    : {len(results):,}")
    print(f"games           : {props['game_number'].nunique()}")
    print(f"weeks in results: {sorted(results['week'].dropna().unique().tolist())}")
    print(f"stats           : {sorted(props['Stat'].dropna().unique().tolist())}")
    print(f"markets         : {sorted(results['Market'].unique().tolist())}")
    if problems:
        print("\nLAYOUT PROBLEMS")
        for problem in problems:
            print(f"  {problem}")
    else:
        print("\nAll sheets matched the expected layout.")


if __name__ == "__main__":
    main()
