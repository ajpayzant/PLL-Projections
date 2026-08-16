"""Verify the postseason-ingest logic in build_warehouse.py without hitting the API.

build_warehouse.py is a top-to-bottom script (importing it would run the whole
collection), so the checked functions and constants are lifted out of it by AST and
exec'd in isolation. That keeps this a test of the REAL code rather than a copy.

What it asserts, using a synthetic event payload that mixes all four observed
seasonSegment values:
  1. "regular" and "post" are ingested; "allstar" and "champseries" are dropped.
  2. Regular-season game_number values are IDENTICAL with and without the playoff
     bracket in the payload — stat tables and saved Google Sheets tabs key on
     game_number, so renumbering would silently corrupt them.
  3. Playoff games are numbered after every regular-season game.
  4. Playoff rows keep their round label / venue / official week, and survive with
     homeTeam/awayTeam null (the bracket is unseeded until the QFs are set).

Run: python scripts/verify_postseason_ingest.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUILD_SCRIPT = Path(__file__).resolve().parent / "build_warehouse.py"

# Names lifted out of build_warehouse.py. Anything these depend on must be listed.
WANTED_FUNCS = {
    "is_included_segment",
    "is_postseason_segment",
    "segment_order_rank",
    "record_observed_segment",
    "parse_event_list_payload",
    "safe_get",
    "to_num_scalar",
    "extract_home_team_obj",
    "extract_away_team_obj",
    "extract_team_id_from_obj",
    "extract_team_name_from_obj",
}
WANTED_ASSIGNS = {
    "COMPETITION_TYPE",
    "POSTSEASON_COMPETITION_TYPES",
    "INCLUDED_COMPETITION_TYPES",
    "OBSERVED_SEGMENT_COUNTS",
}


def load_pipeline_namespace() -> dict:
    tree = ast.parse(BUILD_SCRIPT.read_text(encoding="utf-8"))
    keep: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in WANTED_FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & WANTED_ASSIGNS:
                keep.append(node)

        found_funcs = {n.name for n in keep if isinstance(n, ast.FunctionDef)}
        found_assigns = {
            t.id
            for n in keep if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)
        }
        if found_funcs >= WANTED_FUNCS and found_assigns >= WANTED_ASSIGNS:
            break

    missing = WANTED_FUNCS - {n.name for n in keep if isinstance(n, ast.FunctionDef)}
    if missing:
        raise RuntimeError(f"build_warehouse.py no longer defines: {sorted(missing)}")

    ns: dict = {"pd": pd, "np": np, "os": __import__("os")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(BUILD_SCRIPT), "exec"), ns)
    return ns


def _event(eid, num_id, segment, day, home=None, away=None, description=None, week=None):
    """One event-list item, shaped like the API's payload."""
    return {
        "eventId": eid,
        "id": num_id,
        "slugname": eid,
        "year": 2026,
        "seasonSegment": segment,
        # 2026-08-01 is 1785628800; one day = 86400s.
        "startTime": 1785628800 + day * 86400,
        "homeTeam": ({"officialId": home, "location": home} if home else None),
        "awayTeam": ({"officialId": away, "location": away} if away else None),
        "eventStatus": 3 if day < 16 else 0,
        "description": description,
        "week": week,
        "venue": "Subaru Park" if segment == "post" else "Home Field",
        "location": "Chester, PA" if segment == "post" else "Somewhere",
    }


REGULAR_EVENTS = [
    _event("2026-ev-40", 40, "regular", 0, "PHI", "NY", week=12),
    _event("2026-ev-41", 41, "regular", 1, "CAL", "DEN", week=12),
    _event("2026-ev-47", 47, "regular", 15, "PHI", "MD", week=13),
    _event("2026-ev-48", 48, "regular", 15, "CAL", "DEN", week=13),
]
# Excluded segments. The All-Star Game sits MID-season on purpose: if it were
# ingested it would push every later regular-season game_number up by one.
EXCLUDED_EVENTS = [
    _event("2026-ev-as", 99, "allstar", 7, "EAST", "WEST", description="All-Star Game"),
    _event("2026-cs-01", 5, "champseries", -150, "PHI", "NY", description="Champ Series"),
]
# The published bracket: teams are null until seeding is decided.
POSTSEASON_EVENTS = [
    _event("2026-ev-49", 49, "post", 28, None, None, description="Quarterfinal", week=14),
    _event("2026-ev-50", 50, "post", 29, None, None, description="Quarterfinal", week=14),
    _event("2026-ev-51", 51, "post", 37, None, None, description="Semifinal", week=15),
    _event("2026-ev-52", 52, "post", 37, None, None, description="Semifinal", week=15),
    _event("2026-ev-53", 53, "post", 50, None, None, description="Championship", week=16),
]


def _payload(items):
    return {"data": {"items": items}}


def main() -> int:
    ns = load_pipeline_namespace()
    parse = ns["parse_event_list_payload"]
    failures: list[str] = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    print("Segment allowlist:")
    check("included = regular + post",
          ns["INCLUDED_COMPETITION_TYPES"] == {"regular", "post"},
          str(sorted(ns["INCLUDED_COMPETITION_TYPES"])))
    check("allstar excluded", not ns["is_included_segment"]("allstar"))
    check("champseries excluded", not ns["is_included_segment"]("champseries"))
    check("missing segment excluded", not ns["is_included_segment"](None))
    check("post is postseason", ns["is_postseason_segment"]("post"))
    check("regular is not postseason", not ns["is_postseason_segment"]("regular"))
    check("postseason sorts after regular",
          ns["segment_order_rank"]("post") > ns["segment_order_rank"]("regular"))

    # Regular season only — the pre-postseason state of the world.
    print("\nRegular season only (baseline):")
    before = parse(_payload(REGULAR_EVENTS + EXCLUDED_EVENTS), 2026)
    check("excluded segments dropped", len(before) == len(REGULAR_EVENTS),
          f"{len(before)} rows from {len(REGULAR_EVENTS + EXCLUDED_EVENTS)} events")
    baseline = dict(zip(before["slug"], before["game_number"]))
    check("numbered 1..N", sorted(baseline.values()) == list(range(1, len(REGULAR_EVENTS) + 1)),
          str(baseline))

    # Now with the bracket. The interleaved order is deliberate: the parser must not
    # depend on the API returning segments in any particular order.
    print("\nWith the playoff bracket:")
    mixed = [REGULAR_EVENTS[0], POSTSEASON_EVENTS[0], EXCLUDED_EVENTS[0],
             REGULAR_EVENTS[2], POSTSEASON_EVENTS[4], REGULAR_EVENTS[1],
             *POSTSEASON_EVENTS[1:4], REGULAR_EVENTS[3], EXCLUDED_EVENTS[1]]
    after = parse(_payload(mixed), 2026)
    check("regular + post ingested, others dropped",
          len(after) == len(REGULAR_EVENTS) + len(POSTSEASON_EVENTS),
          f"{len(after)} rows")

    numbers = dict(zip(after["slug"], after["game_number"]))
    drift = {s: (baseline[s], numbers.get(s)) for s in baseline if numbers.get(s) != baseline[s]}
    check("regular-season game_number UNCHANGED", not drift, str(drift) or "no drift")

    post_rows = after[after["competition_type"] == "post"]
    reg_rows = after[after["competition_type"] == "regular"]
    check("playoff numbers all follow regular-season numbers",
          post_rows["game_number"].min() > reg_rows["game_number"].max(),
          f"post starts at {int(post_rows['game_number'].min())}, "
          f"regular ends at {int(reg_rows['game_number'].max())}")
    check("playoff games kept despite null teams",
          len(post_rows) == len(POSTSEASON_EVENTS)
          and post_rows["home_team_id_raw"].isna().all(),
          f"{len(post_rows)} rows, all teams TBD")
    check("round labels preserved",
          list(post_rows.sort_values("game_number")["round_label"]) ==
          ["Quarterfinal", "Quarterfinal", "Semifinal", "Semifinal", "Championship"],
          str(list(post_rows.sort_values("game_number")["round_label"])))
    check("official weeks preserved",
          sorted(set(post_rows["official_week"].dropna().astype(int))) == [14, 15, 16],
          str(sorted(set(post_rows["official_week"].dropna().astype(int)))))
    check("venues preserved", post_rows["venue"].notna().all())

    print("\nObserved-segment tally (feeds the Block 10 QC rows):")
    for (season, seg), n in sorted(ns["OBSERVED_SEGMENT_COUNTS"].items()):
        print(f"  {season} {seg}: {n}")
    check("every segment observed, including excluded ones",
          {seg for _, seg in ns["OBSERVED_SEGMENT_COUNTS"]} ==
          {"regular", "post", "allstar", "champseries"})

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("All postseason-ingest checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
