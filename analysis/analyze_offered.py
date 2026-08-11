"""Accuracy of the projections *that were actually priced into the book*.

The projections workbook contains every rostered player, but only a handful per
team per game reached the board: the main offensive players, the starting goalie,
and the faceoff specialist. Judging the model on all 1,885 rows therefore mixes
two different questions — "is the model good?" and "did the model lose money?" —
and the second one only concerns the offered subset.

The workbook does not flag which props went live, so the offered set is
reconstructed here as a **proxy**:

* the top ``OFFENSIVE_PER_TEAM`` A/M by projected Points, per team per game;
* the highest-projected goalie on each team (the starter);
* the highest-projected FO player on each team (the specialist).

``sensitivity()`` re-runs the headline numbers across a range of roster depths so
a conclusion that depends on where the cut falls is visible as such. If an
authoritative list of offered props exists, it should replace this proxy — the
filter is deliberately isolated in :func:`offered_mask` for that reason.

Two further scope corrections applied here and not in ``analyze_accuracy.py``:

* **Active weeks only.** Props ran weeks 4-6 and 8-12. Week 7 was the All-Star
  break and weeks 13+ had not been played, so their zero rows are not evidence.
  The projections workbook only covers weeks 8-12 (74.7% of season handle).
* **SOG is not a market.** Shots on goal are projected but never offered, so SOG
  accuracy is a model-quality signal only and is excluded from book exposure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_accuracy import add_bet_economics, summarize

DATA = Path(__file__).parent / "data"

OFFENSIVE_PER_TEAM = 5
"""How many A/M skaters per team are treated as offered.

The stated range is 3-7 "main offensive players"; 5 is the midpoint. See
:func:`sensitivity` for how much the conclusions move across 3-7.
"""

OFFERED_STATS = ("Points", "Goals", "Assists", "Saves", "FO Wins")
"""Stats with a corresponding market in the P&L file. SOG has none."""

WEEKEND_TO_WEEK = {
    "2026-07-10": 8, "2026-07-11": 8, "2026-07-12": 8,
    "2026-07-17": 9, "2026-07-18": 9, "2026-07-19": 9,
    "2026-07-25": 10, "2026-07-26": 10,
    "2026-07-31": 11, "2026-08-01": 11, "2026-08-02": 11,
    "2026-08-07": 12, "2026-08-08": 12, "2026-08-09": 12,
}
"""Game dates → league week. Derived from game numbering: games 24-45 fall into
five slates of 4-5 games, matching weeks 8-12 of the results workbook."""


def load_graded() -> pd.DataFrame:
    props = pd.read_parquet(DATA / "props.parquet")
    props = props[props["graded"]].copy()
    props["week"] = props["game_date"].dt.strftime("%Y-%m-%d").map(WEEKEND_TO_WEEK)
    return props


def offered_mask(props: pd.DataFrame, offensive_per_team: int) -> pd.Series:
    """True for rows belonging to a player plausibly on the board.

    Selection is per player (not per prop): once a player is judged offered, all
    of his markets count, which mirrors how the board is actually built.
    """
    # Rank offensive players by their Points projection — the primary market.
    points = props[(props["Stat"] == "Points") & props["Pos"].isin(["A", "M"])]
    ranked = points.groupby(["game_number", "Team"])["Projection"].rank(
        ascending=False, method="first"
    )
    offensive = points.loc[ranked <= offensive_per_team, ["game_number", "Team", "Player"]]

    # The starter is the goalie the model expects to play; likewise the FO specialist.
    def _top(stat: str, position: str) -> pd.DataFrame:
        rows = props[(props["Stat"] == stat) & (props["Pos"] == position)]
        best = rows.groupby(["game_number", "Team"])["Projection"].rank(
            ascending=False, method="first"
        )
        return rows.loc[best == 1, ["game_number", "Team", "Player"]]

    keep = pd.concat([offensive, _top("Saves", "G"), _top("FO Wins", "FO")])
    keys = set(map(tuple, keep.itertuples(index=False, name=None)))
    ids = list(zip(props["game_number"], props["Team"], props["Player"]))
    return pd.Series([key in keys for key in ids], index=props.index)


def offered(props: pd.DataFrame, offensive_per_team: int = OFFENSIVE_PER_TEAM) -> pd.DataFrame:
    """The graded props that were plausibly priced into the book."""
    subset = props[offered_mask(props, offensive_per_team)]
    return subset[subset["Stat"].isin(OFFERED_STATS)].copy()


def sensitivity(props: pd.DataFrame) -> pd.DataFrame:
    """Headline calibration across plausible roster depths.

    If the story only holds at one depth it is an artefact of the proxy, not a
    finding about the model.
    """
    rows = []
    for depth in range(3, 8):
        subset = offered(props, depth)
        live = subset[~subset["push"]]
        rows.append(
            {
                "offensive_per_team": depth,
                "n_props": len(subset),
                "players_per_team_game": round(
                    subset.groupby(["game_number", "Team"])["Player"].nunique().mean(), 2
                ),
                "bias": round(subset["error"].mean(), 3),
                "mae": round(subset["abs_error"].mean(), 3),
                "over_rate": round(live["over_hit"].mean(), 3),
                "fair_p_over": round(subset["fair_p_over"].mean(), 3),
                "calib_gap": round(live["calib_error"].mean(), 3),
            }
        )
    return pd.DataFrame(rows)


CORE = [
    "n", "proj_mean", "actual_mean", "bias", "mae", "rmse",
    "over_hit_rate", "fair_p_over_mean", "calib_gap", "p10_p90_coverage",
]


def show(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")
    display = frame[columns].copy()
    for column in display.columns:
        if display[column].dtype.kind == "f":
            display[column] = display[column].round(3)
    print(display.to_string(index=False))


def main() -> None:
    everything = add_bet_economics(load_graded())
    book = offered(everything)

    print(f"All graded props          : {len(everything):,}")
    print(f"Offered proxy (weeks 8-12): {len(book):,}  ({len(book) / len(everything):.1%})")
    print(
        "Players per team-game     : "
        f"{book.groupby(['game_number', 'Team'])['Player'].nunique().mean():.1f}"
    )
    print(f"Games                     : {book['game_number'].nunique()}")

    show("PROXY SENSITIVITY (does the story depend on the cutoff?)",
         sensitivity(everything), list(sensitivity(everything).columns))

    show("OFFERED BOOK — OVERALL", summarize(book.assign(all="OFFERED"), ["all"]), ["all"] + CORE)
    show("ALL PROJECTIONS — OVERALL (for contrast)",
         summarize(everything.assign(all="ALL"), ["all"]), ["all"] + CORE)

    show("OFFERED BOOK — BY STAT", summarize(book, ["Stat"]), ["Stat"] + CORE)
    show("OFFERED BOOK — BY POSITION", summarize(book, ["Pos"]), ["Pos"] + CORE)
    show("OFFERED BOOK — BY STAT x POSITION (n>=20)",
         summarize(book, ["Stat", "Pos"]).query("n >= 20"), ["Stat", "Pos"] + CORE)

    # Is the low-rate under-projection still present once bench players are gone?
    bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
    labels = ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", "3-5", "5+"]
    for frame, name in ((book, "OFFERED BOOK"), (everything, "ALL PROJECTIONS")):
        frame["proj_bin"] = pd.cut(frame["Projection"], bins=bins, labels=labels)
        table = summarize(frame, ["proj_bin"]).sort_values("proj_bin")
        table["ratio"] = table["actual_mean"] / table["proj_mean"]
        show(f"{name} — BY PROJECTION SIZE", table, ["proj_bin"] + CORE + ["ratio"])

    show("OFFERED BOOK — BY WEEK", summarize(book, ["week"]).sort_values("week"),
         ["week"] + CORE)

    # Goalie starters only: the σ diagnostic that the P&L on Saves MS turns on.
    goalies = book[book["Stat"] == "Saves"]
    print(f"\n{'=' * 104}\nSTARTING GOALIES (offered subset)\n{'=' * 104}")
    print(f"n                : {len(goalies)}")
    print(f"projected sigma  : {goalies['Projection'].std():.2f}")
    print(f"actual sigma     : {goalies['Actual Result'].std():.2f}")
    print(f"projected range  : {goalies['Projection'].min():.1f} - {goalies['Projection'].max():.1f}")
    print(f"actual range     : {goalies['Actual Result'].min():.0f} - {goalies['Actual Result'].max():.0f}")
    print(f"bias             : {goalies['error'].mean():+.2f}")
    print(f"mean P10-P90 span: {(goalies['P90'] - goalies['P10']).mean():.1f}")
    print(f"P10-P90 coverage : {((goalies['Actual Result'] >= goalies['P10']) & (goalies['Actual Result'] <= goalies['P90'])).mean():.1%}")

    excluded = everything[~everything.index.isin(book.index)]
    print(f"\n{'=' * 104}\nEXCLUDED BY THE PROXY (model quality, not book exposure)\n{'=' * 104}")
    print(f"n: {len(excluded):,}")
    print(excluded.groupby("Pos").agg(
        n=("error", "size"), bias=("error", "mean"),
        proj=("Projection", "mean"), actual=("Actual Result", "mean"),
    ).round(3).to_string())

    book.to_parquet(DATA / "offered_scored.parquet", index=False)
    print(f"\nWrote {DATA / 'offered_scored.parquet'}")


if __name__ == "__main__":
    main()
