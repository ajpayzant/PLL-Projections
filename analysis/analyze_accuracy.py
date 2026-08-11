"""Projection accuracy, line quality, and probability calibration.

Three distinct questions, deliberately kept apart because they have different
fixes:

1. **Point accuracy** — is the projected mean right? (bias / MAE / RMSE)
2. **Line quality** — given the projection, was the *line* set where it should
   be? A projection can be unbiased while the line is still systematically
   beatable, because the line is a rounded .5 threshold, not the mean.
3. **Calibration** — does a stated 45% chance happen 45% of the time? This is
   what actually drives P&L, since the odds are derived from it.

A model can pass (1) and fail (3): getting the mean right while getting the
*shape* of the distribution wrong produces correctly-centred, badly-priced props.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MIN_SAMPLE = 30
"""Smallest cell reported without a warning. Below this, hit-rate noise on a
~50/50 prop swamps any real edge, so a cell is shown but flagged."""


def american_to_prob(odds: pd.Series) -> pd.Series:
    """Implied probability from American odds, vig included."""
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def american_profit(odds: float, stake: float = 1.0) -> float:
    """Profit on a winning bet of ``stake`` at ``odds`` (excludes the stake)."""
    return stake * (odds / 100.0 if odds > 0 else 100.0 / -odds)


def load() -> pd.DataFrame:
    props = pd.read_parquet(DATA / "props.parquet")
    return props[props["graded"]].copy()


def add_bet_economics(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-prop hold and the house result had one unit been bet on each side.

    ``theoretical_hold`` is the overround actually built into the two prices,
    which is the correct benchmark for realized hold — not the 7.5% target,
    since rounding to a real price never lands exactly on target.
    """
    out = frame.copy()
    out["p_over_priced"] = american_to_prob(out["Over Odds"])
    out["p_under_priced"] = american_to_prob(out["Under Odds"])
    out["overround"] = out["p_over_priced"] + out["p_under_priced"]
    out["theoretical_hold"] = 1.0 - 1.0 / out["overround"]

    # House P&L per $1 staked on the side that won, assuming balanced action.
    over_profit = out["Over Odds"].apply(american_profit)
    under_profit = out["Under Odds"].apply(american_profit)
    won_over = out["over_won"].astype("boolean").fillna(False)
    out["house_pnl_over_bettor"] = np.where(won_over, -over_profit, 1.0)
    out["house_pnl_under_bettor"] = np.where(won_over, 1.0, -under_profit)
    out.loc[out["push"], ["house_pnl_over_bettor", "house_pnl_under_bettor"]] = 0.0

    # Closing-line-style edge: our own fair probability vs what happened.
    out["fair_p_over"] = pd.to_numeric(out["Fair P(Over)"], errors="coerce")
    out["over_hit"] = won_over.astype(float)
    out["calib_error"] = out["over_hit"] - out["fair_p_over"]
    return out


def summarize(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Accuracy, line quality, and calibration for each group."""
    rows = []
    for keys, group in frame.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(group)
        live = group[~group["push"]]
        rows.append(
            {
                **dict(zip(by, keys)),
                "n": n,
                "proj_mean": group["Projection"].mean(),
                "actual_mean": group["Actual Result"].mean(),
                # Positive bias = the model projected too LOW (actual came in higher).
                "bias": group["error"].mean(),
                "mae": group["abs_error"].mean(),
                "rmse": float(np.sqrt((group["error"] ** 2).mean())),
                "over_hit_rate": live["over_hit"].mean() if len(live) else np.nan,
                "fair_p_over_mean": group["fair_p_over"].mean(),
                "calib_gap": live["calib_error"].mean() if len(live) else np.nan,
                "theo_hold": group["theoretical_hold"].mean(),
                # Realized hold if action had been split evenly across both sides.
                "hold_balanced": (
                    group[["house_pnl_over_bettor", "house_pnl_under_bettor"]]
                    .mean(axis=1)
                    .mean()
                ),
                # Worst case: all action on whichever side actually won.
                "hold_if_all_over": group["house_pnl_over_bettor"].mean(),
                "hold_if_all_under": group["house_pnl_under_bettor"].mean(),
                "p10_p90_coverage": (
                    (group["Actual Result"] >= group["P10"])
                    & (group["Actual Result"] <= group["P90"])
                ).mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def show(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    display = frame[columns].copy()
    for column in display.columns:
        if display[column].dtype.kind == "f":
            display[column] = display[column].round(3)
    print(display.to_string(index=False))
    thin = frame[frame["n"] < MIN_SAMPLE]
    if len(thin):
        print(f"  ({len(thin)} group(s) below n={MIN_SAMPLE} — treat as noise)")


def main() -> None:
    props = add_bet_economics(load())
    core = [
        "n", "proj_mean", "actual_mean", "bias", "mae", "rmse",
        "over_hit_rate", "fair_p_over_mean", "calib_gap",
        "theo_hold", "hold_balanced", "p10_p90_coverage",
    ]

    print(f"Graded props: {len(props):,} across {props['game_number'].nunique()} games")
    print(f"Date range  : {props['game_date'].min():%Y-%m-%d} to {props['game_date'].max():%Y-%m-%d}")

    overall = summarize(props.assign(all="ALL"), ["all"])
    show("OVERALL", overall, ["all"] + core)

    show("BY STAT", summarize(props, ["Stat"]), ["Stat"] + core)
    show("BY POSITION", summarize(props, ["Pos"]), ["Pos"] + core)
    show(
        "BY STAT x POSITION (n>=20)",
        summarize(props, ["Stat", "Pos"]).query("n >= 20"),
        ["Stat", "Pos"] + core,
    )

    # Line-level view: does the model beat the number it set?
    props["line_gap"] = props["Projection"] - props["Main Line"]
    bins = [-np.inf, -1.0, -0.5, -0.15, 0.15, 0.5, 1.0, np.inf]
    labels = ["<-1", "-1:-0.5", "-0.5:-0.15", "±0.15", "0.15:0.5", "0.5:1", ">1"]
    props["line_gap_bin"] = pd.cut(props["line_gap"], bins=bins, labels=labels)
    show(
        "BY PROJECTION-MINUS-LINE (is the priced edge real?)",
        summarize(props, ["line_gap_bin"]),
        ["line_gap_bin"] + core,
    )

    # Favourite-longshot: are cheap overs mispriced?
    prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    props["fair_bin"] = pd.cut(props["fair_p_over"], bins=prob_bins)
    show(
        "CALIBRATION BY FAIR P(OVER) BUCKET",
        summarize(props, ["fair_bin"]),
        ["fair_bin"] + core,
    )

    show("BY WEEK", summarize(props, ["game_date"]).sort_values("game_date"),
         ["game_date"] + core)

    # The binned columns are Categoricals of Intervals, which parquet cannot
    # represent; they are re-derived on load anyway, so store them as labels.
    for column in ("line_gap_bin", "fair_bin"):
        props[column] = props[column].astype(str)
    props.to_parquet(DATA / "props_scored.parquet", index=False)
    print(f"\nWrote {DATA / 'props_scored.parquet'}")


if __name__ == "__main__":
    main()
