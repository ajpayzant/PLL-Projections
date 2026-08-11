"""Re-price the 2026 offered book through the fixed distribution logic.

The unit tests in ``scripts/test_prop_distributions.py`` prove each fix behaves
correctly in isolation. This answers the question that actually decides P&L: do
the calibration gaps close on the real props that were priced and graded?

Method
------
For every graded offered prop (weeks 8-12), take the engine's own projected mean
and re-draw the distribution using the current code, then read fair P(Over) off
the draw at the line that was actually offered. Comparing that to the realized
over-rate isolates the distribution change, because the projected mean is held
fixed at whatever the engine produced at the time -- the same input, a different
shape.

Two things this deliberately does NOT do:

* It does not re-run the full projection pipeline. Means are held at their
  original values on purpose; if means moved, the comparison would conflate a
  shape fix with a projection change and prove nothing about either.
* It does not re-derive the offered subset. It reads
  ``data/offered_scored.parquet`` as written by ``analyze_offered.py``, so the
  proxy and its documented 3-7 sensitivity carry over unchanged.

The headline metric is the calibration gap: realized over-rate minus stated fair
P(Over). It correlates -0.92 with realized hold across the four O/U markets, so
it is the leading indicator for whether these fixes make money.

Run: python analysis/reprice_offered.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import projection_engine_v3 as E

DATA = Path(__file__).parent / "data"

# 60k per prop: the standard error on a fair probability is then ~0.2 points,
# an order of magnitude below the gaps being measured.
DRAWS = 60_000
SEED = 20260810

# The engine keys zero rates and dispersion by position and stat; map the
# workbook's labels onto them. Saves and FO wins have no zero-inflation -- a
# starting goalie facing zero shots is not a real scenario, and the goalie draw
# handles short outings through the playing-time mixture instead.
STAT_TO_PHI_KEY = {
    "Goals": "goals",
    "Assists": "assists",
    "Points": "points",
    "Saves": "saves",
    "FO Wins": "fo_wins",
}
ZERO_INFLATED = {"Goals": "goals", "Assists": "assists"}


def _draw(rng: np.random.Generator, row: pd.Series) -> np.ndarray | None:
    """Redraw one prop's distribution with the current engine code."""
    stat, mu, pos = row["Stat"], float(row["Projection"]), row["Pos"]
    if not np.isfinite(mu) or mu <= 0:
        return None

    if stat == "Saves":
        # Same call the simulator makes, including the playing-time mixture.
        csr_ratio = 1.0          # no per-goalie clean-save-rate here; use baseline
        phi = float(np.clip(E.PHI_PLAYER["saves"] * csr_ratio, 70.0, 220.0))
        return E._draw_goalie_saves(rng, mu, phi, DRAWS)

    if stat in ZERO_INFLATED:
        phi_key = ZERO_INFLATED[stat]
        phi = E.PHI_PLAYER[phi_key]
        prior = E.ZERO_RATE.get(f"{pos}_{phi_key}")
        if prior is None:
            return None
        z = E._solve_excess_zero(mu, phi, E._cap_zero_rate(prior, mu))
        nb_n, nb_p = E._negbinom_params(mu / max(1.0 - z, 0.01), phi)
        is_zero = rng.random(DRAWS) < z
        return np.where(is_zero, 0.0,
                        rng.negative_binomial(nb_n, nb_p, DRAWS).astype(float))

    phi_key = STAT_TO_PHI_KEY.get(stat)
    if phi_key is None:
        return None
    nb_n, nb_p = E._negbinom_params(mu, E.PHI_PLAYER.get(phi_key, 2.0))
    return rng.negative_binomial(nb_n, nb_p, DRAWS).astype(float)


def _draw_old(rng: np.random.Generator, row: pd.Series) -> np.ndarray | None:
    """The pre-fix behaviour, for a like-for-like comparison.

    Reproduces the two defects together, since they were shipped together:
    zero-inflation stacked on the NegBin's own zeros, and no goalie playing-time
    mixture. Uses phi=4.0 for assists, the value in place at the time.
    """
    stat, mu, pos = row["Stat"], float(row["Projection"]), row["Pos"]
    if not np.isfinite(mu) or mu <= 0:
        return None

    if stat == "Saves":
        nb_n, nb_p = E._negbinom_params(mu, 120.0)
        return rng.negative_binomial(nb_n, nb_p, DRAWS).astype(float)

    if stat in ZERO_INFLATED:
        phi_key = ZERO_INFLATED[stat]
        phi = 4.0 if phi_key == "assists" else E.PHI_PLAYER[phi_key]
        z = E.ZERO_RATE.get(f"{pos}_{phi_key}")
        if z is None:
            return None
        nb_n, nb_p = E._negbinom_params(mu / max(1.0 - z, 0.01), phi)
        is_zero = rng.random(DRAWS) < z
        return np.where(is_zero, 0.0,
                        rng.negative_binomial(nb_n, nb_p, DRAWS).astype(float))

    phi_key = STAT_TO_PHI_KEY.get(stat)
    if phi_key is None:
        return None
    nb_n, nb_p = E._negbinom_params(mu, E.PHI_PLAYER.get(phi_key, 2.0))
    return rng.negative_binomial(nb_n, nb_p, DRAWS).astype(float)


def reprice(props: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for _, row in props.iterrows():
        line = float(row["Main Line"])
        if not np.isfinite(line):
            continue
        new, old = _draw(rng, row), _draw_old(rng, row)
        if new is None or old is None:
            continue
        mu = float(row["Projection"])
        out.append({
            "Stat": row["Stat"], "Pos": row["Pos"], "week": row["week"],
            "Projection": mu, "Actual Result": row["Actual Result"],
            "line": line, "push": bool(row["push"]), "over_hit": row["over_hit"],
            "p_over_old": float((old > line).mean()),
            "p_over_new": float((new > line).mean()),
            "mean_old": float(old.mean()), "mean_new": float(new.mean()),
            # Poisson floor, only meaningful on a 0.5 line where Over == "at least one".
            "poisson_floor": (1.0 - math.exp(-mu)) if line == 0.5 else np.nan,
        })
    return pd.DataFrame(out)


def _gap_table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    live = frame[~frame["push"] & frame["over_hit"].notna()]
    g = live.groupby(by, observed=True).apply(lambda d: pd.Series({
        "n": len(d),
        "proj_mean": d["Projection"].mean(),
        "actual_mean": d["Actual Result"].mean(),
        "over_rate": d["over_hit"].mean(),
        "p_over_old": d["p_over_old"].mean(),
        "p_over_new": d["p_over_new"].mean(),
        "gap_old": d["over_hit"].mean() - d["p_over_old"].mean(),
        "gap_new": d["over_hit"].mean() - d["p_over_new"].mean(),
    }), include_groups=False)
    g["improved"] = g["gap_new"].abs() < g["gap_old"].abs()
    return g.round(4)


def main() -> None:
    props = pd.read_parquet(DATA / "offered_scored.parquet")
    priced = reprice(props)

    print(f"Offered graded props re-priced: {len(priced):,} of {len(props):,}")
    print(f"Draws per prop: {DRAWS:,}\n")

    print("=" * 100)
    print("MEANS MUST NOT MOVE  (a shape fix that changes the projection is a bug)")
    print("=" * 100)
    means = priced.groupby("Stat").apply(lambda d: pd.Series({
        "n": len(d),
        "projected": d["Projection"].mean(),
        "sim_mean_old": d["mean_old"].mean(),
        "sim_mean_new": d["mean_new"].mean(),
        "drift": d["mean_new"].mean() - d["mean_old"].mean(),
    }), include_groups=False).round(4)
    print(means.to_string())
    worst = means["drift"].abs().max()
    print(f"\nLargest mean drift across stats: {worst:.4f} "
          f"({'OK' if worst < 0.15 else 'INVESTIGATE'})")

    print("\n" + "=" * 100)
    print("CALIBRATION GAP BY STAT   (realized over-rate minus stated fair P(Over))")
    print("=" * 100)
    print(_gap_table(priced, ["Stat"]).to_string())

    print("\n" + "=" * 100)
    print("CALIBRATION GAP BY STAT x POSITION  (n>=20)")
    print("=" * 100)
    table = _gap_table(priced, ["Stat", "Pos"])
    print(table[table["n"] >= 20].to_string())

    print("\n" + "=" * 100)
    print("POISSON FLOOR CHECK  (0.5 lines only: fair P(Over) must not sit far below")
    print("1 - exp(-mu). This is the check bias/MAE and P10-P90 coverage cannot make.)")
    print("=" * 100)
    half = priced[priced["line"] == 0.5].copy()
    if len(half):
        half["breach_old"] = half["poisson_floor"] - half["p_over_old"]
        half["breach_new"] = half["poisson_floor"] - half["p_over_new"]
        print(half.groupby("Stat").apply(lambda d: pd.Series({
            "n": len(d),
            "mean_mu": d["Projection"].mean(),
            "floor": d["poisson_floor"].mean(),
            "p_over_old": d["p_over_old"].mean(),
            "p_over_new": d["p_over_new"].mean(),
            "worst_breach_old": d["breach_old"].max(),
            "worst_breach_new": d["breach_new"].max(),
            "pct_breaching_5pts_old": (d["breach_old"] > 0.05).mean(),
            "pct_breaching_5pts_new": (d["breach_new"] > 0.05).mean(),
        }), include_groups=False).round(4).to_string())

    print("\n" + "=" * 100)
    print("OVERALL")
    print("=" * 100)
    live = priced[~priced["push"] & priced["over_hit"].notna()]
    for label, column in (("before", "p_over_old"), ("after", "p_over_new")):
        gap = live["over_hit"].mean() - live[column].mean()
        print(f"  {label:6s}: fair P(Over) {live[column].mean():.4f}  "
              f"realized {live['over_hit'].mean():.4f}  gap {gap:+.4f}")

    priced.to_parquet(DATA / "offered_repriced.parquet", index=False)
    print(f"\nWrote {DATA / 'offered_repriced.parquet'}")


if __name__ == "__main__":
    main()
