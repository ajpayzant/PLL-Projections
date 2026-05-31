# PLL GitHub Warehouse Builder Files

Copy these files into the root of the GitHub repository:

- `scripts/build_warehouse.py`
- `requirements.txt`
- `runtime.txt`
- `.github/workflows/update-data.yml`
- `.gitignore` optional

The builder writes the app-ready DuckDB database to:

`data/analytics_database/pll_warehouse.duckdb`

and writes exported CSV/Parquet artifacts to:

`data/curated_data/all_requested_seasons/`

This matches the GitHub-ready `app.py`, which expects the warehouse inside the repository `data/` folder.

Required GitHub secret:

`PLL_BEARER_TOKEN`

Manual workflow run:

GitHub → Actions → Update PLL Warehouse → Run workflow

Scheduled workflow runs:

- Monday 05:00 UTC, intended to represent Sunday midnight EST
- Friday 13:00 UTC, intended to represent Friday 8 AM EST
