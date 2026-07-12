"""Diagnostic: shots vs SOG, team vs starter goalie."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
import duckdb
con = duckdb.connect("data/analytics_database/pll_warehouse.duckdb", read_only=True)

print("=" * 70)
print("SHOTS / SOG / SHOTS-FACED — league averages by season")
print("=" * 70)

tgs = con.sql("""
SELECT season,
       AVG(shots) AS shots_per_game,
       AVG(shots_on_goal) AS sog_per_game,
       AVG(shots_on_goal * 1.0 / NULLIF(shots,0)) AS sog_rate,
       AVG(saves + goals_against) AS opp_sog_via_team_ga,
       COUNT(*) AS n
FROM clean.team_game_stats
GROUP BY 1 ORDER BY 1
""").df()
print(tgs.to_string())

print()
print("=" * 70)
print("Opponent SOG (team-level) vs starter goalie shots-faced")
print("=" * 70)

q = con.sql("""
WITH goalies AS (
    SELECT game_id, team_id, player_id, full_name,
           COALESCE(saves,0)+COALESCE(goals_against,0) AS sf
    FROM clean.player_game_stats
    WHERE UPPER(position)='G'
),
starter AS (
    SELECT game_id, team_id,
           ARG_MAX(player_id, sf) AS starter_id,
           MAX(sf) AS starter_sf,
           SUM(sf) AS team_sf
    FROM goalies GROUP BY 1,2
),
tg_opp AS (
    SELECT tg.game_id, tg.season, tg.team_id, opp.shots_on_goal AS opp_sog
    FROM clean.team_game_stats tg
    JOIN clean.team_game_stats opp
      ON tg.game_id = opp.game_id AND tg.team_id <> opp.team_id
)
SELECT t.season,
       COUNT(*) AS n,
       AVG(t.opp_sog) AS opp_sog_team,
       AVG(s.team_sf) AS goalie_team_sf,
       AVG(s.starter_sf) AS starter_sf,
       AVG(s.starter_sf * 1.0 / NULLIF(s.team_sf,0)) AS starter_share,
       AVG(s.team_sf * 1.0 / NULLIF(t.opp_sog,0)) AS goalie_sf_over_opp_sog
FROM tg_opp t
JOIN starter s USING(game_id, team_id)
GROUP BY 1 ORDER BY 1
""").df()
print(q.to_string())

print()
print("Note: goalie_sf_over_opp_sog < 1 → goalies collectively see fewer shots than")
print("team opp_sog because empty-net/misc are excluded from goalie SF.")
print("starter_share tells us what fraction of that goalie SF was the starter's.")

print()
print("=" * 70)
print("Actual save-projection input decomposition (2025 TEST data)")
print("=" * 70)

# What are the shots_ewm values seen in the engine? Sample by pulling ewms from
# a fresh RatingBuilder at the START of each 2025 game.
q = con.sql("""
SELECT season,
       AVG(shots) AS team_shots,
       AVG(shots_on_goal) AS team_sog,
       AVG(saves + goals_against) AS team_ga_denom_starter_plus_backup
FROM clean.team_game_stats
WHERE season = 2025
""").df()
print("2025 season averages:")
print(q.to_string())
