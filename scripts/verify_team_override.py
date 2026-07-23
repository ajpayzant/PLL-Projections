"""
Verify that team-level two_pt_rate_ewm and sog_rate_ewm overrides actually
change projection output. If output does not move, a UI slider would be a
dead control -> do NOT add it. If it moves in the expected direction and
magnitude, the slider is real -> add it.
"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from projection_engine_v3 import ProjectionEngine

eng = ProjectionEngine()
eng.load()
games = eng.upcoming_games()
g = games[0]
h, a = str(g["home_team_id"]), str(g["away_team_id"])
gd = g.get("game_date")
print(f"Game: {h} vs {a}  date={gd}")


def proj(tro=None):
    r = eng.project(home_team_id=h, away_team_id=a, game_date=gd,
                    team_rating_overrides=tro)
    return r.home_proj

base = proj()
print("\n== BASELINE (home team) ==")
print(f"  proj_shots ={base.proj_shots:.2f}  proj_sog={base.proj_sog:.2f}  "
      f"proj_goals={base.proj_goals:.2f}  proj_2pt={base.proj_2pt_goals:.2f}  "
      f"proj_scores={base.proj_scores:.2f}")

# ── SOG rate override: push to high end 0.80 ─────────────────────────
sog = proj({h: {"sog_rate_ewm": 0.80}})
print("\n== sog_rate_ewm -> 0.80 ==")
print(f"  proj_shots ={sog.proj_shots:.2f}  proj_sog={sog.proj_sog:.2f}  "
      f"proj_goals={sog.proj_goals:.2f}  proj_2pt={sog.proj_2pt_goals:.2f}  "
      f"proj_scores={sog.proj_scores:.2f}")
print(f"  dSOG={sog.proj_sog-base.proj_sog:+.2f}  dGOALS={sog.proj_goals-base.proj_goals:+.2f}  "
      f"dSCORES={sog.proj_scores-base.proj_scores:+.2f}")

# ── 2pt rate override: push to high end 0.30 ─────────────────────────
two = proj({h: {"two_pt_rate_ewm": 0.30}})
print("\n== two_pt_rate_ewm -> 0.30 ==")
print(f"  proj_shots ={two.proj_shots:.2f}  proj_sog={two.proj_sog:.2f}  "
      f"proj_goals={two.proj_goals:.2f}  proj_2pt={two.proj_2pt_goals:.2f}  "
      f"proj_scores={two.proj_scores:.2f}")
print(f"  d2PT={two.proj_2pt_goals-base.proj_2pt_goals:+.2f}  "
      f"dGOALS={two.proj_goals-base.proj_goals:+.2f}  "
      f"dSCORES={two.proj_scores-base.proj_scores:+.2f}")

# ── 2pt rate override low: 0.02 ──────────────────────────────────────
twol = proj({h: {"two_pt_rate_ewm": 0.02}})
print("\n== two_pt_rate_ewm -> 0.02 ==")
print(f"  proj_2pt={twol.proj_2pt_goals:.2f}  proj_scores={twol.proj_scores:.2f}  "
      f"d2PT={twol.proj_2pt_goals-base.proj_2pt_goals:+.2f}  "
      f"dSCORES={twol.proj_scores-base.proj_scores:+.2f}")

print("\n== VERDICT ==")
sog_moves = abs(sog.proj_sog - base.proj_sog) > 0.05
two_moves = abs(two.proj_2pt_goals - base.proj_2pt_goals) > 0.02
print(f"  sog_rate_ewm override moves output : {sog_moves}")
print(f"  two_pt_rate_ewm override moves output: {two_moves}")
