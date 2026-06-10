"""
PLL Projection App
==================
Entry point. Run with:  streamlit run projection_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="PLL Projections",
    page_icon="🥍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main .block-container { padding-top:1rem; padding-bottom:2rem; max-width:1800px; }
  h1,h2,h3 { letter-spacing:-0.02em; }
  .pll-card {
    border:1px solid rgba(148,163,184,.20); border-radius:12px; padding:14px 18px;
    background:linear-gradient(160deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
    box-shadow:0 4px 16px rgba(0,0,0,.10); margin-bottom:10px;
  }
  .pll-card-label { color:#94a3b8; font-size:.78rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
  .pll-card-value { font-size:1.4rem; font-weight:800; color:#f1f5f9; line-height:1.15; }
  .pll-card-sub   { color:#94a3b8; font-size:.78rem; margin-top:4px; line-height:1.4; }
  .note-text { color:#64748b; font-size:.80rem; font-style:italic; }
  .section-header {
    font-size:.72rem; font-weight:700; letter-spacing:.10em; text-transform:uppercase;
    color:#64748b; border-bottom:1px solid rgba(148,163,184,.15);
    padding-bottom:4px; margin:18px 0 8px;
  }
  .highlight { color:#34d399; font-weight:600; }
  .badge {
    display:inline-block; padding:2px 8px; border-radius:6px;
    font-size:.72rem; font-weight:700; margin-right:4px;
  }
  .badge-blue  { background:#1d4ed8; color:#fff; }
  .badge-green { background:#059669; color:#fff; }
  .badge-amber { background:#d97706; color:#fff; }
  .badge-purple{ background:#7c3aed; color:#fff; }
</style>
""", unsafe_allow_html=True)

st.title("🥍 PLL Projections")
st.markdown("##### Monte Carlo projection system for Premier Lacrosse League games")
st.markdown("---")

# -- Page navigation cards ---------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        '<div class="pll-card">'
        '<div class="pll-card-label">Page 1</div>'
        '<div class="pll-card-value">Projections</div>'
        '<div class="pll-card-sub">Select a game, run 20,000 simulations, view team stats, '
        'sim distributions, and per-player projection summary. '
        'Adjust team ratings, save/load sessions, export to Excel.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="pll-card">'
        '<div class="pll-card-label">Page 2</div>'
        '<div class="pll-card-value">Depth Charts</div>'
        '<div class="pll-card-sub">Mark players inactive (scratches/injuries), set goalie starters, '
        'adjust usage multipliers, override positions (e.g. Attack playing Midfield), '
        'and tune individual rating inputs per player.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="pll-card">'
        '<div class="pll-card-label">Page 3</div>'
        '<div class="pll-card-value">Player Props</div>'
        '<div class="pll-card-sub">Goals, assists, points, SOG, saves, FO wins with American odds. '
        'Alternate line ladder, milestone props (1+, 2+, 3+), '
        'market line comparison with edge calculator.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        '<div class="pll-card">'
        '<div class="pll-card-label">Page 4</div>'
        '<div class="pll-card-value">Game Lines</div>'
        '<div class="pll-card-sub">Moneyline, spread, total, team totals with American odds. '
        'Override any line, alternate spread/total tables, '
        'score probability grid heatmap.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# -- Workflow ----------------------------------------------------------------
st.markdown('<div class="section-header">Recommended Workflow</div>', unsafe_allow_html=True)
st.markdown("""
1. **Projections** &rarr; pick a game &rarr; click **Run Projection**
2. **Depth Charts** &rarr; deactivate any scratched/injured players, adjust usage if needed
3. **Projections** &rarr; click **Update Projection** to apply roster changes
4. **Player Props** &rarr; review every player's prop lines and compare vs market
5. **Game Lines** &rarr; final moneyline, spread, total output
""")

st.markdown("---")

# -- Engine overview ---------------------------------------------------------
st.markdown('<div class="section-header">How the Engine Works</div>', unsafe_allow_html=True)

e1, e2 = st.columns(2)
with e1:
    st.markdown("""
**Team Projection Model**
- Ratings built from per-game stats using exponentially weighted moving averages (EWM)
  with half-lives tuned to autocorrelation per stat (FO% most persistent, goals least)
- Team goals use a truncated Normal distribution (var/mean = 0.854, slight underdispersion)
  with a baseline of ~11.25 goals/game per team
- Possession chain model: faceoff win rate drives time-of-possession, which drives
  offensive sequences, which drive shots, which drive goals
- No hardcoded home-field advantage (not statistically significant in PLL data, t=1.62)

**Player Projection Model**
- Credibility-weighted blend (Bühlmann-Straub): player's own EWM history (65%),
  career mean (35%), and position prior — mixed by `gp/(gp+15)` so veterans
  are trusted more than rookies
- Position priors (POS_DEFAULTS) represent the average share at each position:
  Attack ~20% of team goals, Midfield ~13%, SSDM ~2%, etc.
- Player goals/assists use a zero-inflated near-Poisson distribution calibrated
  to empirical data: var/mean ~ 1.02 for attackers (nearly Poisson, not NegBin)
- Zero-inflation rates measured from data: A=20%, M=39%, FO=78%, SSDM=84%, D=95%
  Elite scorers (Holman, Fields, etc.) use their personal career zero rates (~6-12%)
""")
with e2:
    st.markdown("""
**Simulation**
- 20,000 Monte Carlo simulations per projection run
- Team goals conditioned together (home and away drawn jointly)
- Player goals conditioned on team total draw to maintain consistency:
  raw per-player draws are rescaled so they sum to the team total each sim
- Faceoff specialists get a separate NegBin draw for FO wins
- Goalies: saves drawn from NegBin against opponent's projected SOG

**Pricing**
- All lines forced to x.5 (e.g. 1.5, 2.5) to eliminate push risk on integer scoring
- Main prop line selected as the most balanced line (closest to 50/50 over/under)
- Vig/margin applied symmetrically: `price = (fair_prob / total) * (1 + margin)`
- Default margin: **7.5%** (standard sportsbook level). Adjustable on any page,
  synced globally across all pages
- Alternate lines and milestone props (1+, 2+, 3+) priced at the same margin
""")

st.markdown("---")

# -- Key features since revisions --------------------------------------------
st.markdown('<div class="section-header">Features & Recent Revisions</div>', unsafe_allow_html=True)

f1, f2 = st.columns(2)
with f1:
    st.markdown("""
**Depth Chart Controls**
- Active/inactive toggle per player (deactivated players are fully excluded)
- Usage multiplier (0.0 to 2.5x) to scale any player's expected volume
- Goalie starter designation
- **Position override**: reassign a player's position for projections
  (e.g. an Attack player lining up at Midfield). Changes POS_DEFAULTS,
  position caps, and zero-inflation rates used in the sim
- Rating overrides per player: directly set goal share, assist share,
  shooting %, 2PT rate, save %, or FO win rate, bypassing the credibility blend

**Session Persistence**
- Save/load all overrides, depth chart, and game selection as a JSON file
- Excel export: Game Lines, Player Props, Depth Chart, Team Projections, Metadata
  (includes blank "Actual Result" column for tracking model accuracy over the season)
""")
with f2:
    st.markdown("""
**Prop Pricing Fix (June 2026)**
- Corrected NegBin parameterization: `p = n/(mu+n)` instead of `p = phi/(mu+phi)`,
  eliminating an 11% mean inflation bug for non-integer phi values
- Recalibrated `PHI_PLAYER["goals"]` from 1.8 to 40 (near-Poisson) after measuring
  actual 2022+ data: attackers show var/mean = 1.02, midfielders 1.07 — nearly Poisson
- Effect: a player projected at 1.96 goals now prices at ~-150 on Over 1.5
  (was -115). Alternate lines and milestone props move proportionally.
  The old parameterization over-dispersed high scorers, inflating their zero
  spike and pushing prop lines too far toward the under

**Gameday Roster Integration (when available)**
- Scraper (`pll_gameday_roster_cache.py`) pulls official PLL gameday rosters
  from `api.stats.premierlacrosseleague.com` once posted
- Priority: gameday roster > current official roster > historical fallback
- Roster source shown in Depth Chart sidebar (green = gameday, blue = official current)
""")

st.markdown("---")
st.markdown(
    '<span class="note-text">'
    "Engine v3 &nbsp;·&nbsp; Possession-chain model &nbsp;·&nbsp; "
    "20,000 Monte Carlo simulations &nbsp;·&nbsp; "
    "Zero-inflated near-Poisson distributions (calibrated 2022-2025) &nbsp;·&nbsp; "
    "Credibility-weighted player ratings (Buehlmann-Straub)"
    "</span>",
    unsafe_allow_html=True,
)
