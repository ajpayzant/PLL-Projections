"""
PLL Projection App — Entry Point
=================================
Streamlit multi-page app for PLL game projections, player props, and betting lines.
Run with:  streamlit run projection_app.py
"""
import os
import streamlit as st

st.set_page_config(
    page_title="PLL Projection Engine",
    page_icon="🥍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared CSS injected once at app root ──────────────────────────────────
st.markdown("""
<style>
  /* Layout */
  .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1800px; }
  section[data-testid="stSidebar"] { min-width: 260px; max-width: 320px; }

  /* Typography */
  h1, h2, h3 { letter-spacing: -0.02em; }

  /* Metric cards */
  .pll-card {
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 12px;
    padding: 12px 16px;
    background: linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    margin-bottom: 8px;
  }
  .pll-card-label {
    color: #94a3b8; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
  }
  .pll-card-value {
    font-size: 1.5rem; font-weight: 800; color: #f1f5f9; line-height: 1.1;
  }
  .pll-card-sub { color: #94a3b8; font-size: 0.78rem; margin-top: 3px; }

  /* Odds pills */
  .odds-fav  { background:#16a34a; color:#fff; border-radius:6px; padding:2px 8px; font-weight:700; font-size:0.85rem; }
  .odds-dog  { background:#2563eb; color:#fff; border-radius:6px; padding:2px 8px; font-weight:700; font-size:0.85rem; }
  .odds-even { background:#475569; color:#fff; border-radius:6px; padding:2px 8px; font-weight:700; font-size:0.85rem; }

  /* Over / Under badges */
  .over-badge  { color:#16a34a; font-weight:700; }
  .under-badge { color:#dc2626; font-weight:700; }

  /* Section divider */
  .pll-divider { border-top: 1px solid rgba(148,163,184,0.15); margin: 16px 0; }

  /* Player row highlight */
  .player-starter { font-weight: 700; }
  .player-inactive { color: #64748b; text-decoration: line-through; }

  /* Small note text */
  .note-text { color: #64748b; font-size: 0.80rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── Landing page ──────────────────────────────────────────────────────────
st.title("🥍 PLL Projection Engine")
st.markdown("##### Monte Carlo projection system for Premier Lacrosse League games")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="pll-card">
      <div class="pll-card-label">📊 Projections</div>
      <div class="pll-card-value">Game Totals</div>
      <div class="pll-card-sub">Team goals, scores, spreads, win %</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="pll-card">
      <div class="pll-card-label">👤 Player Props</div>
      <div class="pll-card-value">Prop Lines</div>
      <div class="pll-card-sub">Goals, assists, points, saves, FO wins</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pll-card">
      <div class="pll-card-label">📋 Depth Charts</div>
      <div class="pll-card-value">Roster Control</div>
      <div class="pll-card-sub">Active/inactive, starters, usage</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="pll-card">
      <div class="pll-card-label">💰 Game Lines</div>
      <div class="pll-card-value">Market Output</div>
      <div class="pll-card-sub">Moneyline, spread, total with hold</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
**How to use:**
1. Go to **Projections** → select an upcoming game → run the projection
2. Go to **Depth Charts** → mark any players inactive or adjust usage
3. Return to **Projections** → re-run to see updated projections
4. Go to **Player Props** → review individual prop lines and milestone prices
5. Go to **Game Lines** → review final market output, override lines as needed

<span class="note-text">Engine: v3 possession-chain model · 20,000 Monte Carlo simulations · Negative Binomial + truncated Normal distributions</span>
""", unsafe_allow_html=True)
