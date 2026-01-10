# app.py
"""
Top-Down Macro Strategy Dashboard V2

Systematic regime identification and cross-asset transmission analysis.
"""

import datetime as dt
import logging

import streamlit as st

import config
from data.fred import FREDClient
from data.yfinance import MarketClient
from data.yfinance import YFinanceClient  
from models.regime import RegimeEngine
    
# Import tab renderers
from views.regime_tab import render_regime_tab
from views.transmission_tab import render_transmission_tab
from views.rates_tab import render_rates_tab
from views.scoreboard_tab import render_scoreboard_tab
from views.nowcast_tab import render_nowcast_tab
from views.surprises_tab import render_surprises_tab
from views.summary_tab import render_summary_tab
from views.sectors_tab import render_sectors_tab
from views.heatmap_tab import render_heatmap_tab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Macro Terminal V2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Top-Down Macro Strategy")
st.caption(
    "Systematic regime identification and cross-asset transmission. "
    "Research-driven framework for macro analysis."
)

# ========================================
# SIDEBAR CONTROLS
# ========================================
with st.sidebar:
    st.subheader("Controls")
    
    # Date range
    market_start = st.date_input(
        "Market data start",
        value=dt.date(2020, 1, 1),
        help="Start date for market data (ETFs, equities)"
    )
    
    lookback_years = st.slider(
        "Macro lookback (years)",
        min_value=5,
        max_value=80,
        value=10,
        step=1,
        help="Historical window for charts"
    )
    
    st.divider()
    
    # Chart options
    st.subheader("Chart Options")
    
    normalize_overlays = st.checkbox(
        "Normalize overlays",
        value=True,
        help="Index overlays to 100 for visual comparison"
    )
    
    show_beta = st.checkbox(
        "Show rolling beta",
        value=True,
        help="Display rolling beta in transmission tab"
    )
    
    beta_window_months = st.slider(
        "Beta window (months)",
        min_value=12,
        max_value=60,
        value=24,
        step=6,
        help="Window for rolling beta calculation"
    )
    
    st.divider()
    
    # Info
    st.subheader("Data Sources")
    st.caption(
        "**FRED**: Macro & rates data\n\n"
        "**YFinance**: Market data (ETFs, equities)\n\n"
        "**Note**: This is for research/education. "
        "For production trading, use paid data feeds."
    )
    
    # Refresh button
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ========================================
# DATA LOADING
# ========================================
@st.cache_data(ttl=3600, show_spinner=True)
def load_all_data(market_start_str: str):
    """
    Load all FRED and market data.
    
    Args:
        market_start_str: Start date for market data
        
    Returns:
        Tuple of (fred_bundle, market_bundle)
    """
    logger.info("Loading data...")
    
    # Get API key from secrets
    try:
        fred_key = st.secrets["FRED_API_KEY"]
    except Exception as e:
        st.error(
            "Missing FRED API key. Add FRED_API_KEY to .streamlit/secrets.toml\n\n"
            f"Error: {e}"
        )
        st.stop()
    
    # Initialize clients
    fred_client = FREDClient(api_key=fred_key)
    market_client = MarketClient()
    
    # Load FRED data
    with st.spinner("Loading FRED data..."):
        fred_bundle = fred_client.get_bundle(config.DEFAULT_FRED_IDS)
    
    # Load market data
    with st.spinner("Loading market data..."):
        market_bundle = market_client.get_bundle(
            config.DEFAULT_ASSET_KEYS,
            start=market_start_str
        )
    
    logger.info(f"Loaded {len(fred_bundle)} FRED series, {len(market_bundle)} market assets")
    
    return fred_bundle, market_bundle


# Load data
fred_bundle, market_bundle = load_all_data(str(market_start))

# Check if we got any data
if not fred_bundle:
    st.error("Failed to load FRED data. Check your API key and connection.")
    st.stop()

if not market_bundle:
    st.warning("Failed to load market data. Some features may be unavailable.")

# DEBUG: Check what actually loaded
st.sidebar.divider()
st.sidebar.subheader("Debug Info")
st.sidebar.write(f"FRED series loaded: {len(fred_bundle)}")
st.sidebar.write(f"Market assets loaded: {len(market_bundle)}")

with st.sidebar.expander("Show loaded assets"):
    st.write("**FRED:**")
    st.write(list(fred_bundle.keys()))
    st.write("**Market:**")
    st.write(list(market_bundle.keys()))

# ========================================
# REGIME CLASSIFICATION
# ========================================
@st.cache_data(ttl=600, show_spinner=False)
def classify_current_regime(fred_data_dict):
    """
    Run regime engine on current data.
    
    Args:
        fred_data_dict: Serialized fred_bundle for caching
        
    Returns:
        RegimeState
    """
    logger.info("Classifying regime...")
    
    engine = RegimeEngine()
    
    # Convert back to dict of DataFrames
    regime_data = {k: v.df for k, v in fred_data_dict.items()}
    
    return engine.classify(regime_data)


# Classify regime
with st.spinner("Analyzing regime..."):
    current_regime = classify_current_regime(fred_bundle)

# Convert RegimeState to dict for summary tab
regime_result = {
    "regime": current_regime.macro.value.title(),  # 'reflation' -> 'Reflation'
    "growth_score": current_regime.growth_score,
    "inflation_score": current_regime.inflation_score,
    "liquidity_score": current_regime.liquidity_score,
    "growth_classification": current_regime.growth.value.title(),  # 'accelerating' -> 'Accelerating'
    "inflation_classification": current_regime.inflation.value.title(),  # 'rising' -> 'Rising'
    "liquidity_classification": current_regime.liquidity.value.title(),  # 'tightening' -> 'Tightening'
    "curve_regime": current_regime.curve_regime,
}

# ========================================
# RECESSION DATA
# ========================================
usrec = None
if "USREC" in fred_bundle:
    usrec = fred_bundle["USREC"].df["Value"].astype(float)

# ========================================
# TABS
# ========================================
tabs = st.tabs([
    "Summary",         
    "GIP Regime",
    "Heatmap",         
    "Nowcast",
    "Surprises",
    "Sectors",          
    "Transmission",
    "Rates & Credit",
    "Scoreboard"
])

# Tab 1: Summary (NEW)
with tabs[0]:
    render_summary_tab(
        fred_bundle=fred_bundle,
        regime_result=regime_result,
        usrec=usrec,
        lookback_years=lookback_years
    )

# Tab 2: GIP Regime
with tabs[1]:
    render_regime_tab(
        fred_bundle=fred_bundle,
        current_regime=current_regime,
        usrec=usrec,
        lookback_years=lookback_years
    )

# Tab 3: Heatmap
with tabs[2]:
    render_heatmap_tab(
        fred_bundle=fred_bundle,
        lookback_years=lookback_years
    )

# Tab 4: Nowcast (NEW)
with tabs[3]:
    render_nowcast_tab(
        fred_bundle=fred_bundle,
        usrec=usrec,
        lookback_years=lookback_years
    )

# Tab 5: Surprises
with tabs[4]:
    render_surprises_tab(
        fred_bundle=fred_bundle,
        fred_client=None,  # No longer needed - we build our own index
        usrec=usrec,
        lookback_years=lookback_years
    )

# Tab 6: Sectors (NEW)
with tabs[5]:
    render_sectors_tab(
        yf_client=YFinanceClient(),
        lookback_years=lookback_years
    )

# Tab 7: Transmission
with tabs[6]:
    render_transmission_tab(
        fred_bundle=fred_bundle,
        market_bundle=market_bundle,
        usrec=usrec,
        lookback_years=lookback_years,
        normalize=normalize_overlays,
        show_beta=show_beta,
        beta_window=beta_window_months
    )

# Tab 8: Rates & Credit
with tabs[7]:
    render_rates_tab(
        fred_bundle=fred_bundle,
        market_bundle=market_bundle,
        current_regime=current_regime,
        usrec=usrec,
        lookback_years=lookback_years
    )

# Tab 9: Scoreboard
with tabs[8]:
    render_scoreboard_tab(
        fred_bundle=fred_bundle,
        current_regime=current_regime
    )


# ========================================
# FOOTER
# ========================================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Last updated: {current_regime.timestamp.strftime('%Y-%m-%d %H:%M')}")

with col2:
    st.caption(f"Data confidence: {current_regime.confidence:.1%}")

with col3:
    st.caption("Research-driven macro framework")
