# views/summary_tab.py
"""
Executive Summary Tab - One-page overview of key signals
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from data.fred import SeriesFrame


def render_summary_tab(
    fred_bundle: Dict[str, SeriesFrame],
    regime_result: Dict,
    usrec: Optional[pd.Series],
    lookback_years: int
):
    """
    Executive summary with key signals and changes.
    """
    
    st.header("Executive Summary")
    st.caption(f"Dashboard snapshot as of {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # -----------------------------
    # Educational Framework
    # -----------------------------
    with st.expander("Understanding This Dashboard"):
        st.markdown("""
        **The GIP Framework**
        
        This dashboard classifies the macroeconomic environment using a Growth-Inflation-Policy (GIP) 
        framework. Rather than focusing on absolute levels, the system measures **momentum** - whether 
        conditions are accelerating or decelerating relative to recent history.
        
        | Regime | Growth | Inflation | Typical Asset Response |
        |--------|--------|-----------|------------------------|
        | Goldilocks | Accelerating | Falling | Risk assets outperform |
        | Reflation | Accelerating | Rising | Real assets, value sectors |
        | Stagflation | Decelerating | Rising | Defensive, cash |
        | Deflation | Decelerating | Falling | Duration, quality |
        
        **Key Principle: Second Derivative Matters**
        
        Markets price in the *change* in data, not the level. GDP growing at 3% (down from 4%) 
        registers as "Decelerating" even though 3% is historically strong. This distinction drives 
        tactical positioning decisions.
        
        **Data Sources**
        
        All economic data is sourced from the Federal Reserve Economic Data (FRED) database. 
        Market data is sourced from Yahoo Finance. Signals are computed using z-scores normalized 
        against post-2021 history to avoid COVID-era distortions.
        """)
    
    # -----------------------------
    # Current Regime Box
    # -----------------------------
    st.subheader("Current Macro Regime")
    
    regime = regime_result.get("regime", "Unknown")
    growth_class = regime_result.get("growth_classification", "Unknown")
    inflation_class = regime_result.get("inflation_classification", "Unknown")
    liquidity_class = regime_result.get("liquidity_classification", "Unknown")
    curve_regime = regime_result.get("curve_regime", "Unknown")
    
    # Regime color mapping
    regime_colors = {
        "Goldilocks": "#2ecc71",  # Green
        "Reflation": "#f39c12",   # Orange
        "Stagflation": "#e74c3c", # Red
        "Deflation": "#3498db",   # Blue
    }
    
    regime_color = regime_colors.get(regime, "#95a5a6")
    
    # Main regime display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: {regime_color}20;
                border-left: 4px solid {regime_color};
                padding: 20px;
                border-radius: 4px;
            ">
                <h2 style="margin: 0; color: {regime_color};">{regime}</h2>
                <p style="margin: 5px 0 0 0; font-size: 14px;">Current Macro Regime</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("**Regime Components:**")
        
        component_col1, component_col2, component_col3 = st.columns(3)
        
        with component_col1:
            growth_icon = "+" if growth_class == "Accelerating" else ("-" if growth_class == "Decelerating" else "=")
            st.metric(
                "Growth", 
                growth_class, 
                growth_icon,
                help="Composite of Industrial Production, Retail Sales, Employment, and Hours Worked. "
                     "Accelerating = z-score > 0.5 | Decelerating = z-score < -0.5"
            )
        
        with component_col2:
            inflation_icon = "+" if inflation_class == "Rising" else ("-" if inflation_class == "Falling" else "=")
            st.metric(
                "Inflation", 
                inflation_class, 
                inflation_icon,
                help="Composite of Core CPI, Core PCE (2x weight), and Breakeven Inflation. "
                     "Rising = z-score > 0.5 | Falling = z-score < -0.5"
            )
        
        with component_col3:
            liquidity_icon = "+" if liquidity_class == "Easing" else ("-" if liquidity_class == "Tightening" else "=")
            st.metric(
                "Liquidity", 
                liquidity_class, 
                liquidity_icon,
                help="Composite of Real Yields (inverted), Credit Spreads (inverted), NFCI (inverted), "
                     "and 2Y Treasury (inverted). Falling inputs = Easing conditions."
            )

    st.caption(
        "Regime classification is based on MOMENTUM (rate of change), not absolute levels. "
        "A strong economy that is slowing registers as 'Decelerating' even if growth remains above trend. "
        "See Nowcast tab for level-based quadrant analysis."
    )
    
    st.divider()
    
    # -----------------------------
    # Key Signals Grid
    # -----------------------------
    st.subheader("Key Signals")
    
    with st.expander("Signal Interpretation Guide"):
        st.markdown("""
        **Reading the Signals**
        
        Each signal displays a z-score measuring standard deviations from the historical mean:
        
        | Z-Score | Interpretation | Typical Action |
        |---------|----------------|----------------|
        | > +2.0 | Extremely elevated | High conviction signal |
        | +1.0 to +2.0 | Above average | Moderate signal |
        | -1.0 to +1.0 | Normal range | No action required |
        | -1.0 to -2.0 | Below average | Moderate signal |
        | < -2.0 | Extremely depressed | High conviction signal |
        
        **Color Coding**
        
        - **Green**: Favorable for risk assets
        - **Red**: Unfavorable for risk assets  
        - **Blue**: Neutral or mixed implications
        
        Note: For inflation, *falling* is favorable (green) while *rising* is unfavorable (red).
        """)
    
    signals = _compute_key_signals(fred_bundle, regime_result)
    
    # Display signals in a grid
    cols = st.columns(3)
    
    for i, signal in enumerate(signals):
        with cols[i % 3]:
            if signal["status"] == "positive":
                st.success(f"**{signal['name']}**: {signal['message']}")
            elif signal["status"] == "negative":
                st.error(f"**{signal['name']}**: {signal['message']}")
            else:
                st.info(f"**{signal['name']}**: {signal['message']}")
    
    st.divider()
    
    # -----------------------------
    # Risk Watchlist
    # -----------------------------
    st.subheader("Risk Watchlist")
    
    st.caption(
        "Automated screening for elevated risk conditions. Triggers are based on "
        "z-score thresholds, percentile rankings, and regime classifications."
    )
    
    risks = _compute_risk_watchlist(fred_bundle, regime_result)
    
    if risks:
        for risk in risks:
            severity_color = {"high": "red", "medium": "orange", "low": "blue"}.get(risk["severity"], "gray")
            st.markdown(
                f"- **[{risk['severity'].upper()}]** {risk['description']}"
            )
        
        with st.expander("Risk Trigger Thresholds"):
            st.markdown("""
            | Risk Type | Trigger Condition |
            |-----------|-------------------|
            | Stagflation | Current regime = Stagflation |
            | Inflation Spike | Inflation z-score > 1.5 |
            | Growth Collapse | Growth z-score < -1.5 |
            | Liquidity Crunch | Liquidity z-score < -1.0 |
            | Valuation Headwind | 10Y Real Yield > 2.0% |
            | Recession Signal | 2s10s Curve < -20bps |
            """)
    else:
        st.success("No elevated risks detected")
    
    st.divider()
    
    # -----------------------------
    # Positioning Implications
    # -----------------------------
    st.subheader("Positioning Implications")
    
    st.caption(
        "Suggested positioning based on historical asset class performance in each regime. "
        "These are directional guides, not investment recommendations."
    )
    
    implications = _get_regime_implications(regime, growth_class, inflation_class, liquidity_class)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Favored:**")
        for item in implications.get("favored", []):
            st.markdown(f"- {item}")
    
    with col2:
        st.markdown("**Avoid:**")
        for item in implications.get("avoid", []):
            st.markdown(f"- {item}")
    
    with st.expander("Regime Playbook Reference"):
        st.markdown("""
        **Goldilocks (Growth Up, Inflation Down)**
        
        The optimal environment for risk assets. Earnings grow while discount rates remain stable. 
        Favor beta exposure, growth factors, and cyclical sectors. Duration is less critical.
        
        **Reflation (Growth Up, Inflation Up)**
        
        Strong growth but rising price pressures. Nominal earnings benefit but real returns compress. 
        Favor real assets (commodities, TIPS), value sectors, and short duration. Financials benefit 
        from steeper curves.
        
        **Stagflation (Growth Down, Inflation Up)**
        
        The most challenging environment. Earnings decline while costs rise. Cash and defensive 
        positioning outperform. Quality factor and stable dividend payers provide relative safety. 
        Avoid leverage and duration.
        
        **Deflation (Growth Down, Inflation Down)**
        
        Recessionary conditions. Flight to quality dominates. Long-duration Treasuries outperform 
        as rates fall. Avoid cyclical exposure and credit risk. USD typically strengthens.
        """)
    
    st.divider()
    
    # -----------------------------
    # Data Freshness
    # -----------------------------
    st.subheader("Data Freshness")
    
    st.caption(
        "Monitoring data pipeline integrity. Economic data updates on varying schedules: "
        "daily (yields, spreads), weekly (claims, NFCI), monthly (CPI, payrolls), quarterly (GDP)."
    )
    
    freshness = _check_data_freshness(fred_bundle)
    
    stale_series = [f for f in freshness if f["days_old"] > 7]
    
    if stale_series:
        st.warning(f"{len(stale_series)} series have data older than 7 days")
        with st.expander("View stale data"):
            for s in stale_series:
                st.markdown(f"- **{s['series']}**: Last update {s['days_old']} days ago ({s['last_date']})")
    else:
        st.success("All data is current (within 7 days)")


def _compute_key_signals(fred_bundle: Dict, regime_result: Dict) -> list:
    """Compute key signals from data"""
    signals = []
    
    # Growth signal
    growth_score = regime_result.get("growth_score", 0)
    if growth_score > 1.0:
        signals.append({
            "name": "Growth Momentum",
            "message": f"Strong acceleration (z={growth_score:.1f})",
            "status": "positive"
        })
    elif growth_score < -1.0:
        signals.append({
            "name": "Growth Momentum",
            "message": f"Sharp deceleration (z={growth_score:.1f})",
            "status": "negative"
        })
    else:
        signals.append({
            "name": "Growth Momentum",
            "message": f"Stable (z={growth_score:.1f})",
            "status": "neutral"
        })
    
    # Inflation signal
    inflation_score = regime_result.get("inflation_score", 0)
    if inflation_score > 1.0:
        signals.append({
            "name": "Inflation Pressure",
            "message": f"Accelerating (z={inflation_score:.1f})",
            "status": "negative"
        })
    elif inflation_score < -1.0:
        signals.append({
            "name": "Inflation Pressure",
            "message": f"Decelerating (z={inflation_score:.1f})",
            "status": "positive"
        })
    else:
        signals.append({
            "name": "Inflation Pressure",
            "message": f"Stable (z={inflation_score:.1f})",
            "status": "neutral"
        })
    
    # Liquidity signal
    liquidity_score = regime_result.get("liquidity_score", 0)
    if liquidity_score > 0.5:
        signals.append({
            "name": "Financial Conditions",
            "message": f"Easing (z={liquidity_score:.1f})",
            "status": "positive"
        })
    elif liquidity_score < -0.5:
        signals.append({
            "name": "Financial Conditions",
            "message": f"Tightening (z={liquidity_score:.1f})",
            "status": "negative"
        })
    else:
        signals.append({
            "name": "Financial Conditions",
            "message": f"Neutral (z={liquidity_score:.1f})",
            "status": "neutral"
        })
    
    # Real yields signal
    if "DFII10" in fred_bundle:
        real_yield = fred_bundle["DFII10"].df["Value"].dropna()
        if not real_yield.empty:
            current_ry = real_yield.iloc[-1]
            if current_ry > 2.0:
                signals.append({
                    "name": "Real Yields",
                    "message": f"Elevated at {current_ry:.2f}% (headwind for risk)",
                    "status": "negative"
                })
            elif current_ry < 0.5:
                signals.append({
                    "name": "Real Yields",
                    "message": f"Low at {current_ry:.2f}% (supportive for risk)",
                    "status": "positive"
                })
            else:
                signals.append({
                    "name": "Real Yields",
                    "message": f"Moderate at {current_ry:.2f}%",
                    "status": "neutral"
                })
    
    # Curve signal
    curve_regime = regime_result.get("curve_regime", "Unknown")
    if "Bear" in curve_regime:
        signals.append({
            "name": "Yield Curve",
            "message": f"{curve_regime} (rates rising)",
            "status": "negative"
        })
    elif "Bull" in curve_regime:
        signals.append({
            "name": "Yield Curve",
            "message": f"{curve_regime} (rates falling)",
            "status": "positive"
        })
    else:
        signals.append({
            "name": "Yield Curve",
            "message": "Neutral",
            "status": "neutral"
        })
    
    # Credit spreads signal
    if "BAMLH0A0HYM2" in fred_bundle:
        hy_spread = fred_bundle["BAMLH0A0HYM2"].df["Value"].dropna()
        if not hy_spread.empty and len(hy_spread) > 252:
            current = hy_spread.iloc[-1]
            pctl = (hy_spread <= current).sum() / len(hy_spread)
            if pctl > 0.8:
                signals.append({
                    "name": "Credit Spreads",
                    "message": f"Wide ({pctl:.0%} percentile) - stress elevated",
                    "status": "negative"
                })
            elif pctl < 0.2:
                signals.append({
                    "name": "Credit Spreads",
                    "message": f"Tight ({pctl:.0%} percentile) - complacent",
                    "status": "neutral"
                })
            else:
                signals.append({
                    "name": "Credit Spreads",
                    "message": f"Normal ({pctl:.0%} percentile)",
                    "status": "positive"
                })
    
    return signals


def _compute_risk_watchlist(fred_bundle: Dict, regime_result: Dict) -> list:
    """Identify elevated risks"""
    risks = []
    
    # Stagflation risk
    regime = regime_result.get("regime", "")
    if regime == "Stagflation":
        risks.append({
            "severity": "high",
            "description": "Currently in Stagflation regime - weak growth with sticky inflation"
        })
    
    # Inflation re-acceleration
    inflation_score = regime_result.get("inflation_score", 0)
    if inflation_score > 1.5:
        risks.append({
            "severity": "high",
            "description": f"Inflation momentum elevated (z-score: {inflation_score:.1f})"
        })
    
    # Growth deceleration
    growth_score = regime_result.get("growth_score", 0)
    if growth_score < -1.5:
        risks.append({
            "severity": "high",
            "description": f"Growth momentum deteriorating rapidly (z-score: {growth_score:.1f})"
        })
    
    # Liquidity tightening
    liquidity_score = regime_result.get("liquidity_score", 0)
    if liquidity_score < -1.0:
        risks.append({
            "severity": "medium",
            "description": f"Financial conditions tightening (z-score: {liquidity_score:.1f})"
        })
    
    # Real yields elevated
    if "DFII10" in fred_bundle:
        real_yield = fred_bundle["DFII10"].df["Value"].dropna()
        if not real_yield.empty and real_yield.iloc[-1] > 2.0:
            risks.append({
                "severity": "medium",
                "description": f"Real yields elevated at {real_yield.iloc[-1]:.2f}% - headwind for valuations"
            })
    
    # Curve inversion
    if "DGS2" in fred_bundle and "DGS10" in fred_bundle:
        dgs2 = fred_bundle["DGS2"].df["Value"].dropna()
        dgs10 = fred_bundle["DGS10"].df["Value"].dropna()
        if not dgs2.empty and not dgs10.empty:
            spread = dgs10.iloc[-1] - dgs2.iloc[-1]
            if spread < -0.2:
                risks.append({
                    "severity": "medium",
                    "description": f"Yield curve inverted ({spread*100:.0f}bps) - historical recession signal"
                })
    
    return risks


def _get_regime_implications(regime: str, growth: str, inflation: str, liquidity: str) -> Dict:
    """Get positioning implications for current regime"""
    
    implications = {
        "Goldilocks": {
            "favored": [
                "Risk assets (equities, credit)",
                "Growth/momentum factors",
                "Cyclical sectors (Tech, Discretionary, Industrials)",
                "Duration neutral"
            ],
            "avoid": [
                "Defensive positioning",
                "Cash/short-term instruments",
                "Inflation hedges (unless cheap)"
            ]
        },
        "Reflation": {
            "favored": [
                "Real assets (commodities, TIPS)",
                "Value and cyclical sectors",
                "Financials, Energy, Materials",
                "Short duration"
            ],
            "avoid": [
                "Long duration bonds",
                "Growth/tech at high multiples",
                "Defensive sectors"
            ]
        },
        "Stagflation": {
            "favored": [
                "Cash and short-term instruments",
                "Defensive sectors (Utilities, Healthcare, Staples)",
                "Gold and inflation hedges",
                "Quality factor"
            ],
            "avoid": [
                "Cyclical risk",
                "High-beta equities",
                "Long duration",
                "Credit risk"
            ]
        },
        "Deflation": {
            "favored": [
                "Long duration treasuries",
                "Quality and low-volatility factors",
                "Defensive sectors",
                "USD cash"
            ],
            "avoid": [
                "Cyclical sectors",
                "Commodities",
                "High yield credit",
                "Emerging markets"
            ]
        }
    }
    
    return implications.get(regime, {"favored": [], "avoid": []})


def _check_data_freshness(fred_bundle: Dict) -> list:
    """Check how fresh each data series is"""
    freshness = []
    today = pd.Timestamp.now()
    
    for key, sf in fred_bundle.items():
        if hasattr(sf, 'df') and not sf.df.empty:
            last_date = sf.df.index.max()
            days_old = (today - last_date).days
            freshness.append({
                "series": key,
                "last_date": last_date.strftime("%Y-%m-%d"),
                "days_old": days_old
            })
    
    return sorted(freshness, key=lambda x: x["days_old"], reverse=True)