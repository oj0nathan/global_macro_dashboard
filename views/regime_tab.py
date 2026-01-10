# views/regime_tab.py
"""
GIP Regime Dashboard Tab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional

from data.fred import SeriesFrame, compute_transforms
from models.regime import RegimeState, compute_macro_pulse

def render_regime_tab(
    fred_bundle: Dict[str, SeriesFrame],
    current_regime: RegimeState,
    usrec: Optional[pd.Series],
    lookback_years: int
):
    """
    Render the GIP Regime tab.
    
    Shows:
    - Current regime classification
    - Growth indicators
    - Inflation indicators  
    - Liquidity indicators
    - Curve regime
    """
    
    st.header("Growth-Inflation-Policy Regime")

    # Educational panel
    with st.expander("Understanding the GIP Framework"):
        st.markdown("""
        **The Core Principle: Momentum Over Levels**
        
        This framework measures the *rate of change* in economic conditions, not absolute levels. 
        Markets discount the future - by the time GDP prints at 3%, equities have already priced it. 
        What moves assets is whether that 3% is accelerating toward 4% or decelerating toward 2%.
        
        > "Define everything into regimes to see the highest probability for returns. 
        > Then identify any discontinuity from past returns."
        
        ---
        
        **The Three Pillars**
        
        | Pillar | What It Measures | Key Inputs | Weighting Logic |
        |--------|------------------|------------|-----------------|
        | Growth | Real economic acceleration | INDPRO, RSAFS, PAYEMS, UNRATE, AWHMAN | Equal weight; UNRATE inverted |
        | Inflation | Price pressure momentum | Core CPI, Core PCE, 5Y/10Y Breakevens | Core PCE 2x weight (Fed's target) |
        | Liquidity | Financial conditions transmission | Real Yields, HY Spreads, NFCI, 2Y Treasury | All inverted (falling = easing) |
        
        ---
        
        **Z-Score Methodology**
        
        Each input is converted to a z-score measuring standard deviations from historical mean:
        
        1. Compute 1-month, 3-month, and 6-month percent changes
        2. Filter history to post-June 2021 (avoids COVID distortion)
        3. Calculate z-score: (Current - Mean) / StdDev
        4. Average across time horizons
        5. Clip to +/- 2.0 (prevents outlier dominance)
        
        | Z-Score | Classification | Interpretation |
        |---------|----------------|----------------|
        | > +0.5 | Accelerating/Rising/Easing | Momentum is positive |
        | -0.5 to +0.5 | Stable/Neutral | No clear direction |
        | < -0.5 | Decelerating/Falling/Tightening | Momentum is negative |
        
        ---
        
        **Macro Quadrant Classification**
        
        The Growth and Inflation scores combine to determine the macro regime:
        
        | Quadrant | Growth | Inflation | Market Implication |
        |----------|--------|-----------|-------------------|
        | Goldilocks | Positive | Negative | Optimal for risk assets; earnings grow, rates stable |
        | Reflation | Positive | Positive | Favor real assets; nominal growth but inflation drag |
        | Stagflation | Negative | Positive | Defensive posture; worst environment for equities |
        | Deflation | Negative | Negative | Duration outperforms; recessionary conditions |
        
        ---
        
        **Curve Regime Analysis**
        
        The yield curve shape reflects market expectations for growth and policy:
        
        | Regime | Mechanism | Signal |
        |--------|-----------|--------|
        | Bull Steepener | Front-end falls faster | Fed easing expectations; early cycle |
        | Bear Steepener | Long-end rises faster | Inflation/term premium pressure |
        | Bull Flattener | Long-end falls faster | Growth scare; flight to safety |
        | Bear Flattener | Front-end rises faster | Fed hawkish; late cycle |
        
        Classification uses 63-day (3-month) changes in 2Y and 10Y yields. A spread change 
        greater than +/-10bps triggers a steepening/flattening label.
        """)
    
    # -----------------------------
    # Regime Summary Cards
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Macro Regime",
            current_regime.macro.value.replace("_", " ").title(),
            help="Quadrant classification based on Growth x Inflation momentum. "
                 "Goldilocks (G+/I-), Reflation (G+/I+), Stagflation (G-/I+), Deflation (G-/I-)."
        )
    
    with col2:
        delta_color = "normal" if current_regime.growth_score > 0 else "inverse"
        st.metric(
            "Growth",
            current_regime.growth.value.title(),
            delta=f"Score: {current_regime.growth_score:+.2f}",
            delta_color=delta_color,
            help="Composite z-score of Industrial Production, Retail Sales, Payrolls, "
                 "Unemployment (inverted), and Hours Worked. Range: -2 to +2."
        )
    
    with col3:
        delta_color = "inverse" if current_regime.inflation_score > 0 else "normal"
        st.metric(
            "Inflation",
            current_regime.inflation.value.title(),
            delta=f"Score: {current_regime.inflation_score:+.2f}",
            delta_color=delta_color,
            help="Composite z-score of Core CPI, Core PCE (2x weight), and Breakeven Inflation. "
                 "Rising inflation = negative for risk assets. Range: -2 to +2."
        )
    
    with col4:
        delta_color = "normal" if current_regime.liquidity_score > 0 else "inverse"
        st.metric(
            "Liquidity",
            current_regime.liquidity.value.title(),
            delta=f"Score: {current_regime.liquidity_score:+.2f}",
            delta_color=delta_color,
            help="Composite of Real Yields (2x), HY Spreads (1.5x), NFCI (1.5x), 2Y Treasury (1x). "
                 "All inverted: falling inputs = positive score = easier conditions."
        )
    
    # Confidence bar
    st.progress(
        current_regime.confidence,
        text=f"Data Confidence: {current_regime.confidence:.1%} ({int(current_regime.confidence*100)}/100)"
    )
    
    st.caption(
        f"Last updated: {current_regime.timestamp.strftime('%Y-%m-%d')} | "
        f"Curve: {current_regime.curve_regime.replace('_', ' ').title()} "
        f"(2s10s: {current_regime.curve_2s10s:.0f}bps)"
    )
    
    st.divider()
    
    # -----------------------------
    # Growth Section
    # -----------------------------
    st.subheader("Growth Indicators")
    
    with st.expander("Growth Methodology"):
        st.markdown("""
        **What We Measure**
        
        The growth composite captures real economic activity momentum across production, 
        consumption, and labor markets. We deliberately exclude financial variables to 
        isolate the "real" economy.
        
        | Indicator | FRED ID | Frequency | Why It Matters |
        |-----------|---------|-----------|----------------|
        | Industrial Production | INDPRO | Monthly | Proxy for business investment and inventory cycles |
        | Retail Sales | RSAFS | Monthly | Best monthly proxy for PCE (68% of GDP) |
        | Nonfarm Payrolls | PAYEMS | Monthly | Broadest measure of labor demand |
        | Unemployment Rate | UNRATE | Monthly | Labor slack indicator (inverted in scoring) |
        | Avg Weekly Hours | AWHMAN | Monthly | Labor intensity; leading indicator of hiring |
        
        **Interpretation**
        
        - **Accelerating (Score > +0.5)**: Economy gaining momentum; favor cyclicals
        - **Stable (-0.5 to +0.5)**: Steady state; positioning depends on other factors
        - **Decelerating (Score < -0.5)**: Economy losing momentum; reduce risk
        
        Note: A score of +2.0 means growth is 2 standard deviations stronger than 
        the post-2021 average - an extremely rare reading.
        """)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        if "INDPRO" in fred_bundle:
            indpro = fred_bundle["INDPRO"]
            tf = compute_transforms(indpro)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "Industrial Production (YoY)",
                    f"{last['YoY']:.2f}%",
                    delta=f"{last['Chg_3']:+.2f}pp (3m)",
                    help="Year-over-year change in industrial output index. "
                         "Captures manufacturing, mining, and utilities. "
                         "3m delta shows recent momentum shift."
                )
                
                # Chart
                fig = _line_chart(
                    tf["YoY"],
                    "Industrial Production (YoY %)",
                    usrec,
                    lookback_years
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        if "RSAFS" in fred_bundle:
            rsafs = fred_bundle["RSAFS"]
            tf = compute_transforms(rsafs)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "Retail Sales (YoY)",
                    f"{last['YoY']:.2f}%",
                    delta=f"{last['Chg_3']:+.2f}pp (3m)",
                    help="Year-over-year change in retail sales. "
                         "Primary monthly proxy for consumer spending (PCE). "
                         "Receives highest weight (40%) in GDP nowcast."
                )
                
                fig = _line_chart(
                    tf["YoY"],
                    "Retail Sales (YoY %)",
                    usrec,
                    lookback_years
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Labor market
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        if "PAYEMS" in fred_bundle:
            payems = fred_bundle["PAYEMS"]
            tf = compute_transforms(payems)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "Nonfarm Payrolls",
                    f"{last['Value']/1000:.1f}M",
                    delta=f"{last['Chg_1']:+.0f}K (MoM)",  
                    help="Total nonfarm employment in thousands. "
                         "Monthly change (MoM) is the headline NFP number. "
                         "Positive = job gains; negative = job losses."
                )
    
    with col_l2:
        if "UNRATE" in fred_bundle:
            unrate = fred_bundle["UNRATE"]
            tf = compute_transforms(unrate)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "Unemployment Rate",
                    f"{last['Value']:.1f}%",
                    delta=f"{last['Chg_3']:+.1f}bps (3m)",
                    delta_color="inverse",
                    help="Civilian unemployment rate. Rising unemployment is negative "
                         "for growth (hence inverted delta color). "
                         "Fed targets maximum employment alongside price stability."
                )
    
    st.divider()
    
    # -----------------------------
    # Inflation Section
    # -----------------------------
    st.subheader("Inflation Indicators")
    
    with st.expander("Inflation Methodology"):
        st.markdown("""
        **What We Measure**
        
        The inflation composite tracks price pressure momentum using both realized 
        inflation (CPI, PCE) and market expectations (breakevens).
        
        | Indicator | FRED ID | Weight | Why It Matters |
        |-----------|---------|--------|----------------|
        | Core CPI | CPILFESL | 1.0x | Widely watched; excludes food/energy |
        | Core PCE | PCEPILFE | 2.0x | Fed's explicit target measure |
        | 5Y Breakeven | T5YIE | 1.2x | Market inflation expectations (medium-term) |
        | 10Y Breakeven | T10YIE | 1.0x | Market inflation expectations (long-term) |
        
        **Why Core PCE Gets 2x Weight**
        
        The Federal Reserve explicitly targets Core PCE at 2%. Their policy decisions - 
        rate hikes, cuts, balance sheet - are anchored to this measure. When Core PCE 
        moves, the Fed reacts. CPI matters for headlines; PCE matters for policy.
        
        **Breakeven Inflation**
        
        Breakevens = Nominal Treasury Yield - TIPS Yield. This spread represents what 
        bond investors expect inflation to average. Rising breakevens signal the market 
        is pricing in higher future inflation.
        
        **Interpretation**
        
        - **Rising (Score > +0.5)**: Inflation accelerating; negative for duration, mixed for equities
        - **Stable (-0.5 to +0.5)**: Inflation contained; neutral signal
        - **Falling (Score < -0.5)**: Disinflation; positive for duration, supports multiples
        """)
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        if "CPIAUCSL" in fred_bundle:
            cpi = fred_bundle["CPIAUCSL"]
            tf = compute_transforms(cpi)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "CPI (Headline, NSA)",
                    f"{last['YoY']:.2f}%",
                    delta=f"{last['Chg_3']:+.2f}pp (3m)",
                    delta_color="inverse",
                    help="Consumer Price Index, year-over-year. Includes food and energy. "
                         "NSA = Not Seasonally Adjusted. This is the 'headline' inflation "
                         "number reported in media."
                )
                
                fig = _line_chart(
                    tf["YoY"],
                    "CPI (YoY %, NSA)",
                    usrec,
                    lookback_years
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with col_i2:
        if "PCEPILFE" in fred_bundle:
            pce = fred_bundle["PCEPILFE"]
            tf = compute_transforms(pce)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "Core PCE (Fed's Preferred)",
                    f"{last['YoY']:.2f}%",
                    delta=f"{last['Chg_3']:+.2f}pp (3m)",
                    delta_color="inverse",
                    help="Personal Consumption Expenditures excluding food and energy. "
                         "The Federal Reserve's explicit 2% target. Receives 2x weight "
                         "in the inflation composite score."
                )
                
                fig = _line_chart(
                    tf["YoY"],
                    "Core PCE (YoY %)",
                    usrec,
                    lookback_years
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Breakevens
    if "T10YIE" in fred_bundle:
        t10 = fred_bundle["T10YIE"]
        tf = compute_transforms(t10)
        
        if not tf.empty:
            last = tf.dropna().iloc[-1]
            st.metric(
                "10Y Breakeven Inflation",
                f"{last['Value']:.2f}%",
                delta=f"{last['Chg_3']:+.0f}bps (3m)",
                help="Market-implied 10-year inflation expectation. "
                     "Calculated as 10Y Nominal Treasury minus 10Y TIPS yield. "
                     "Rising breakevens = market expects higher inflation."
            )
    
    st.divider()
    
    # -----------------------------
    # Liquidity Section
    # -----------------------------
    st.subheader("Liquidity / Financial Conditions")
    
    with st.expander("Liquidity Methodology"):
        st.markdown("""
        **The Core Insight**
        
        > "Real rates pushed higher - ES struggled and gold came under pressure. 
        > When real rates fell, liquidity spilled into equities, gold and tech."
        
        Liquidity measures whether financial conditions support or hinder risk-taking. 
        Unlike growth and inflation (where "up" is directionally clear), liquidity 
        inputs are **inverted** - falling values indicate easier conditions.
        
        | Indicator | FRED ID | Weight | Inversion Logic |
        |-----------|---------|--------|-----------------|
        | 10Y Real Yield | DFII10 | 2.0x | Lower real rates = cheaper capital = easier |
        | HY Credit Spread | BAMLH0A0HYM2 | 1.5x | Tighter spreads = less risk aversion = easier |
        | Chicago Fed NFCI | NFCI | 1.5x | Lower NFCI = looser financial conditions |
        | 2Y Treasury | DGS2 | 1.0x | Lower front-end = Fed easing expectations |
        
        **Why Real Yields Matter Most (2x Weight)**
        
        Real yields represent the true cost of capital after inflation. When real yields 
        rise, discount rates increase, compressing equity multiples - especially for 
        long-duration assets like growth stocks. The 10Y real yield is the single most 
        important variable for tech/growth relative performance.
        
        **Interpretation**
        
        - **Easing (Score > +0.5)**: Conditions supportive for risk assets
        - **Neutral (-0.5 to +0.5)**: Mixed signals; other factors dominate
        - **Tightening (Score < -0.5)**: Headwind for risk assets; favor quality/defense
        """)
    
    col_liq1, col_liq2 = st.columns(2)
    
    with col_liq1:
        if "NFCI" in fred_bundle:
            nfci = fred_bundle["NFCI"]
            tf = compute_transforms(nfci)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "Chicago Fed NFCI",
                    f"{last['Value']:.2f}",
                    delta=f"{last['Chg_3']:+.2f} (3m)",
                    delta_color="inverse",
                    help="National Financial Conditions Index. Zero = average conditions. "
                         "Positive = tighter than average. Negative = looser than average. "
                         "Inverted delta: falling NFCI (green) = easing conditions."
                )
                
                fig = _line_chart(
                    nfci.df["Value"],
                    "Chicago Fed NFCI (level)",
                    usrec,
                    lookback_years
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with col_liq2:
        if "DFII10" in fred_bundle:
            real = fred_bundle["DFII10"]
            tf = compute_transforms(real)
            
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "10Y Real Yield",
                    f"{last['Value']:.2f}%",
                    delta=f"{last['Chg_3']:+.0f}bps (3m)",
                    delta_color="inverse",
                    help="10-Year TIPS yield (inflation-adjusted borrowing cost). "
                         "Receives 2x weight in liquidity score. "
                         "Rising real yields = tightening = headwind for valuations."
                )
                
                fig = _line_chart(
                    real.df["Value"],
                    "10Y Real Yield (%)",
                    usrec,
                    lookback_years
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer context
    st.caption(
        "Gray shaded areas indicate NBER-defined recessions. "
        "All z-scores are calculated against post-June 2021 history to avoid COVID-era distortions."
    )

    st.divider()
    
    # -----------------------------
    # Macro Pulse Section
    # -----------------------------
    st.subheader("Macro Pulse Signals")
    
    with st.expander("Understanding the Macro Pulse"):
        st.markdown("""
**What is the Macro Pulse?**

The Macro Pulse visualizes the **historical evolution** of growth and inflation momentum. 
While the regime scores above show the current snapshot, the pulse shows how we got here 
and the trajectory of change.

---

**How It's Calculated**

For each month in history, we compute:

1. **YoY momentum** for each indicator (1m, 3m, 6m changes)
2. **Z-score normalize** against a rolling 36-month window
3. **Weighted average** across indicators (same weights as regime scoring)
4. **Scale to -100/+100** range for visualization

---

**Reading the Chart**

| Pulse Value | Interpretation |
|-------------|----------------|
| +50 to +100 | Very strong momentum (rare, typically 1-2 std devs above normal) |
| +20 to +50 | Above-average momentum |
| -20 to +20 | Normal range |
| -50 to -20 | Below-average momentum |
| -100 to -50 | Very weak momentum (rare, typically 1-2 std devs below normal) |

---

**Key Patterns to Watch**

| Pattern | Signal | Implication |
|---------|--------|-------------|
| Both rising | Reflation building | Real assets, value, cyclicals |
| Growth rising, Inflation falling | Goldilocks forming | Risk-on, tech, growth |
| Growth falling, Inflation rising | Stagflation risk | Defensive, cash, gold |
| Both falling | Deflation/recession | Duration, quality, bonds |
| Crossover (Growth crosses Inflation) | Regime transition | Watch for confirmation |

---

**Divergences**

When Growth and Inflation pulses diverge significantly, it often precedes:
- Policy response (Fed reacting to the lagging indicator)
- Market rotation (sectors repricing the regime shift)
- Volatility expansion (uncertainty about direction)

---

**Limitations**

- Backward-looking (measures what happened, not what will happen)
- Z-score normalization means extremes are relative to recent history
- Monthly frequency misses intra-month dynamics
- COVID period (2020-2021) may still distort some readings
""")
    
    # Compute pulse data
    pulse_data = {k: sf.df for k, sf in fred_bundle.items()}
    pulse_df = compute_macro_pulse(pulse_data, lookback_years=lookback_years)
    
    if not pulse_df.empty:
        # Current readings
        latest = pulse_df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            growth_pulse = latest["Growth_Pulse"]
            delta_color = "normal" if growth_pulse > 0 else "inverse"
            st.metric(
                "Growth Pulse",
                f"{growth_pulse:+.0f}",
                delta="Strong" if abs(growth_pulse) > 50 else ("Moderate" if abs(growth_pulse) > 20 else "Weak"),
                delta_color=delta_color,
                help="Current growth momentum on -100 to +100 scale. "
                     "Positive = accelerating growth. Negative = decelerating."
            )
        
        with col2:
            inflation_pulse = latest["Inflation_Pulse"]
            delta_color = "inverse" if inflation_pulse > 0 else "normal"
            st.metric(
                "Inflation Pulse",
                f"{inflation_pulse:+.0f}",
                delta="Strong" if abs(inflation_pulse) > 50 else ("Moderate" if abs(inflation_pulse) > 20 else "Weak"),
                delta_color=delta_color,
                help="Current inflation momentum on -100 to +100 scale. "
                     "Positive = rising inflation. Negative = falling inflation."
            )
        
        with col3:
            st.metric(
                "Pulse Regime",
                latest["Regime"],
                help="Regime classification based on pulse values. "
                     "Growth > 0 and Inflation <= 0 = Goldilocks, etc."
            )
        
        # Main pulse chart
        fig = go.Figure()
        
        # Growth Pulse
        fig.add_trace(go.Scatter(
            x=pulse_df.index,
            y=pulse_df["Growth_Pulse"],
            mode="lines",
            name="Growth Pulse",
            line=dict(color="#888888", width=2.5),
        ))
        
        # Inflation Pulse
        fig.add_trace(go.Scatter(
            x=pulse_df.index,
            y=pulse_df["Inflation_Pulse"],
            mode="lines",
            name="Inflation Pulse",
            line=dict(color="#1E90FF", width=2.5),
        ))
        
        # Zero line
        fig.add_hline(y=0, line=dict(color="white", width=1, dash="dash"), opacity=0.5)
        
        # Reference lines at +/- 50
        fig.add_hline(y=50, line=dict(color="green", width=1, dash="dot"), opacity=0.3)
        fig.add_hline(y=-50, line=dict(color="red", width=1, dash="dot"), opacity=0.3)
        
        # Recession shading
        if usrec is not None and not usrec.empty:
            start_date = pulse_df.index.min()
            end_date = pulse_df.index.max()
            rec = usrec[(usrec.index >= start_date) & (usrec.index <= end_date)]
            
            in_rec = False
            rec_start = None
            
            for t, v in rec.items():
                if (v == 1) and (not in_rec):
                    in_rec = True
                    rec_start = t
                if (v == 0) and in_rec:
                    in_rec = False
                    fig.add_vrect(
                        x0=rec_start,
                        x1=t,
                        fillcolor="rgba(128, 128, 128, 0.2)",
                        line_width=0,
                        layer="below"
                    )
            
            if in_rec and rec_start is not None:
                fig.add_vrect(
                    x0=rec_start,
                    x1=end_date,
                    fillcolor="rgba(128, 128, 128, 0.2)",
                    line_width=0,
                    layer="below"
                )
        
        fig.update_layout(
            title="Macro Pulse Signals: USA",
            yaxis_title="Signal",
            height=450,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            yaxis=dict(range=[-110, 110], dtick=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime history
        with st.expander("Regime History"):
            # Count regime occurrences
            regime_counts = pulse_df["Regime"].value_counts()
            total_months = len(pulse_df)
            
            st.markdown("**Time Spent in Each Regime:**")
            
            regime_data = []
            for regime in ["Goldilocks", "Reflation", "Stagflation", "Deflation"]:
                count = regime_counts.get(regime, 0)
                pct = count / total_months * 100
                regime_data.append({
                    "Regime": regime,
                    "Months": count,
                    "% of Period": f"{pct:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(regime_data), use_container_width=True, hide_index=True)
            
            # Recent regime transitions
            st.markdown("**Recent Regime Transitions:**")
            
            # Find transitions
            pulse_df["Regime_Shift"] = pulse_df["Regime"] != pulse_df["Regime"].shift(1)
            transitions = pulse_df[pulse_df["Regime_Shift"]].tail(10)
            
            if len(transitions) > 1:
                trans_data = []
                for i, (date, row) in enumerate(transitions.iterrows()):
                    if i > 0:  # Skip first row (no prior regime)
                        prior_regime = pulse_df.loc[:date, "Regime"].iloc[-2] if len(pulse_df.loc[:date]) > 1 else "N/A"
                        trans_data.append({
                            "Date": date.strftime("%Y-%m"),
                            "From": prior_regime,
                            "To": row["Regime"],
                            "Growth": f"{row['Growth_Pulse']:+.0f}",
                            "Inflation": f"{row['Inflation_Pulse']:+.0f}"
                        })
                
                if trans_data:
                    st.dataframe(pd.DataFrame(trans_data), use_container_width=True, hide_index=True)
            else:
                st.info("No regime transitions in the selected period.")
    
    else:
        st.warning("Insufficient data to compute Macro Pulse. Need at least 4 years of history.")
    
    st.divider()

def _line_chart(
    series: pd.Series,
    title: str,
    usrec: Optional[pd.Series],
    lookback_years: int
) -> Optional[go.Figure]:
    """Create a simple line chart with recession shading"""
    
    series = series.dropna()
    if series.empty:
        return None
    
    # Determine date range
    end = series.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    series = series[(series.index >= start) & (series.index <= end)]
    
    if series.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=title,
        line=dict(width=2)
    ))
    
    # Recession shading
    if usrec is not None and not usrec.empty:
        rec = usrec[(usrec.index >= start) & (usrec.index <= end)]
        
        in_rec = False
        rec_start = None
        
        for t, v in rec.items():
            if (v == 1) and (not in_rec):
                in_rec = True
                rec_start = t
            if (v == 0) and in_rec:
                in_rec = False
                fig.add_vrect(
                    x0=rec_start,
                    x1=t,
                    fillcolor="rgba(200,200,200,0.2)",
                    line_width=0,
                    layer="below"
                )
        
        if in_rec and rec_start is not None:
            fig.add_vrect(
                x0=rec_start,
                x1=end,
                fillcolor="rgba(200,200,200,0.2)",
                line_width=0,
                layer="below"
            )
    
    # Layout
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        hovermode="x unified"
    )
    fig.update_xaxes(range=[start, end])
    
    return fig