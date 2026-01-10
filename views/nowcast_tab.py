# views/nowcast_tab.py
"""
Nowcast Dashboard Tab

Displays real-time GDP and inflation nowcasts with:
- Comparison to Atlanta Fed GDPNow
- Component contributions
- Growth-Inflation quadrant positioning
- Historical tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional

from data.fred import SeriesFrame, compute_transforms
from models.nowcast import quick_nowcast, NowcastResult


def render_nowcast_tab(
    fred_bundle: Dict[str, SeriesFrame],
    usrec: Optional[pd.Series],
    lookback_years: int
):
    """
    Render the Nowcast tab.
    
    Shows:
    - GDP nowcast vs Atlanta Fed
    - Inflation nowcast vs target
    - Component contributions
    - GIP quadrant
    """
    
    st.header("Real-Time Activity & Inflation")
    st.caption(
        "Growth activity pulse and inflation momentum estimates. "
        "Activity index compared with Atlanta Fed GDPNow for reference."
    )

    # Educational panel
    with st.expander("Understanding the Activity Pulse"):
        st.markdown("""
**What This Is (and Isn't)**

This tab provides a **real-time activity pulse** - a momentum indicator showing whether 
economic growth is accelerating or decelerating. It is NOT a formal GDP nowcast.

| Feature | Our Activity Pulse | Atlanta Fed GDPNow |
|---------|-------------------|-------------------|
| Inputs | 5 key indicators | 100+ series |
| Method | Fixed weights | Bayesian dynamic factors |
| Updates | When we refresh | Multiple times daily |
| Goal | Directional signal | Precise GDP estimate |

We show Atlanta Fed GDPNow for **reference**, not as a benchmark to beat.

---

**Key Improvement: Real Retail Sales**

Retail Sales (RSAFS) is reported in **nominal dollars**. If inflation is 3%, nominal 
retail sales can grow 3% with zero real growth. This contaminates the signal.

We fix this by **deflating** retail sales using CPI:

`Real Retail Sales = Nominal Retail Sales / CPI × 100`

This isolates true volume growth from price effects.

---

**Component Weights**

| Indicator | Weight | Why |
|-----------|--------|-----|
| Real Retail Sales | 40% | Consumption is 68% of GDP; retail is best monthly proxy |
| Nonfarm Payrolls | 30% | Jobs = income = spending capacity |
| Industrial Production | 15% | Business investment and inventory proxy |
| Avg Weekly Hours | 10% | Leading indicator of hiring/firing |
| Manufacturing Emp | 5% | Small weight (declining sector share) |

---

**3-Month Annualized Growth**

We use 3-month annualized rates (same unit as GDP SAAR):

`3M Annualized = ((1 + 3-month % change)^4 - 1) × 100`

This smooths monthly noise while remaining responsive to trends.

---

**Inflation Nowcast**

The inflation estimate uses momentum weighting:

`Nowcast = (0.6 × Current YoY) + (0.4 × 3M Annualized)`

- **60% weight on YoY**: The "anchor" - where inflation is today
- **40% weight on 3M**: The "scout" - where inflation is heading

If 3M < YoY, inflation is decelerating.
""")
    
    # Run nowcast
    with st.spinner("Computing nowcasts..."):
        nowcast_results = quick_nowcast(fred_bundle)
    
    if not nowcast_results:
        st.error("Failed to compute nowcasts. Check data availability.")
        return
    
    # -----------------------------
    # Summary Cards
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "gdp" in nowcast_results:
            gdp = nowcast_results["gdp"]
            
            # Delta vs Atlanta Fed (for reference only)
            delta_str = None
            if "gdp_comparison" in nowcast_results:
                comp = nowcast_results["gdp_comparison"]
                if comp.atlanta_fed is not None:
                    delta_str = f"ATL Fed ref: {comp.atlanta_fed:.2f}%"
            
            st.metric(
                "Activity Pulse",
                f"{gdp.estimate:.2f}%",
                delta=delta_str,
                delta_color="off",  # Neutral color - it's reference, not comparison
                help=f"Real-time growth activity estimate for {gdp.quarter}. "
                     f"Based on real retail sales, payrolls, production, hours worked. "
                     f"Atlanta Fed shown for reference (different methodology)."
            )
    
    with col2:
        if "inflation" in nowcast_results:
            inf = nowcast_results["inflation"]
            
            # Delta vs Fed target
            vs_target = inf.estimate - 2.0
            delta_color = "inverse" if vs_target > 0 else "normal"
            
            st.metric(
                "Inflation Nowcast (YoY)",
                f"{inf.estimate:.2f}%",
                delta=f"vs 2% target: {vs_target:+.2f}pp",
                delta_color=delta_color,
                help="Momentum-weighted inflation projection based on Core CPI. "
                     "Formula: (0.6 x Current YoY) + (0.4 x 3M Annualized). "
                     "Delta shows distance from Fed's 2% target."
            )
    
    with col3:
        if "quadrant" in nowcast_results:
            quad = nowcast_results["quadrant"]
            
            # Color code by quadrant (removed emoji for professional tone)
            quadrant_indicators = {
                "Goldilocks": "[G]",
                "Reflation": "[R]",
                "Stagflation": "[S]",
                "Deflation": "[D]"
            }
            indicator = quadrant_indicators.get(quad["quadrant"], "[?]")
            
            st.metric(
                "Macro Quadrant",
                f"{indicator} {quad['quadrant']}",
                delta=quad["risk_posture"],
                help=f"{quad['description']} | Based on GDP vs 2.0% threshold and Inflation vs 2.5% threshold."
            )
    
    st.divider()
    
    # -----------------------------
    # GDP Nowcast Detail
    # -----------------------------
    st.subheader("Growth Activity Pulse")
    
    col_gdp1, col_gdp2 = st.columns([2, 1])
    
    with col_gdp1:
        # Chart: Our nowcast vs Atlanta Fed history
        if "GDPNOW" in fred_bundle:
            fig = _plot_gdp_nowcast_history(
                fred_bundle["GDPNOW"].df,
                nowcast_results.get("gdp"),
                lookback_years
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Atlanta Fed GDPNow data not available for comparison chart.")
    
    with col_gdp2:
        # Component contributions
        if "gdp" in nowcast_results and nowcast_results["gdp"].contributions:
            st.markdown("**Component Contributions**")
            
            contrib_df = _format_gdp_contributions(nowcast_results["gdp"].contributions)
            if contrib_df is not None:
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)
            
            # Data freshness
            gdp = nowcast_results["gdp"]
            st.caption(f"Data vintage: {gdp.data_vintage.strftime('%Y-%m-%d')}")

            days_stale = (pd.Timestamp.now() - gdp.data_vintage).days
            if days_stale > 45:
                st.warning(f"Data is {days_stale} days old - nowcast may be stale. Check FRED data freshness.")
    
    # Methodology expander
    with st.expander("Activity Pulse Methodology"):
        st.markdown("""
**What Makes This Different**

Unlike a formal GDP nowcast (which maps to BEA accounting identities), this is an 
**activity momentum indicator**. Think of it as a real-time thermometer for economic growth.

---

**Critical: Nominal to Real Conversion**

Retail Sales is reported in nominal dollars. We deflate using CPI:

`Real Retail Sales = Nominal RSAFS / CPIAUCSL × 100`

| Without Deflation | With Deflation |
|-------------------|----------------|
| 3% inflation + 0% real growth = "3% growth" | 3% inflation + 0% real growth = "0% growth" |
| Signal contaminated by prices | Pure volume/activity signal |

---

**Component Details**

| Component | Source | Treatment | Weight |
|-----------|--------|-----------|--------|
| Retail Sales | RSAFS | **Deflated by CPI** | 40% |
| Payrolls | PAYEMS | Level (already real) | 30% |
| Industrial Production | INDPRO | Index (already real) | 15% |
| Avg Weekly Hours | AWHMAN | Level (already real) | 10% |
| Manufacturing Emp | MANEMP | Level (already real) | 5% |

---

**Why Not Match Atlanta Fed?**

Atlanta Fed GDPNow uses:
- 100+ input series
- Bayesian dynamic factor model
- Component-by-component GDP mapping
- Multiple daily updates

Our goal is different: **simple, interpretable, directional signal**.

If you need precise GDP estimates, use Atlanta Fed. If you need a quick read on 
whether activity is accelerating or decelerating, use this pulse.

---

**Limitations**

- No services sector input (majority of economy)
- Fixed weights don't adapt to structural changes
- 3-month lag in some inputs
- Doesn't capture financial conditions or sentiment
""")
          
    st.divider()
    
    # -----------------------------
    # Inflation Nowcast Detail
    # -----------------------------
    st.subheader("Inflation Nowcast")
    
    col_inf1, col_inf2 = st.columns([2, 1])
    
    with col_inf1:
        # Chart: Core CPI trend with nowcast projection
        if "CPILFESL" in fred_bundle:
            fig = _plot_inflation_trend(
                fred_bundle["CPILFESL"].df,
                nowcast_results.get("inflation"),
                fred_bundle.get("T5YIE"),
                lookback_years
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col_inf2:
        # Inflation details
        if "inflation" in nowcast_results:
            inf = nowcast_results["inflation"]
            
            st.markdown("**Inflation Momentum**")
            
            if inf.contributions:
                for component, data in inf.contributions.items():
                    trend_label = "Accelerating" if data.get("trend") == "Accelerating" else "Decelerating"
                    trend_color = "red" if trend_label == "Accelerating" else "green"
                    
                    st.markdown(f"**{component}** - {trend_label}")
                    st.write(f"- Current YoY: {data.get('current_yoy', 0):.2f}%")
                    st.write(f"- 3M Annualized: {data.get('3m_annualized', 0):.2f}%")
                    
                    # Interpretation
                    yoy = data.get('current_yoy', 0)
                    ann_3m = data.get('3m_annualized', 0)
                    if ann_3m < yoy:
                        st.caption("3M < YoY indicates deceleration")
                    elif ann_3m > yoy:
                        st.caption("3M > YoY indicates acceleration")
                    st.write("")
        
        # Market comparison
        if "inflation_vs_market" in nowcast_results:
            mkt = nowcast_results["inflation_vs_market"]
            if mkt.get("breakeven") is not None:
                st.markdown("**vs Market Expectations**")
                st.write(f"5Y Breakeven: {mkt['breakeven']:.2f}%")
                st.write(f"Nowcast vs BE: {mkt['vs_market']:+.2f}pp")
                st.write(f"Position: {mkt['interpretation']}")
    
    with st.expander("Inflation Nowcast Methodology"):
        st.markdown("""
**Momentum-Weighted Approach**

Unlike GDP (which we estimate as a weighted sum of components), inflation 
is projected using momentum weighting:

`Inflation Nowcast = (0.6 x Current YoY) + (0.4 x 3M Annualized)`

| Component | Weight | Role |
|-----------|--------|------|
| Current YoY | 60% | The "anchor" - where inflation is today |
| 3M Annualized | 40% | The "scout" - where inflation is heading |

---

**Interpreting Momentum**

| 3M vs YoY | Signal | Implication |
|-----------|--------|-------------|
| 3M < YoY | Decelerating | Recent prints below trend; disinflation |
| 3M = YoY | Stable | Inflation running at consistent pace |
| 3M > YoY | Accelerating | Recent prints above trend; re-acceleration |

---

**Market Comparison (Breakevens)**

We compare our nowcast to 5-Year breakeven inflation:

- **Breakeven**: Market's expected average inflation over 5 years
- **Nowcast vs BE**: Positive = we see more inflation than market prices

| Nowcast vs Market | Interpretation |
|-------------------|----------------|
| Nowcast >> BE (+0.5pp+) | Market may be underpricing inflation risk |
| Nowcast ~= BE | Consensus alignment |
| Nowcast << BE (-0.5pp+) | Market may be overpricing inflation risk |

---

**Data Source**

Primary input is Core CPI (CPILFESL) - CPI excluding food and energy. 
This is more stable than headline CPI and closer to what the Fed watches, 
though the Fed officially targets Core PCE.
""")
    
    st.divider()
    
    # -----------------------------
    # GIP Quadrant Visualization
    # -----------------------------
    st.subheader("Growth-Inflation Quadrant")
    
    with st.expander("Understanding the Quadrant"):
        st.markdown("""
        **Level-Based Classification**
        
        This quadrant classifies the economy based on **absolute levels**, not momentum:
        
        | Threshold | Value | Rationale |
        |-----------|-------|-----------|
        | GDP | 2.0% | US potential/trend growth rate |
        | Inflation | 2.5% | Fed target (2%) + tolerance buffer |
        
        ---
        
        **The Four Quadrants**
        
        | Quadrant | GDP | Inflation | Economic State | Asset Implication |
        |----------|-----|-----------|----------------|-------------------|
        | Goldilocks | > 2% | < 2.5% | Strong growth, contained prices | Risk-on; equities, credit |
        | Reflation | > 2% | > 2.5% | Strong growth, rising prices | Real assets, commodities, value |
        | Stagflation | < 2% | > 2.5% | Weak growth, high prices | Defensive; cash, gold |
        | Deflation | < 2% | < 2.5% | Weak growth, falling prices | Duration; long bonds, quality |
        
        ---
        
        **Nowcast Quadrant vs Regime Quadrant**
        
        This dashboard has two quadrant classifications:
        
        | Feature | Nowcast (This Tab) | Regime (GIP Tab) |
        |---------|-------------------|------------------|
        | Basis | Absolute levels | Z-score momentum |
        | GDP Test | Is GDP > 2%? | Is GDP accelerating? |
        | Inflation Test | Is Inflation > 2.5%? | Is Inflation rising? |
        | Timeframe | Structural (quarters) | Tactical (weeks/months) |
        
        **Use Both Together:**
        - Nowcast tells you "where are we structurally?"
        - Regime tells you "what direction are we moving?"
        
        A Goldilocks nowcast with Decelerating regime = economy still strong but losing momentum.
        """)
    
    if "quadrant" in nowcast_results:
        fig = _plot_gip_quadrant(nowcast_results["quadrant"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        quad = nowcast_results["quadrant"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Current Position:** {quad['quadrant']}")
            st.write(quad["description"])
        
        with col2:
            st.success(f"**Suggested Posture:** {quad['risk_posture']}")
            
            # Add context from research
            if quad["quadrant"] == "Reflation":
                st.caption(
                    "Research note: 'Growth accelerates faster than inflation' - "
                    "Higher long-end yields create less drag on equities when growth outpaces inflation."
                )
            elif quad["quadrant"] == "Stagflation":
                st.caption(
                    "Research warning: 'Inflation runs ahead of growth, pushing long end "
                    "higher for wrong reasons' - Equities struggle when growth cannot offset inflation drag."
                )
            elif quad["quadrant"] == "Goldilocks":
                st.caption(
                    "Optimal environment: Earnings grow (GDP positive) while discount rates "
                    "remain stable (inflation contained). Supports multiple expansion."
                )
            elif quad["quadrant"] == "Deflation":
                st.caption(
                    "Recessionary conditions: Falling demand compresses both growth and prices. "
                    "Duration outperforms as rates fall; credit risk rises."
                )
                
    st.caption(
        "Quadrant based on ABSOLUTE LEVELS (GDP vs 2% threshold, Inflation vs 2.5% threshold). "
        "See GIP Regime tab for momentum-based classification using z-scores."
    )


def _plot_gdp_nowcast_history(
    atlanta_fed_df: pd.DataFrame,
    our_nowcast: Optional[NowcastResult],
    lookback_years: int
) -> Optional[go.Figure]:
    """Plot Atlanta Fed GDPNow history with our current estimate"""
    
    if "Value" not in atlanta_fed_df.columns:
        return None
    
    series = atlanta_fed_df["Value"].dropna()
    if series.empty:
        return None
    
    # Filter to lookback
    end = series.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    series = series[(series.index >= start)]
    
    fig = go.Figure()
    
    # Atlanta Fed history
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name="Atlanta Fed GDPNow",
        line=dict(color="steelblue", width=2)
    ))
    
    # Our nowcast point
    if our_nowcast:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp.now()],
            y=[our_nowcast.estimate],
            mode="markers",
            name="Our Nowcast",
            marker=dict(color="coral", size=12, symbol="diamond")
        ))
        
        # Confidence band as horizontal line
        fig.add_hline(
            y=our_nowcast.estimate,
            line=dict(color="coral", width=1, dash="dash"),
            annotation_text=f"Our: {our_nowcast.estimate:.2f}%"
        )
    
    # Zero line
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
    
    # Trend threshold
    fig.add_hline(
        y=2.0, 
        line=dict(color="green", width=1, dash="dot"),
        annotation_text="Trend (2%)"
    )
    
    fig.update_layout(
        title="Activity Pulse vs Atlanta Fed GDPNow (Reference)",
        yaxis_title="Real GDP Growth (SAAR %)",
        height=350,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def _plot_inflation_trend(
    core_cpi_df: pd.DataFrame,
    our_nowcast: Optional[NowcastResult],
    breakeven_sf: Optional[SeriesFrame],
    lookback_years: int
) -> Optional[go.Figure]:
    """Plot core CPI YoY trend with breakeven comparison"""
    
    if "Value" not in core_cpi_df.columns:
        return None
    
    # Compute YoY
    series = core_cpi_df["Value"].dropna()
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()
    
    if yoy.empty:
        return None
    
    # Filter to lookback
    end = yoy.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    yoy = yoy[(yoy.index >= start)]
    
    fig = go.Figure()
    
    # Core CPI YoY
    fig.add_trace(go.Scatter(
        x=yoy.index,
        y=yoy.values,
        mode="lines",
        name="Core CPI YoY",
        line=dict(color="steelblue", width=2)
    ))
    
    # Breakeven if available
    if breakeven_sf is not None and "Value" in breakeven_sf.df.columns:
        be = breakeven_sf.df["Value"].dropna()
        be = be[(be.index >= start)]
        
        fig.add_trace(go.Scatter(
            x=be.index,
            y=be.values,
            mode="lines",
            name="5Y Breakeven",
            line=dict(color="orange", width=1.5, dash="dash"),
            yaxis="y1"
        ))
    
    # Fed target
    fig.add_hline(
        y=2.0,
        line=dict(color="green", width=2, dash="dot"),
        annotation_text="Fed Target (2%)"
    )
    
    # Our nowcast point
    if our_nowcast:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp.now()],
            y=[our_nowcast.estimate],
            mode="markers",
            name="Nowcast",
            marker=dict(color="coral", size=12, symbol="diamond")
        ))
    
    fig.update_layout(
        title="Core Inflation Trend",
        yaxis_title="YoY %",
        height=350,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig

def _format_gdp_contributions(contributions: Dict) -> Optional[pd.DataFrame]:
    """Format GDP contributions for display"""
    
    rows = []
    for series_key, data in contributions.items():
        # Show if series was deflated to real
        real_indicator = "✓" if data.get("is_real", True) else "nominal"
        
        rows.append({
            "Indicator": series_key,
            "Growth": f"{data.get('growth', 0):.2f}%",
            "Weight": f"{data.get('weight', 0)*100:.0f}%",
            "Contrib": f"{data.get('contribution', 0):.2f}pp",
            "Real": real_indicator
        })
    
    if not rows:
        return None
    
    return pd.DataFrame(rows)


def _plot_gip_quadrant(quadrant_data: Dict) -> go.Figure:
    """Plot Growth-Inflation quadrant with current position"""
    
    fig = go.Figure()
    
    # Quadrant backgrounds
    quadrant_colors = {
        "Goldilocks": "rgba(0, 200, 0, 0.15)",
        "Reflation": "rgba(255, 200, 0, 0.15)",
        "Stagflation": "rgba(255, 0, 0, 0.15)",
        "Deflation": "rgba(0, 0, 255, 0.15)"
    }
    
    # Define quadrant boundaries
    gdp_center = 2.0
    inf_center = 2.5
    
    # Goldilocks (right side: high growth, low inflation)
    fig.add_shape(type="rect", x0=gdp_center, y0=0, x1=6, y1=inf_center,
                  fillcolor=quadrant_colors["Goldilocks"], line_width=0, layer="below")
    fig.add_annotation(x=4, y=1.25, text="Goldilocks", showarrow=False, 
                       font=dict(size=14, color="green"))
    
    # Reflation (right side: high growth, high inflation)
    fig.add_shape(type="rect", x0=gdp_center, y0=inf_center, x1=6, y1=6,
                  fillcolor=quadrant_colors["Reflation"], line_width=0, layer="below")
    fig.add_annotation(x=4, y=4.5, text="Reflation", showarrow=False,
                       font=dict(size=14, color="orange"))
    
    # Stagflation (left side: low growth, high inflation)
    fig.add_shape(type="rect", x0=-2, y0=inf_center, x1=gdp_center, y1=6,
                  fillcolor=quadrant_colors["Stagflation"], line_width=0, layer="below")
    fig.add_annotation(x=0, y=4.5, text="Stagflation", showarrow=False,
                       font=dict(size=14, color="red"))
    
    # Deflation (left side: low growth, low inflation)
    fig.add_shape(type="rect", x0=-2, y0=0, x1=gdp_center, y1=inf_center,
                  fillcolor=quadrant_colors["Deflation"], line_width=0, layer="below")
    fig.add_annotation(x=0, y=1.25, text="Deflation", showarrow=False,
                       font=dict(size=14, color="blue"))
    
    # Dividing lines
    fig.add_vline(x=gdp_center, line=dict(color="gray", width=1, dash="dash"))
    fig.add_hline(y=inf_center, line=dict(color="gray", width=1, dash="dash"))
    
    # Current position
    gdp_now = quadrant_data["gdp_nowcast"]
    inf_now = quadrant_data["inflation_nowcast"]
    
    fig.add_trace(go.Scatter(
        x=[gdp_now],
        y=[inf_now],
        mode="markers+text",
        marker=dict(color="red", size=20, symbol="star"),
        text=["NOW"],
        textposition="top center",
        name="Current Position"
    ))
    
    fig.update_layout(
        title="Growth-Inflation Quadrant",
        xaxis_title="GDP Growth Nowcast (%)",
        yaxis_title="Inflation Nowcast (%)",
        xaxis=dict(range=[-1, 5], dtick=1),
        yaxis=dict(range=[0, 5], dtick=1),
        height=400,
        showlegend=False
    )
    
    return fig