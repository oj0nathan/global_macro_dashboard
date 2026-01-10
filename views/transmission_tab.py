# views/transmission_tab.py
"""
Macro to Micro Transmission Tab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional

import config
from data.fred import SeriesFrame, compute_transforms
from data.yfinance import resample_to_month_end, normalize_index, rolling_beta


def render_transmission_tab(
    fred_bundle: Dict[str, SeriesFrame],
    market_bundle: Dict[str, SeriesFrame],
    usrec: Optional[pd.Series],
    lookback_years: int,
    normalize: bool,
    show_beta: bool,
    beta_window: int
):
    """
    Render Macro to Micro Transmission tab.
    
    Shows how macro indicators transmit to sector/factor performance.
    """
    
    st.header("Macro to Micro Transmission")
    st.caption(
        "Overlays are aligned to month-end to avoid false precision. "
        "Use relative performance and rolling beta for inference."
    )

    # Educational panel
    with st.expander("Understanding Macro-to-Micro Transmission"):
        st.markdown("""
**The Core Framework**

> "All asset classes are connected. Nothing operates in a silo. You are riding blind 
> if you are only focusing on a single asset class or single data point."

Macro indicators transmit to asset prices through economic channels. This tab visualizes 
those transmission mechanisms by pairing macro data with market ratios that should 
theoretically respond to that data.

---

**Why Use Relative Performance (Ratios)?**

Absolute performance conflates macro sensitivity with broad market direction. 
A sector can rise 10% simply because the market rose 12% - that tells you nothing 
about macro transmission.

Ratios isolate the *relative* response:

`XLY/XLP = Consumer Discretionary vs Consumer Staples`

When retail sales are strong, consumers shift spending from necessities (staples) 
to wants (discretionary). The ratio captures this rotation regardless of whether 
the market is up or down.

---

**Transmission Pairs Explained**

| Macro Indicator | Market Ratio | Economic Logic |
|-----------------|--------------|----------------|
| Retail Sales | XLY/XLP | Strong retail = consumer confidence, discretionary outperforms |
| Industrial Production | XLI/SPY | Industrial strength = cyclical sectors benefit |
| Payrolls | IWM/SPY | Strong labor = domestic growth, small caps outperform |
| Core CPI | GLD/TLT | Rising inflation = gold hedge outperforms nominal bonds |
| Breakevens | XLE/SPY | Inflation expectations = energy (real asset) outperforms |
| Real Yields | QQQ/SPY | Lower real yields = growth/tech outperforms (duration) |
| HY Spreads | HYG/LQD | Tighter spreads = credit risk appetite, HY outperforms IG |

---

**Reading the Charts**

| Element | Color | Interpretation |
|---------|-------|----------------|
| Left axis (blue) | Macro indicator | YoY change for growth/inflation; Level for rates |
| Right axis (orange) | Market ratio | Relative performance of the pair |
| Co-movement | Both rise/fall together | Positive transmission |
| Divergence | Move in opposite directions | Transmission breakdown or lead/lag |

---

**Lead/Lag Relationships**

Some macro indicators lead market ratios; others lag:

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| Macro leads market | Macro change predicts rotation | Position early |
| Market leads macro | Market pricing in expectations | Macro confirms or denies |
| Contemporaneous | Move together | No timing edge |
| Breakdown | Historical relationship failing | Regime shift or noise |

---

**Normalization Option**

When "Normalize" is enabled, both series are indexed to 100 at the start of the 
lookback period. This makes visual comparison easier when the series have very 
different scales (e.g., CPI at 3% vs ratio at 1.2).

---

**Caveats**

- Correlation ≠ causation
- Relationships can break down during regime shifts
- Month-end alignment introduces lag
- Single-factor models oversimplify complex markets
- Use as one input among many, not as sole signal
""")
    
    # Settings explanation
    with st.expander("Chart Settings Guide"):
        st.markdown("""
**Normalize**

When enabled, both the macro indicator and market ratio are rebased to 100 at the 
start of the lookback period. This helps visualize co-movement when the series have 
very different absolute values.

- **On**: Better for visual comparison of direction/trend
- **Off**: Better for seeing actual levels/magnitudes

---

**Show Rolling Beta**

Beta measures the sensitivity of the market ratio to the macro indicator:

`Beta = Cov(Ratio Returns, Macro Changes) / Var(Macro Changes)`

Interpretation:
- **Beta = 1**: Ratio moves 1% for each 1% macro move (baseline sensitivity)
- **Beta > 1**: Ratio MORE sensitive than historical average
- **Beta < 1**: Ratio LESS sensitive than historical average
- **Beta < 0**: Inverse relationship (rare, often signals regime shift)

---

**Beta Window**

The lookback period (in months) for calculating rolling beta.

| Window | Trade-off |
|--------|-----------|
| 12 months | Responsive but noisy |
| 24 months | Balanced (default) |
| 36 months | Stable but slow to adapt |

Shorter windows detect regime changes faster but produce more false signals.
""")
    
    # Precompute month-end market data
    market_me = {}
    for k, sf in market_bundle.items():
        try:
            me = resample_to_month_end(sf.df)["Value"]
            market_me[k] = me
        except Exception as e:
            st.warning(f"Failed to resample {k}: {e}")

    # For growth/inflation: Use YoY
    # For rates/credit: Use levels
    macro_series = {}
    
    # Collect all macro keys used in transmission pairs
    macro_keys_needed = set(pair[0] for pair in config.TRANSMISSION_PAIRS)
    
    for mk in macro_keys_needed:
        if mk not in fred_bundle:
            continue
        
        try:
            sf = fred_bundle[mk]
            
            # Decide whether to use YoY or level
            if sf.unit in ["index", "level"] and sf.chg_unit == "pct":
                # Growth/inflation indicators: Use YoY
                tf = compute_transforms(sf)
                if "YoY" in tf.columns:
                    macro_series[mk] = tf["YoY"].dropna()
            else:
                # Rates/credit spreads: Use level
                macro_series[mk] = sf.df["Value"].dropna()
                
        except Exception as e:
            st.warning(f"Failed to prepare {mk}: {e}")
    
    # Transmission pair context
    PAIR_CONTEXT = {
        "RSAFS": {
            "logic": "Retail sales measure consumer spending on goods. Strong retail indicates confident consumers willing to spend on discretionary items rather than just necessities.",
            "expected": "When retail sales accelerate, XLY (discretionary) should outperform XLP (staples).",
            "breakdown": "If retail strong but XLY/XLP falling, may indicate margin pressure or rotation to services."
        },
        "INDPRO": {
            "logic": "Industrial production captures manufacturing, mining, and utilities output. It's a proxy for business investment and inventory cycles.",
            "expected": "When IP accelerates, industrials (XLI) should outperform the broad market (SPY).",
            "breakdown": "If IP strong but XLI lagging, may indicate concerns about sustainability or input costs."
        },
        "PAYEMS": {
            "logic": "Nonfarm payrolls measure labor market strength. Small caps (IWM) are more domestically focused and labor-intensive than large caps.",
            "expected": "When payrolls accelerate, small caps should outperform large caps.",
            "breakdown": "If payrolls strong but IWM/SPY falling, may indicate wage pressure concerns or profit margin compression."
        },
        "CPILFESL": {
            "logic": "Core CPI measures underlying inflation pressure. Gold is a traditional inflation hedge; bonds suffer from inflation (purchasing power erosion).",
            "expected": "When core CPI rises, gold (GLD) should outperform long bonds (TLT).",
            "breakdown": "If CPI rising but GLD/TLT falling, may indicate real rate dominance or deflation expectations."
        },
        "T10YIE": {
            "logic": "Breakeven inflation reflects market inflation expectations. Energy stocks benefit from inflation (commodity prices rise).",
            "expected": "When breakevens rise, energy (XLE) should outperform the broad market.",
            "breakdown": "If breakevens rising but XLE lagging, may indicate supply concerns or demand destruction fears."
        },
        "DFII10": {
            "logic": "Real yields represent the true cost of capital. Tech/growth stocks have long duration (distant cash flows) and are highly sensitive to discount rates.",
            "expected": "When real yields fall, tech (QQQ) should outperform the broad market.",
            "breakdown": "If real yields falling but QQQ lagging, may indicate growth concerns or earnings disappointments."
        },
        "BAMLH0A0HYM2": {
            "logic": "HY spreads measure credit risk appetite. When spreads tighten, investors are accepting less compensation for credit risk.",
            "expected": "When HY spreads tighten, HY bonds (HYG) should outperform IG bonds (LQD).",
            "breakdown": "If spreads tightening but HYG/LQD falling, may indicate duration effects or liquidity concerns."
        }
    }
    
    # Render transmission pairs
    for macro_key, num_key, den_key, title in config.TRANSMISSION_PAIRS:
        st.markdown(f"### {title}")
        
        # Add context for this pair
        if macro_key in PAIR_CONTEXT:
            ctx = PAIR_CONTEXT[macro_key]
            with st.expander(f"Understanding: {title}"):
                st.markdown(f"""
**Economic Logic**

{ctx['logic']}

---

**Expected Relationship**

{ctx['expected']}

---

**When Relationship Breaks Down**

{ctx['breakdown']}

---

**Data Sources**

- **Macro**: {config.FRED_SERIES[macro_key].label} ({macro_key}) from FRED
- **Numerator**: {config.YF_ASSETS[num_key].label} ({num_key}) from Yahoo Finance
- **Denominator**: {config.YF_ASSETS[den_key].label} ({den_key}) from Yahoo Finance
""")
        
        # Get data
        m = macro_series.get(macro_key)
        num = market_me.get(num_key)
        den = market_me.get(den_key)
        
        if m is None:
            st.info(f"Missing macro data: {macro_key}")
            continue
        
        if num is None:
            st.info(f"Missing market data: {num_key}")
            continue
        
        if den is None:
            st.info(f"Missing market data: {den_key}")
            continue
        
        # Compute relative performance
        try:
            rel = (num / den).dropna()
            rel = rel.replace([float('inf'), float('-inf')], float('nan')).dropna()
        except Exception as e:
            st.error(f"Failed to compute relative for {title}: {e}")
            continue
        
        # Align
        m2, rel2 = m.align(rel, join="inner")
        
        if len(m2) < 24:
            st.info(f"Insufficient data for {title} (need 24+ months after alignment)")
            continue
        
        # Determine label based on series type
        if macro_key in config.REGIME_GROWTH_KEYS + config.REGIME_INFLATION_KEYS:
            macro_label = f"{config.FRED_SERIES[macro_key].label} (YoY)"
        else:
            macro_label = f"{config.FRED_SERIES[macro_key].label} (level)"
        
        # Calculate correlation for context
        try:
            corr = m2.corr(rel2)
            corr_str = f"Correlation: {corr:.2f}"
            
            if corr > 0.5:
                corr_interpretation = "Strong positive relationship"
            elif corr > 0.2:
                corr_interpretation = "Moderate positive relationship"
            elif corr > -0.2:
                corr_interpretation = "Weak/no relationship"
            elif corr > -0.5:
                corr_interpretation = "Moderate negative relationship"
            else:
                corr_interpretation = "Strong negative relationship"
        except:
            corr_str = ""
            corr_interpretation = ""
        
        # Plot overlay
        fig = _overlay_chart(
            left=m2,
            right=rel2,
            left_name=f"{config.FRED_SERIES[macro_key].label} YoY",
            right_name=f"{config.YF_ASSETS[num_key].label} / {config.YF_ASSETS[den_key].label}",
            title=title,
            usrec=usrec,
            lookback_years=lookback_years,
            normalize=normalize
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation
        if corr_str:
            st.caption(f"{corr_str} | {corr_interpretation}")
        
        # Rolling beta
        if show_beta and len(m2) >= beta_window:
            try:
                # Convert to monthly returns
                rel_ret = rel2.pct_change().dropna()
                mac = m2.loc[rel_ret.index].dropna()
                rel_ret, mac = rel_ret.align(mac, join="inner")
                
                if len(rel_ret) >= beta_window:
                    b = rolling_beta(rel_ret, mac, beta_window)
                    
                    # Plot beta
                    b_fig = _line_chart(
                        b,
                        f"Rolling Beta ({beta_window}m window)",
                        usrec,
                        lookback_years
                    )
                    
                    if b_fig:
                        with st.expander("Rolling Beta Analysis"):
                            st.plotly_chart(b_fig, use_container_width=True)
                            
                            if not b.dropna().empty:
                                current_beta = b.dropna().iloc[-1]
                                avg_beta = b.dropna().mean()
                                std_beta = b.dropna().std()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Current Beta", 
                                        f"{current_beta:.2f}",
                                        help="Current sensitivity of market ratio to macro indicator"
                                    )
                                with col2:
                                    st.metric(
                                        "Average Beta", 
                                        f"{avg_beta:.2f}",
                                        help="Historical average sensitivity"
                                    )
                                with col3:
                                    # Z-score of current beta
                                    if std_beta > 0:
                                        beta_z = (current_beta - avg_beta) / std_beta
                                        st.metric(
                                            "Beta Z-Score",
                                            f"{beta_z:.2f}",
                                            help="How unusual is current beta? >2 or <-2 is extreme."
                                        )
                                
                                # Interpretation
                                if current_beta > avg_beta + std_beta:
                                    st.info(
                                        "Current sensitivity is ABOVE average - market ratio responding "
                                        "more strongly to macro changes than usual."
                                    )
                                elif current_beta < avg_beta - std_beta:
                                    st.warning(
                                        "Current sensitivity is BELOW average - transmission mechanism "
                                        "may be weakening or other factors dominating."
                                    )
                                
                                if current_beta < 0 and avg_beta > 0:
                                    st.error(
                                        "Beta has flipped negative - relationship has inverted. "
                                        "This often signals a regime change or structural shift."
                                    )
                                
            except Exception as e:
                st.warning(f"Failed to compute beta: {e}")
    
    st.divider()
    
    # Summary section
    st.subheader("Transmission Summary")
    
    with st.expander("How to Use This Tab"):
        st.markdown("""
**Workflow for Transmission Analysis**

1. **Identify Active Macro Drivers**
   - Which macro indicators are moving significantly?
   - Use GIP Regime and Nowcast tabs for context

2. **Check Expected Transmission**
   - Find the transmission pair for that macro indicator
   - Is the market ratio responding as expected?

3. **Assess Relationship Strength**
   - High correlation + expected direction = transmission working
   - Low correlation or unexpected direction = investigate

4. **Consider Lead/Lag**
   - If macro leads market: Position ahead of market catch-up
   - If market leads macro: Market pricing in expectations

5. **Monitor Beta for Regime Changes**
   - Rising beta = increasing sensitivity
   - Falling beta = decreasing sensitivity
   - Negative beta = relationship inversion

---

**Integration with Other Tabs**

| This Tab Shows | Cross-Reference With |
|----------------|----------------------|
| Consumer transmission | Heatmap (Retail Sales percentile) |
| Industrial transmission | GIP Regime (Growth score) |
| Inflation transmission | Rates tab (Real yield decomposition) |
| Credit transmission | Rates tab (Credit spreads) |

---

**Common Patterns**

| Pattern | Interpretation | Example |
|---------|----------------|---------|
| Macro up, ratio up | Transmission working | Strong retail → XLY outperforms |
| Macro up, ratio flat | Transmission delayed or priced in | IP rising but XLI already moved |
| Macro up, ratio down | Transmission broken | CPI rising but GLD lagging |
| Macro down, ratio down | Transmission working (inverse) | Payrolls weak → IWM underperforms |

---

**Red Flags**

- Persistent divergence between macro and expected market response
- Beta flipping from positive to negative
- Correlation breakdown during non-recession periods
- Multiple transmission mechanisms failing simultaneously

These patterns often precede larger market dislocations.
""")
    
    st.caption(
        "Transmission analysis shows historical relationships, not guaranteed future behavior. "
        "Always cross-reference with current positioning and sentiment indicators."
    )


def _overlay_chart(
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
    title: str,
    usrec: Optional[pd.Series],
    lookback_years: int,
    normalize: bool
) -> Optional[go.Figure]:
    """Create overlay chart with dual y-axes"""
    
    # Determine date range
    end = min(left.index.max(), right.index.max())
    start = end - pd.DateOffset(years=lookback_years)
    
    left = left[(left.index >= start) & (left.index <= end)]
    right = right[(right.index >= start) & (right.index <= end)]
    
    # Align
    left, right = left.align(right, join="inner")
    
    if left.empty or right.empty:
        return None
    
    # Normalize if requested
    if normalize:
        left_plot = normalize_index(left, base=100)
        right_plot = normalize_index(right, base=100)
        y1_title = f"{left_name} (index=100)"
        y2_title = f"{right_name} (index=100)"
    else:
        left_plot = left
        right_plot = right
        y1_title = left_name
        y2_title = right_name
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=left_plot.index,
        y=left_plot.values,
        mode="lines",
        name=left_name,
        yaxis="y1",
        line=dict(width=2, color="steelblue")
    ))
    
    fig.add_trace(go.Scatter(
        x=right_plot.index,
        y=right_plot.values,
        mode="lines",
        name=right_name,
        yaxis="y2",
        line=dict(width=2, color="coral")
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
                    fillcolor="rgba(200,200,200,0.15)",
                    line_width=0,
                    layer="below"
                )
        
        if in_rec and rec_start is not None:
            fig.add_vrect(
                x0=rec_start,
                x1=end,
                fillcolor="rgba(200,200,200,0.15)",
                line_width=0,
                layer="below"
            )
    
    # Layout
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(
            title=dict(text=y1_title, font=dict(color="steelblue")), 
            tickfont=dict(color="steelblue")
        ),
        yaxis2=dict(
            title=dict(text=y2_title, font=dict(color="coral")),  
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(range=[start, end])
    
    return fig


def _line_chart(
    series: pd.Series,
    title: str,
    usrec: Optional[pd.Series],
    lookback_years: int
) -> Optional[go.Figure]:
    """Simple line chart"""
    
    series = series.dropna()
    if series.empty:
        return None
    
    end = series.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    series = series[(series.index >= start) & (series.index <= end)]
    
    if series.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=title,
        line=dict(width=2)
    ))
    
    # Add reference lines
    fig.add_hline(y=1, line=dict(color="gray", width=1, dash="dash"),
                  annotation_text="Baseline (β=1)", annotation_position="bottom right")
    fig.add_hline(y=0, line=dict(color="red", width=1, dash="dot"),
                  annotation_text="Zero", annotation_position="bottom right")
    
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
    
    fig.update_layout(
        title=title,
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        hovermode="x unified"
    )
    
    fig.update_xaxes(range=[start, end])
    
    return fig