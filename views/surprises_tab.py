# views/surprises_tab.py
"""
Economic Surprises & Calendar Tab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional

from data.fred import SeriesFrame
from models.surprises import SurpriseEngine


def render_surprises_tab(
    fred_bundle: Dict[str, SeriesFrame],
    fred_client,  # Keep for compatibility but not used
    usrec: Optional[pd.Series],
    lookback_years: int
):
    """
    Render Economic Surprises & Calendar tab.
    """
    
    st.header("Economic Surprises & Calendar")
    st.caption(
        "Track how economic data comes in relative to expectations. "
        "Surprise indices aggregate beats/misses across all major releases."
    )
    
    # Educational panel
    with st.expander("Understanding Economic Surprises"):
        st.markdown("""
**The Core Concept**

> "Part of connecting economic data to markets is looking at how data and earnings 
> come up above or below expectations."

Markets don't react to data levels - they react to data *relative to expectations*. 
A 200K jobs print is bullish if consensus was 150K, but bearish if consensus was 250K. 
The surprise, not the level, drives the immediate market response.

---

**Why Surprises Matter**

| Surprise | Market Reaction | Mechanism |
|----------|-----------------|-----------|
| Positive (beat) | Analysts revise up | Growth expectations increase |
| Negative (miss) | Analysts revise down | Growth expectations decrease |
| Persistent positive | Momentum builds | "Economy stronger than thought" |
| Persistent negative | Momentum fades | "Economy weaker than thought" |

---

**Our Methodology**

Since Bloomberg/Reuters consensus estimates require expensive subscriptions, 
we construct our own surprise index using time-series expectations:

1. **Expected Value**: For each month, compute the average of same-month 
   year-over-year changes from the prior 5 years (captures seasonality)
   
2. **Surprise Calculation**: 
   `Surprise = (Actual YoY - Expected YoY) / Historical StdDev`
   
3. **Standardization**: Divide by expanding standard deviation for comparability
   across indicators with different volatilities

4. **Aggregation**: Equal-weighted average across growth and inflation components

---

**Components**

| Category | Indicators | Interpretation |
|----------|------------|----------------|
| Growth | Industrial Production, Retail Sales | Beat = economy stronger |
| Labor | Nonfarm Payrolls, Unemployment (inv) | Beat = labor market tighter |
| Inflation | CPI, Core CPI, Core PCE | Inverted: lower = positive |

Note: Inflation is inverted because lower-than-expected inflation is generally 
positive for markets (less Fed tightening risk).

---

**Interpretation Guide**

| Index Level | Signal | Implication |
|-------------|--------|-------------|
| > +1.0 | Strong beats | Analysts too pessimistic, likely upgrades |
| +0.5 to +1.0 | Modest beats | Economy tracking above consensus |
| -0.5 to +0.5 | In line | No revision pressure |
| -0.5 to -1.0 | Modest misses | Economy tracking below consensus |
| < -1.0 | Strong misses | Analysts too optimistic, likely downgrades |

---

**Limitations**

- Uses time-series expectations, not actual consensus forecasts
- Equal weighting may not reflect market sensitivity
- Monthly frequency misses intra-month dynamics
- Doesn't capture revision surprises (only initial releases)
""")
    
    # Initialize surprise engine
    surprise_engine = SurpriseEngine()
    
    # Build our own surprise index
    st.subheader("Economic Surprise Index")
    
    surprise_df, component_details = surprise_engine.build_aggregate_surprise_index(fred_bundle)
    
    if not surprise_df.empty and "Aggregate" in surprise_df.columns:
        # Current metrics
        current = surprise_df["Aggregate"].dropna().iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Level",
                f"{current:.2f}",
                help="Standardized surprise index. Positive = data beating expectations. "
                     "Range typically -2 to +2, with 0 = in-line with expectations."
            )
        
        with col2:
            if "Aggregate_3M" in surprise_df.columns:
                avg_3m = surprise_df["Aggregate_3M"].dropna().iloc[-1]
                st.metric(
                    "3-Month Average",
                    f"{avg_3m:.2f}",
                    delta=f"{current - avg_3m:+.2f} vs avg",
                    help="Rolling 3-month average of surprise index. "
                         "Smooths monthly volatility. Delta shows current vs average."
                )
        
        with col3:
            if len(surprise_df) >= 6:
                recent = surprise_df["Aggregate"].iloc[-3:].mean()
                prior = surprise_df["Aggregate"].iloc[-6:-3].mean()
                trend = "Improving" if recent > prior else "Deteriorating"
                trend_delta = recent - prior
                st.metric(
                    "Trend", 
                    trend,
                    delta=f"{trend_delta:+.2f}",
                    help="Compares last 3 months vs prior 3 months. "
                         "Improving = recent surprises more positive. "
                         "Deteriorating = recent surprises more negative."
                )
        
        with col4:
            if len(surprise_df) >= 12:
                hist = surprise_df["Aggregate"].dropna()
                pctl = (hist <= current).sum() / len(hist)
                st.metric(
                    "Historical Percentile", 
                    f"{pctl:.0%}",
                    help="Where current reading ranks vs all historical readings. "
                         "90th percentile = current beats 90% of history."
                )
        
        # Interpretation
        if current > 0.5:
            st.success(
                "Data is significantly beating expectations - economy stronger than consensus. "
                "Watch for analyst upgrades and positive momentum."
            )
        elif current > 0:
            st.info(
                "Data is modestly beating expectations - economy tracking slightly above consensus."
            )
        elif current > -0.5:
            st.warning(
                "Data is modestly missing expectations - economy tracking slightly below consensus."
            )
        else:
            st.error(
                "Data is significantly missing expectations - economy weaker than consensus. "
                "Watch for analyst downgrades and negative momentum."
            )
        
        # Chart
        fig = _plot_surprise_index(surprise_df, usrec, lookback_years)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Component breakdown
        with st.expander("Component Breakdown"):
            st.markdown("""
**Individual Indicator Surprises**

Each component shows its latest surprise value (standardized) and historical percentile.
Positive surprise = data came in above time-series expectation.
""")
            
            if component_details:
                comp_rows = []
                for key, details in component_details.items():
                    # Determine signal
                    surprise_val = details['latest_surprise']
                    if surprise_val > 0.5:
                        signal = "Strong Beat"
                    elif surprise_val > 0:
                        signal = "Beat"
                    elif surprise_val > -0.5:
                        signal = "Miss"
                    else:
                        signal = "Strong Miss"
                    
                    comp_rows.append({
                        "Indicator": key,
                        "Category": details["category"],
                        "Latest Surprise": f"{details['latest_surprise']:.2f}",
                        "Signal": signal,
                        "Percentile": f"{details['percentile']:.0%}" if not pd.isna(details['percentile']) else "N/A"
                    })
                
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                st.caption(
                    "Note: Inflation indicators are inverted (lower actual = positive surprise) "
                    "since below-expectation inflation is typically market-positive."
                )
    
    else:
        st.warning("Insufficient data to compute surprise index. Need at least 24 months of history.")
    
    st.divider()
    
    # -----------------------------
    # Upcoming Releases
    # -----------------------------
    st.subheader("Upcoming Economic Releases")
    
    # Import calendar client here to avoid circular imports
    from data.calendar import CalendarClient
    calendar_client = CalendarClient()
    
    with st.expander("Why Release Dates Matter"):
        st.markdown("""
**Pre-Release Positioning**

> "You always want to watch how data comes in above or below expectations... 
> watch the market response to the expectation vs actual matrix."

Markets often move in anticipation of key releases. Understanding the calendar helps:

---

**Release Importance Tiers**

| Tier | Examples | Market Impact |
|------|----------|---------------|
| High | NFP, CPI, Core PCE, FOMC, GDP | Can move markets 1%+ intraday |
| Medium | Retail Sales, Industrial Production, PPI | Modest impact, confirms trends |
| Low | Regional surveys, secondary indicators | Context only, rarely market-moving |

---

**Key Release Timing (US)**

| Release | Typical Day | Time (ET) |
|---------|-------------|-----------|
| Nonfarm Payrolls | First Friday | 8:30 AM |
| CPI | Mid-month | 8:30 AM |
| Retail Sales | Mid-month | 8:30 AM |
| Core PCE | End of month | 8:30 AM |
| GDP | End of month | 8:30 AM |
| FOMC Decision | 8x per year | 2:00 PM |

---

**Pre-Release Strategy**

- Reduce position size before high-impact releases if uncertain
- Monitor options markets for implied volatility around releases
- Watch revisions to prior data (often as important as new data)
- Consider clustering effects (multiple releases in same week)
""")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        days_ahead = st.selectbox(
            "Look ahead",
            options=[7, 14, 30],
            index=1,
            format_func=lambda x: f"{x} days",
            help="How far ahead to show scheduled releases"
        )
    
    with col2:
        importance_filter = st.multiselect(
            "Importance",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
            help="Filter by release importance. High = market-moving events."
        )
    
    upcoming_df = calendar_client.get_upcoming_releases(
        days_ahead=days_ahead,
        importance_filter=importance_filter if importance_filter else None
    )

    if not upcoming_df.empty:
        upcoming_df = upcoming_df.drop_duplicates(subset=['Date', 'Indicator'], keep='first')
    
    if not upcoming_df.empty:
        def highlight_importance(row):
            if row["Importance"] == "High":
                return ["background-color: rgba(255, 100, 100, 0.2)"] * len(row)
            elif row["Importance"] == "Medium":
                return ["background-color: rgba(255, 200, 100, 0.2)"] * len(row)
            return [""] * len(row)
        
        styled_df = upcoming_df.style.apply(highlight_importance, axis=1)
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 50 + len(upcoming_df) * 35)
        )
        
        high_importance = upcoming_df[upcoming_df["Importance"] == "High"]
        if not high_importance.empty:
            st.info(
                f"Key releases ahead: {', '.join(high_importance['Indicator'].tolist())}"
            )
    else:
        st.info("No major releases scheduled in the selected window.")
    
    st.divider()
    
    # -----------------------------
    # Recent Releases
    # -----------------------------
    st.subheader("Recent Releases")
    
    with st.expander("Reading Recent Releases"):
        st.markdown("""
**What to Look For**

| Column | Description | Usage |
|--------|-------------|-------|
| Actual | Most recent data point | The headline number |
| Prior | Previous period's value | Baseline for comparison |
| Change | Actual - Prior | Direction and magnitude |

---

**Beyond the Headline**

- **Revisions**: Was prior period revised up or down?
- **Composition**: What drove the number (e.g., full-time vs part-time jobs)?
- **Trend**: Is this continuing or reversing recent direction?
- **Context**: How does it fit with other data?

---

**Market Reaction Timing**

- Initial reaction (first 5 minutes): Algorithms parsing headline
- Secondary reaction (5-30 minutes): Humans reading details
- Tertiary reaction (30+ minutes): Positioning and narrative adjustment
""")
    
    recent_df = calendar_client.get_recent_releases(fred_bundle, days_back=30)

    if not recent_df.empty:
        recent_df = recent_df.drop_duplicates(subset=['Date', 'Indicator'], keep='first')
    
    if not recent_df.empty:
        display_df = recent_df.copy()
        
        for col in ["Actual", "Prior", "Change"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )
        
        st.dataframe(
            display_df[["Date", "Indicator", "Category", "Actual", "Prior", "Change"]],
            use_container_width=True,
            hide_index=True,
            height=min(500, 50 + len(display_df) * 35)
        )
    else:
        st.info("No recent release data available.")
    
    st.divider()
    
    # -----------------------------
    # Framework Reference
    # -----------------------------
    st.subheader("Expectations vs Actual Framework")
    
    with st.expander("The Complete Framework"):
        st.markdown("""
**From the Research**

> "When a data print comes out, there are several scenarios: data coming in line, 
> above, or below expectations. From here, you want to map the various market 
> pricing and its response."

---

**The Expectations vs Actual Matrix**

| Data Print | Growth Indicator | Inflation Indicator |
|------------|------------------|---------------------|
| Above Expectations | Risk-on, cyclicals rally | Risk-off, short duration |
| In Line | Neutral, positioning unwind | Neutral |
| Below Expectations | Risk-off, defensives rally | Risk-on, long duration |

---

**Regime-Dependent Reactions**

The same surprise can have opposite effects depending on the macro regime:

| Regime | Strong Jobs Print | Weak Jobs Print |
|--------|-------------------|-----------------|
| Fed Hiking | Bearish (more hikes) | Bullish (pause) |
| Fed Easing | Bullish (soft landing) | Bearish (hard landing) |
| Inflation High | Mixed (wage pressure) | Bullish (demand cooling) |
| Inflation Low | Bullish (growth) | Bearish (recession) |

---

**Second-Order Effects**

Beyond the immediate reaction, consider:

1. **Fed Reaction Function**: Will this change Fed expectations?
2. **Earnings Implications**: Does this affect corporate profits?
3. **Positioning**: How was the market positioned going in?
4. **Narrative**: Does this confirm or challenge the prevailing story?

---

**Market Response Analysis**

After each release, observe:

| Response Pattern | Interpretation |
|------------------|----------------|
| Initial spike, holds | Market accepts the surprise |
| Initial spike, fades | Surprise was priced in |
| Initial spike, reverses | Positioning unwind or rejection |
| No reaction | Surprise was fully expected |
""")
    
    st.markdown("""
**Market Response Matrix:**

The market's reaction depends on both the surprise direction AND the current regime.
""")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Above Expectations**")
        st.markdown("""
- **Growth data**: Risk-on, cyclicals, small caps
- **Inflation data**: Risk-off, short duration, banks
- **Labor data**: Depends on Fed posture
- **Watch**: Initial spike direction + follow-through
""")
    
    with col2:
        st.markdown("**In Line**")
        st.markdown("""
- Generally neutral for direction
- Positioning unwind can dominate
- Watch price action for sentiment clues
- Often "sell the news" if priced in
""")
    
    with col3:
        st.markdown("**Below Expectations**")
        st.markdown("""
- **Growth data**: Risk-off, defensives, quality
- **Inflation data**: Risk-on, long duration, tech
- **Labor data**: Duration bid, dovish Fed
- **Watch**: Is it "bad is good" (Fed pivot) or "bad is bad"?
""")
    
    st.caption(
        "Context matters: A strong jobs print in a Fed-tightening regime is different "
        "from the same print in an easing regime. Always consider the reaction function."
    )


def _plot_surprise_index(
    df: pd.DataFrame,
    usrec: Optional[pd.Series],
    lookback_years: int
) -> Optional[go.Figure]:
    """Plot surprise index with components"""
    
    if "Aggregate" not in df.columns:
        return None
    
    series = df["Aggregate"].dropna()
    if series.empty:
        return None
    
    end = series.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    plot_df = df[(df.index >= start)]
    
    if plot_df.empty:
        return None
    
    fig = go.Figure()
    
    # Main aggregate line
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Aggregate"],
        mode="lines",
        name="Surprise Index",
        line=dict(color="steelblue", width=2)
    ))
    
    # 3-month average if available
    if "Aggregate_3M" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["Aggregate_3M"],
            mode="lines",
            name="3M Average",
            line=dict(color="orange", width=1, dash="dash")
        ))
    
    # Zero line
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"),
                  annotation_text="In Line", annotation_position="bottom right")
    
    # Reference lines for interpretation
    fig.add_hline(y=0.5, line=dict(color="green", width=1, dash="dot"),
                  annotation_text="Beating", annotation_position="top right")
    fig.add_hline(y=-0.5, line=dict(color="red", width=1, dash="dot"),
                  annotation_text="Missing", annotation_position="bottom right")
    
    # Shaded regions
    fig.add_hrect(
        y0=0, y1=plot_df["Aggregate"].max() * 1.1 if plot_df["Aggregate"].max() > 0 else 2,
        fillcolor="rgba(0, 150, 0, 0.05)",
        line_width=0,
        layer="below"
    )
    fig.add_hrect(
        y0=plot_df["Aggregate"].min() * 1.1 if plot_df["Aggregate"].min() < 0 else -2, y1=0,
        fillcolor="rgba(150, 0, 0, 0.05)",
        line_width=0,
        layer="below"
    )
    
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
                    x0=rec_start, x1=t,
                    fillcolor="rgba(128, 128, 128, 0.2)",
                    line_width=0, layer="below"
                )
        
        if in_rec and rec_start:
            fig.add_vrect(
                x0=rec_start, x1=end,
                fillcolor="rgba(128, 128, 128, 0.2)",
                line_width=0, layer="below"
            )
    
    fig.update_layout(
        title="Economic Surprise Index (Constructed from FRED Data)",
        yaxis_title="Standardized Surprise",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig