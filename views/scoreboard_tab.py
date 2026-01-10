# views/scoreboard_tab.py
"""
Data Scoreboard Tab
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict

import config
from data.fred import compute_transforms, SeriesFrame
from models.regime import RegimeState


def render_scoreboard_tab(
    fred_bundle: Dict[str, SeriesFrame],
    current_regime: RegimeState
):
    """Render macro data scoreboard"""
    
    st.header("Macro Scoreboard")
    st.caption(
        "Bias-reduction layer: review levels, momentum, and historical context before forming a narrative."
    )
    
    # Educational panel
    with st.expander("Understanding the Scoreboard"):
        st.markdown("""
**Purpose**

> "Before trying to predict the future, you have to correctly analyze the present. 
> I cannot tell you how many people just get analyzing the present wrong."

The scoreboard is a bias-reduction tool. It presents raw data without narrative or 
interpretation, forcing you to confront the actual numbers before forming conclusions.

---

**Why This Matters**

Cognitive biases that affect macro analysis:

| Bias | Description | How Scoreboard Helps |
|------|-------------|----------------------|
| Confirmation | Seeking data that confirms existing view | See ALL data, not cherry-picked |
| Recency | Overweighting recent events | Z-scores and percentiles show historical context |
| Anchoring | Fixating on specific levels | Multiple timeframes (1m, 3m, YoY) |
| Narrative | Fitting data to a story | Raw numbers, no interpretation |

---

**How to Use**

1. **Scan for Extremes**: Look for Z-scores beyond +/-1.5 or percentiles beyond 20/80
2. **Check Momentum**: Compare Chg_1 vs Chg_3 - is momentum accelerating or decelerating?
3. **Cross-Category**: Are growth, inflation, and financial conditions telling consistent stories?
4. **Date Check**: When was each series last updated? Stale data can mislead.

---

**Data Flow**
```
FRED API → Raw Data → Transforms → Scoreboard
                         ↓
                    Z-scores (post-2021)
                    Percentiles (full history)
                    Period changes
```

All transformations are mechanical and consistent across series.
""")
    
    # Build scoreboard
    rows = []
    
    for key, sf in fred_bundle.items():
        if key == "USREC":  # Skip recession indicator
            continue
        
        try:
            tf = compute_transforms(sf)
        except Exception as e:
            st.warning(f"Transform failed for {key}: {e}")
            continue
        
        if tf.empty:
            continue
        
        tf_clean = tf.dropna()
        if tf_clean.empty:
            continue
            
        last = tf_clean.iloc[-1]
        
        rows.append({
            "Series": sf.name,
            "Ticker": key,
            "Category": config.FRED_SERIES[key].category,
            "Freq": sf.freq,
            "Last": float(last.get("Value", float('nan'))),
            "Chg_1": float(last.get("Chg_1", float('nan'))),
            "Chg_3": float(last.get("Chg_3", float('nan'))),
            "YoY": float(last.get("YoY", float('nan'))),
            "Z": float(last.get("Z", float('nan'))),
            "Pctl": float(last.get("Pctl", float('nan'))),
            "Unit(level)": sf.unit,
            "Unit(chg)": sf.chg_unit,
            "Last Date": last.name.strftime("%Y-%m-%d") if hasattr(last.name, "strftime") else str(last.name)
        })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        st.info("No data available for scoreboard.")
        return
    
    # Summary statistics
    st.subheader("Quick Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hot_count = len(df[df["Z"] > 1.5])
        st.metric(
            "Hot Readings (Z > 1.5)",
            hot_count,
            help="Number of indicators with Z-score above 1.5 (significantly above historical average)"
        )
    
    with col2:
        cold_count = len(df[df["Z"] < -1.5])
        st.metric(
            "Cold Readings (Z < -1.5)",
            cold_count,
            help="Number of indicators with Z-score below -1.5 (significantly below historical average)"
        )
    
    with col3:
        high_pctl = len(df[df["Pctl"] > 0.8])
        st.metric(
            "High Percentile (>80th)",
            high_pctl,
            help="Number of indicators in top 20% of historical readings"
        )
    
    with col4:
        low_pctl = len(df[df["Pctl"] < 0.2])
        st.metric(
            "Low Percentile (<20th)",
            low_pctl,
            help="Number of indicators in bottom 20% of historical readings"
        )
    
    st.divider()
    
    # Filters
    st.subheader("Data Table")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Category filter
        cats = sorted(df["Category"].unique())
        selected = st.multiselect(
            "Filter categories",
            options=cats,
            default=cats,
            help="Select categories to display. Categories match the GIP framework components."
        )
    
    with col2:
        # Extreme filter
        extreme_filter = st.selectbox(
            "Highlight extremes",
            options=["All", "Hot Only (Z > 1.5)", "Cold Only (Z < -1.5)", "Extremes Only"],
            index=0,
            help="Filter to show only extreme readings"
        )
    
    view = df[df["Category"].isin(selected)].copy()
    
    # Apply extreme filter
    if extreme_filter == "Hot Only (Z > 1.5)":
        view = view[view["Z"] > 1.5]
    elif extreme_filter == "Cold Only (Z < -1.5)":
        view = view[view["Z"] < -1.5]
    elif extreme_filter == "Extremes Only":
        view = view[(view["Z"] > 1.5) | (view["Z"] < -1.5)]
    
    if view.empty:
        st.info("No data matches the current filters.")
    else:
        # Format units
        def fmt_unit(u):
            return {"pct": "%", "bps": "bps", "level": "lvl", "index": "idx"}.get(u, u)
        
        view["Unit(chg)"] = view["Unit(chg)"].apply(fmt_unit)
        view["Unit(level)"] = view["Unit(level)"].apply(fmt_unit)
        
        # Round numeric columns
        for col in ["Last", "Chg_1", "Chg_3", "YoY", "Z", "Pctl"]:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce")
        
        # Create display dataframe with formatting
        display_df = view.copy()
        
        # Format for display
        display_df["Last"] = display_df["Last"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df["Chg_1"] = display_df["Chg_1"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
        display_df["Chg_3"] = display_df["Chg_3"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
        display_df["YoY"] = display_df["YoY"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
        display_df["Z"] = display_df["Z"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
        display_df["Pctl"] = display_df["Pctl"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
        
        # Display
        st.dataframe(
            display_df[[
                "Series", "Ticker", "Category", "Freq", "Last",
                "Chg_1", "Chg_3", "YoY", "Z", "Pctl",
                "Unit(level)", "Unit(chg)", "Last Date"
            ]],
            use_container_width=True,
            hide_index=True,
            height=min(600, 50 + len(display_df) * 35)
        )
        
        # Data freshness warning
        if "Last Date" in view.columns:
            dates = pd.to_datetime(view["Last Date"])
            oldest = dates.min()
            days_old = (pd.Timestamp.now() - oldest).days
            
            if days_old > 30:
                st.warning(
                    f"Some data is {days_old} days old. Check 'Last Date' column for data freshness."
                )
    
    st.divider()
    
    # Column definitions
    with st.expander("Column Definitions"):
        st.markdown("""
**Identification Columns**

| Column | Description |
|--------|-------------|
| Series | Human-readable indicator name |
| Ticker | FRED series ID (for API reference) |
| Category | GIP framework category (Growth, Inflation, Financial, Labor) |
| Freq | Data frequency (D=Daily, W=Weekly, M=Monthly, Q=Quarterly) |

---

**Value Columns**

| Column | Description | Example |
|--------|-------------|---------|
| Last | Most recent observation | 3.45 |
| Chg_1 | 1-period change | +0.12 (rose by 0.12 from prior period) |
| Chg_3 | 3-period change | -0.25 (fell by 0.25 over 3 periods) |
| YoY | Year-over-year change | +2.50 (up 2.5 from same period last year) |

---

**Context Columns**

| Column | Description | Interpretation |
|--------|-------------|----------------|
| Z | Z-score on YoY | >1.5 = Hot, <-1.5 = Cold |
| Pctl | Historical percentile on YoY | 85% = Higher than 85% of history |

---

**Unit Columns**

| Column | Description | Values |
|--------|-------------|--------|
| Unit(level) | Units for Last column | %, bps, lvl, idx |
| Unit(chg) | Units for Chg/YoY columns | %, bps, lvl |

---

**Metadata**

| Column | Description |
|--------|-------------|
| Last Date | Date of most recent observation |
""")
    
    with st.expander("Interpretation Guide"):
        st.markdown("""
**Z-Score Interpretation**

| Z-Score | Interpretation | Statistical Meaning |
|---------|----------------|---------------------|
| > +2.0 | Extremely hot | Top ~2% of observations |
| +1.5 to +2.0 | Very hot | Top ~7% of observations |
| +0.5 to +1.5 | Above average | Above median |
| -0.5 to +0.5 | Normal range | Near historical average |
| -1.5 to -0.5 | Below average | Below median |
| -2.0 to -1.5 | Very cold | Bottom ~7% of observations |
| < -2.0 | Extremely cold | Bottom ~2% of observations |

Note: Z-scores are calculated on post-2021 data only to avoid COVID distortions.

---

**Percentile Interpretation**

| Percentile | Interpretation |
|------------|----------------|
| 90-100% | Historically very high |
| 70-90% | Above average |
| 30-70% | Normal range |
| 10-30% | Below average |
| 0-10% | Historically very low |

Note: Percentiles use full available history.

---

**Momentum Analysis**

Compare Chg_1 vs Chg_3 to assess momentum:

| Pattern | Chg_1 vs Chg_3 | Signal |
|---------|----------------|--------|
| Accelerating | Chg_1 > Chg_3/3 | Momentum building |
| Decelerating | Chg_1 < Chg_3/3 | Momentum fading |
| Stable | Chg_1 ≈ Chg_3/3 | Steady trend |
| Reversal | Opposite signs | Potential inflection |

---

**Cross-Category Consistency**

Look for alignment across categories:

| Pattern | Interpretation |
|---------|----------------|
| All hot | Overheating, watch for Fed response |
| All cold | Slowdown, watch for policy support |
| Growth hot, Inflation cold | Goldilocks |
| Growth cold, Inflation hot | Stagflation risk |
| Mixed signals | Transition period, wait for clarity |
""")
    
    with st.expander("Data Sources & Methodology"):
        st.markdown("""
**Data Sources**

All economic data is sourced from FRED (Federal Reserve Economic Data) via their API.
Market data is sourced from Yahoo Finance.

---

**Transform Calculations**

| Transform | Formula |
|-----------|---------|
| Chg_1 | Current - Prior period |
| Chg_3 | Current - 3 periods ago |
| YoY | Current - Same period last year |
| Z-score | (YoY - Mean(YoY)) / StdDev(YoY) |
| Percentile | Rank(YoY) / Count(YoY) |

---

**Z-Score Window**

Z-scores are calculated using data from 2021-01-01 onwards only. This avoids 
the COVID period distortions that would otherwise dominate the statistics.

---

**Unit Conversions**

| Original Unit | Change Unit | Notes |
|---------------|-------------|-------|
| Index/Level | Percent change | YoY = (Current/Prior - 1) × 100 |
| Percent | Basis points | YoY = (Current - Prior) × 100 |
| Basis points | Basis points | YoY = Current - Prior |

---

**Update Frequency**

| Frequency Code | Meaning | Typical Lag |
|----------------|---------|-------------|
| D | Daily | 1 day |
| W | Weekly | 1 week |
| M | Monthly | 2-4 weeks |
| Q | Quarterly | 4-8 weeks |
""")
    
    st.divider()
    
    # Regime context
    st.subheader("Current Regime Context")
    
    with st.expander("Regime Summary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**GIP Classification**")
            
            st.markdown(f"""
| Component | State | Score |
|-----------|-------|-------|
| **Macro Regime** | {current_regime.macro.value.replace('_', ' ').title()} | - |
| Growth | {current_regime.growth.value.title()} | {current_regime.growth_score:+.2f} |
| Inflation | {current_regime.inflation.value.title()} | {current_regime.inflation_score:+.2f} |
| Liquidity | {current_regime.liquidity.value.title()} | {current_regime.liquidity_score:+.2f} |
""")
            
            st.metric(
                "Regime Confidence",
                f"{current_regime.confidence:.0%}",
                help="How clearly the data supports this regime classification. "
                     "Higher = more conviction. Lower = mixed signals."
            )
        
        with col2:
            st.markdown("**Score Interpretation**")
            st.markdown("""
| Score Range | Interpretation |
|-------------|----------------|
| > +1.0 | Strong positive momentum |
| +0.5 to +1.0 | Moderate positive |
| -0.5 to +0.5 | Neutral/mixed |
| -1.0 to -0.5 | Moderate negative |
| < -1.0 | Strong negative momentum |

---

**Confidence Levels**

| Confidence | Meaning |
|------------|---------|
| > 70% | High conviction, clear regime |
| 50-70% | Moderate conviction |
| < 50% | Low conviction, mixed signals |
""")
    
    # Scoreboard vs Other Views
    with st.expander("Scoreboard vs Other Dashboard Views"):
        st.markdown("""
**How This Tab Differs**

| Tab | Purpose | Abstraction Level |
|-----|---------|-------------------|
| Summary | Quick regime snapshot | High (synthesized) |
| GIP Regime | Regime classification | High (model output) |
| Heatmap | Visual percentile scan | Medium (normalized) |
| Nowcast | GDP/Inflation estimates | Medium (aggregated) |
| **Scoreboard** | Raw data review | **Low (source data)** |

---

**When to Use Scoreboard**

- **Verify other tabs**: Cross-check regime classification against raw data
- **Identify drivers**: Which specific indicators are driving the aggregate scores?
- **Spot anomalies**: Find data points that don't fit the narrative
- **Due diligence**: Before making decisions, review the actual numbers

---

**Workflow Integration**

1. Start with Summary tab for quick orientation
2. Deep dive into relevant tabs (Rates, Sectors, etc.)
3. **End with Scoreboard** to verify conclusions against raw data
4. If Scoreboard contradicts your view, investigate the discrepancy
""")
    
    st.caption(
        "The scoreboard presents raw data without interpretation. "
        "Form your own conclusions before reading narrative content in other tabs."
    )