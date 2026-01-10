# views/heatmap_tab.py
"""
Economic Data Heatmap - Visual tracker of growth drivers across time periods
Uses historical percentile ranks for color coding (professional approach)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from data.fred import SeriesFrame


@dataclass
class HeatmapIndicator:
    """Configuration for a heatmap indicator"""
    fred_key: str
    label: str
    source: str
    measurement: str  # MoM, YoY, QoQ, Level, Index
    invert: bool = False  # True if lower is better (e.g., unemployment, claims)
    multiplier: float = 1.0  # For display formatting
    decimals: int = 1
    suffix: str = ""  # %, k, etc.
    transform: str = "level"  # level, yoy, mom, diff


# Define indicators for the heatmap
HEATMAP_INDICATORS: List[HeatmapIndicator] = [
    # GDP & Nowcast
    HeatmapIndicator("GDPNOW", "GDPNow", "Atlanta Fed", "QoQ", False, 1, 2, "%", "level"),
    
    # Labor Market
    HeatmapIndicator("PAYEMS", "Non-Farm Payrolls", "BLS", "MoM Chg", False, 1, 0, "k", "diff"),
    HeatmapIndicator("MANEMP", "Manufacturing Emp", "BLS", "MoM Chg", False, 1, 0, "k", "diff"),
    HeatmapIndicator("ICSA", "Initial Claims", "DOL", "Weekly", True, 0.001, 0, "k", "level"),
    HeatmapIndicator("AWHMAN", "Avg Hours Worked", "BLS", "Level", False, 1, 1, "", "level"),
    
    # Growth Indicators
    HeatmapIndicator("INDPRO", "Industrial Production", "Fed", "YoY", False, 1, 2, "%", "yoy"),
    HeatmapIndicator("RSAFS", "Retail Sales", "Census", "YoY", False, 1, 2, "%", "yoy"),
    
    # Inflation
    HeatmapIndicator("CPIAUCSL", "CPI Headline", "BLS", "YoY", True, 1, 2, "%", "yoy"),
    HeatmapIndicator("CPILFESL", "CPI Core", "BLS", "YoY", True, 1, 2, "%", "yoy"),
    HeatmapIndicator("PCEPILFE", "Core PCE", "BEA", "YoY", True, 1, 2, "%", "yoy"),
    HeatmapIndicator("T10YIE", "10Y Breakeven", "FRED", "Level", True, 1, 2, "%", "level"),
    
    # Financial Conditions
    HeatmapIndicator("NFCI", "Chicago Fed NFCI", "Chicago Fed", "Level", True, 1, 2, "", "level"),
    HeatmapIndicator("DFII10", "10Y Real Yield", "FRED", "Level", True, 1, 2, "%", "level"),
    HeatmapIndicator("BAMLH0A0HYM2", "HY Credit Spread", "FRED", "Level", True, 100, 0, "bps", "level"),
    HeatmapIndicator("BAMLC0A0CM", "IG Credit Spread", "FRED", "Level", True, 100, 0, "bps", "level"),
]


def render_heatmap_tab(
    fred_bundle: Dict[str, SeriesFrame],
    lookback_years: int
):
    """
    Render the Economic Heatmap tab.
    """
    
    st.header("Economic Data Heatmap")
    st.caption("Track growth drivers across multiple periods. Colors based on historical percentile rank.")
    
    with st.expander("Understanding the Heatmap"):
        st.markdown("""
**Purpose**

The heatmap provides a rapid visual assessment of where each economic indicator 
sits relative to its own history. Rather than asking "is 3% CPI high?", we ask 
"where does 3% CPI rank in the last 10 years of readings?"

This approach solves a critical problem: different indicators have different scales, 
different volatilities, and different "normal" ranges. Percentile ranking normalizes 
everything to a common 0-100 scale.

---

**Percentile Calculation**

For each value, we compute:
```
Percentile = (Count of historical values below current) / (Total historical observations) x 100
```

A reading of 85% means the current value exceeds 85% of all historical observations 
in the lookback window.

---

**Color Scale (Quintile-Based)**

| Percentile Range | Color | Interpretation |
|------------------|-------|----------------|
| 80-100% | Dark Green | Top quintile - very strong |
| 60-80% | Light Green | Above average |
| 40-60% | Yellow | Normal range |
| 20-40% | Orange | Below average |
| 0-20% | Red | Bottom quintile - very weak |

---

**Inverted Indicators**

For indicators where *lower is better*, the color logic is inverted:

| Indicator Type | Inversion | Logic |
|----------------|-----------|-------|
| Growth (GDP, Payrolls, Production) | No | Higher = better = green |
| Inflation (CPI, PCE, Breakevens) | Yes | Lower = better = green |
| Unemployment, Claims | Yes | Lower = better = green |
| Credit Spreads | Yes | Tighter = better = green |
| Real Yields, NFCI | Yes | Lower = easier = green |

---

**Column Definitions**

| Column | Description |
|--------|-------------|
| T-6 to T-1 | Historical period values (most recent 6 observations) |
| Latest | Most recent data point |
| 6P Mean | Average of last 6 periods |
| 3P Mean | Average of last 3 periods |
| 6P Spd | Speed: Change from T-6 to Latest (momentum) |
| 3P Spd | Speed: Change from T-3 to Latest (recent momentum) |

---

**Data Transformations**

Raw data is transformed before percentile calculation:

| Transform | Applied To | Calculation |
|-----------|------------|-------------|
| YoY | CPI, PCE, Production, Sales | Year-over-year % change |
| Diff | Payrolls, Mfg Employment | Month-over-month level change |
| Level | GDP, Claims, Yields, Spreads | No transformation |
""")
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        num_periods = st.slider(
            "Periods to display", 
            4, 12, 6, 
            key="heatmap_periods",
            help="Number of historical periods to show in the heatmap grid"
        )
    with col2:
        hist_window_years = st.slider(
            "Historical window (years)", 
            3, 20, 10, 
            key="heatmap_history",
            help="Lookback period for percentile calculation. Longer = more context, but may include different regimes"
        )
    
    # Build the heatmap data
    heatmap_data, percentile_data = _build_heatmap_data(
        fred_bundle, 
        HEATMAP_INDICATORS, 
        num_periods,
        hist_window_years
    )
    
    if heatmap_data.empty:
        st.warning("No data available for heatmap")
        return

    # Display the heatmap
    _render_styled_heatmap(heatmap_data, percentile_data, HEATMAP_INDICATORS, num_periods)
    
    st.divider()
    
    # Summary statistics
    _render_heatmap_summary(heatmap_data, percentile_data, HEATMAP_INDICATORS)
    
    # Distribution explorer
    _render_distribution_explorer(fred_bundle, HEATMAP_INDICATORS, hist_window_years)


def _compute_historical_percentile(value: float, historical_series: pd.Series) -> float:
    """
    Compute the percentile rank of a value within its historical distribution.
    
    Returns a value between 0 and 100.
    """
    if pd.isna(value) or len(historical_series.dropna()) < 10:
        return np.nan
    
    historical = historical_series.dropna()
    percentile = (historical < value).sum() / len(historical) * 100
    return percentile


def _compute_z_score(value: float, historical_series: pd.Series) -> float:
    """
    Compute z-score of a value relative to historical distribution.
    """
    if pd.isna(value) or len(historical_series.dropna()) < 10:
        return np.nan
    
    historical = historical_series.dropna()
    mean = historical.mean()
    std = historical.std()
    
    if std == 0:
        return 0
    
    return (value - mean) / std


def _transform_series(series: pd.Series, transform: str) -> pd.Series:
    """
    Apply transformation to raw series.
    """
    if transform == "yoy":
        # Year-over-year percentage change
        return series.pct_change(12) * 100
    elif transform == "mom":
        # Month-over-month percentage change
        return series.pct_change(1) * 100
    elif transform == "diff":
        # Simple difference (for payrolls, etc.)
        return series.diff(1)
    elif transform == "qoq":
        # Quarter-over-quarter
        return series.pct_change(3) * 100
    else:
        # Level - no transform
        return series


def _build_heatmap_data(
    fred_bundle: Dict[str, SeriesFrame],
    indicators: List[HeatmapIndicator],
    num_periods: int,
    hist_window_years: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the heatmap DataFrame with values and percentiles.
    
    Returns:
        Tuple of (values_df, percentiles_df)
    """
    value_rows = []
    percentile_rows = []
    
    for ind in indicators:
        if ind.fred_key not in fred_bundle:
            continue
        
        sf = fred_bundle[ind.fred_key]
        raw_series = sf.df["Value"].dropna()
        
        if len(raw_series) < 24:  # Need at least 2 years
            continue
        
        # Apply transformation
        transformed = _transform_series(raw_series, ind.transform).dropna()
        
        if len(transformed) < num_periods + 12:
            continue
        
        # Historical window for percentile calculation
        hist_periods = hist_window_years * 12  # Approximate monthly
        if len(transformed) > hist_periods:
            historical = transformed.iloc[-hist_periods:]
        else:
            historical = transformed
        
        # Get last N periods for display
        recent = transformed.iloc[-(num_periods):].values
        recent_indices = transformed.iloc[-(num_periods):].index
        
        # Apply multiplier for display
        recent_display = recent * ind.multiplier
        
        # Calculate percentiles for each period value
        recent_percentiles = []
        for val in recent:
            pctl = _compute_historical_percentile(val, historical)
            recent_percentiles.append(pctl)
        
        # Calculate statistics
        mean_3p = np.mean(recent[-3:]) * ind.multiplier if len(recent) >= 3 else np.nan
        mean_6p = np.mean(recent[-6:]) * ind.multiplier if len(recent) >= 6 else np.nan
        
        # Speed (change over period) - also compute percentile
        speed_3p = (recent[-1] - recent[-3]) * ind.multiplier if len(recent) >= 3 else np.nan
        speed_6p = (recent[-1] - recent[-6]) * ind.multiplier if len(recent) >= 6 else np.nan
        
        # Speed percentiles (based on historical speed distribution)
        hist_speed_3 = transformed.diff(3).dropna()
        hist_speed_6 = transformed.diff(6).dropna()
        
        speed_3p_raw = (recent[-1] - recent[-3]) if len(recent) >= 3 else np.nan
        speed_6p_raw = (recent[-1] - recent[-6]) if len(recent) >= 6 else np.nan
        
        speed_3p_pctl = _compute_historical_percentile(speed_3p_raw, hist_speed_3) if not pd.isna(speed_3p_raw) else np.nan
        speed_6p_pctl = _compute_historical_percentile(speed_6p_raw, hist_speed_6) if not pd.isna(speed_6p_raw) else np.nan
        
        # Mean percentiles
        mean_3p_pctl = _compute_historical_percentile(np.mean(recent[-3:]), historical) if len(recent) >= 3 else np.nan
        mean_6p_pctl = _compute_historical_percentile(np.mean(recent[-6:]), historical) if len(recent) >= 6 else np.nan
        
        # Build value row
        value_row = {
            "Indicator": ind.label,
            "Source": ind.source,
            "Measurement": ind.measurement,
            "fred_key": ind.fred_key,
        }
        
        # Build percentile row
        pctl_row = {
            "Indicator": ind.label,
            "fred_key": ind.fred_key,
        }
        
        # Add period columns
        for i, (val, pctl) in enumerate(zip(recent_display, recent_percentiles)):
            period_label = f"T-{num_periods - i}" if i < num_periods - 1 else "Latest"
            value_row[period_label] = val
            pctl_row[period_label] = pctl
        
        # Add statistics
        value_row["6P Mean"] = mean_6p
        value_row["3P Mean"] = mean_3p
        value_row["6P Spd"] = speed_6p
        value_row["3P Spd"] = speed_3p
        
        pctl_row["6P Mean"] = mean_6p_pctl
        pctl_row["3P Mean"] = mean_3p_pctl
        pctl_row["6P Spd"] = speed_6p_pctl
        pctl_row["3P Spd"] = speed_3p_pctl
        
        value_rows.append(value_row)
        percentile_rows.append(pctl_row)
    
    return pd.DataFrame(value_rows), pd.DataFrame(percentile_rows)


def _percentile_to_color(percentile: float, invert: bool = False) -> str:
    """
    Convert percentile to color based on quintile.
    
    Args:
        percentile: 0-100 percentile rank
        invert: If True, lower percentile = better (green)
    
    Returns:
        CSS background-color string
    """
    if pd.isna(percentile):
        return "background-color: #333333"  # Gray for missing
    
    # Flip percentile if inverted (lower = better)
    if invert:
        percentile = 100 - percentile
    
    # Color gradient based on quintile
    if percentile >= 80:
        # Top quintile - dark green
        return "background-color: rgba(0, 150, 0, 0.8)"
    elif percentile >= 60:
        # Above average - light green
        return "background-color: rgba(50, 180, 50, 0.6)"
    elif percentile >= 40:
        # Average - yellow
        return "background-color: rgba(200, 200, 0, 0.5)"
    elif percentile >= 20:
        # Below average - orange
        return "background-color: rgba(220, 140, 0, 0.6)"
    else:
        # Bottom quintile - red
        return "background-color: rgba(200, 0, 0, 0.7)"


def _render_styled_heatmap(
    values_df: pd.DataFrame,
    percentiles_df: pd.DataFrame,
    indicators: List[HeatmapIndicator],
    num_periods: int
):
    """
    Render the heatmap with percentile-based color styling.
    """
    # Create indicator lookup
    ind_lookup = {ind.label: ind for ind in indicators}
    
    # Get display columns (exclude metadata)
    meta_cols = ["Indicator", "Source", "Measurement", "fred_key"]
    data_cols = [c for c in values_df.columns if c not in meta_cols]
    
    # Build styled HTML table
    html = """
    <style>
        .heatmap-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .heatmap-table th {
            background-color: #1a1a1a;
            color: #ffa500;
            padding: 10px 6px;
            text-align: center;
            border: 1px solid #444;
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        .heatmap-table td {
            padding: 8px 6px;
            text-align: center;
            border: 1px solid #444;
            color: white;
            font-weight: 500;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }
        .heatmap-table tr:hover {
            outline: 2px solid #ffa500;
        }
        .indicator-cell {
            text-align: left !important;
            font-weight: bold;
            background-color: #1a1a1a !important;
            color: #fff;
            min-width: 140px;
        }
        .source-cell {
            text-align: center !important;
            background-color: #1a1a1a !important;
            color: #888;
            font-size: 10px;
        }
        .measurement-cell {
            background-color: #1a1a1a !important;
            color: #888;
            font-size: 10px;
        }
        .speed-col {
            font-style: italic;
        }
    </style>
    <table class="heatmap-table">
    <thead>
        <tr>
            <th>Indicator</th>
            <th>Source</th>
            <th>Type</th>
    """
    
    # Add period headers
    for col in data_cols:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    # Add data rows
    for idx, (_, value_row) in enumerate(values_df.iterrows()):
        indicator = ind_lookup.get(value_row["Indicator"])
        if not indicator:
            continue
        
        pctl_row = percentiles_df.iloc[idx]
        
        suffix = indicator.suffix
        decimals = indicator.decimals
        
        html += f"""
        <tr>
            <td class="indicator-cell">{value_row['Indicator']}</td>
            <td class="source-cell">{value_row['Source']}</td>
            <td class="measurement-cell">{value_row['Measurement']}</td>
        """
        
        for col in data_cols:
            value = value_row[col]
            percentile = pctl_row.get(col, np.nan)
            
            # Determine if this is a speed column
            is_speed = "Spd" in col
            
            # Get color based on percentile
            color = _percentile_to_color(percentile, indicator.invert)
            
            # Format display value
            if pd.isna(value):
                display_val = "-"
            else:
                # Add sign for speed columns
                if is_speed:
                    if decimals == 0:
                        display_val = f"{value:+.0f}{suffix}"
                    elif decimals == 1:
                        display_val = f"{value:+.1f}{suffix}"
                    else:
                        display_val = f"{value:+.2f}{suffix}"
                else:
                    if decimals == 0:
                        display_val = f"{value:.0f}{suffix}"
                    elif decimals == 1:
                        display_val = f"{value:.1f}{suffix}"
                    else:
                        display_val = f"{value:.2f}{suffix}"
            
            speed_class = ' class="speed-col"' if is_speed else ''
            html += f'<td{speed_class} style="{color}">{display_val}</td>'
        
        html += "</tr>"
    
    html += "</tbody></table>"
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
    <div style="display: flex; gap: 15px; margin-top: 10px; font-size: 11px; justify-content: center;">
        <span style="background-color: rgba(0, 150, 0, 0.8); padding: 2px 8px; color: white;">Top 20%</span>
        <span style="background-color: rgba(50, 180, 50, 0.6); padding: 2px 8px; color: white;">60-80%</span>
        <span style="background-color: rgba(200, 200, 0, 0.5); padding: 2px 8px; color: black;">40-60%</span>
        <span style="background-color: rgba(220, 140, 0, 0.6); padding: 2px 8px; color: white;">20-40%</span>
        <span style="background-color: rgba(200, 0, 0, 0.7); padding: 2px 8px; color: white;">Bottom 20%</span>
    </div>
    """, unsafe_allow_html=True)


def _render_heatmap_summary(
    values_df: pd.DataFrame, 
    percentiles_df: pd.DataFrame,
    indicators: List[HeatmapIndicator]
):
    """
    Render summary statistics below the heatmap.
    """
    st.subheader("Aggregate Conditions")
    
    with st.expander("Aggregation Methodology"):
        st.markdown("""
        **Category Scoring**
        
        Indicators are grouped into four categories. Each category score is the 
        average percentile of its constituent indicators, adjusted for inversion.
        
        | Category | Indicators Included |
        |----------|---------------------|
        | Growth | GDPNow, Payrolls, Industrial Production, Retail Sales, Mfg Employment, Hours Worked |
        | Labor Market | Payrolls, Unemployment, Initial Claims, Hours Worked |
        | Inflation | CPI Headline, CPI Core, Core PCE, 10Y Breakeven |
        | Financial Conditions | NFCI, 10Y Real Yield, HY Spread, IG Spread |
        
        ---
        
        **Score Interpretation**
        
        | Average Percentile | Label | Implication |
        |--------------------|-------|-------------|
        | 70-100% | Strong | Conditions significantly above average |
        | 55-69% | Above Avg | Conditions moderately favorable |
        | 45-54% | Neutral | Conditions near historical average |
        | 30-44% | Below Avg | Conditions moderately weak |
        | 0-29% | Weak | Conditions significantly below average |
        
        ---
        
        **Momentum Arrows**
        
        The arrows indicate how many indicators are improving vs deteriorating:
        - Improving: 3-period speed percentile >= 50 (momentum above median)
        - Deteriorating: 3-period speed percentile < 50 (momentum below median)
        
        ---
        
        **Special Case: Inflation**
        
        For inflation, the interpretation is inverted:
        - "Contained" = High percentile (after inversion) = Low inflation readings
        - "Elevated" = Low percentile (after inversion) = High inflation readings
        """)
    
    ind_lookup = {ind.label: ind for ind in indicators}
    
    # Categorize indicators
    growth_indicators = ["GDPNow", "Non-Farm Payrolls", "Industrial Production", "Retail Sales", 
                        "Manufacturing Emp", "Avg Hours Worked"]
    labor_indicators = ["Non-Farm Payrolls", "Unemployment Rate", "Initial Claims", "Avg Hours Worked"]
    inflation_indicators = ["CPI Headline", "CPI Core", "Core PCE", "10Y Breakeven"]
    financial_indicators = ["Chicago Fed NFCI", "10Y Real Yield", "HY Credit Spread", "IG Credit Spread"]
    
    def calc_category_score(category_names: List[str]) -> Tuple[float, int, int]:
        """Calculate average percentile for a category, accounting for inversion."""
        scores = []
        improving = 0
        deteriorating = 0
        
        for _, row in percentiles_df.iterrows():
            if row["Indicator"] not in category_names:
                continue
            
            indicator = ind_lookup.get(row["Indicator"])
            if not indicator:
                continue
            
            latest_pctl = row.get("Latest", np.nan)
            speed_pctl = row.get("3P Spd", np.nan)
            
            if not pd.isna(latest_pctl):
                # Adjust for inversion
                adjusted_pctl = (100 - latest_pctl) if indicator.invert else latest_pctl
                scores.append(adjusted_pctl)
            
            if not pd.isna(speed_pctl):
                adjusted_speed = (100 - speed_pctl) if indicator.invert else speed_pctl
                if adjusted_speed >= 50:
                    improving += 1
                else:
                    deteriorating += 1
        
        avg_score = np.mean(scores) if scores else np.nan
        return avg_score, improving, deteriorating
    
    # Calculate scores for each category
    growth_score, growth_up, growth_down = calc_category_score(growth_indicators)
    labor_score, labor_up, labor_down = calc_category_score(labor_indicators)
    inflation_score, infl_up, infl_down = calc_category_score(inflation_indicators)
    financial_score, fin_up, fin_down = calc_category_score(financial_indicators)
    
    def score_to_label(score: float) -> Tuple[str, str]:
        """Convert score to label and color."""
        if pd.isna(score):
            return "N/A", "gray"
        elif score >= 70:
            return "Strong", "green"
        elif score >= 55:
            return "Above Avg", "lightgreen"
        elif score >= 45:
            return "Neutral", "yellow"
        elif score >= 30:
            return "Below Avg", "orange"
        else:
            return "Weak", "red"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        label, color = score_to_label(growth_score)
        st.metric(
            "Growth Conditions", 
            label,
            delta=f"{growth_up}+ {growth_down}-",
            help=f"Average adjusted percentile: {growth_score:.0f}%. "
                 f"Based on: GDPNow, Payrolls, Industrial Production, Retail Sales, Mfg Employment, Hours Worked."
        )
    
    with col2:
        label, color = score_to_label(labor_score)
        st.metric(
            "Labor Market", 
            label,
            delta=f"{labor_up}+ {labor_down}-",
            help=f"Average adjusted percentile: {labor_score:.0f}%. "
                 f"Based on: Payrolls, Unemployment (inv), Initial Claims (inv), Hours Worked."
        )
    
    with col3:
        # For inflation, LOWER is better, so invert the label logic
        infl_label = "Elevated" if inflation_score < 40 else ("Contained" if inflation_score > 60 else "Moderate")
        st.metric(
            "Inflation Pressure", 
            infl_label,
            delta=f"{infl_up}+ {infl_down}-",
            help=f"Average adjusted percentile: {inflation_score:.0f}% (higher = less pressure). "
                 f"Based on: CPI Headline, CPI Core, Core PCE, 10Y Breakeven (all inverted)."
        )
    
    with col4:
        label, color = score_to_label(financial_score)
        st.metric(
            "Financial Conditions", 
            label,
            delta=f"{fin_up}+ {fin_down}-",
            help=f"Average adjusted percentile: {financial_score:.0f}%. "
                 f"Based on: NFCI, 10Y Real Yield, HY Spread, IG Spread (all inverted - lower = easier)."
        )


def _render_distribution_explorer(
    fred_bundle: Dict[str, SeriesFrame],
    indicators: List[HeatmapIndicator],
    hist_window_years: int
):
    """
    Interactive explorer to see where current values sit in distribution.
    """
    st.subheader("Distribution Explorer")
    
    with st.expander("Explore Individual Indicator Distributions"):
        
        st.markdown("""
        **Purpose**
        
        The distribution explorer visualizes where the current reading falls within 
        the full historical distribution. This provides deeper context than the 
        heatmap percentile alone.
        
        **Interpretation Guide**
        
        | Position | Percentile | Signal Strength |
        |----------|------------|-----------------|
        | Far right tail | > 90% | Extreme high - potential mean reversion |
        | Right of center | 60-90% | Above average - favorable (or concerning if inverted) |
        | Near center | 40-60% | Normal range - no strong signal |
        | Left of center | 10-40% | Below average - concerning (or favorable if inverted) |
        | Far left tail | < 10% | Extreme low - potential mean reversion |
        
        The Z-score provides additional context: readings beyond +/- 2.0 are statistically 
        unusual (outside 95% of observations).
        """)
        
        st.divider()
        
        # Dropdown to select indicator
        ind_labels = [ind.label for ind in indicators if ind.fred_key in fred_bundle]
        selected = st.selectbox(
            "Select Indicator", 
            ind_labels,
            help="Choose an indicator to view its historical distribution"
        )
        
        # Find the indicator
        selected_ind = next((ind for ind in indicators if ind.label == selected), None)
        
        if selected_ind and selected_ind.fred_key in fred_bundle:
            sf = fred_bundle[selected_ind.fred_key]
            raw_series = sf.df["Value"].dropna()
            transformed = _transform_series(raw_series, selected_ind.transform).dropna()
            
            # Historical window
            hist_periods = hist_window_years * 12
            if len(transformed) > hist_periods:
                historical = transformed.iloc[-hist_periods:]
            else:
                historical = transformed
            
            current_value = transformed.iloc[-1] * selected_ind.multiplier
            percentile = _compute_historical_percentile(transformed.iloc[-1], historical)
            z_score = _compute_z_score(transformed.iloc[-1], historical)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Value", 
                    f"{current_value:.{selected_ind.decimals}f}{selected_ind.suffix}",
                    help="Most recent observation after transformation"
                )
            with col2:
                st.metric(
                    "Percentile", 
                    f"{percentile:.0f}%",
                    help="Percentage of historical values below current reading"
                )
            with col3:
                st.metric(
                    "Z-Score", 
                    f"{z_score:+.2f}",
                    help="Standard deviations from historical mean. Beyond +/-2 is unusual."
                )
            with col4:
                hist_mean = historical.mean() * selected_ind.multiplier
                st.metric(
                    f"{hist_window_years}Y Mean", 
                    f"{hist_mean:.{selected_ind.decimals}f}{selected_ind.suffix}",
                    help=f"Average value over the {hist_window_years}-year lookback window"
                )
            
            # Histogram
            import plotly.graph_objects as go
            
            hist_values = historical * selected_ind.multiplier
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=hist_values,
                nbinsx=30,
                name="Historical Distribution",
                marker_color="rgba(100, 100, 200, 0.6)"
            ))
            
            # Add current value line
            fig.add_vline(
                x=current_value,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"Current: {current_value:.{selected_ind.decimals}f}",
                annotation_position="top"
            )
            
            # Add mean line
            fig.add_vline(
                x=hist_mean,
                line_dash="dot",
                line_color="yellow",
                line_width=2,
                annotation_text=f"Mean: {hist_mean:.{selected_ind.decimals}f}",
                annotation_position="bottom"
            )
            
            fig.update_layout(
                title=f"{selected} - Historical Distribution ({hist_window_years} Years)",
                xaxis_title=f"Value ({selected_ind.suffix})" if selected_ind.suffix else "Value",
                yaxis_title="Frequency",
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if selected_ind.invert:
                if percentile < 30:
                    st.success(f"Current value is in the **bottom 30%** historically - this is **favorable** for this indicator.")
                elif percentile > 70:
                    st.error(f"Current value is in the **top 30%** historically - this is **concerning** for this indicator.")
                else:
                    st.info(f"Current value is near historical average.")
            else:
                if percentile > 70:
                    st.success(f"Current value is in the **top 30%** historically - this is **favorable** for this indicator.")
                elif percentile < 30:
                    st.error(f"Current value is in the **bottom 30%** historically - this is **concerning** for this indicator.")
                else:
                    st.info(f"Current value is near historical average.")
            
            # Inversion note
            if selected_ind.invert:
                st.caption(
                    f"Note: {selected} is an inverted indicator (lower = better). "
                    f"In the heatmap, low percentiles appear green."
                )