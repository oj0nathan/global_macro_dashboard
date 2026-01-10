# views/sectors_tab.py
"""
Sector and Factor Performance Heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional

from data.yfinance import YFinanceClient


# Sector ETF mapping
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Disc.": "XLY",
    "Consumer Stap.": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Comm. Services": "XLC",
}

# Factor ETF mapping
FACTOR_ETFS = {
    "Momentum": "MTUM",
    "Value": "VLUE",
    "Quality": "QUAL",
    "Size (Small)": "SIZE",
    "Low Vol": "USMV",
    "High Beta": "SPHB",
}

# Benchmark
BENCHMARK = "SPY"


def render_sectors_tab(
    yf_client: YFinanceClient,
    lookback_years: int
):
    """
    Render Sector and Factor performance tab.
    """
    
    st.header("Sector & Factor Performance")
    st.caption(
        "Track relative performance across sectors and factors to identify regime shifts and leadership changes."
    )
    
    # Educational panel
    with st.expander("Understanding Sector & Factor Rotation"):
        st.markdown("""
**The Core Insight**

> "Sector performance over the last 60 trading days has actually been much more supportive 
> of risk-taking... Industrials are leading, with financials and tech close behind. 
> Risk is still being recycled into equities."

Sector and factor leadership reveals what the market is pricing in about the macro regime. 
When cyclicals lead, the market expects growth. When defensives lead, the market is cautious.

---

**Why Relative Performance Matters**

Absolute returns tell you how much money you made. Relative returns tell you what the 
market believes about the future. A sector can be up 10% but still be "lagging" if the 
benchmark is up 15%.

| Metric | What It Tells You |
|--------|-------------------|
| Absolute return | P&L on the position |
| Relative return (vs SPY) | Market's sector preference |
| Factor return | Market's style preference |

---

**Sector Implications by Macro Regime**

| Regime | Leading Sectors | Lagging Sectors | Rationale |
|--------|-----------------|-----------------|-----------|
| Goldilocks | Tech, Discretionary, Industrials | Utilities, Staples | Growth + low inflation = risk-on |
| Reflation | Energy, Materials, Financials | Tech (high P/E), Utilities | Inflation benefits real assets |
| Stagflation | Utilities, Healthcare, Staples | Discretionary, Industrials | Defensive + pricing power |
| Deflation | Utilities, Staples, Healthcare | Energy, Materials, Financials | Safety + duration |

---

**Factor Definitions**

| Factor | ETF | What It Captures | Best Environment |
|--------|-----|------------------|------------------|
| Momentum | MTUM | Winners keep winning | Trending markets, low vol |
| Value | VLUE | Cheap vs fundamentals | Early recovery, reflation |
| Quality | QUAL | High ROE, low debt | Late cycle, uncertainty |
| Size (Small) | SIZE | Small cap premium | Risk-on, domestic growth |
| Low Vol | USMV | Stable, defensive | Risk-off, late cycle |
| High Beta | SPHB | Amplified market moves | Maximum risk appetite |

---

**Reading Factor Leadership**

> "What it is showing, over the past 3 months, is a clear shift in appetite. Growth and 
> momentum have been slipping down the rankings, while concentration and value have moved 
> into the lead. That isn't a high-risk-on setup."

Factor leadership shifts often precede sector rotation:

| Factor Shift | Signal | Implication |
|--------------|--------|-------------|
| Momentum → Value | Risk appetite fading | Reduce growth exposure |
| Quality → High Beta | Risk appetite rising | Add cyclical exposure |
| Low Vol leading | Defensive positioning | Late cycle caution |
| Small Cap leading | Domestic optimism | Favor US over international |

---

**Cross-Checking with Macro**

Always validate sector/factor signals against macro indicators:

- Sectors say "risk-on" but spreads widening? → Divergence, be cautious
- Factors say "defensive" but curve steepening? → Transition period
- Leadership narrow (few sectors leading)? → Late cycle, selectivity matters
""")
    
    # Time period selector
    period_options = {
        "1 Week": 5,
        "1 Month": 21,
        "3 Months": 63,
        "6 Months": 126,
        "1 Year": 252,
        "YTD": "YTD"
    }
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_period = st.selectbox(
            "Performance Period",
            options=list(period_options.keys()),
            index=2,  # Default to 3 months
            help="Timeframe for calculating returns. 3 months (63 trading days) balances "
                 "noise reduction with responsiveness to regime changes."
        )
    
    with col2:
        st.caption(
            f"Showing {selected_period} returns relative to SPY. "
            "Green bars = outperforming benchmark. Red bars = underperforming."
        )
    
    st.divider()
    
    # -----------------------------
    # Sector Heatmap
    # -----------------------------
    st.subheader("Sector Performance")
    
    with st.expander("Sector ETF Reference"):
        st.markdown("""
**SPDR Select Sector ETFs**

These ETFs track the 11 GICS (Global Industry Classification Standard) sectors 
of the S&P 500:

| Sector | Ticker | Key Holdings | Cyclicality |
|--------|--------|--------------|-------------|
| Technology | XLK | AAPL, MSFT, NVDA | Cyclical |
| Financials | XLF | BRK.B, JPM, V | Cyclical |
| Healthcare | XLV | UNH, JNJ, LLY | Defensive |
| Consumer Disc. | XLY | AMZN, TSLA, HD | Cyclical |
| Consumer Stap. | XLP | PG, KO, PEP | Defensive |
| Energy | XLE | XOM, CVX | Cyclical |
| Industrials | XLI | CAT, UNP, HON | Cyclical |
| Materials | XLB | LIN, APD, SHW | Cyclical |
| Utilities | XLU | NEE, DUK, SO | Defensive |
| Real Estate | XLRE | PLD, AMT, EQIX | Interest-rate sensitive |
| Comm. Services | XLC | META, GOOGL, NFLX | Mixed |

---

**Cyclical vs Defensive**

- **Cyclicals** (Tech, Discretionary, Industrials, Financials, Materials, Energy): 
  Outperform when economy expanding
- **Defensives** (Utilities, Staples, Healthcare): 
  Outperform when economy slowing or uncertain
""")
    
    sector_data = _fetch_sector_data(yf_client, lookback_years)
    
    if sector_data is not None and not sector_data.empty:
        # Calculate returns for selected period
        sector_returns = _calculate_returns(sector_data, period_options[selected_period])
        
        if sector_returns is not None:
            # Relative to SPY
            spy_return = sector_returns.get(BENCHMARK, 0)
            relative_returns = {k: v - spy_return for k, v in sector_returns.items() if k != BENCHMARK}
            
            # Display heatmap
            fig = _create_heatmap(relative_returns, f"Sector Returns vs SPY ({selected_period})")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top/Bottom performers
            col1, col2 = st.columns(2)
            
            sorted_sectors = sorted(relative_returns.items(), key=lambda x: x[1], reverse=True)
            
            with col1:
                st.markdown("**Leaders (vs SPY):**")
                for name, ret in sorted_sectors[:3]:
                    st.markdown(f"- {name}: {ret:+.1f}%")
            
            with col2:
                st.markdown("**Laggards (vs SPY):**")
                for name, ret in sorted_sectors[-3:]:
                    st.markdown(f"- {name}: {ret:+.1f}%")
            
            # Interpretation
            _interpret_sector_leadership(sorted_sectors)
    
    else:
        st.warning("Unable to fetch sector data")
    
    st.divider()
    
    # -----------------------------
    # Factor Heatmap
    # -----------------------------
    st.subheader("Factor Performance")
    
    with st.expander("Factor ETF Reference"):
        st.markdown("""
**iShares Factor ETFs**

Factor investing targets specific return drivers identified by academic research:

| Factor | Ticker | Strategy | Academic Basis |
|--------|--------|----------|----------------|
| Momentum | MTUM | Buy recent winners | Jegadeesh & Titman (1993) |
| Value | VLUE | Buy cheap stocks (P/B, P/E) | Fama & French (1992) |
| Quality | QUAL | Buy profitable, stable firms | Novy-Marx (2013) |
| Size | SIZE | Tilt toward small caps | Fama & French (1992) |
| Low Vol | USMV | Buy low-volatility stocks | Ang et al. (2006) |
| High Beta | SPHB | Buy high-beta stocks | Frazzini & Pedersen (2014) |

---

**Factor Combinations**

| Combination | Signal | Interpretation |
|-------------|--------|----------------|
| Momentum + High Beta leading | Maximum risk-on | Bull market, chase winners |
| Quality + Low Vol leading | Maximum risk-off | Bear market, protect capital |
| Value + Size leading | Reflation | Early cycle, mean reversion |
| Momentum + Quality leading | Late bull | Selective risk-taking |

---

**Factor Crowding Warning**

When one factor dominates for extended periods, crowding risk increases. 
Watch for sharp reversals when crowded factors unwind.
""")
    
    factor_data = _fetch_factor_data(yf_client, lookback_years)
    
    if factor_data is not None and not factor_data.empty:
        factor_returns = _calculate_returns(factor_data, period_options[selected_period])
        
        if factor_returns is not None:
            spy_return = factor_returns.get(BENCHMARK, 0)
            relative_returns = {k: v - spy_return for k, v in factor_returns.items() if k != BENCHMARK}
            
            fig = _create_heatmap(relative_returns, f"Factor Returns vs SPY ({selected_period})")
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("**Factor Leadership Interpretation:**")
            
            sorted_factors = sorted(relative_returns.items(), key=lambda x: x[1], reverse=True)
            leader = sorted_factors[0][0]
            laggard = sorted_factors[-1][0]
            
            interpretations = {
                "Momentum": "Trending regime - follow the leaders, momentum strategies working",
                "Value": "Mean reversion expected - early cycle, cheap stocks attracting flows",
                "Quality": "Flight to safety - late cycle, preference for stable earnings",
                "Size (Small)": "Risk appetite elevated - domestic growth optimism, small caps leading",
                "Low Vol": "Defensive positioning - risk-off sentiment, volatility aversion",
                "High Beta": "Maximum risk appetite - aggressive positioning, leveraged beta"
            }
            
            laggard_interpretations = {
                "Momentum": "Momentum unwinding - trend reversal possible",
                "Value": "Growth preferred - value trap concerns",
                "Quality": "Risk-on environment - quality premium not rewarded",
                "Size (Small)": "Large cap preference - flight to liquidity",
                "Low Vol": "Risk-on environment - defensive stocks ignored",
                "High Beta": "Risk-off environment - high beta punished"
            }
            
            st.info(f"**Leader: {leader}** - {interpretations.get(leader, '')}")
            st.warning(f"**Laggard: {laggard}** - {laggard_interpretations.get(laggard, '')}")
            
            # Risk appetite inference
            _infer_risk_appetite(sorted_factors)
    
    else:
        st.warning("Unable to fetch factor data")
    
    st.divider()
    
    # -----------------------------
    # Rolling Performance Chart
    # -----------------------------
    st.subheader("Rolling Relative Performance")
    
    with st.expander("Reading the Relative Performance Chart"):
        st.markdown("""
**How to Interpret**

This chart shows each sector's performance *relative to SPY*, indexed to 100 at the 
start of the lookback period.

| Line Position | Meaning |
|---------------|---------|
| Above 100 | Outperforming SPY over the period |
| Below 100 | Underperforming SPY over the period |
| Rising | Gaining relative strength |
| Falling | Losing relative strength |

---

**What to Watch For**

- **Trend Changes**: When a line crosses from below to above 100 (or vice versa)
- **Divergences**: When sector lines diverge sharply from each other
- **Convergence**: When all sectors move toward 100 (market becoming less rotational)
- **Leadership Shifts**: When a lagging sector becomes a leader (regime change signal)

---

**Research Application**

> "You want to look at cross-sectional (relative) fundamentals and cross-sectional 
> momentum (returns). Relative relationships send you a signal just as much as 
> absolute levels."
""")
    
    if sector_data is not None and not sector_data.empty:
        # Select sectors to display
        selected_sectors = st.multiselect(
            "Select sectors to compare",
            options=list(SECTOR_ETFS.keys()),
            default=["Technology", "Energy", "Financials", "Utilities"],
            help="Choose sectors to plot on the relative performance chart. "
                 "Default selection spans cyclicals and defensives for comparison."
        )
        
        if selected_sectors:
            fig = _plot_rolling_performance(sector_data, selected_sectors, SECTOR_ETFS, lookback_years)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one sector to display the chart")
    
    st.divider()
    
    # -----------------------------
    # Regime Inference Summary
    # -----------------------------
    st.subheader("Market Regime Inference")
    
    with st.expander("Methodology"):
        st.markdown("""
**Voting-Based Regime Classification**

We infer the market's implied regime by examining which sectors and factors are leading:

**Cyclical Score**: Average relative return of cyclical sectors (Tech, Discretionary, 
Industrials, Financials, Materials, Energy)

**Defensive Score**: Average relative return of defensive sectors (Utilities, Staples, Healthcare)

**Risk Appetite**: Based on factor leadership (Momentum, High Beta = risk-on; Quality, 
Low Vol = risk-off)

---

**Classification Logic**

| Cyclical vs Defensive | Factor Signal | Inferred Regime |
|-----------------------|---------------|-----------------|
| Cyclicals leading | Risk-on factors | Goldilocks/Reflation |
| Cyclicals leading | Risk-off factors | Transition (watch) |
| Defensives leading | Risk-off factors | Stagflation/Deflation |
| Defensives leading | Risk-on factors | Transition (watch) |

---

**Limitations**

- Based on price action, not fundamentals
- Lagging indicator (reflects what happened, not what will happen)
- Can be distorted by single-stock moves (e.g., AAPL in Tech)
- Should be cross-checked with macro indicators
""")
    
    if sector_data is not None and factor_data is not None:
        sector_returns = _calculate_returns(sector_data, period_options[selected_period])
        factor_returns = _calculate_returns(factor_data, period_options[selected_period])
        
        if sector_returns and factor_returns:
            spy_return = sector_returns.get(BENCHMARK, 0)
            
            # Calculate cyclical vs defensive
            cyclical_sectors = ["Technology", "Financials", "Consumer Disc.", "Industrials", "Materials", "Energy"]
            defensive_sectors = ["Utilities", "Consumer Stap.", "Healthcare"]
            
            cyclical_returns = [sector_returns.get(s, 0) - spy_return for s in cyclical_sectors if s in sector_returns]
            defensive_returns = [sector_returns.get(s, 0) - spy_return for s in defensive_sectors if s in sector_returns]
            
            cyclical_avg = np.mean(cyclical_returns) if cyclical_returns else 0
            defensive_avg = np.mean(defensive_returns) if defensive_returns else 0
            
            # Factor risk appetite
            spy_factor = factor_returns.get(BENCHMARK, 0)
            risk_on_factors = ["Momentum", "High Beta", "Size (Small)"]
            risk_off_factors = ["Quality", "Low Vol"]
            
            risk_on_avg = np.mean([factor_returns.get(f, 0) - spy_factor for f in risk_on_factors if f in factor_returns])
            risk_off_avg = np.mean([factor_returns.get(f, 0) - spy_factor for f in risk_off_factors if f in factor_returns])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sector_bias = "Cyclical" if cyclical_avg > defensive_avg else "Defensive"
                st.metric(
                    "Sector Bias",
                    sector_bias,
                    delta=f"{cyclical_avg - defensive_avg:+.1f}%",
                    help="Cyclical sectors avg relative return minus Defensive sectors avg. "
                         "Positive = cyclicals leading. Negative = defensives leading."
                )
            
            with col2:
                factor_bias = "Risk-On" if risk_on_avg > risk_off_avg else "Risk-Off"
                st.metric(
                    "Factor Bias",
                    factor_bias,
                    delta=f"{risk_on_avg - risk_off_avg:+.1f}%",
                    help="Risk-on factors avg relative return minus Risk-off factors avg. "
                         "Positive = risk-on. Negative = risk-off."
                )
            
            with col3:
                # Infer regime
                if cyclical_avg > 0 and risk_on_avg > 0:
                    inferred = "Risk-On"
                    color = "normal"
                elif defensive_avg > 0 and risk_off_avg > 0:
                    inferred = "Risk-Off"
                    color = "inverse"
                else:
                    inferred = "Mixed"
                    color = "off"
                
                st.metric(
                    "Market Posture",
                    inferred,
                    help="Combined inference from sector and factor leadership. "
                         "Risk-On = cyclicals + risk-on factors leading. "
                         "Risk-Off = defensives + risk-off factors leading."
                )
            
            # Interpretation
            if inferred == "Risk-On":
                st.success(
                    "Market positioning is risk-on: Cyclical sectors and risk-seeking factors leading. "
                    "Consistent with Goldilocks or Reflation regime expectations."
                )
            elif inferred == "Risk-Off":
                st.warning(
                    "Market positioning is risk-off: Defensive sectors and safety factors leading. "
                    "Consistent with Stagflation or Deflation regime expectations."
                )
            else:
                st.info(
                    "Market positioning is mixed: Sector and factor signals diverging. "
                    "May indicate regime transition or rotational market."
                )


def _interpret_sector_leadership(sorted_sectors: list):
    """Provide interpretation of sector leadership"""
    
    leaders = [s[0] for s in sorted_sectors[:3]]
    laggards = [s[0] for s in sorted_sectors[-3:]]
    
    cyclicals = {"Technology", "Financials", "Consumer Disc.", "Industrials", "Materials", "Energy"}
    defensives = {"Utilities", "Consumer Stap.", "Healthcare"}
    
    cyclical_leaders = sum(1 for s in leaders if s in cyclicals)
    defensive_leaders = sum(1 for s in leaders if s in defensives)
    
    st.markdown("**Sector Rotation Signal:**")
    
    if cyclical_leaders >= 2:
        st.success(
            f"Cyclical leadership ({cyclical_leaders}/3 top sectors are cyclical). "
            "Market favoring growth-sensitive sectors - consistent with risk-on positioning."
        )
    elif defensive_leaders >= 2:
        st.warning(
            f"Defensive leadership ({defensive_leaders}/3 top sectors are defensive). "
            "Market favoring safety - consistent with risk-off or late-cycle positioning."
        )
    else:
        st.info(
            "Mixed sector leadership - no clear cyclical/defensive tilt. "
            "Market may be in transition or rotating within sectors."
        )


def _infer_risk_appetite(sorted_factors: list):
    """Infer risk appetite from factor leadership"""
    
    risk_on = {"Momentum", "High Beta", "Size (Small)"}
    risk_off = {"Quality", "Low Vol"}
    
    top_3 = [f[0] for f in sorted_factors[:3]]
    
    risk_on_count = sum(1 for f in top_3 if f in risk_on)
    risk_off_count = sum(1 for f in top_3 if f in risk_off)
    
    st.markdown("**Risk Appetite Assessment:**")
    
    if risk_on_count >= 2:
        st.success(
            f"Risk-on factor regime ({risk_on_count}/3 top factors are risk-seeking). "
            "High conviction for beta exposure."
        )
    elif risk_off_count >= 2:
        st.warning(
            f"Risk-off factor regime ({risk_off_count}/3 top factors are defensive). "
            "Market favoring capital preservation over return maximization."
        )
    else:
        st.info(
            "Mixed factor leadership - unclear risk appetite signal. "
            "Consider other indicators for positioning guidance."
        )


def _fetch_sector_data(yf_client: YFinanceClient, lookback_years: int) -> Optional[pd.DataFrame]:
    """Fetch sector ETF data"""
    tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
    
    try:
        data = {}
        for ticker in tickers:
            df = yf_client.get_price_history(ticker, years=lookback_years)
            if df is not None and not df.empty:
                data[ticker] = df["Close"]
        
        if data:
            return pd.DataFrame(data)
        return None
    except Exception as e:
        return None


def _fetch_factor_data(yf_client: YFinanceClient, lookback_years: int) -> Optional[pd.DataFrame]:
    """Fetch factor ETF data"""
    tickers = list(FACTOR_ETFS.values()) + [BENCHMARK]
    
    try:
        data = {}
        for ticker in tickers:
            df = yf_client.get_price_history(ticker, years=lookback_years)
            if df is not None and not df.empty:
                data[ticker] = df["Close"]
        
        if data:
            return pd.DataFrame(data)
        return None
    except Exception as e:
        return None


def _calculate_returns(df: pd.DataFrame, period) -> Optional[Dict]:
    """Calculate returns for given period"""
    if df.empty:
        return None
    
    returns = {}
    
    if period == "YTD":
        # Get first trading day of year
        current_year = df.index.max().year
        ytd_data = df[df.index.year == current_year]
        if len(ytd_data) < 2:
            return None
        
        for col in df.columns:
            start_val = ytd_data[col].iloc[0]
            end_val = ytd_data[col].iloc[-1]
            if start_val > 0:
                returns[col] = (end_val / start_val - 1) * 100
    else:
        # Period is number of trading days
        if len(df) < period:
            period = len(df) - 1
        
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > period:
                start_val = series.iloc[-period-1]
                end_val = series.iloc[-1]
                if start_val > 0:
                    returns[col] = (end_val / start_val - 1) * 100
    
    # Map ticker to name
    ticker_to_name = {v: k for k, v in {**SECTOR_ETFS, **FACTOR_ETFS, BENCHMARK: BENCHMARK}.items()}
    named_returns = {ticker_to_name.get(k, k): v for k, v in returns.items()}
    
    return named_returns


def _create_heatmap(returns: Dict, title: str) -> go.Figure:
    """Create a horizontal bar heatmap"""
    
    sorted_items = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    # Color scale: red for negative, green for positive
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition='auto'
    ))
    
    fig.add_vline(x=0, line=dict(color="white", width=1))
    
    fig.update_layout(
        title=title,
        xaxis_title="Relative Return (%)",
        height=max(300, len(names) * 35),
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def _plot_rolling_performance(
    df: pd.DataFrame,
    sectors: list,
    mapping: Dict,
    lookback_years: int
) -> Optional[go.Figure]:
    """Plot rolling relative performance"""
    
    fig = go.Figure()
    
    spy = df[BENCHMARK]
    
    for sector in sectors:
        ticker = mapping.get(sector)
        if ticker and ticker in df.columns:
            # Calculate relative performance (indexed to 100)
            sector_series = df[ticker]
            relative = (sector_series / spy) * 100
            relative = relative / relative.iloc[0] * 100  # Rebase to 100
            
            fig.add_trace(go.Scatter(
                x=relative.index,
                y=relative.values,
                mode='lines',
                name=sector
            ))
    
    fig.add_hline(y=100, line=dict(color="gray", width=1, dash="dash"),
                  annotation_text="SPY (benchmark)", annotation_position="bottom right")
    
    fig.update_layout(
        title="Relative Performance vs SPY (Indexed to 100)",
        yaxis_title="Relative Performance",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig