# views/rates_tab.py
"""
Rates & Credit Dashboard Tab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from plotly.subplots import make_subplots
from enum import Enum
from typing import Dict, Optional

from data.fred import SeriesFrame, compute_transforms
from data.yfinance import resample_to_month_end
from models.regime import RegimeState

# -----------------------------
# Curve Regime Definitions
# -----------------------------
class CurveRegime(Enum):
    BULL_STEEPENER = "Bull Steepener"
    BEAR_STEEPENER = "Bear Steepener"
    BULL_FLATTENER = "Bull Flattener"
    BEAR_FLATTENER = "Bear Flattener"
    STEEPENER_TWIST = "Steepener Twist"
    FLATTENER_TWIST = "Flattener Twist"
    NEUTRAL = "Neutral"


REGIME_COLORS = {
    CurveRegime.BULL_STEEPENER: "#00CED1",
    CurveRegime.BEAR_STEEPENER: "#FFA500",
    CurveRegime.BULL_FLATTENER: "#9370DB",
    CurveRegime.BEAR_FLATTENER: "#DC143C",
    CurveRegime.STEEPENER_TWIST: "#32CD32",
    CurveRegime.FLATTENER_TWIST: "#FF69B4",
    CurveRegime.NEUTRAL: "#808080",
}

REGIME_IMPLICATIONS = {
    CurveRegime.BULL_STEEPENER: {"equity": "Bullish", "description": "Front-end falling (Fed easing)", "sectors": "Cyclicals, Financials, Small Caps"},
    CurveRegime.BEAR_STEEPENER: {"equity": "Mixed", "description": "Long-end rising (inflation/term premium)", "sectors": "Real Assets, Energy, Value"},
    CurveRegime.BULL_FLATTENER: {"equity": "Defensive", "description": "Long-end falling (growth scare)", "sectors": "Defensives, Quality, Duration"},
    CurveRegime.BEAR_FLATTENER: {"equity": "Bearish", "description": "Front-end rising (Fed hawkish)", "sectors": "Cash, Short Duration"},
    CurveRegime.STEEPENER_TWIST: {"equity": "Neutral", "description": "Both ends moving, spread widening", "sectors": "Depends on driver"},
    CurveRegime.FLATTENER_TWIST: {"equity": "Cautious", "description": "Both ends moving, spread narrowing", "sectors": "Quality, Low Beta"},
    CurveRegime.NEUTRAL: {"equity": "Neutral", "description": "No significant curve movement", "sectors": "Market weight"},
}


def _classify_curve_regime(dgs2_chg: float, dgs10_chg: float, spread_chg: float, threshold: float = 10.0) -> CurveRegime:
    if abs(spread_chg) < threshold:
        return CurveRegime.NEUTRAL
    if spread_chg > threshold:  # Steepening
        if dgs2_chg < -threshold and dgs10_chg > -threshold:
            return CurveRegime.BULL_STEEPENER
        elif dgs10_chg > threshold and dgs2_chg < threshold:
            return CurveRegime.BEAR_STEEPENER
        return CurveRegime.STEEPENER_TWIST
    if spread_chg < -threshold:  # Flattening
        if dgs10_chg < -threshold and dgs2_chg > -threshold:
            return CurveRegime.BULL_FLATTENER
        elif dgs2_chg > threshold and dgs10_chg < threshold:
            return CurveRegime.BEAR_FLATTENER
        return CurveRegime.FLATTENER_TWIST
    return CurveRegime.NEUTRAL


def _compute_rolling_regime(dgs2: pd.Series, dgs10: pd.Series, lookback_days: int = 63) -> pd.DataFrame:
    df = pd.DataFrame({"DGS2": dgs2, "DGS10": dgs10}).dropna()
    df["Spread"] = df["DGS10"] - df["DGS2"]
    df["DGS2_Chg"] = df["DGS2"].diff(lookback_days) * 100
    df["DGS10_Chg"] = df["DGS10"].diff(lookback_days) * 100
    df["Spread_Chg"] = df["Spread"].diff(lookback_days) * 100
    
    regimes, colors = [], []
    for _, row in df.iterrows():
        if pd.isna(row["DGS2_Chg"]):
            regime = CurveRegime.NEUTRAL
        else:
            regime = _classify_curve_regime(row["DGS2_Chg"], row["DGS10_Chg"], row["Spread_Chg"])
        regimes.append(regime)
        colors.append(REGIME_COLORS[regime])
    
    df["Regime"] = regimes
    df["Regime_Name"] = [r.value for r in regimes]
    df["Regime_Color"] = colors
    return df

def render_rates_tab(
    fred_bundle: Dict[str, SeriesFrame],
    market_bundle: Dict[str, SeriesFrame],
    current_regime: RegimeState,
    usrec: Optional[pd.Series],
    lookback_years: int
):
    """Render Rates & Credit tab"""
    
    st.header("Rates, Curve & Credit")
    
    # Educational overview
    with st.expander("Understanding Rates & Credit"):
        st.markdown("""
**The Core Framework**

Fixed income markets transmit monetary policy to the real economy. This tab tracks 
three interconnected signals:

1. **Yield Curve Shape**: Reflects growth and policy expectations
2. **Real Rate Decomposition**: Separates growth vs inflation drivers
3. **Credit Spreads**: Measures risk appetite and financial stress

---

**Why Rates Matter for Equities**

> "Real rates pushed higher - ES struggled and gold came under pressure. 
> When real rates fell, liquidity spilled into equities, gold and tech."

The transmission mechanism:

| Rate Move | Mechanism | Equity Impact |
|-----------|-----------|---------------|
| Real rates up | Discount rates rise, multiples compress | Negative (especially growth/tech) |
| Real rates down | Discount rates fall, multiples expand | Positive (especially growth/tech) |
| Breakevens up | Inflation pressure, margin squeeze | Mixed (favors real assets) |
| Breakevens down | Disinflation, demand concerns | Mixed (watch for deflation) |

---

**Yield Curve as Macro Barometer**

> "The yield curve is a direct reflection of the macro regime. Growth, inflation, 
> policy, liquidity, and credit all get priced into yields."

The curve shape tells you about market expectations:

| Curve Shape | Signal | Historical Implication |
|-------------|--------|------------------------|
| Steep (positive) | Growth optimism, Fed behind | Early cycle, risk-on |
| Flat | Uncertainty, transition | Late cycle, caution |
| Inverted (negative) | Recession expectations | Historically precedes recessions |

---

**Credit Spreads as Fear Gauge**

Credit spreads measure the premium investors demand for taking default risk:

| Spread Level | Signal | Positioning |
|--------------|--------|-------------|
| Tight (<300bps HY) | Complacency, risk-on | Watch for reversal |
| Normal (300-500bps) | Balanced | Neutral |
| Wide (>500bps) | Stress, risk-off | Opportunity or crisis |

Spread direction matters as much as level - rapidly widening spreads signal 
deteriorating conditions even from tight levels.
""")
    
    # -----------------------------
    # Curve Overview
    # -----------------------------
    st.subheader("Yield Curve")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Compute curve regime consistently with the expander
        if "DGS2" in fred_bundle and "DGS10" in fred_bundle:
            regime_df_top = _compute_rolling_regime(
                fred_bundle["DGS2"].df["Value"].dropna(),
                fred_bundle["DGS10"].df["Value"].dropna(),
                lookback_days=63
            )
            curve_regime_display = regime_df_top["Regime"].iloc[-1].value
        else:
            curve_regime_display = "N/A"
        
        st.metric(
            "Curve Regime",
            curve_regime_display,
            help="Classification based on 3-month (63-day) changes in 2Y and 10Y yields. "
                 "Bull/Bear refers to bond prices (Bull = yields falling). "
                 "Steep/Flat refers to spread direction."
        )
    
    with col2:
        if "DGS2" in fred_bundle and "DGS10" in fred_bundle:
            dgs2 = fred_bundle["DGS2"].df["Value"].iloc[-1]
            dgs10 = fred_bundle["DGS10"].df["Value"].iloc[-1]
            spread = (dgs10 - dgs2) * 100  # Convert to bps
            
            spread_status = "Inverted" if spread < 0 else ("Flat" if spread < 25 else "Normal")
            st.metric(
                "2s10s Spread",
                f"{spread:.0f}bps",
                help=f"10Y minus 2Y Treasury yield. Currently: {spread_status}. "
                     f"Negative spread historically signals recession risk within 12-18 months."
            )
    
    with col3:
        if "DGS10" in fred_bundle:
            tf = compute_transforms(fred_bundle["DGS10"])
            if not tf.empty:
                last = tf.dropna().iloc[-1]
                st.metric(
                    "10Y Yield",
                    f"{last['Value']:.2f}%",
                    delta=f"{last['Chg_3']:+.0f}bps (3m)",
                    help="10-Year Treasury yield - the benchmark risk-free rate. "
                         "Rising yields = tightening conditions. "
                         "Falling yields = easing conditions or growth concerns."
                )

    # Curve chart
    if "DGS2" in fred_bundle and "DGS10" in fred_bundle:
        dgs2 = fred_bundle["DGS2"].df["Value"]
        dgs10 = fred_bundle["DGS10"].df["Value"]
        spread = (dgs10 - dgs2).dropna() * 100  # Convert to bps
        
        fig = _line_chart(spread, "2s10s Spread (bps)", usrec, lookback_years)
        if fig:
            # Add zero line for inversion reference
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Inversion", annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)

        # Curve Regime Analysis
        st.subheader("Curve Regime Analysis")
        
        with st.expander("Curve Regime Methodology"):
            st.markdown("""
**Classification Logic**

The curve regime is determined by examining 3-month (63-day) changes in both 
the 2-Year and 10-Year Treasury yields:

| Regime | 2Y Move | 10Y Move | Spread | Driver |
|--------|---------|----------|--------|--------|
| Bull Steepener | Down >10bps | Stable/Down less | Widens | Fed easing expectations |
| Bear Steepener | Stable | Up >10bps | Widens | Inflation/term premium |
| Bull Flattener | Stable | Down >10bps | Narrows | Growth scare, flight to safety |
| Bear Flattener | Up >10bps | Stable/Up less | Narrows | Fed hawkish, hiking cycle |
| Twist | Both move significantly | - | Varies | Complex dynamics |
| Neutral | Both <10bps change | - | <10bps | Range-bound |

---

**Equity Implications**

| Regime | Equity Stance | Rationale |
|--------|---------------|-----------|
| Bull Steepener | Bullish | Easier policy, early cycle recovery |
| Bear Steepener | Mixed | Growth strong but inflation risk |
| Bull Flattener | Defensive | Growth concerns, prefer quality |
| Bear Flattener | Bearish | Policy tightening, late cycle |

---

**Research Context**

> "Bear steepening driven by rising term premium, rather than genuine growth 
> optimism, is where risk begins to build. The long end sells off even as the 
> front end eases, duration risk overtakes growth risk."

Key insight: Not all steepening is equity-friendly. The *driver* matters more 
than the shape itself.
""")
        
        st.markdown("""
**From the research:** "The yield curve is a direct reflection of the macro regime. 
Bear flattening driven by rising term premium is where risk begins to build."
""")
        
        # Compute rolling regime
        regime_df = _compute_rolling_regime(dgs2.dropna(), dgs10.dropna(), lookback_days=63)
        
        # Filter to lookback
        end_date = regime_df.index.max()
        start_date = end_date - pd.DateOffset(years=lookback_years)
        regime_df = regime_df[regime_df.index >= start_date]
        
        # Current regime
        current_curve_regime = regime_df["Regime"].iloc[-1]
        impl = REGIME_IMPLICATIONS[current_curve_regime]
        color = REGIME_COLORS[current_curve_regime]
        
        # Display current regime
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: {color}30;
                    border-left: 4px solid {color};
                    padding: 12px;
                    border-radius: 4px;
                ">
                    <h4 style="margin: 0; color: {color};">{current_curve_regime.value}</h4>
                    <p style="margin: 5px 0 0 0; font-size: 11px;">Current Regime</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.metric(
                "Equity Stance", 
                impl["equity"],
                help="Historical equity market behavior during this curve regime"
            )
        
        with col3:
            st.info(f"**{impl['description']}**  \nFavored: {impl['sectors']}")
        
        # Get SPY for overlay
        spy_data = None
        if "SPY" in market_bundle:
            spy = market_bundle["SPY"].df["Value"].dropna()
            spy_data = spy[(spy.index >= start_date) & (spy.index <= end_date)]
        
        # Dual-panel chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=("S&P 500" if spy_data is not None else "", "2s10s Spread by Regime")
        )
        
        # Top: SPY
        if spy_data is not None and not spy_data.empty:
            fig.add_trace(
                go.Scatter(x=spy_data.index, y=spy_data.values, mode='lines',
                            name='SPY', line=dict(color='white', width=1.5), showlegend=False),
                row=1, col=1
            )
        
        # Bottom: Spread bars by regime
        for regime in CurveRegime:
            mask = regime_df["Regime"] == regime
            if mask.any():
                subset = regime_df[mask]
                fig.add_trace(
                    go.Bar(x=subset.index, y=subset["Spread"] * 100, name=regime.value,
                            marker_color=REGIME_COLORS[regime], opacity=0.85),
                    row=2, col=1
                )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Recession shading
        if usrec is not None and not usrec.empty:
            rec = usrec[(usrec.index >= start_date) & (usrec.index <= end_date)]
            in_rec = False
            rec_start = None
            for t, v in rec.items():
                if v == 1 and not in_rec:
                    in_rec = True
                    rec_start = t
                if v == 0 and in_rec:
                    in_rec = False
                    for r in [1, 2]:
                        fig.add_vrect(x0=rec_start, x1=t, fillcolor="rgba(128,128,128,0.25)",
                                        line_width=0, layer="below", row=r, col=1)
            if in_rec and rec_start:
                for r in [1, 2]:
                    fig.add_vrect(x0=rec_start, x1=end_date, fillcolor="rgba(128,128,128,0.25)",
                                    line_width=0, layer="below", row=r, col=1)
        
        fig.update_layout(
            height=500,
            barmode='overlay',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            hovermode="x unified",
            margin=dict(l=50, r=50, t=60, b=40)
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime statistics
        with st.expander("Regime Statistics"):
            regime_counts = regime_df["Regime_Name"].value_counts()
            total_days = len(regime_df)
            
            stats_data = []
            for regime in CurveRegime:
                name = regime.value
                if name in regime_counts.index:
                    days = regime_counts[name]
                    pct = days / total_days * 100
                    impl = REGIME_IMPLICATIONS[regime]
                    stats_data.append({
                        "Regime": name, 
                        "Days": days, 
                        "% Time": f"{pct:.1f}%",
                        "Equity Stance": impl["equity"]
                    })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            st.caption(
                f"Statistics based on {lookback_years}-year lookback period. "
                f"Regime classification uses 63-day (3-month) rolling changes with 10bps threshold."
            )
    
    st.divider()
    
    # -----------------------------
    # Real Rates
    # -----------------------------
    st.subheader("Real Rates & Inflation Expectations")
    
    with st.expander("Understanding Real Rates"):
        st.markdown("""
**The Fisher Equation**

`Nominal Yield = Real Yield + Inflation Expectations`

Or equivalently:

`Real Yield = Nominal Yield - Breakeven Inflation`

---

**Why Real Rates Matter**

Real yields represent the true cost of capital after adjusting for inflation. 
They are the single most important variable for:

- **Equity Valuations**: Higher real rates = higher discount rates = lower multiples
- **Growth vs Value**: Growth stocks have longer duration, more sensitive to real rates
- **Gold**: Negative real rates historically bullish for gold (opportunity cost argument)
- **Tech/Momentum**: Extremely sensitive to real rate moves

---

**Interpreting Changes**

| Scenario | Nominal | Real | Breakeven | Interpretation |
|----------|---------|------|-----------|----------------|
| Tightening | Up | Up | Flat | Fed hawkish, risk-off |
| Reflation | Up | Flat | Up | Growth + inflation, favor real assets |
| Goldilocks Easing | Down | Down | Down | Disinflation + rate cuts, risk-on |
| Growth Scare | Down | Flat/Up | Down | Deflation fear, flight to safety |

---

**Data Sources**

- **10Y Nominal (DGS10)**: Constant maturity Treasury yield
- **10Y Real (DFII10)**: 10-Year TIPS yield (inflation-protected)
- **10Y Breakeven (T10YIE)**: Nominal minus TIPS = market inflation expectation
""")

    if "DFII10" in fred_bundle and "DGS10" in fred_bundle and "T10YIE" in fred_bundle:
        real = fred_bundle["DFII10"].df["Value"]
        nominal = fred_bundle["DGS10"].df["Value"]
        breakeven = fred_bundle["T10YIE"].df["Value"]
        
        # Align using DataFrame approach
        df_aligned = pd.concat(
            [real, nominal, breakeven], 
            axis=1, 
            keys=['real', 'nominal', 'breakeven']
        )
        df_aligned = df_aligned.dropna()
        
        # Extract aligned series
        real = df_aligned['real']
        nominal = df_aligned['nominal']
        breakeven = df_aligned['breakeven']
        
        # Check if we have enough data
        if len(real) < 63:
            st.info("Insufficient data for real rate analysis (need 63+ days)")
        else:
            # Recent changes (3 months = 63 trading days)
            real_chg = real.diff(63).iloc[-1] * 100  # Convert to bps
            be_chg = breakeven.diff(63).iloc[-1] * 100
            nom_chg = nominal.diff(63).iloc[-1] * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "10Y Real Yield",
                    f"{real.iloc[-1]:.2f}%",
                    delta=f"{real_chg:+.0f}bps (3m)",
                    delta_color="inverse",
                    help="10-Year TIPS yield. Rising real yields = tightening financial conditions. "
                         "Key driver of tech/growth relative performance. Inverted delta: rising = red."
                )

            with col2:
                st.metric(
                    "10Y Nominal",
                    f"{nominal.iloc[-1]:.2f}%",
                    delta=f"{nom_chg:+.0f}bps (3m)",
                    help="10-Year Treasury yield. The headline rate most commonly quoted. "
                         "Decompose into real + breakeven to understand the driver."
                )

            with col3:
                st.metric(
                    "10Y Breakeven",
                    f"{breakeven.iloc[-1]:.2f}%",
                    delta=f"{be_chg:+.0f}bps (3m)",
                    delta_color="inverse",
                    help="Market-implied 10-year inflation expectation. "
                         "Rising breakevens = inflation pressure. Inverted delta: rising = red."
                )

            # Interpretation
            if real_chg > 5 and be_chg < 0:  # 5 bps threshold
                st.warning(
                    "Real rates rising while inflation expectations fall = TIGHTENING liquidity. "
                    "Headwind for risk assets, especially growth/tech."
                )
            elif real_chg < -5 and be_chg < -5:
                st.success(
                    "Real rates falling with disinflation = EASING conditions (Goldilocks). "
                    "Supportive for risk assets."
                )
            elif real_chg > 5 and be_chg > 5:
                st.info(
                    "Both real rates and breakevens rising = REFLATION. "
                    "Favor real assets, commodities, value over growth."
                )
            elif real_chg < -5 and be_chg > 5:
                st.info(
                    "Real rates falling but breakevens rising = MONETARY ACCOMMODATION with inflation. "
                    "Mixed signal - watch Fed response."
                )

            # Chart: Real vs nominal
            fig = _overlay_dual(
                real, nominal,
                "10Y Real Yield", "10Y Nominal Yield",
                "Real vs Nominal Yields",
                usrec, lookback_years
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()


    # -----------------------------
    # Real Rate Decomposition (Research-driven)
    # -----------------------------
    st.subheader("Nominal Yield Decomposition")
    
    with st.expander("Why This Matters"):
        st.markdown("""
**The Key Insight**

> "Real rates pushed higher - ES struggled... When real rates fell, 
> liquidity spilled into equities, gold and tech"

The nominal yield can rise for two very different reasons:

**Scenario 1: Real Rates Rising (inflation expectations stable)**
- Tightening financial conditions
- Headwind for risk assets, especially growth/tech
- Duration is painful
- Fed is restrictive or market expects tightening

**Scenario 2: Inflation Expectations Rising (real rates stable)**
- Reflation trade
- Favor real assets, commodities, TIPS
- Different sector implications
- Growth may be strong enough to offset

**Critical Point:** The SAME move in nominal yields has opposite implications 
depending on the decomposition. A 50bps rise driven by real rates is very 
different from a 50bps rise driven by breakevens.

---

**Attribution Analysis**

We decompose the 3-month change in nominal yields into:

`Nominal Change = Real Rate Change + Breakeven Change`

If real rates account for >60% of the nominal move, the move is "real-rate driven."
If breakevens account for >60%, the move is "inflation-expectations driven."
""")
    
    # Get the data
    if "DGS10" in fred_bundle and "DFII10" in fred_bundle and "T10YIE" in fred_bundle:
        nominal = fred_bundle["DGS10"].df["Value"].dropna()
        real = fred_bundle["DFII10"].df["Value"].dropna()
        breakeven = fred_bundle["T10YIE"].df["Value"].dropna()
        
        # Align all series
        df_decomp = pd.DataFrame({
            "Nominal": nominal,
            "Real": real,
            "Breakeven": breakeven
        }).dropna()
        
        if not df_decomp.empty and len(df_decomp) > 63:
            # Current levels
            current_nominal = df_decomp["Nominal"].iloc[-1]
            current_real = df_decomp["Real"].iloc[-1]
            current_be = df_decomp["Breakeven"].iloc[-1]
            
            # 3-month changes
            chg_nominal = (df_decomp["Nominal"].iloc[-1] - df_decomp["Nominal"].iloc[-63]) * 100  # bps
            chg_real = (df_decomp["Real"].iloc[-1] - df_decomp["Real"].iloc[-63]) * 100
            chg_be = (df_decomp["Breakeven"].iloc[-1] - df_decomp["Breakeven"].iloc[-63]) * 100
            
            # Attribution
            st.markdown("**3-Month Change Attribution:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Nominal 10Y Change",
                    f"{chg_nominal:+.0f} bps",
                    help="Total change in 10Y Treasury yield over past 3 months"
                )
            
            with col2:
                if abs(chg_nominal) >= 5:
                    pct_real = chg_real/chg_nominal*100
                    delta_str = f"{pct_real:.0f}% of move"
                else:
                    delta_str = None
                
                st.metric(
                    "Real Rate Contribution",
                    f"{chg_real:+.0f} bps",
                    delta=delta_str,
                    help="Change in 10Y TIPS yield. Represents the 'true' cost of capital component."
                )
            
            with col3:
                if abs(chg_nominal) >= 5:
                    pct_be = chg_be/chg_nominal*100
                    delta_str = f"{pct_be:.0f}% of move"
                else:
                    delta_str = None
                
                st.metric(
                    "Inflation Exp. Contribution",
                    f"{chg_be:+.0f} bps",
                    delta=delta_str,
                    help="Change in 10Y breakeven inflation. Represents the inflation premium component."
                )
            
            # Interpretation
            st.markdown("**Interpretation:**")
            
            if abs(chg_nominal) < 10:
                st.info("Nominal yields relatively unchanged over past 3 months - no strong signal")
            elif chg_nominal > 0:
                if chg_real > chg_be:
                    st.error(
                        f"Nominal yields UP driven primarily by REAL RATES (+{chg_real:.0f}bps). "
                        "This is tightening financial conditions - headwind for risk assets, "
                        "especially long-duration equities (tech/growth)."
                    )
                else:
                    st.warning(
                        f"Nominal yields UP driven primarily by INFLATION EXPECTATIONS (+{chg_be:.0f}bps). "
                        "This is reflation - favor real assets, commodities, value over growth."
                    )
            else:  # chg_nominal < 0
                if abs(chg_real) > abs(chg_be):
                    st.success(
                        f"Nominal yields DOWN driven primarily by REAL RATES ({chg_real:.0f}bps). "
                        "This is easing financial conditions - supportive for risk assets."
                    )
                else:
                    st.warning(
                        f"Nominal yields DOWN driven primarily by INFLATION EXPECTATIONS ({chg_be:.0f}bps). "
                        "This could signal growth concerns - watch for deflation risk."
                    )
            
            # Decomposition chart
            st.markdown("**Historical Decomposition:**")
            
            # Filter to lookback period
            end = df_decomp.index.max()
            start = end - pd.DateOffset(years=lookback_years)
            plot_df = df_decomp[(df_decomp.index >= start)]
            
            fig = go.Figure()
            
            # Stacked area for decomposition
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df["Real"],
                name="Real Yield",
                fill='tozeroy',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(65, 105, 225, 0.5)'  # Royal blue
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df["Real"] + plot_df["Breakeven"],
                name="+ Breakeven = Nominal",
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 165, 0, 0.5)'  # Orange
            ))
            
            # Overlay nominal line for clarity
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df["Nominal"],
                name="Nominal (actual)",
                mode='lines',
                line=dict(color='white', width=2, dash='dot')
            ))
            
            # Add recession shading if available
            if usrec is not None and not usrec.empty:
                rec = usrec[(usrec.index >= start) & (usrec.index <= end)]
                in_rec = False
                rec_start = None
                
                for t, v in rec.items():
                    if v == 1 and not in_rec:
                        in_rec = True
                        rec_start = t
                    if v == 0 and in_rec:
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
                title="10Y Nominal Yield = Real Yield + Breakeven Inflation",
                yaxis_title="Yield (%)",
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Research context
            st.markdown("**Quick Reference:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
**Real Rate Driven Moves:**
- Rising = Tightening (risk-off)
- Falling = Easing (risk-on)
- Tech/Growth most sensitive
- Watch: DFII10 (10Y TIPS)
""")
            
            with col2:
                st.markdown("""
**Breakeven Driven Moves:**
- Rising = Inflation pressure
- Falling = Disinflation/deflation risk
- Commodities/Energy sensitive
- Watch: T10YIE, inflation swaps
""")
    
    st.divider()
    
    # -----------------------------
    # Credit
    # -----------------------------
    st.subheader("Credit Spreads")
    
    with st.expander("Understanding Credit Spreads"):
        st.markdown("""
**What Are Credit Spreads?**

Credit spreads measure the additional yield investors demand for holding 
corporate bonds instead of risk-free Treasuries. The spread compensates for:

- Default risk
- Liquidity risk
- Recovery uncertainty

---

**Option-Adjusted Spread (OAS)**

OAS adjusts the spread for embedded options (callable bonds). It provides 
a cleaner measure of pure credit risk premium.

---

**Spread Interpretation**

| HY OAS Level | Environment | Signal |
|--------------|-------------|--------|
| <300bps | Very tight | Risk-on, complacency |
| 300-400bps | Normal | Balanced conditions |
| 400-600bps | Elevated | Caution, stress emerging |
| >600bps | Wide | Crisis conditions, opportunity or risk |

---

**Spreads as Leading Indicator**

Credit markets often lead equity markets:
- Widening spreads before equity selloffs = warning signal
- Tight spreads during equity rally = confirmation
- Divergence (equities up, spreads widening) = caution

---

**IG vs HY Spreads**

| Type | Risk Profile | Typical Spread | Sensitivity |
|------|--------------|----------------|-------------|
| Investment Grade (IG) | Lower default risk | 80-150bps | Rates + Credit |
| High Yield (HY) | Higher default risk | 300-500bps | Mostly Credit |

HY spreads are a purer measure of risk appetite since they're less 
sensitive to duration/rates.
""")

    col1, col2 = st.columns(2)

    with col1:
        if "BAMLH0A0HYM2" in fred_bundle:
            hy = fred_bundle["BAMLH0A0HYM2"]
            hy_values = hy.df["Value"].dropna()
            
            if len(hy_values) >= 63:
                current_value = hy_values.iloc[-1]
                prev_value = hy_values.iloc[-63]
                change_3m_bps = (current_value - prev_value) * 100
                
                # Percentile calculation
                pctl = (hy_values <= current_value).sum() / len(hy_values) * 100
                
                st.metric(
                    "High Yield OAS",
                    f"{current_value * 100:.0f}bps",
                    delta=f"{change_3m_bps:+.0f}bps (3m)",
                    delta_color="inverse",
                    help=f"BofA High Yield Option-Adjusted Spread. "
                         f"Current percentile: {pctl:.0f}%. "
                         f"Widening spreads (red) = risk-off. Tightening (green) = risk-on."
                )
                
                fig = _line_chart(hy.df["Value"] * 100, "HY OAS (bps)", usrec, lookback_years)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "BAMLC0A0CM" in fred_bundle:
            ig = fred_bundle["BAMLC0A0CM"]
            ig_values = ig.df["Value"].dropna()
            
            if len(ig_values) >= 63:
                current_value = ig_values.iloc[-1]
                prev_value = ig_values.iloc[-63]
                change_3m_bps = (current_value - prev_value) * 100
                
                # Percentile calculation
                pctl = (ig_values <= current_value).sum() / len(ig_values) * 100
                
                st.metric(
                    "Investment Grade OAS",
                    f"{current_value * 100:.0f}bps",
                    delta=f"{change_3m_bps:+.0f}bps (3m)",
                    delta_color="inverse",
                    help=f"BofA Investment Grade Option-Adjusted Spread. "
                         f"Current percentile: {pctl:.0f}%. "
                         f"More sensitive to rates than HY."
                )
                
                fig = _line_chart(ig.df["Value"] * 100, "IG OAS (bps)", usrec, lookback_years)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # Credit interpretation
    if "BAMLH0A0HYM2" in fred_bundle:
        hy_current = fred_bundle["BAMLH0A0HYM2"].df["Value"].dropna().iloc[-1] * 100
        if hy_current < 300:
            st.info(
                f"HY spreads tight at {hy_current:.0f}bps - markets pricing minimal default risk. "
                "Watch for complacency."
            )
        elif hy_current > 500:
            st.warning(
                f"HY spreads elevated at {hy_current:.0f}bps - credit stress evident. "
                "Monitor for further widening or stabilization."
            )

    st.divider()
        
    # -----------------------------
    # FX & Vol
    # -----------------------------
    st.subheader("FX & Volatility")
    
    with st.expander("FX & Volatility Context"):
        st.markdown("""
**USD Index (UUP)**

The US Dollar is a key macro variable:

| USD Move | Typical Driver | Implication |
|----------|----------------|-------------|
| Strengthening | Risk-off, Fed hawkish, growth scare | Headwind for EM, commodities |
| Weakening | Risk-on, Fed dovish, global growth | Tailwind for EM, commodities |

---

**VIX (Volatility Index)**

The VIX measures implied volatility of S&P 500 options:

| VIX Level | Environment | Interpretation |
|-----------|-------------|----------------|
| <15 | Low vol | Complacency, potential for spike |
| 15-20 | Normal | Balanced conditions |
| 20-30 | Elevated | Uncertainty, hedging demand |
| >30 | High | Fear, potential capitulation |

---

**Cross-Asset Signals**

Watch for divergences:
- VIX low + USD strong = risk-off building quietly
- VIX low + USD weak = classic risk-on
- VIX high + USD strong = crisis mode
- VIX high + USD weak = unusual, policy intervention likely
""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "UUP" in market_bundle:
            uup = market_bundle["UUP"]
            uup_me = resample_to_month_end(uup.df)["Value"]
            
            if len(uup_me) >= 3:
                current = uup_me.iloc[-1]
                prev = uup_me.iloc[-3] if len(uup_me) >= 3 else uup_me.iloc[0]
                chg = (current / prev - 1) * 100
                
                st.metric(
                    "USD Index (UUP)",
                    f"{current:.2f}",
                    delta=f"{chg:+.1f}% (3m)",
                    help="Invesco DB US Dollar Index ETF. Tracks USD vs basket of major currencies."
                )
            
            fig = _line_chart(uup_me, "USD Index (UUP, month-end)", usrec, lookback_years)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "VIX" in market_bundle:
            vix = market_bundle["VIX"]
            vix_me = resample_to_month_end(vix.df)["Value"]
            
            if len(vix_me) >= 1:
                current = vix_me.iloc[-1]
                
                # VIX level interpretation
                if current < 15:
                    level_desc = "Low"
                elif current < 20:
                    level_desc = "Normal"
                elif current < 30:
                    level_desc = "Elevated"
                else:
                    level_desc = "High"
                
                st.metric(
                    "VIX",
                    f"{current:.1f}",
                    delta=level_desc,
                    help="CBOE Volatility Index. Measures S&P 500 implied volatility (30-day)."
                )
            
            fig = _line_chart(vix_me, "VIX (month-end)", usrec, lookback_years)
            if fig:
                # Add reference lines
                fig.add_hline(y=20, line_dash="dash", line_color="yellow", 
                             annotation_text="Elevated (20)")
                fig.add_hline(y=30, line_dash="dash", line_color="red",
                             annotation_text="High (30)")
                st.plotly_chart(fig, use_container_width=True)


def _line_chart(series, title, usrec, lookback_years):
    """Simple line chart with recession shading"""
    series = series.dropna()
    if series.empty:
        return None
    
    end = series.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    series = series[(series.index >= start) & (series.index <= end)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(width=2)))
    
    if usrec is not None:
        rec = usrec[(usrec.index >= start) & (usrec.index <= end)]
        in_rec = False
        rec_start = None
        for t, v in rec.items():
            if (v == 1) and (not in_rec):
                in_rec = True
                rec_start = t
            if (v == 0) and in_rec:
                in_rec = False
                fig.add_vrect(x0=rec_start, x1=t, fillcolor="rgba(200,200,200,0.2)", line_width=0, layer="below")
        if in_rec and rec_start:
            fig.add_vrect(x0=rec_start, x1=end, fillcolor="rgba(200,200,200,0.2)", line_width=0, layer="below")
    
    fig.update_layout(title=title, height=300, margin=dict(l=10,r=10,t=40,b=10), showlegend=False, hovermode="x unified")
    fig.update_xaxes(range=[start, end])
    return fig


def _overlay_dual(left, right, left_name, right_name, title, usrec, lookback_years):
    """Overlay chart"""
    end = min(left.index.max(), right.index.max())
    start = end - pd.DateOffset(years=lookback_years)
    left = left[(left.index >= start) & (left.index <= end)]
    right = right[(right.index >= start) & (right.index <= end)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=left.index, y=left.values, mode="lines", name=left_name, yaxis="y1", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=right.index, y=right.values, mode="lines", name=right_name, yaxis="y2", line=dict(color="coral")))
    
    fig.update_layout(
        title=title, 
        height=350,
        yaxis=dict(
            title=dict(text=left_name, font=dict(color="steelblue")),  
            tickfont=dict(color="steelblue")
        ),
        yaxis2=dict(
            title=dict(text=right_name, font=dict(color="coral")),  
            tickfont=dict(color="coral"),
            overlaying="y", 
            side="right"
        ),
        hovermode="x unified"
    )
    fig.update_xaxes(range=[start, end])
    return fig