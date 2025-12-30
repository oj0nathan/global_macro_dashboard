# app.py
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
from data import (
    DataClient, compute_transforms, resample_to_month_end,
    normalize_index, rolling_beta
)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Top-Down Macro Strategy", layout="wide")

st.title("Top-Down Macro Strategy")
st.caption("Macro dashboard for regime identification and cross-asset transmission. Visual layer is not a trading system.")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Controls")
    market_start_date = st.date_input("Market start date", value=dt.date(2020, 12, 30))
    lookback_years = st.slider("Macro lookback (years)", min_value=5, max_value=80, value=10, step=1)

    normalize_overlays = st.checkbox("Normalize overlays (index to 100)", value=True)
    show_beta = st.checkbox("Show rolling beta (monthly)", value=True)
    beta_window_months = st.slider("Rolling beta window (months)", min_value=12, max_value=60, value=24, step=1)

    st.divider()
    st.subheader("Data")
    st.write(
        "This dashboard uses FRED for macro/rates and yfinance for ETFs/commodities/FX. "
        "For production, replace yfinance with a paid feed where required."
    )

# -----------------------------
# Caching
# -----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def load_bundles(start_date_str: str):
    fred_key = st.secrets.get("FRED_API_KEY", None)
    client = DataClient(fred_api_key=fred_key)

    fred_bundle = client.get_fred_bundle(config.DEFAULT_FRED_IDS)
    yf_bundle = client.get_yf_bundle(config.DEFAULT_ASSET_KEYS, start=start_date_str)

    # Precompute transforms for scoreboard
    macro_rows = []
    for k, sf in fred_bundle.items():
        tf = compute_transforms(sf)
        if tf.empty:
            continue
        last = tf.dropna().iloc[-1]
        macro_rows.append({
            "key": k,
            "Series": sf.name,
            "Category": config.FRED_SERIES[k].category,
            "Freq": sf.freq,
            "Last": float(last.get("Value", np.nan)),
            "Chg_1": float(last.get("Chg_1", np.nan)),
            "Chg_3": float(last.get("Chg_3", np.nan)),
            "YoY": float(last.get("YoY", np.nan)),
            "Z": float(last.get("Z", np.nan)),
            "Pctl": float(last.get("Pctl", np.nan)),
            "Unit(level)": sf.unit,
            "Unit(chg/yoy)": sf.chg_unit,
            "Last Date": last.name.date() if hasattr(last.name, "date") else last.name,
        })

    scoreboard = pd.DataFrame(macro_rows)
    return fred_bundle, yf_bundle, scoreboard

# -----------------------------
# Plot helpers
# -----------------------------
def _date_range_from_lookback(end: pd.Timestamp, years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = end - pd.DateOffset(years=years)
    return start, end

def add_recession_shading(fig: go.Figure, usrec: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    """
    Add NBER recession shading WITHOUT dragging the axis back to the 1800s.
    We only add shapes within [start, end] and force x-axis range later.
    """
    if usrec is None or usrec.empty:
        return fig

    s = usrec.copy()
    s = s[(s.index >= start) & (s.index <= end)]
    if s.empty:
        return fig

    in_rec = False
    rec_start = None
    for t, v in s.items():
        if (v == 1) and (not in_rec):
            in_rec = True
            rec_start = t
        if (v == 0) and in_rec:
            in_rec = False
            rec_end = t
            fig.add_vrect(x0=rec_start, x1=rec_end, fillcolor="rgba(200,200,200,0.15)", line_width=0)

    if in_rec and rec_start is not None:
        fig.add_vrect(x0=rec_start, x1=end, fillcolor="rgba(200,200,200,0.15)", line_width=0)

    return fig

def line_fig(series: pd.Series, title: str, usrec: pd.Series | None, lookback_years: int):
    series = series.dropna()
    if series.empty:
        return None

    end = series.index.max()
    start, end = _date_range_from_lookback(end, lookback_years)
    series = series[(series.index >= start) & (series.index <= end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=title))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(range=[start, end])  # CRITICAL: prevent shading from expanding axis

    if usrec is not None:
        fig = add_recession_shading(fig, usrec, start, end)

    return fig

def overlay_fig(left: pd.Series, right: pd.Series, left_name: str, right_name: str,
                title: str, usrec: pd.Series | None, lookback_years: int, normalize: bool):
    left, right = left.align(right, join="inner")
    left, right = left.dropna(), right.dropna()
    if left.empty or right.empty:
        return None

    end = min(left.index.max(), right.index.max())
    start, end = _date_range_from_lookback(end, lookback_years)
    left = left[(left.index >= start) & (left.index <= end)]
    right = right[(right.index >= start) & (right.index <= end)]
    left, right = left.align(right, join="inner")
    if left.empty or right.empty:
        return None

    if normalize:
        left_plot = normalize_index(left)
        right_plot = normalize_index(right)
        y1_title = f"{left_name} (indexed)"
        y2_title = f"{right_name} (indexed)"
    else:
        left_plot = left
        right_plot = right
        y1_title = left_name
        y2_title = right_name

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=left_plot.index, y=left_plot.values, mode="lines", name=left_name, yaxis="y1"))
    fig.add_trace(go.Scatter(x=right_plot.index, y=right_plot.values, mode="lines", name=right_name, yaxis="y2"))

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title=y1_title),
        yaxis2=dict(title=y2_title, overlaying="y", side="right"),
    )
    fig.update_xaxes(range=[start, end])  # CRITICAL

    if usrec is not None:
        fig = add_recession_shading(fig, usrec, start, end)

    return fig

def infer_gip_regime(g_yoy: float, g_mom3: float, pi_yoy: float, pi_mom3: float) -> str:
    # Acceleration-based quadrant (classic GIP framing)
    g_acc = "improving" if g_mom3 > 0 else "worsening"
    pi_acc = "rising" if pi_mom3 > 0 else "falling"

    if (g_mom3 > 0) and (pi_mom3 < 0):
        return "Goldilocks: Growth improving, inflation falling"
    if (g_mom3 > 0) and (pi_mom3 > 0):
        return "Reflation: Growth improving, inflation rising"
    if (g_mom3 < 0) and (pi_mom3 > 0):
        return "Stagflation: Growth worsening, inflation rising"
    return "Deflation: Growth worsening, inflation falling"

# -----------------------------
# Load
# -----------------------------
fred_bundle, yf_bundle, scoreboard = load_bundles(str(market_start_date))

# Pull recession series for shading
usrec = None
if "USREC" in fred_bundle:
    usrec = fred_bundle["USREC"].df["Value"].astype(float)

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["GIP Regime", "Sector Transmission", "Rates Drivers", "Commodities", "Scoreboard"])

# ========== TAB 1: GIP ==========
with tabs[0]:
    col1, col2 = st.columns(2)

    # Growth = INDPRO YoY
    indpro = fred_bundle.get("INDPRO")
    cpi = fred_bundle.get("CPIAUCSL")
    nfci = fred_bundle.get("NFCI")

    if indpro and cpi:
        ind_tf = compute_transforms(indpro)
        cpi_tf = compute_transforms(cpi)

        g_last = ind_tf.dropna().iloc[-1]
        pi_last = cpi_tf.dropna().iloc[-1]

        g_yoy = float(g_last["YoY"])
        g_mom3 = float(g_last["Chg_3"])  # 3m pct change in YoY proxy (your display convention)
        pi_yoy = float(pi_last["YoY"])
        pi_mom3 = float(pi_last["Chg_3"])

        with col1:
            st.subheader("Growth (Industrial Production)")
            st.metric("Industrial Production YoY", f"{g_yoy:.2f}%", delta=f"{g_mom3:+.2f} pp (3m)")
            fig = line_fig(ind_tf["YoY"], "Industrial Production (YoY)", usrec, lookback_years)
            if fig: st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Inflation (CPI)")
            st.metric("CPI YoY (NSA)", f"{pi_yoy:.2f}%", delta=f"{pi_mom3:+.2f} pp (3m)")
            fig = line_fig(cpi_tf["YoY"], "CPI (YoY, NSA)", usrec, lookback_years)
            if fig: st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Current Regime")
        st.write(infer_gip_regime(g_yoy, g_mom3, pi_yoy, pi_mom3))

    # Liquidity / NFCI
    if nfci:
        st.subheader("Liquidity / Financial Conditions (baseline)")
        nfci_tf = compute_transforms(nfci)
        last = nfci_tf.dropna().iloc[-1]
        st.metric("NFCI (higher = tighter)", f"{float(last['Value']):.2f}", delta=f"{float(last['Chg_3']):+.2f} (approx 3m)")
        fig = line_fig(nfci.df["Value"], "Chicago Fed NFCI", usrec, lookback_years)
        if fig: st.plotly_chart(fig, use_container_width=True)

# ========== TAB 2: Transmission ==========
with tabs[1]:
    st.subheader("Macro to Micro Transmission")
    st.caption("Overlays are aligned to month-end to avoid false precision. For inference, use relative performance and rolling beta.")

    # Precompute month-end market
    market_me = {}
    for k, sf in yf_bundle.items():
        me = resample_to_month_end(sf.df)["Value"]
        market_me[k] = me

    # Macro YoY series (month-end shifted already in data.py)
    macro_yoy = {}
    for mk in ["RSAFS", "INDPRO", "UNRATE"]:
        sf = fred_bundle.get(mk)
        if sf is None:
            continue
        tf = compute_transforms(sf)
        if "YoY" in tf:
            macro_yoy[mk] = tf["YoY"].dropna()

    for macro_key, num_key, den_key, title in config.TRANSMISSION_PAIRS:
        st.markdown(f"### {title}")

        m = macro_yoy.get(macro_key)
        num = market_me.get(num_key)
        den = market_me.get(den_key)

        if m is None or num is None or den is None:
            st.info("Missing inputs for this plot.")
            continue

        rel = (num / den).dropna()
        rel = rel.replace([np.inf, -np.inf], np.nan).dropna()

        # Align
        m2, rel2 = m.align(rel, join="inner")
        if len(m2) < 24:
            st.info("Insufficient overlapping data for this plot (need ~24+ months after alignment).")
            continue

        # Plot overlay (macro YoY vs relative equity proxy)
        fig = overlay_fig(
            left=m2,
            right=rel2,
            left_name=f"{config.FRED_SERIES[macro_key].label} YoY",
            right_name=f"{config.YF_ASSETS[num_key].label} / {config.YF_ASSETS[den_key].label}",
            title=title,
            usrec=usrec,
            lookback_years=lookback_years,
            normalize=normalize_overlays
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        if show_beta:
            # rolling beta of relative proxy returns vs macro (both monthly)
            rel_ret = rel2.pct_change().dropna()
            mac = m2.loc[rel_ret.index].dropna()
            rel_ret, mac = rel_ret.align(mac, join="inner")
            if len(rel_ret) >= beta_window_months:
                b = rolling_beta(rel_ret, mac, beta_window_months)
                b_fig = line_fig(b, f"Rolling beta (monthly, {beta_window_months}m)", usrec, lookback_years)
                if b_fig:
                    st.plotly_chart(b_fig, use_container_width=True)

# ========== TAB 3: Rates ==========
with tabs[2]:
    st.subheader("Cost of Capital, Curve, and Credit")

    # Month-end aligned overlays
    real_yield = fred_bundle.get("DFII10")
    qqq = yf_bundle.get("QQQ")
    if real_yield and qqq:
        ry_me = resample_to_month_end(real_yield.df)["Value"]
        qqq_me = resample_to_month_end(qqq.df)["Value"]
        fig = overlay_fig(
            ry_me, qqq_me,
            "10Y Real Yield (level)", "QQQ (level)",
            "Real Yield (10Y, DFII10) vs Nasdaq 100 (QQQ) — month-end aligned",
            usrec, lookback_years, normalize_overlays
        )
        if fig: st.plotly_chart(fig, use_container_width=True)

    # Yield curve spread
    dgs10 = fred_bundle.get("DGS10")
    dgs2 = fred_bundle.get("DGS2")
    if dgs10 and dgs2:
        spread = (dgs10.df["Value"] - dgs2.df["Value"]).dropna()
        fig = line_fig(spread, "UST Curve (10Y - 2Y)", usrec, lookback_years)
        if fig: st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        hy = fred_bundle.get("BAMLH0A0HYM2")
        if hy:
            fig = line_fig(hy.df["Value"], "High Yield OAS (level)", usrec, lookback_years)
            if fig: st.plotly_chart(fig, use_container_width=True)

    with c2:
        dxy = yf_bundle.get("DXY")
        if dxy:
            dxy_me = resample_to_month_end(dxy.df)["Value"]
            fig = line_fig(dxy_me, "DXY (month-end)", usrec, lookback_years)
            if fig: st.plotly_chart(fig, use_container_width=True)

    with c3:
        vix = yf_bundle.get("VIX")
        if vix:
            vix_me = resample_to_month_end(vix.df)["Value"]
            fig = line_fig(vix_me, "VIX (month-end)", usrec, lookback_years)
            if fig: st.plotly_chart(fig, use_container_width=True)

# ========== TAB 4: Commodities ==========
with tabs[3]:
    st.subheader("Commodity Proxies")

    real_yield = fred_bundle.get("DFII10")
    gld = yf_bundle.get("GLD")
    if real_yield and gld:
        ry_me = resample_to_month_end(real_yield.df)["Value"]
        gld_me = resample_to_month_end(gld.df)["Value"]
        fig = overlay_fig(
            ry_me, gld_me,
            "10Y Real Yield (level)", "Gold (GLD)",
            "Real Yield (DFII10) vs Gold — month-end aligned",
            usrec, lookback_years, normalize_overlays
        )
        if fig: st.plotly_chart(fig, use_container_width=True)

    wti = yf_bundle.get("WTI")
    xle = yf_bundle.get("XLE")
    spy = yf_bundle.get("SPY")
    if wti and xle and spy:
        wti_me = resample_to_month_end(wti.df)["Value"]
        xle_rel = (resample_to_month_end(xle.df)["Value"] / resample_to_month_end(spy.df)["Value"]).dropna()
        fig = overlay_fig(
            wti_me, xle_rel,
            "WTI (level)", "XLE/SPY (relative)",
            "WTI Crude vs Energy relative (XLE/SPY) — month-end aligned",
            usrec, lookback_years, normalize_overlays
        )
        if fig: st.plotly_chart(fig, use_container_width=True)

# ========== TAB 5: Scoreboard ==========
with tabs[4]:
    st.subheader("Macro Scoreboard")
    st.caption("Bias-reduction layer: review levels, momentum, and historical context before forming a narrative.")

    if scoreboard.empty:
        st.info("Scoreboard is empty (data fetch/transforms returned no rows).")
    else:
        # Filters
        cats = sorted(scoreboard["Category"].unique().tolist())
        selected = st.multiselect("Filter categories", options=cats, default=cats)

        view = scoreboard[scoreboard["Category"].isin(selected)].copy()

        # Round / format
        def _fmt_unit(u: str) -> str:
            return {"pct": "%", "bps": "bps", "level": "level"}.get(u, u)

        view["Unit(chg/yoy)"] = view["Unit(chg/yoy)"].apply(_fmt_unit)

        # Sensible rounding
        for col in ["Last", "Chg_1", "Chg_3", "YoY", "Z", "Pctl"]:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce")

        st.dataframe(
            view[[
                "Series", "Category", "Freq", "Last",
                "Chg_1", "Chg_3", "YoY", "Z", "Pctl",
                "Unit(level)", "Unit(chg/yoy)", "Last Date"
            ]],
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Series definitions / notes"):
        st.write("Rates/Credit changes are in **bps**. Index/price changes are in **%**. Monthly macro dates are shifted to **month-end** for alignment.")
