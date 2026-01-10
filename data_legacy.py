# data.py
import os
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from pandas.tseries.offsets import MonthEnd, QuarterEnd

import config

logger = logging.getLogger("macro_terminal")
logger.setLevel(logging.INFO)

# -----------------------------
# Containers
# -----------------------------
@dataclass
class SeriesFrame:
    df: pd.DataFrame          # index = datetime
    name: str
    source: str               # "FRED" / "YF"
    freq: str                 # "D","W","M","Q"
    unit: str                 # "index","rate","level","price"
    chg_unit: str             # "pct","bps","level"


# -----------------------------
# Helpers
# -----------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out

def _shift_period_end(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Shift monthly/quarterly macro dates to period end to align with month-end markets."""
    out = df.copy()
    if freq == "M":
        out.index = pd.to_datetime(out.index) + MonthEnd(0)
    elif freq == "Q":
        out.index = pd.to_datetime(out.index) + QuarterEnd(0)
    return out

def _infer_periods(freq: str) -> Dict[str, int]:
    if freq == "M":
        return {"m1": 1, "m3": 3, "y1": 12}
    if freq == "Q":
        return {"m1": 1, "m3": 2, "y1": 4}
    if freq == "W":
        return {"m1": 4, "m3": 13, "y1": 52}
    return {"m1": 21, "m3": 63, "y1": 252}  # daily

def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    return df.resample("M").last().dropna()

def normalize_index(series: pd.Series, base: float = 100.0) -> pd.Series:
    s = series.dropna().copy()
    if s.empty:
        return s
    return (s / s.iloc[0]) * base

def compute_transforms(sf: SeriesFrame) -> pd.DataFrame:
    """
    Standardized table:
      - Value
      - Chg_1 / Chg_3 (pct for index/price; bps for rate; raw for level)
      - YoY (pct for index/price; bps for rate; raw for level)
      - Z (rolling z-score on YoY)
      - Pctl (historical percentile on YoY)
    """
    df = sf.df.copy()
    df.columns = ["Value"]
    df = _ensure_datetime_index(df)

    p = _infer_periods(sf.freq)

    if sf.chg_unit == "pct":
        df["Chg_1"] = df["Value"].pct_change(p["m1"]) * 100.0
        df["Chg_3"] = df["Value"].pct_change(p["m3"]) * 100.0
        df["YoY"]   = df["Value"].pct_change(p["y1"]) * 100.0
    elif sf.chg_unit == "bps":
        # Value is already in % terms (e.g., yield = 4.25)
        # Differences in % * 100 => bps
        df["Chg_1"] = df["Value"].diff(p["m1"]) * 100.0
        df["Chg_3"] = df["Value"].diff(p["m3"]) * 100.0
        df["YoY"]   = df["Value"].diff(p["y1"]) * 100.0
    else:  # "level"
        df["Chg_1"] = df["Value"].diff(p["m1"])
        df["Chg_3"] = df["Value"].diff(p["m3"])
        df["YoY"]   = df["Value"].diff(p["y1"])

    # Rolling Z on YoY (use longer window for daily series)
    if sf.freq in ("M", "W"):
        roll = df["YoY"].rolling(60, min_periods=24)
    else:
        roll = df["YoY"].rolling(252, min_periods=60)

    mu = roll.mean()
    sd = roll.std(ddof=0).replace(0, np.nan)
    df["Z"] = (df["YoY"] - mu) / sd

    df["Pctl"] = df["YoY"].expanding(min_periods=24).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

    return df.dropna(how="all")

def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Beta(y~x) via cov/var."""
    y, x = y.align(x, join="inner")
    if len(y) < window:
        return pd.Series(index=y.index, dtype=float)
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


# -----------------------------
# Data client
# -----------------------------
class DataClient:
    def __init__(self, fred_api_key: Optional[str] = None):
        key = fred_api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise ValueError(
                "Missing FRED API key. Put FRED_API_KEY in .streamlit/secrets.toml "
                "or export env var FRED_API_KEY."
            )
        self.fred = Fred(api_key=key)

    def get_fred(self, series_key: str) -> SeriesFrame:
        spec = config.FRED_SERIES[series_key]
        logger.info(f"FRED: fetching {spec.label} ({spec.id})")
        s = self.fred.get_series(spec.id)
        df = pd.DataFrame(s, columns=["Value"])
        df.index.name = "Date"
        df = _ensure_datetime_index(df).dropna()

        # Crucial: align monthly/quarterly macro to period end
        df = _shift_period_end(df, spec.freq)

        return SeriesFrame(
            df=df,
            name=spec.label,
            source="FRED",
            freq=spec.freq,
            unit=spec.unit,
            chg_unit=spec.chg_unit,
        )

    def get_fred_bundle(self, series_keys: Iterable[str]) -> Dict[str, SeriesFrame]:
        out: Dict[str, SeriesFrame] = {}
        for key in series_keys:
            try:
                out[key] = self.get_fred(key)
            except Exception as e:
                logger.exception(f"Failed to fetch FRED series {key}: {e}")
        return out

    def get_yf_bundle(self, asset_keys: Iterable[str], start: Optional[str] = None) -> Dict[str, SeriesFrame]:
        keys = list(asset_keys)
        tickers = [config.YF_ASSETS[k].ticker for k in keys]
        ticker_to_key = {config.YF_ASSETS[k].ticker: k for k in keys}

        logger.info(f"YF: downloading {len(tickers)} tickers")
        raw = yf.download(
            tickers=tickers,
            start=start,
            interval="1d",
            auto_adjust=True,
            threads=False,
            progress=False,
        )

        # Select close prices robustly
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close" in raw.columns.get_level_values(0)):
                px = raw["Close"].copy()
            elif ("Adj Close" in raw.columns.get_level_values(0)):
                px = raw["Adj Close"].copy()
            else:
                px = raw.xs(raw.columns.levels[0][0], level=0, axis=1).copy()
        else:
            # Single ticker returns columns like Open/High/Low/Close/Volume
            if "Close" in raw.columns:
                px = raw["Close"].to_frame()
                px.columns = [tickers[0]]
            elif "Adj Close" in raw.columns:
                px = raw["Adj Close"].to_frame()
                px.columns = [tickers[0]]
            else:
                # fallback: first numeric column
                col = [c for c in raw.columns if c.lower() != "volume"][0]
                px = raw[col].to_frame()
                px.columns = [tickers[0]]

        px = _ensure_datetime_index(px).dropna(how="all").ffill()

        out: Dict[str, SeriesFrame] = {}
        for tkr in px.columns:
            k = ticker_to_key.get(tkr, tkr)
            df = pd.DataFrame(px[tkr]).rename(columns={tkr: "Value"}).dropna()
            out[k] = SeriesFrame(
                df=df,
                name=config.YF_ASSETS[k].label if k in config.YF_ASSETS else tkr,
                source="YF",
                freq="D",
                unit="price",
                chg_unit="pct",
            )
        return out