# data/fred.py
"""
FRED data client with enhanced error handling and caching.
1. Align the dates 
2. Making sure we use the last ending period for the value
3. Calculate:
Percent (pct): Used for indices and stock prices. It calculates the fractional change using .pct_change() and multiplies by 100 to convert to a percentage.
Basis Points (bps): Used for interest rates or bond yields that are already expressed as percentages. It calculates the raw difference using .diff() and multiplies by 100 to express the change in basis points.
Level: Used for raw values (e.g., economic levels). It calculates the simple absolute difference between periods. 

4. Find regime scoring using a rolling Z score, window of 50 period for monhtly / weekly data, a 252 period of daily data 
- Normalization to use to get the rolling mean and standard deviation -> Afterwards Z score formula is applied 
- Replaces a standard deviation of - with NaN to prevent calculation errors in flat data 

5.Calculates an Expanding Percentile (df["Pctl"]) to show how the current YoY value ranks against every previous historical reading:
- Expanding Window: Unlike the Z-score, this uses an "expanding" window that grows with each new data point.
- Ranking: It uses .rank(pct=True) to convert the current value into a percentile between 0 and 1 (where 1.0 is the all-time high). 
- This is for the heatmap display 

"""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from fredapi import Fred
from pandas.tseries.offsets import MonthEnd, QuarterEnd

import config

logger = logging.getLogger(__name__)

# -----------------------------
# Data container
# -----------------------------
@dataclass
class SeriesFrame:
    """
    Container for a single time series
    Different data types require different change calculations:
    pct: For indices (CPI, INDPRO) → Use percent change: (new-old)/old × 100
    bps: For rates (10Y yield, unemployment) → Use basis point diff: (new-old) × 100
    level: For levels (Payrolls) → Use raw difference: new-old
    """
    df: pd.DataFrame          # index = datetime, columns = ["Value"]
    name: str                 # i.e "Industrial Production"
    source: str               # "FRED"
    freq: str                 # "D","W","M","Q"
    unit: str                 # "index","rate","level","price"
    chg_unit: str             # "pct","bps","level"


# -----------------------------
# Transform functions
# -----------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is datetime and sorted"""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    
    # Assumption 1: Remove timezone info to avoid alignment issues
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    
    out = out.sort_index()
    return out


def _shift_period_end(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Shift monthly/quarterly macro dates to period end.
    Assumption 2: Aligns macro data to month-end market data.
    """
    out = df.copy()
    if freq == "M":
        out.index = pd.to_datetime(out.index) + MonthEnd(0)
    elif freq == "Q":
        out.index = pd.to_datetime(out.index) + QuarterEnd(0)
    return out


def _infer_periods(freq: str) -> Dict[str, int]:
    """Infer periods for different frequencies"""
    if freq == "M":
        return {"m1": 1, "m3": 3, "y1": 12}
    if freq == "Q":
        return {"m1": 1, "m3": 2, "y1": 4}
    if freq == "W":
        return {"m1": 4, "m3": 13, "y1": 52}
    return {"m1": 21, "m3": 63, "y1": 252}  # daily


def compute_transforms(sf: SeriesFrame) -> pd.DataFrame:
    """
    Computes standardized financial transformations for time-series analysis.

    Returns a DataFrame with the following columns:
    - Value: The raw input data.
    - Chg_1 / Chg_3: 1 and 3-period momentum. Uses pct_change for prices/indices 
      and diff * 100 for rates (basis points).
    - YoY: Year-over-year change, calculated based on the inferred frequency.
    - Z: Rolling Z-score of YoY (60-period for M/W, 252 for daily). Measures 
      how many standard deviations the current YoY is from its recent mean.
    - Pctl: Historical percentile of YoY using an expanding window. Ranks the 
      current YoY against all available history (0.0 to 1.0).

    Args:
        sf: A SeriesFrame object containing data, frequency, and change unit type.
    
    | Column | What It Is | How It's Used |
    |--------|------------|---------------|
    | Value | Raw value | Display |
    | Chg_1 | 1-period change | MoM momentum |
    | Chg_3 | 3-period change | Quarterly momentum |
    | YoY | Year-over-year change | Trend identification |
    | Z | Rolling z-score on YoY | Regime scoring |
    | Pctl | Historical percentile | Heatmap coloring |
    """
    df = sf.df.copy()
    df.columns = ["Value"]
    df = _ensure_datetime_index(df)

    p = _infer_periods(sf.freq)

    # Compute changes based on unit type
    if sf.chg_unit == "pct":
        # Percent changes for indices/prices
        df["Chg_1"] = df["Value"].pct_change(p["m1"]) * 100.0
        df["Chg_3"] = df["Value"].pct_change(p["m3"]) * 100.0
        df["YoY"]   = df["Value"].pct_change(p["y1"]) * 100.0
    elif sf.chg_unit == "bps":
        # Basis point changes for rates (value already in %)
        df["Chg_1"] = df["Value"].diff(p["m1"]) * 100.0
        df["Chg_3"] = df["Value"].diff(p["m3"]) * 100.0
        df["YoY"]   = df["Value"].diff(p["y1"]) * 100.0
    else:  # "level"
        # Raw changes for levels
        df["Chg_1"] = df["Value"].diff(p["m1"])
        df["Chg_3"] = df["Value"].diff(p["m3"])
        df["YoY"]   = df["Value"].diff(p["y1"])

    # Rolling Z-score on YoY
    if sf.freq in ("M", "W"):
        roll = df["YoY"].rolling(60, min_periods=24)
    else:
        roll = df["YoY"].rolling(252, min_periods=60)

    mu = roll.mean()
    sd = roll.std(ddof=0).replace(0, np.nan)
    df["Z"] = (df["YoY"] - mu) / sd

    # Historical percentile on YoY tells you how the current Year-over-Year (YoY) value 
    # compares to all previous values in the dataset Unlike a "rolling" window that only looks at the last \(N\) months, 
    # an expanding window starts at the beginning of your data and grows by one row at a time.min_periods=24 
    # ensures the calculation only starts once you have at least 2 years (24 months) of data, 
    # providing a statistically valid baseline.
    df["Pctl"] = df["YoY"].expanding(min_periods=24).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

    return df.dropna(how="all")


# -----------------------------
# FRED Client
# -----------------------------
class FREDClient:
    """
    A validated wrapper for the FRED API that fetches and standardizes economic data.

    Features:
    - Secure API key management via environment variables.
    - Automatic conversion of FRED series into structured DataFrames.
    - Period alignment (shifting dates to month/quarter-end) for data consistency.
    - Resilience: get_bundle allows multiple fetches to continue even if one series fails.
    - Metadata tagging: Returns SeriesFrame objects with pre-defined frequency and unit types.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise ValueError(
                "Missing FRED API key. Set FRED_API_KEY in environment or pass to constructor."
            )
        self.fred = Fred(api_key=key)
    
    def get_series(self, series_key: str) -> SeriesFrame:
        """
        Fetch a single FRED series.
        
        Args:
            series_key: Key from config.FRED_SERIES
            
        Returns:
            SeriesFrame with data and metadata
        """
        if series_key not in config.FRED_SERIES:
            raise ValueError(f"Unknown series key: {series_key}")
        
        spec = config.FRED_SERIES[series_key]
        logger.info(f"Fetching FRED: {spec.label} ({spec.id})")
        
        try:
            s = self.fred.get_series(spec.id)
        except Exception as e:
            logger.error(f"FRED API error for {spec.id}: {e}")
            raise
        
        if s is None or s.empty:
            raise ValueError(f"No data returned for {spec.id}")
        
        # Convert to DataFrame
        df = pd.DataFrame(s, columns=["Value"])
        df.index.name = "Date"
        df = _ensure_datetime_index(df).dropna()
        
        # CRITICAL: Shift to period end for monthly/quarterly
        df = _shift_period_end(df, spec.freq)
        
        # Data quality check
        if len(df) < 10:
            logger.warning(f"Only {len(df)} observations for {spec.id}")
        
        return SeriesFrame(
            df=df,
            name=spec.label,
            source="FRED",
            freq=spec.freq,
            unit=spec.unit,
            chg_unit=spec.chg_unit,
        )
    
    def get_bundle(self, series_keys: List[str]) -> Dict[str, SeriesFrame]:
        """
        Fetch multiple FRED series.
        
        Args:
            series_keys: List of keys from config.FRED_SERIES
            
        Returns:
            Dict mapping series_key -> SeriesFrame
        """
        out: Dict[str, SeriesFrame] = {}
        failed = []
        
        for key in series_keys:
            try:
                out[key] = self.get_series(key)
            except Exception as e:
                logger.warning(f"Failed to fetch {key}: {e}")
                failed.append(key)
        
        if failed:
            logger.warning(f"Failed series: {', '.join(failed)}")
        
        logger.info(f"Successfully fetched {len(out)}/{len(series_keys)} series")
        return out