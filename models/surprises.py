# models/surprises.py
"""
Expectations vs Actual framework.

Research quote:
"Part of connecting economic data to markets is looking at how data and earnings 
come up above or below expectations."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SurpriseMetrics:
    """Surprise statistics for a given series"""
    series_key: str
    actual: float
    expected: float
    surprise: float
    surprise_pct: float
    historical_percentile: float
    timestamp: pd.Timestamp


class SurpriseEngine:
    """
    Generate expectations and measure surprises.
    
    Since we don't have Bloomberg consensus, we use time-series models
    to create "implied expectations" based on recent trends.
    """
    
    def __init__(self, method: str = "rolling_avg"):
        self.method = method
    
    def _rolling_expectation(
        self, 
        series: pd.Series, 
        window: int = 3,
        seasonal: bool = True
    ) -> pd.Series:
        """
        Rolling average forecast on YoY changes (not levels).
        This prevents index drift from creating spurious surprises.
        """
        # Convert to YoY changes first
        if len(series) >= 13:
            yoy_change = series.pct_change(12) * 100  # YoY % change
        else:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        if seasonal and len(yoy_change) >= 24:
            expectations = []
            for i in range(len(yoy_change)):
                if i < 24:  # Need 2 years for seasonal
                    expectations.append(np.nan)
                else:
                    month = yoy_change.index[i].month
                    # Look at same month YoY changes (last 5 years)
                    past_same_month = yoy_change.iloc[:i][
                        yoy_change.iloc[:i].index.month == month
                    ].tail(5)
                    
                    if len(past_same_month) >= 2:
                        expectations.append(past_same_month.mean())
                    else:
                        expectations.append(yoy_change.iloc[i-window:i].mean())
            
            return pd.Series(expectations, index=yoy_change.index)
        else:
            return yoy_change.rolling(window, min_periods=1).mean().shift(1)
    
    def compute_surprise(
        self,
        series: pd.Series,
        series_key: str,
        window: int = 3,
        seasonal: bool = True
    ) -> Tuple[pd.Series, Optional[SurpriseMetrics]]:
        """
        Compute surprise on YoY changes, not levels.
        """
        if len(series) < 13 + window:
            return pd.Series(dtype=float), None
        
        # Work with YoY changes
        yoy = series.pct_change(12) * 100
        
        expected = self._rolling_expectation(series, window, seasonal)
        surprise_raw = yoy - expected
        
        # Use expanding std for more stable normalization
        surprise_std = surprise_raw.expanding(min_periods=24).std()
        surprises = surprise_raw / surprise_std.replace(0, np.nan)
        
        if surprises.dropna().empty:
            return surprises, None
        
        last_idx = surprises.dropna().index[-1]
        actual_yoy = yoy.loc[last_idx]
        exp = expected.loc[last_idx]
        surp = surprises.loc[last_idx]
        
        hist = surprises.dropna()
        if len(hist) > 24:
            pctl = (hist <= surp).sum() / len(hist)
        else:
            pctl = np.nan
        
        if exp != 0:
            surp_pct = (actual_yoy - exp) / abs(exp) * 100
        else:
            surp_pct = np.nan
        
        metrics = SurpriseMetrics(
            series_key=series_key,
            actual=float(actual_yoy),  # Now stores YoY change, not level
            expected=float(exp),
            surprise=float(surp),
            surprise_pct=surp_pct,
            historical_percentile=pctl,
            timestamp=last_idx
        )
        
        return surprises, metrics
    
    def build_aggregate_surprise_index(
        self,
        fred_bundle: Dict,
        growth_keys: List[str] = None,
        inflation_keys: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Build our own economic surprise index from FRED data.
        
        Aggregates standardized surprises across growth and inflation indicators.
        Similar methodology to Citi Economic Surprise Index.
        
        Returns:
            - DataFrame with aggregate surprise index
            - Dict with component details
        """
        if growth_keys is None:
            growth_keys = ["INDPRO", "RSAFS", "PAYEMS", "UNRATE"]
        if inflation_keys is None:
            inflation_keys = ["CPIAUCSL", "CPILFESL", "PCEPILFE"]
        
        all_surprises = {}
        component_details = {}
        
        # Process growth indicators
        for key in growth_keys:
            if key not in fred_bundle:
                continue
            
            sf = fred_bundle[key]
            if not hasattr(sf, 'df') or sf.df.empty:
                continue
            
            series = sf.df["Value"].dropna()
            if len(series) < 24:
                continue
            
            # For unemployment, invert (lower than expected = positive surprise)
            if key == "UNRATE":
                series = -series
            
            surprises, metrics = self.compute_surprise(series, key)
            
            if not surprises.dropna().empty:
                all_surprises[key] = surprises
                if metrics:
                    component_details[key] = {
                        "category": "Growth",
                        "latest_surprise": metrics.surprise,
                        "percentile": metrics.historical_percentile
                    }
        
        # Process inflation indicators
        for key in inflation_keys:
            if key not in fred_bundle:
                continue
            
            sf = fred_bundle[key]
            if not hasattr(sf, 'df') or sf.df.empty:
                continue
            
            series = sf.df["Value"].dropna()
            if len(series) < 24:
                continue
            
            # For inflation, invert (lower than expected = positive surprise for markets)
            series_inverted = -series
            
            surprises, metrics = self.compute_surprise(series_inverted, key)
            
            if not surprises.dropna().empty:
                all_surprises[key] = surprises
                if metrics:
                    component_details[key] = {
                        "category": "Inflation",
                        "latest_surprise": metrics.surprise,
                        "percentile": metrics.historical_percentile
                    }
        
        if not all_surprises:
            return pd.DataFrame(), {}
        
        # Combine into single index
        df = pd.DataFrame(all_surprises)
        
        # Resample to common frequency (monthly) and forward fill
        df = df.resample("M").last().ffill()
        
        # Aggregate: equal-weighted mean
        df["Aggregate"] = df.mean(axis=1, skipna=True)
        
        # Add rolling metrics
        if len(df) >= 12:
            df["Aggregate_3M"] = df["Aggregate"].rolling(3).mean()
            df["Aggregate_12M"] = df["Aggregate"].rolling(12).mean()
        
        return df, component_details


# Helper functions for decomposition
def real_rate_decomposition(
    nominal_10y: pd.Series,
    breakeven_10y: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decompose nominal yield into real rate and inflation expectations.
    """
    real = nominal_10y - breakeven_10y
    dnominal = nominal_10y.diff(63)
    dbreakeven = breakeven_10y.diff(63)
    
    return real, dnominal, dbreakeven