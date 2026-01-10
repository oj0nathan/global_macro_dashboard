# regime.py
"""
Regime classification engine following Capital Flows methodology.

Key insight from research:
"Economic Mechanics/Structural Dynamics Set Probable Distribution/Skew for Expectations vs actual"

We need to classify regimes on MULTIPLE dimensions:
1. Growth acceleration (not just level)
2. Inflation momentum (not just level)  
3. Liquidity transmission (real rates + credit)
4. Curve regime (bull/bear steep/flat)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

from config import (
    GrowthRegime, InflationRegime, LiquidityRegime, MacroRegime,
    REGIME_GROWTH_KEYS, REGIME_INFLATION_KEYS, REGIME_LIQUIDITY_KEYS
)

@dataclass
class RegimeState:
    """Current regime classification"""
    macro: MacroRegime
    growth: GrowthRegime
    inflation: InflationRegime
    liquidity: LiquidityRegime
    
    growth_score: float      # -1 to 1 (composite z-score)
    inflation_score: float   # -1 to 1
    liquidity_score: float   # -1 to 1
    
    confidence: float        # 0 to 1 (based on data consistency)
    
    # Curve regime
    curve_2s10s: float
    curve_regime: str  # "bull_steep", "bear_steep", "bull_flat", "bear_flat"
    
    timestamp: pd.Timestamp

class RegimeEngine:
    """
    Composite scoring across multiple series.
    
    Philosophy from research:
    "Define everything into regimes to see the highest probability for returns. 
    Then you want to identify any discontinuity from past returns."
    """
    
    def __init__(self, lookback_months: int = 60):
        self.lookback = lookback_months
    
    def _compute_momentum_score(self, series: pd.Series, periods: list[int]) -> float:
        """
        Momentum score based on multiple horizons.
        Returns z-score normalized composite.
        
        Filter to post-2021 to avoid COVID distortion.
        """
        if len(series) < max(periods) + 24:
            return np.nan
        
        scores = []
        for p in periods:
            mom = series.pct_change(p).iloc[-1] * 100  # Convert to %
            
            # Z-score vs trailing distribution (post-COVID to avoid distortion)
            hist = series.pct_change(p).dropna()
            
            # Filter to post-June 2021 to avoid COVID spike/recovery skew
            hist_filtered = hist[hist.index >= "2021-06-01"]
            
            # Fallback if not enough post-COVID data
            if len(hist_filtered) < 12:
                hist_filtered = hist.tail(36)  # Last 3 years
            
            if len(hist_filtered) < 12:
                continue
            
            z = (mom - hist_filtered.mean()) / hist_filtered.std()
            scores.append(z)
        
        if not scores:
            return np.nan
        
        # Composite (equal weight) with clip to prevent extreme values
        return np.clip(np.mean(scores), -2, 2)
    
    def _growth_composite(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, GrowthRegime]:
        """
        Aggregate growth momentum across INDPRO, RSAFS, PAYEMS, UNRATE.
        
        Research note: "Growth and labour market is holding up (consumer spending held up well)"
        We need ACCELERATION not just levels.
        """
        scores = []
        weights = []
        
        for key in REGIME_GROWTH_KEYS:
            if key not in data:
                continue
            
            df = data[key]
            if "Value" not in df.columns or len(df) < 36:
                continue
            
            s = df["Value"].dropna()
            
            # For unemployment, invert (rising = bad)
            if key == "UNRATE":
                s = -s
            
            # 1m, 3m, 6m momentum
            score = self._compute_momentum_score(s, periods=[1, 3, 6])
            
            if not np.isnan(score):
                scores.append(score)
                weights.append(1.0)  # Could use regime_weight from config
        
        if not scores:
            return np.nan, GrowthRegime.STABLE
        
        composite = np.average(scores, weights=weights)
        
        # Classify
        if composite > 0.5:
            regime = GrowthRegime.ACCELERATING
        elif composite < -0.5:
            regime = GrowthRegime.DECELERATING
        else:
            regime = GrowthRegime.STABLE
        
        return composite, regime
    
    def _inflation_composite(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, InflationRegime]:
        """
        Inflation momentum: Core CPI, Core PCE, breakevens.
        
        Research: "Inflation tail risks resurface... inflation swaps drift higher"
        We care about DIRECTION more than level.
        """
        scores = []
        weights = []
        
        for key in REGIME_INFLATION_KEYS:
            if key not in data:
                continue
            
            df = data[key]
            if "Value" not in df.columns or len(df) < 36:
                continue
            
            s = df["Value"].dropna()
            
            # For breakevens (daily), use shorter periods
            if key in ["T5YIE", "T10YIE"]:
                score = self._compute_momentum_score(s, periods=[21, 63, 126])  # 1m, 3m, 6m in trading days
            else:
                score = self._compute_momentum_score(s, periods=[1, 3, 6])
            
            if not np.isnan(score):
                scores.append(score)
                # Weight Core PCE higher (Fed's preferred)
                w = 2.0 if key == "PCEPILFE" else 1.0
                weights.append(w)
        
        if not scores:
            return np.nan, InflationRegime.STABLE
        
        composite = np.average(scores, weights=weights)
        
        if composite > 0.5:
            regime = InflationRegime.RISING
        elif composite < -0.5:
            regime = InflationRegime.FALLING
        else:
            regime = InflationRegime.STABLE
        
        return composite, regime
    
    def _liquidity_composite(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, LiquidityRegime]:
        """
        Liquidity = f(real rates, credit spreads, financial conditions).
        
        Research: "Real rates pushed higher → ES struggled... When real rates fell, 
        liquidity spilled into equities, gold and tech"
        
        Components:
        - DFII10 (real yield): lower = easier (invert)
        - BAMLH0A0HYM2 (HY OAS): lower = easier (invert)
        - NFCI: lower = easier (invert)
        - 2Y yield change: falling = easier (invert)
        """
        scores = []
        weights = []
        
        # Real yields (invert: falling = easing)
        if "DFII10" in data:
            df = data["DFII10"]
            if "Value" in df.columns and len(df) > 63:
                s = -df["Value"].dropna()  # Negative so falling yields = positive score
                score = self._compute_momentum_score(s, periods=[21, 63])
                if not np.isnan(score):
                    scores.append(score)
                    weights.append(2.0)  # High weight per research
        
        # Credit spreads (invert: tightening = easing)
        if "BAMLH0A0HYM2" in data:
            df = data["BAMLH0A0HYM2"]
            if "Value" in df.columns and len(df) > 63:
                s = -df["Value"].dropna()
                score = self._compute_momentum_score(s, periods=[21, 63])
                if not np.isnan(score):
                    scores.append(score)
                    weights.append(1.5)
        
        # Financial conditions (invert: falling NFCI = easing)
        if "NFCI" in data:
            df = data["NFCI"]
            if "Value" in df.columns and len(df) > 12:
                s = -df["Value"].dropna()
                score = self._compute_momentum_score(s, periods=[4, 13])  # Weekly data
                if not np.isnan(score):
                    scores.append(score)
                    weights.append(1.5)
        
        # Front-end rates (invert: falling = easing)
        if "DGS2" in data:
            df = data["DGS2"]
            if "Value" in df.columns and len(df) > 63:
                s = -df["Value"].dropna()
                score = self._compute_momentum_score(s, periods=[21, 63])
                if not np.isnan(score):
                    scores.append(score)
                    weights.append(1.0)
        
        if not scores:
            return np.nan, LiquidityRegime.NEUTRAL
        
        composite = np.average(scores, weights=weights)
        
        if composite > 0.5:
            regime = LiquidityRegime.EASING
        elif composite < -0.5:
            regime = LiquidityRegime.TIGHTENING
        else:
            regime = LiquidityRegime.NEUTRAL
        
        return composite, regime
    
    def _curve_regime(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, str]:
        """
        Research: "The yield curve is a direct reflection of the macro regime we're in."
        
        Four regimes:
        - Bull steepening: front falls faster (easing + growth expectations)
        - Bear steepening: long rises faster (growth/inflation pressure)
        - Bull flattening: long falls faster (growth scare)
        - Bear flattening: front rises faster (Fed hawkish, inflation)
        """
        if "DGS2" not in data or "DGS10" not in data:
            return np.nan, "unknown"
        
        dgs2 = data["DGS2"]["Value"].dropna()
        dgs10 = data["DGS10"]["Value"].dropna()
        
        dgs2, dgs10 = dgs2.align(dgs10, join="inner")
        
        if len(dgs2) < 126:
            return np.nan, "unknown"
        
        spread = (dgs10 - dgs2).dropna()
        current_spread = spread.iloc[-1]
        
        # 3-month change in spread
        delta_spread = spread.diff(63).iloc[-1]
        
        # 3-month change in levels
        delta_2y = dgs2.diff(63).iloc[-1]
        delta_10y = dgs10.diff(63).iloc[-1]
        
        # Classify
        if delta_spread > 5:  # Steepening (>5bps over 3m)
            if delta_2y < 0:
                regime = "bull_steep"  # Front falling
            else:
                regime = "bear_steep"  # Long rising
        elif delta_spread < -5:  # Flattening
            if delta_10y < 0:
                regime = "bull_flat"  # Long falling (growth scare)
            else:
                regime = "bear_flat"  # Front rising (Fed hawkish)
        else:
            regime = "neutral"
        
        return current_spread, regime
    
    
    def classify(self, data: Dict[str, pd.DataFrame]) -> RegimeState:
        """
        Main entry point: classify current regime.
        
        Input: dict of {series_key: DataFrame with "Value" column}
        Output: RegimeState with all classifications
        """
        # Get component scores
        g_score, g_regime = self._growth_composite(data)
        i_score, i_regime = self._inflation_composite(data)
        l_score, l_regime = self._liquidity_composite(data)
        
        # Macro quadrant (GIP)
        if not np.isnan(g_score) and not np.isnan(i_score):
            if g_score > 0 and i_score < 0:
                macro = MacroRegime.GOLDILOCKS
            elif g_score > 0 and i_score > 0:
                macro = MacroRegime.REFLATION
            elif g_score < 0 and i_score > 0:
                macro = MacroRegime.STAGFLATION
            else:
                macro = MacroRegime.DEFLATION
        else:
            # Fallback if missing data
            macro = MacroRegime.DEFLATION  # Conservative default
        
        # Curve
        curve_spread, curve_regime = self._curve_regime(data)
        
        # Confidence: how many series contributed vs expected
        expected_growth = len(REGIME_GROWTH_KEYS)
        expected_inflation = len(REGIME_INFLATION_KEYS)
        expected_liquidity = len(REGIME_LIQUIDITY_KEYS)
        
        actual_growth = sum(1 for k in REGIME_GROWTH_KEYS if k in data)
        actual_inflation = sum(1 for k in REGIME_INFLATION_KEYS if k in data)
        actual_liquidity = sum(1 for k in REGIME_LIQUIDITY_KEYS if k in data)
        
        confidence = (actual_growth / expected_growth +
                     actual_inflation / expected_inflation +
                     actual_liquidity / expected_liquidity) / 3.0
        
        # Get latest timestamp from data
        timestamps = []
        for df in data.values():
            if not df.empty and hasattr(df.index, "max"):
                timestamps.append(df.index.max())
        
        ts = max(timestamps) if timestamps else pd.Timestamp.now()
        
        return RegimeState(
            macro=macro,
            growth=g_regime,
            inflation=i_regime,
            liquidity=l_regime,
            growth_score=g_score,
            inflation_score=i_score,
            liquidity_score=l_score,
            confidence=confidence,
            curve_2s10s=curve_spread,
            curve_regime=curve_regime,
            timestamp=ts
        )

def _classify_regime(growth: float, inflation: float) -> str:
    """Classify regime based on pulse values."""
    if growth > 0 and inflation <= 0:
        return "Goldilocks"
    elif growth > 0 and inflation > 0:
        return "Reflation"
    elif growth <= 0 and inflation > 0:
        return "Stagflation"
    else:
        return "Deflation"

def compute_macro_pulse(
    data: Dict[str, pd.DataFrame],
    lookback_years: int = 7,
    scale_factor: float = 50.0
) -> pd.DataFrame:
    """
    Compute historical Growth and Inflation pulse signals.
    
    Uses FIXED post-June 2021 baseline (same as RegimeEngine) for consistency.
    """
    
    # Fixed baseline start date (same as RegimeEngine)
    BASELINE_START = "2021-06-01"
    
    def _compute_indicator_pulse(
        series: pd.Series, 
        periods: List[int],
        invert: bool = False
    ) -> pd.Series:
        """
        Compute z-score normalized momentum for a single indicator.
        Uses fixed post-2021 baseline for consistency with RegimeEngine.
        """
        if len(series) < max(periods) + 24:
            return pd.Series(dtype=float)
        
        momentum_scores = []
        
        for p in periods:
            # Compute percent change
            mom = series.pct_change(p) * 100
            
            if invert:
                mom = -mom
            
            # Use FIXED post-2021 baseline (not rolling)
            # This matches RegimeEngine methodology
            baseline = mom[mom.index >= BASELINE_START]
            
            if len(baseline) < 12:
                continue
            
            baseline_mean = baseline.mean()
            baseline_std = baseline.std()
            
            if baseline_std == 0 or pd.isna(baseline_std):
                continue
            
            # Z-score against fixed baseline
            z = (mom - baseline_mean) / baseline_std
            z = z.clip(-3, 3)
            
            momentum_scores.append(z)
        
        if momentum_scores:
            combined = pd.concat(momentum_scores, axis=1).mean(axis=1)
            return combined
        
        return pd.Series(dtype=float)
    
    # ----- GROWTH PULSE -----
    growth_components = []
    growth_weights = []
    
    growth_config = [
        ("INDPRO", [1, 3, 6], False, 1.0),
        ("RSAFS", [1, 3, 6], False, 1.0),
        ("PAYEMS", [1, 3, 6], False, 1.0),
        ("UNRATE", [1, 3, 6], True, 1.0),
        ("AWHMAN", [1, 3, 6], False, 0.8),
    ]
    
    for key, periods, invert, weight in growth_config:
        if key not in data:
            continue
        
        df = data[key]
        if "Value" not in df.columns or len(df) < 48:
            continue
        
        series = df["Value"].dropna()
        
        # Resample to monthly if needed
        if len(series) > 500:
            series = series.resample("M").last()
        
        pulse = _compute_indicator_pulse(series, periods, invert)
        
        if not pulse.empty:
            growth_components.append(pulse * weight)
            growth_weights.append(weight)
    
    # Combine growth components
    if growth_components:
        growth_df = pd.concat(growth_components, axis=1)
        growth_pulse = growth_df.mean(axis=1, skipna=True)
    else:
        growth_pulse = pd.Series(dtype=float)
    
    # ----- INFLATION PULSE -----
    inflation_components = []
    inflation_weights = []
    
    inflation_config = [
        ("CPILFESL", [1, 3, 6], False, 1.0),
        ("PCEPILFE", [1, 3, 6], False, 2.0),
        ("T5YIE", [21, 63, 126], False, 1.2),
        ("T10YIE", [21, 63, 126], False, 1.0),
    ]
    
    for key, periods, invert, weight in inflation_config:
        if key not in data:
            continue
        
        df = data[key]
        if "Value" not in df.columns or len(df) < 48:
            continue
        
        series = df["Value"].dropna()
        
        # For breakevens (daily), resample to monthly
        if key in ["T5YIE", "T10YIE"]:
            series = series.resample("M").last()
            periods = [1, 3, 6]
        
        pulse = _compute_indicator_pulse(series, periods, invert)
        
        if not pulse.empty:
            inflation_components.append(pulse * weight)
            inflation_weights.append(weight)
    
    # Combine inflation components
    if inflation_components:
        inflation_df = pd.concat(inflation_components, axis=1)
        inflation_pulse = inflation_df.mean(axis=1, skipna=True)
    else:
        inflation_pulse = pd.Series(dtype=float)
    
    # ----- COMBINE AND SCALE -----
    if growth_pulse.empty or inflation_pulse.empty:
        return pd.DataFrame()
    
    result = pd.DataFrame({
        "Growth_Pulse": growth_pulse,
        "Inflation_Pulse": inflation_pulse
    }).dropna()
    
    if result.empty:
        return pd.DataFrame()
    
    # Scale to -100 to +100 range
    result["Growth_Pulse"] = (result["Growth_Pulse"] * scale_factor).clip(-100, 100)
    result["Inflation_Pulse"] = (result["Inflation_Pulse"] * scale_factor).clip(-100, 100)
    
    # Filter to lookback period
    end_date = result.index.max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    result = result[result.index >= start_date]
    
    # Add regime classification
    result["Regime"] = result.apply(
        lambda row: _classify_regime(row["Growth_Pulse"], row["Inflation_Pulse"]),
        axis=1
    )
    
    return result