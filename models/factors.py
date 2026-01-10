# models/factors.py
"""
Factor performance analysis.

Research quote:
"Growth and momentum have been slipping down the rankings, while 
concentration and value have moved into the lead. That isn't a 
high-risk-on setup."
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class FactorScores:
    """Factor performance metrics"""
    factor: str
    ret_1m: float      # 1-month excess return vs benchmark (%)
    ret_3m: float      # 3-month excess return vs benchmark (%)
    ret_6m: float      # 6-month excess return vs benchmark (%)
    rank_3m: int       # Rank based on 3M performance
    

def compute_factor_performance(
    factor_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    lookback_days: int = 126
) -> pd.DataFrame:
    """
    Compute factor performance vs benchmark.
    
    Args:
        factor_data: Dict of {factor_key: DataFrame with "Value" column}
        benchmark_data: Benchmark DataFrame (e.g., SPY) with "Value" column
        lookback_days: Historical window for calculations
        
    Returns:
        DataFrame with factor scores, sorted by 3M performance
    """
    results = []
    
    # Benchmark returns
    bench_ret = benchmark_data["Value"].pct_change()
    
    for factor_key, factor_df in factor_data.items():
        if "Value" not in factor_df.columns:
            continue
        
        # Factor returns
        factor_ret = factor_df["Value"].pct_change()
        
        # Align with benchmark
        factor_ret, bench_aligned = factor_ret.align(bench_ret, join="inner")
        
        if len(factor_ret) < lookback_days:
            continue
        
        # Excess returns (factor - benchmark)
        excess = factor_ret - bench_aligned
        
        # Cumulative excess returns over different periods
        if len(excess) >= 21:
            ret_1m = excess.iloc[-21:].sum() * 100  # Convert to %
        else:
            ret_1m = np.nan
        
        if len(excess) >= 63:
            ret_3m = excess.iloc[-63:].sum() * 100
        else:
            ret_3m = np.nan
        
        if len(excess) >= 126:
            ret_6m = excess.iloc[-126:].sum() * 100
        else:
            ret_6m = np.nan
        
        results.append({
            "Factor": factor_key,
            "1M": ret_1m,
            "3M": ret_3m,
            "6M": ret_6m,
        })
    
    df = pd.DataFrame(results)
    
    if not df.empty and "3M" in df.columns:
        df["Rank_3M"] = df["3M"].rank(ascending=False, method="min")
        df = df.sort_values("Rank_3M")
    
    return df


def infer_regime_from_factors(scores: pd.DataFrame) -> str:
    """
    Infer risk appetite regime from factor leadership.
    
    Research logic:
    - Risk-on: Momentum, Size (small cap), Growth leading
    - Risk-off: Quality, Value, Low Vol leading
    - Mixed: No clear pattern
    
    Args:
        scores: DataFrame from compute_factor_performance()
        
    Returns:
        Regime string
    """
    if scores.empty or len(scores) < 3:
        return "Unknown (insufficient data)"
    
    # Top 3 factors by 3M performance
    top_3 = scores.head(3)["Factor"].tolist()
    
    # Define factor groups
    risk_on_factors = ["MTUM", "SIZE", "IWM"]  # Momentum, small cap
    defensive_factors = ["QUAL", "USMV", "VLUE"]  # Quality, low vol, value
    
    risk_on_count = sum(1 for f in top_3 if f in risk_on_factors)
    defensive_count = sum(1 for f in top_3 if f in defensive_factors)
    
    if risk_on_count >= 2:
        return "Risk-On (momentum/growth leading)"
    elif defensive_count >= 2:
        return "Risk-Off (quality/defensive leading)"
    else:
        return "Mixed (rotation/dispersion)"


def compute_sector_relative(
    sector_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    lookback_days: int = 63
) -> pd.DataFrame:
    """
    Compute sector relative strength vs benchmark.
    
    Similar to factor performance but for sectors (XLE, XLF, etc.)
    
    Args:
        sector_data: Dict of {sector_key: DataFrame with "Value"}
        benchmark_data: Benchmark (SPY) with "Value"
        lookback_days: Window for calculations
        
    Returns:
        DataFrame with sector scores
    """
    results = []
    
    bench_ret = benchmark_data["Value"].pct_change()
    
    for sector_key, sector_df in sector_data.items():
        if "Value" not in sector_df.columns:
            continue
        
        sector_ret = sector_df["Value"].pct_change()
        sector_ret, bench_aligned = sector_ret.align(bench_ret, join="inner")
        
        if len(sector_ret) < lookback_days:
            continue
        
        excess = sector_ret - bench_aligned
        
        # Cumulative excess
        if len(excess) >= 21:
            ret_1m = excess.iloc[-21:].sum() * 100
        else:
            ret_1m = np.nan
        
        if len(excess) >= 63:
            ret_3m = excess.iloc[-63:].sum() * 100
        else:
            ret_3m = np.nan
        
        results.append({
            "Sector": sector_key,
            "1M": ret_1m,
            "3M": ret_3m,
        })
    
    df = pd.DataFrame(results)
    
    if not df.empty and "3M" in df.columns:
        df["Rank_3M"] = df["3M"].rank(ascending=False, method="min")
        df = df.sort_values("Rank_3M")
    
    return df


def compute_momentum_indicators(
    price_series: pd.Series,
    windows: List[int] = [21, 63, 126, 252]
) -> pd.DataFrame:
    """
    Compute momentum indicators for a price series.
    
    Args:
        price_series: Price time series
        windows: List of lookback windows (in days)
        
    Returns:
        DataFrame with momentum metrics
    """
    results = []
    
    for window in windows:
        if len(price_series) < window:
            continue
        
        # Total return over window
        ret = (price_series.iloc[-1] / price_series.iloc[-window] - 1) * 100
        
        # Volatility (annualized)
        daily_ret = price_series.pct_change().dropna()
        if len(daily_ret) >= window:
            vol = daily_ret.iloc[-window:].std() * np.sqrt(252) * 100
        else:
            vol = np.nan
        
        # Sharpe-like ratio
        if not np.isnan(vol) and vol > 0:
            sharpe = (ret / window * 252) / vol
        else:
            sharpe = np.nan
        
        results.append({
            "Window": f"{window}d",
            "Return": ret,
            "Vol": vol,
            "Sharpe": sharpe,
        })
    
    return pd.DataFrame(results)