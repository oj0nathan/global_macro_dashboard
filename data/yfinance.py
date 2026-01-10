# data/yfinance.py
"""
Market data client using yfinance.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

import config
from data.fred import SeriesFrame, _ensure_datetime_index

logger = logging.getLogger(__name__)

class MarketClient:
    """
    Market data client for equities, ETFs, commodities, FX.
    """
    
    def __init__(self):
        pass
    
    def get_asset(self, asset_key: str, start: Optional[str] = None) -> SeriesFrame:
        """
        Fetch a single asset's price history.
        
        Args:
            asset_key: Key from config.YF_ASSETS
            start: Start date (YYYY-MM-DD), default is 5 years ago
            
        Returns:
            SeriesFrame with daily prices
        """
        if asset_key not in config.YF_ASSETS:
            raise ValueError(f"Unknown asset key: {asset_key}")
        
        spec = config.YF_ASSETS[asset_key]
        logger.info(f"Fetching YF: {spec.label} ({spec.ticker})")
        
        try:
            ticker = yf.Ticker(spec.ticker)
            df = ticker.history(start=start, interval="1d", auto_adjust=True)
        except Exception as e:
            logger.error(f"YF error for {spec.ticker}: {e}")
            raise
        
        if df is None or df.empty:
            raise ValueError(f"No data returned for {spec.ticker}")
        
        # Extract close price
        if "Close" in df.columns:
            series = df["Close"]
        elif "Adj Close" in df.columns:
            series = df["Adj Close"]
        else:
            # Fallback to first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric columns for {spec.ticker}")
            series = df[numeric_cols[0]]
        
        # Convert to DataFrame
        out_df = pd.DataFrame(series).rename(columns={series.name: "Value"})
        out_df = _ensure_datetime_index(out_df).dropna()
        
        # Forward fill gaps (markets closed on weekends/holidays)
        out_df = out_df.ffill()
        
        # Data quality check
        if len(out_df) < 50:
            logger.warning(f"Only {len(out_df)} observations for {spec.ticker}")
        
        na_pct = out_df["Value"].isna().sum() / len(out_df)
        if na_pct > 0.05:
            logger.warning(f"{spec.ticker} has {na_pct:.1%} missing data")
        
        return SeriesFrame(
            df=out_df,
            name=spec.label,
            source="YF",
            freq="D",
            unit="price",
            chg_unit="pct",
        )
    
    def get_bundle(
        self, 
        asset_keys: List[str], 
        start: Optional[str] = None
    ) -> Dict[str, SeriesFrame]:
        """Fetch multiple assets with better error handling"""
        import time
        
        out: Dict[str, SeriesFrame] = {}
        failed = []
        
        for key in asset_keys:
            try:
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
                out[key] = self.get_asset(key, start=start)
                logger.info(f"✓ Successfully fetched {key}")
                
            except Exception as e:
                logger.warning(f"✗ Failed to fetch {key}: {e}")
                failed.append(key)
        
        if failed:
            logger.warning(f"Failed assets: {', '.join(failed)}")
            # Show this to user in Streamlit
            import streamlit as st
            st.sidebar.warning(f"Failed to load: {', '.join(failed)}")
        
        logger.info(f"Successfully fetched {len(out)}/{len(asset_keys)} assets")
        return out


# -----------------------------
# Helper functions
# -----------------------------
def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to month-end.
    Used to align market data with monthly macro data.
    
    Ensures timezone-naive output.
    """
    df = _ensure_datetime_index(df)
    
    # Double-check timezone is removed
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df.resample("M").last().dropna()


def normalize_index(series: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Normalize series to index (first value = base).
    Useful for overlay charts.
    """
    s = series.dropna().copy()
    if s.empty:
        return s
    return (s / s.iloc[0]) * base


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling beta of y vs x.
    Beta = cov(y, x) / var(x)
    """
    y, x = y.align(x, join="inner")
    if len(y) < window:
        return pd.Series(index=y.index, dtype=float)
    
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0, np.nan)

class YFinanceClient:
    """
    Client for fetching market data from Yahoo Finance.
    """
    
    def __init__(self):
        pass
    
    def get_price_history(
        self, 
        ticker: str, 
        years: int = 5,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch price history for a ticker.
        
        Args:
            ticker: Stock/ETF ticker symbol
            years: Years of history to fetch
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=years * 365)
            
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            return data
            
        except Exception as e:
            logging.warning(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()
    
    def get_multiple_tickers(
        self,
        tickers: list,
        years: int = 5,
        column: str = "Close"
    ) -> pd.DataFrame:
        """
        Fetch price data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            years: Years of history
            column: Which price column to return
            
        Returns:
            DataFrame with tickers as columns
        """
        result = {}
        
        for ticker in tickers:
            df = self.get_price_history(ticker, years)
            if not df.empty and column in df.columns:
                result[ticker] = df[column]
        
        if result:
            return pd.DataFrame(result)
        return pd.DataFrame()