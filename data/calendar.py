# data/calendar.py
"""
Economic calendar and surprise index integration.
- Economic Calendar module for tracking and predicting US economic releases.

Provides:
- RELEASE_SCHEDULE: A registry of major US indicators (CPI, NFP, GDP) and their logic (this is for the surprises tab)
- CalendarClient: An estimator that calculates specific release dates (e.g., 'First Friday')
  and returns a searchable DataFrame of upcoming market-moving events.
- Integration: Maps indicators directly to FRED series IDs for easy data retrieval.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import calendar

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EconomicRelease:
    """Container for an economic release"""
    indicator: str
    date: datetime
    time_et: str          # Release time (Eastern)
    frequency: str        # "M", "W", "Q"
    importance: str       # "High", "Medium", "Low"
    fred_series: str      # Corresponding FRED series
    category: str         # "Growth", "Inflation", "Labor", etc.


# -----------------------------
# Release Schedule Templates
# -----------------------------
# Major US economic releases and their typical schedules - Hard Coded
RELEASE_SCHEDULE = {
    # Labor Market
    "NFP": {
        "name": "Nonfarm Payrolls",
        "schedule": "first_friday",
        "time_et": "08:30",
        "frequency": "M",
        "importance": "High",
        "fred_series": "PAYEMS",
        "category": "Labor"
    },
    "UNEMPLOYMENT": {
        "name": "Unemployment Rate",
        "schedule": "first_friday",
        "time_et": "08:30",
        "frequency": "M",
        "importance": "High",
        "fred_series": "UNRATE",
        "category": "Labor"
    },
    "INITIAL_CLAIMS": {
        "name": "Initial Jobless Claims",
        "schedule": "weekly_thursday",
        "time_et": "08:30",
        "frequency": "W",
        "importance": "Medium",
        "fred_series": "ICSA",
        "category": "Labor"
    },
    
    # Inflation
    "CPI": {
        "name": "CPI (Consumer Price Index)",
        "schedule": "mid_month",
        "day_range": (10, 15),
        "time_et": "08:30",
        "frequency": "M",
        "importance": "High",
        "fred_series": "CPIAUCSL",
        "category": "Inflation"
    },
    "CORE_CPI": {
        "name": "Core CPI",
        "schedule": "mid_month",
        "day_range": (10, 15),
        "time_et": "08:30",
        "frequency": "M",
        "importance": "High",
        "fred_series": "CPILFESL",
        "category": "Inflation"
    },
    "PPI": {
        "name": "PPI (Producer Price Index)",
        "schedule": "mid_month",
        "day_range": (11, 17),
        "time_et": "08:30",
        "frequency": "M",
        "importance": "Medium",
        "fred_series": "PPIACO",
        "category": "Inflation"
    },
    "PCE": {
        "name": "Core PCE Price Index",
        "schedule": "end_month",
        "day_range": (25, 31),
        "time_et": "08:30",
        "frequency": "M",
        "importance": "High",
        "fred_series": "PCEPILFE",
        "category": "Inflation"
    },
    
    # Growth / Activity
    "RETAIL_SALES": {
        "name": "Retail Sales",
        "schedule": "mid_month",
        "day_range": (13, 17),
        "time_et": "08:30",
        "frequency": "M",
        "importance": "High",
        "fred_series": "RSAFS",
        "category": "Growth"
    },
    "INDPRO": {
        "name": "Industrial Production",
        "schedule": "mid_month",
        "day_range": (15, 18),
        "time_et": "09:15",
        "frequency": "M",
        "importance": "Medium",
        "fred_series": "INDPRO",
        "category": "Growth"
    },
    "GDP": {
        "name": "GDP (Advance/Prelim/Final)",
        "schedule": "end_month",
        "day_range": (25, 30),
        "time_et": "08:30",
        "frequency": "Q",
        "importance": "High",
        "fred_series": "GDPC1",
        "category": "Growth"
    },
    
    # Surveys
    "ISM_MFG": {
        "name": "ISM Manufacturing PMI",
        "schedule": "first_business_day",
        "time_et": "10:00",
        "frequency": "M",
        "importance": "High",
        "fred_series": "MANEMP",
        "category": "Survey"
    },
    
    # Housing
    "HOUSING_STARTS": {
        "name": "Housing Starts",
        "schedule": "mid_month",
        "day_range": (16, 20),
        "time_et": "08:30",
        "frequency": "M",
        "importance": "Medium",
        "fred_series": "HOUST",
        "category": "Housing"
    },
}


class CalendarClient:
    """
    Economic calendar client.
    
    Provides:
    - Upcoming release schedule
    - Historical release tracking
    - Integration with FRED for actual values
    """
    
    def __init__(self):
        self.schedule = RELEASE_SCHEDULE
    
    def _get_first_friday(self, year: int, month: int) -> datetime:
        """Get first Friday of the month"""
        cal = calendar.Calendar()
        for day in cal.itermonthdates(year, month):
            if day.month == month and day.weekday() == 4:  # Friday = 4
                return datetime(day.year, day.month, day.day)
        return datetime(year, month, 1)
    
    def _get_first_business_day(self, year: int, month: int) -> datetime:
        """Get first business day of the month"""
        day = datetime(year, month, 1)
        while day.weekday() > 4:  # Skip weekend
            day += timedelta(days=1)
        return day
    
    def _estimate_release_date(
        self, 
        release_key: str, 
        year: int, 
        month: int
    ) -> Optional[datetime]:
        """Estimate release date based on schedule pattern"""
        
        if release_key not in self.schedule:
            return None
        
        spec = self.schedule[release_key]
        schedule_type = spec.get("schedule", "")
        
        if schedule_type == "first_friday":
            return self._get_first_friday(year, month)
        
        elif schedule_type == "first_business_day":
            return self._get_first_business_day(year, month)
        
        elif schedule_type == "weekly_thursday":
            # Return next Thursday
            today = datetime.now()
            days_ahead = 3 - today.weekday()  # Thursday = 3
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        
        elif schedule_type in ("mid_month", "end_month"):
            day_range = spec.get("day_range", (15, 15))
            # Use midpoint of range
            day = (day_range[0] + day_range[1]) // 2
            # Ensure valid day for month
            max_day = calendar.monthrange(year, month)[1]
            day = min(day, max_day)
            return datetime(year, month, day)
        
        return None
    
    def get_upcoming_releases(
        self, 
        days_ahead: int = 14,
        importance_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get upcoming economic releases.
        
        Args:
            days_ahead: Look ahead window in days
            importance_filter: Filter by importance ["High", "Medium", "Low"]
            
        Returns:
            DataFrame with upcoming releases
        """
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)
        
        releases = []
        
        for key, spec in self.schedule.items():
            # Filter by importance
            if importance_filter and spec["importance"] not in importance_filter:
                continue
            
            # Skip weekly for simplicity (except initial claims)
            if spec["frequency"] == "W" and key != "INITIAL_CLAIMS":
                continue
            
            # Check current month and next month
            for month_offset in range(3):
                check_date = today + timedelta(days=30 * month_offset)
                year = check_date.year
                month = check_date.month
                
                release_date = self._estimate_release_date(key, year, month)
                
                if release_date and today <= release_date <= end_date:
                    releases.append({
                        "Date": release_date,
                        "Time (ET)": spec["time_et"],
                        "Indicator": spec["name"],
                        "Category": spec["category"],
                        "Importance": spec["importance"],
                        "FRED Series": spec["fred_series"],
                    })
        
        if not releases:
            return pd.DataFrame()
        
        df = pd.DataFrame(releases)
        df = df.sort_values("Date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        
        return df
    
    def get_recent_releases(
        self,
        fred_bundle: Dict,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Get recent releases with actual values.
        
        Cross-references FRED data to show what was released.
        
        Args:
            fred_bundle: Dict of SeriesFrame from FRED client
            days_back: Look back window
            
        Returns:
            DataFrame with recent releases and values
        """
        cutoff = datetime.now() - timedelta(days=days_back)
        releases = []
        
        for key, spec in self.schedule.items():
            fred_key = spec["fred_series"]
            
            if fred_key not in fred_bundle:
                continue
            
            sf = fred_bundle[fred_key]
            if sf.df.empty:
                continue
            
            # Get most recent data point
            series = sf.df["Value"].dropna()
            if series.empty:
                continue
            
            last_date = series.index[-1]
            last_value = series.iloc[-1]
            
            # Get prior value for comparison
            if len(series) >= 2:
                prior_value = series.iloc[-2]
                change = last_value - prior_value
            else:
                prior_value = None
                change = None
            
            if last_date >= cutoff:
                releases.append({
                    "Date": last_date,
                    "Indicator": spec["name"],
                    "Category": spec["category"],
                    "Actual": last_value,
                    "Prior": prior_value,
                    "Change": change,
                    "FRED Series": fred_key,
                })
        
        if not releases:
            return pd.DataFrame()
        
        df = pd.DataFrame(releases)
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        
        return df


class SurpriseIndexClient:
    """
    Client for economic surprise indices from FRED.
    
    Research: "Economic Surprise Indices" are key for tracking 
    whether data is coming in above or below expectations.
    """
    
    # Surprise indices available on FRED
    SURPRISE_SERIES = {
        "US": "USCESI",       # Citi Economic Surprise Index - US
        # Note: Other regional indices may require Bloomberg
    }
    
    def __init__(self, fred_client):
        """
        Args:
            fred_client: FREDClient instance for data fetching
        """
        self.fred = fred_client
    
    def get_surprise_index(
        self, 
        region: str = "US"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch surprise index for a region.
        Args:
            region: "US" (others may be added)
        Returns:
            DataFrame with surprise index values
        """
        if region not in self.SURPRISE_SERIES:
            logger.warning(f"No surprise index for region: {region}")
            return None
        
        series_id = self.SURPRISE_SERIES[region]
        
        try:
            data = self.fred.fred.get_series(series_id)
            if data is None or data.empty:
                return None
            
            df = pd.DataFrame(data, columns=["Value"])
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch surprise index {series_id}: {e}")
            return None
    
    def analyze_surprise_trend(
        self,
        surprise_df: pd.DataFrame,
        windows: List[int] = [5, 21, 63]
    ) -> Dict:
        """
        Analyze surprise index trend.
        
        Args:
            surprise_df: DataFrame with surprise index
            windows: Rolling windows for analysis (trading days)
            
        Returns:
            Dict with trend metrics
        """
        if surprise_df is None or surprise_df.empty:
            return {}
        
        series = surprise_df["Value"].dropna()
        if len(series) < max(windows):
            return {}
        
        current = series.iloc[-1]
        
        result = {
            "current": current,
            "interpretation": "Positive surprises" if current > 0 else "Negative surprises",
        }
        
        for w in windows:
            if len(series) >= w:
                avg = series.iloc[-w:].mean()
                result[f"avg_{w}d"] = avg
        
        # Trend direction
        if len(series) >= 21:
            recent = series.iloc[-5:].mean()
            prior = series.iloc[-21:-5].mean()
            result["trend"] = "Improving" if recent > prior else "Deteriorating"
        
        # Historical percentile
        if len(series) >= 252:
            pctl = (series <= current).sum() / len(series)
            result["percentile_1y"] = pctl
        
        return result