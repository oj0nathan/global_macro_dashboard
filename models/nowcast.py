# models/nowcast.py
"""
GDP and Inflation Nowcasting Engine.

Research quote:
"The way that I think about GDP is that there are always different 
weighting for the inputs. You always want to know what the individual 
drivers are and how they are moving on a monthly basis."

"How do people come up with these GDP nowcasts? Basically, people take 
the monthly data and extrapolate it to the quarterly data."

This module implements:
1. GDP nowcast using bridge equations (monthly → quarterly)
2. Inflation nowcast using component momentum
3. Comparison with Atlanta Fed GDPNow benchmark
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NowcastResult:
    """Container for nowcast output"""
    estimate: float              # Point estimate (annualized %)
    confidence_band: Tuple[float, float]  # 68% confidence interval
    contributions: Dict[str, float]  # Component contributions
    data_vintage: pd.Timestamp   # Most recent data used
    quarter: str                 # Target quarter (e.g., "2025Q4")
    model_type: str              # "GDP" or "Inflation"


@dataclass 
class NowcastComparison:
    """Comparison with benchmark (Atlanta Fed)"""
    our_estimate: float
    atlanta_fed: Optional[float]
    difference: Optional[float]
    atlanta_fed_date: Optional[pd.Timestamp]


class GDPNowcast:
    """
    Real-time Growth Activity Pulse.
    
    Methodology:
    1. Take monthly growth indicators (INDPRO, RSAFS, PAYEMS, etc.)
    2. Deflate nominal series (Retail Sales) to real terms using CPI
    3. Compute 3-month annualized growth rates
    4. Weight by economic relevance
    5. Aggregate to activity estimate
    
    Note: This is an activity momentum indicator, not a formal GDP nowcast.
    For official GDP tracking, see Atlanta Fed GDPNow.
    """
    
    # Default weights based on GDP composition and empirical relevance
    DEFAULT_WEIGHTS = {
        "INDPRO": 0.15,   # Industrial production → Investment/inventory proxy
        "RSAFS":  0.40,   # Retail sales → PCE proxy (consumption is ~68% of GDP)
        "PAYEMS": 0.30,   # Employment → Broad activity indicator
        "AWHMAN": 0.10,   # Hours worked → Labor intensity
        "MANEMP": 0.05,   # Manufacturing employment → Small weight (declining sector)
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize GDP nowcast model.
        
        Args:
            weights: Custom weights for inputs. If None, uses defaults.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"GDPNowcast initialized with weights: {self.weights}")
    
    def _compute_mom_annualized(self, series: pd.Series) -> pd.Series:
        """
        Compute month-over-month growth, annualized.
        
        For monthly data: ((1 + mom)^12 - 1) * 100
        """
        mom = series.pct_change()
        # Annualize: compound monthly growth to annual
        annualized = ((1 + mom) ** 12 - 1) * 100
        return annualized
    
    def _compute_3m_annualized(self, series: pd.Series) -> pd.Series:
        """
        Compute 3-month annualized growth rate.
        
        More stable than MoM, commonly used for GDP tracking.
        """
        pct_3m = series.pct_change(3)
        # Annualize: (1 + 3m_growth)^4 - 1
        annualized = ((1 + pct_3m) ** 4 - 1) * 100
        return annualized
    
    def _deflate_to_real(
        self, 
        nominal_series: pd.Series, 
        cpi_series: pd.Series
    ) -> pd.Series:
        """
        Convert nominal series to real terms using CPI deflator.
        
        Real = Nominal / CPI * 100
        
        This is critical for Retail Sales which is reported in nominal dollars.
        Without deflation, the "growth" signal is contaminated by inflation.
        """
        # Align the series
        nominal, cpi = nominal_series.align(cpi_series, join="inner")
        
        if cpi.empty or nominal.empty:
            return nominal_series  # Fallback to nominal if can't deflate
        
        # Deflate: Real = Nominal / CPI * 100
        real = (nominal / cpi) * 100
        
        return real
    
    def _get_current_quarter(self) -> str:
        """Get current quarter string (e.g., '2025Q4')"""
        now = pd.Timestamp.now()
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}Q{quarter}"
    
    def _get_quarter_months(self, quarter_str: str) -> List[pd.Timestamp]:
        """Get the 3 months in a quarter"""
        year = int(quarter_str[:4])
        q = int(quarter_str[-1])
        start_month = (q - 1) * 3 + 1
        
        months = []
        for m in range(start_month, start_month + 3):
            months.append(pd.Timestamp(year=year, month=m, day=1))
        return months
    
    def nowcast(
        self, 
        monthly_data: Dict[str, pd.DataFrame],
        target_quarter: Optional[str] = None
    ) -> Optional[NowcastResult]:
        """
        Generate real-time activity estimate.
        
        Args:
            monthly_data: Dict of {series_key: DataFrame with "Value" column}
            target_quarter: Quarter to nowcast (default: current)
            
        Returns:
            NowcastResult with estimate and components
        """
        if target_quarter is None:
            target_quarter = self._get_current_quarter()
        
        # Get CPI for deflation (needed for nominal series like Retail Sales)
        cpi_series = None
        if "CPIAUCSL" in monthly_data:
            cpi_df = monthly_data["CPIAUCSL"]
            if "Value" in cpi_df.columns:
                cpi_series = cpi_df["Value"].dropna()
        
        contributions = {}
        weighted_sum = 0.0
        total_weight_used = 0.0
        latest_dates = []
        
        for series_key, weight in self.weights.items():
            if series_key not in monthly_data:
                logger.warning(f"Missing series for nowcast: {series_key}")
                continue
            
            df = monthly_data[series_key]
            if "Value" not in df.columns or df.empty:
                continue
            
            series = df["Value"].dropna()
            if len(series) < 4:  # Need at least 4 months for 3m growth
                continue
            
            # CRITICAL: Deflate nominal series to real terms
            is_deflated = False
            if series_key == "RSAFS" and cpi_series is not None:
                series = self._deflate_to_real(series, cpi_series)
                is_deflated = True
            
            # Use 3-month annualized growth (more stable)
            growth = self._compute_3m_annualized(series)
            
            # Get most recent value
            latest_growth = growth.dropna().iloc[-1]
            latest_date = growth.dropna().index[-1]
            latest_dates.append(latest_date)
            
            # Compute contribution
            contribution = latest_growth * weight
            contributions[series_key] = {
                "growth": latest_growth,
                "weight": weight,
                "contribution": contribution,
                "as_of": latest_date,
                "is_real": is_deflated or series_key != "RSAFS"  # Track if real or nominal
            }
            
            weighted_sum += contribution
            total_weight_used += weight
        
        if total_weight_used == 0:
            logger.error("No valid data for activity nowcast")
            return None
        
        # Scale by weight used (in case some series missing)
        estimate = weighted_sum / total_weight_used
        
        # Simple confidence band: +/- 1.5% based on typical nowcast uncertainty
        confidence = (estimate - 1.5, estimate + 1.5)
        
        # Data vintage is the earliest "latest date" (most stale input)
        data_vintage = min(latest_dates) if latest_dates else pd.Timestamp.now()
        
        return NowcastResult(
            estimate=estimate,
            confidence_band=confidence,
            contributions=contributions,
            data_vintage=data_vintage,
            quarter=target_quarter,
            model_type="Activity"  # Changed from "GDP"
        )

    def compare_with_atlanta_fed(
        self,
        our_result: NowcastResult,
        atlanta_fed_data: Optional[pd.DataFrame]
    ) -> NowcastComparison:
        """
        Compare our nowcast with Atlanta Fed GDPNow.
        
        Args:
            our_result: Our nowcast result
            atlanta_fed_data: DataFrame with Atlanta Fed GDPNow ("Value" column)
            
        Returns:
            NowcastComparison object
        """
        atlanta_value = None
        atlanta_date = None
        
        if atlanta_fed_data is not None and "Value" in atlanta_fed_data.columns:
            series = atlanta_fed_data["Value"].dropna()
            if not series.empty:
                atlanta_value = series.iloc[-1]
                atlanta_date = series.index[-1]
        
        diff = None
        if atlanta_value is not None:
            diff = our_result.estimate - atlanta_value
        
        return NowcastComparison(
            our_estimate=our_result.estimate,
            atlanta_fed=atlanta_value,
            difference=diff,
            atlanta_fed_date=atlanta_date
        )


class InflationNowcast:
    """
    Inflation nowcasting using component momentum.
    
    Methodology:
    1. Track momentum in key CPI components (shelter, energy, food)
    2. Use recent trends to project next month's print
    3. Compare with breakeven-implied expectations
    
    Research: "Inflation swaps drift higher... inflation expectations"
    """
    
    # CPI component weights (approximate)
    CPI_WEIGHTS = {
        "CUSR0000SAH1": 0.33,     # Shelter (~33% of CPI)
        "CUSR0000SETB01": 0.04,   # Gasoline
        "CUSR0000SAF11": 0.08,    # Food at home
        # Core services ex-shelter is remainder, proxied by wage growth
    }
    
    def __init__(self):
        logger.info("InflationNowcast initialized")
    
    def _compute_mom(self, series: pd.Series) -> float:
        """Compute most recent month-over-month % change"""
        if len(series) < 2:
            return np.nan
        return (series.iloc[-1] / series.iloc[-2] - 1) * 100
    
    def _compute_yoy(self, series: pd.Series) -> float:
        """Compute year-over-year % change"""
        if len(series) < 13:
            return np.nan
        return (series.iloc[-1] / series.iloc[-13] - 1) * 100
    
    def _compute_3m_annualized(self, series: pd.Series) -> float:
        """Compute 3-month annualized rate"""
        if len(series) < 4:
            return np.nan
        pct_3m = series.iloc[-1] / series.iloc[-4] - 1
        return ((1 + pct_3m) ** 4 - 1) * 100
    
    def nowcast(
        self,
        inflation_data: Dict[str, pd.DataFrame],
        headline_key: str = "CPIAUCSL",
        core_key: str = "CPILFESL"
    ) -> Optional[NowcastResult]:
        """
        Generate inflation nowcast.
        
        Uses momentum in headline/core CPI to project trend.
        
        Args:
            inflation_data: Dict of inflation series
            headline_key: Key for headline CPI
            core_key: Key for core CPI
            
        Returns:
            NowcastResult for inflation
        """
        contributions = {}
        
        # Get core CPI (Fed's focus)
        if core_key not in inflation_data:
            logger.warning(f"Missing {core_key} for inflation nowcast")
            return None
        
        core_df = inflation_data[core_key]
        if "Value" not in core_df.columns:
            return None
        
        core = core_df["Value"].dropna()
        if len(core) < 13:
            return None
        
        # Current YoY
        current_yoy = self._compute_yoy(core)
        
        # 3-month annualized (momentum indicator)
        momentum_3m = self._compute_3m_annualized(core)
        
        # Simple projection: weight current YoY vs recent momentum
        # If momentum < YoY, inflation is decelerating
        estimate = 0.6 * current_yoy + 0.4 * momentum_3m
        
        contributions["Core CPI"] = {
            "current_yoy": current_yoy,
            "3m_annualized": momentum_3m,
            "trend": "Decelerating" if momentum_3m < current_yoy else "Accelerating"
        }
        
        # Add headline if available
        if headline_key in inflation_data:
            headline_df = inflation_data[headline_key]
            if "Value" in headline_df.columns:
                headline = headline_df["Value"].dropna()
                if len(headline) >= 13:
                    headline_yoy = self._compute_yoy(headline)
                    headline_3m = self._compute_3m_annualized(headline)
                    contributions["Headline CPI"] = {
                        "current_yoy": headline_yoy,
                        "3m_annualized": headline_3m,
                        "trend": "Decelerating" if headline_3m < headline_yoy else "Accelerating"
                    }
        
        # Confidence band based on typical forecast error
        confidence = (estimate - 0.3, estimate + 0.3)
        
        return NowcastResult(
            estimate=estimate,
            confidence_band=confidence,
            contributions=contributions,
            data_vintage=core.index[-1],
            quarter=self._get_current_month(),
            model_type="Inflation"
        )
    
    def _get_current_month(self) -> str:
        """Get current month string"""
        now = pd.Timestamp.now()
        return now.strftime("%Y-%m")
    
    def compare_with_breakevens(
        self,
        our_result: NowcastResult,
        breakeven_data: Optional[pd.DataFrame],
        horizon: str = "5Y"
    ) -> Dict:
        """
        Compare inflation nowcast with market-implied expectations.
        
        Args:
            our_result: Our inflation nowcast
            breakeven_data: DataFrame with breakeven inflation
            horizon: "5Y" or "10Y"
            
        Returns:
            Dict with comparison metrics
        """
        if breakeven_data is None or "Value" not in breakeven_data.columns:
            return {"breakeven": None, "vs_market": None}
        
        be = breakeven_data["Value"].dropna()
        if be.empty:
            return {"breakeven": None, "vs_market": None}
        
        current_be = be.iloc[-1]
        
        # Our nowcast vs market expectations
        # Note: Breakevens are forward-looking averages, not point estimates
        diff = our_result.estimate - current_be
        
        return {
            "breakeven": current_be,
            "breakeven_horizon": horizon,
            "our_nowcast": our_result.estimate,
            "vs_market": diff,
            "interpretation": "Above market" if diff > 0 else "Below market"
        }


class NowcastDashboard:
    """
    Unified nowcast dashboard combining GDP and inflation.
    
    Provides:
    - Real-time GDP estimate vs Atlanta Fed
    - Inflation trajectory vs Fed target
    - Growth-inflation quadrant positioning
    """
    
    def __init__(self):
        self.gdp_model = GDPNowcast()
        self.inflation_model = InflationNowcast()
    
    def run_full_nowcast(
        self,
        fred_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Run complete nowcast suite.
        
        Args:
            fred_data: Dict of all FRED series {key: DataFrame}
            
        Returns:
            Dict with GDP and inflation nowcasts + analysis
        """
        results = {}
        
        # GDP Nowcast
        gdp_result = self.gdp_model.nowcast(fred_data)
        if gdp_result:
            results["gdp"] = gdp_result
            
            # Compare with Atlanta Fed if available
            if "GDPNOW" in fred_data:
                comparison = self.gdp_model.compare_with_atlanta_fed(
                    gdp_result, 
                    fred_data["GDPNOW"]
                )
                results["gdp_comparison"] = comparison
        
        # Inflation Nowcast
        inflation_result = self.inflation_model.nowcast(fred_data)
        if inflation_result:
            results["inflation"] = inflation_result
            
            # Compare with breakevens
            if "T5YIE" in fred_data:
                be_comparison = self.inflation_model.compare_with_breakevens(
                    inflation_result,
                    fred_data["T5YIE"],
                    horizon="5Y"
                )
                results["inflation_vs_market"] = be_comparison
        
        # Growth-Inflation positioning
        if gdp_result and inflation_result:
            results["quadrant"] = self._determine_quadrant(
                gdp_result.estimate,
                inflation_result.estimate
            )
        
        return results
    
    def _determine_quadrant(
        self, 
        gdp_nowcast: float, 
        inflation_nowcast: float
    ) -> Dict:
        """
        Determine GIP quadrant from nowcasts.
        
        Research: The macro regime quadrant matters more than individual levels.
        """
        # Thresholds (can be refined)
        gdp_trend_threshold = 2.0  # Above 2% = strong growth
        inflation_target = 2.5     # Fed target + buffer
        
        if gdp_nowcast > gdp_trend_threshold:
            if inflation_nowcast < inflation_target:
                quadrant = "Goldilocks"
                description = "Strong growth, contained inflation"
                risk_posture = "Risk-on, favor cyclicals"
            else:
                quadrant = "Reflation"
                description = "Strong growth, rising inflation"
                risk_posture = "Favor real assets, inflation hedges"
        else:
            if inflation_nowcast > inflation_target:
                quadrant = "Stagflation"
                description = "Weak growth, sticky inflation"
                risk_posture = "Defensive, cash, short duration"
            else:
                quadrant = "Deflation"
                description = "Weak growth, low inflation"
                risk_posture = "Long duration, quality"
        
        return {
            "quadrant": quadrant,
            "description": description,
            "risk_posture": risk_posture,
            "gdp_nowcast": gdp_nowcast,
            "inflation_nowcast": inflation_nowcast
        }


# Convenience function for quick nowcast
def quick_nowcast(fred_bundle: Dict) -> Dict:
    """
    Quick nowcast from FRED data bundle.
    
    Args:
        fred_bundle: Dict of {key: SeriesFrame} from FREDClient
        
    Returns:
        Dict with nowcast results
    """
    # Convert SeriesFrame to DataFrame dict
    data = {}
    for key, sf in fred_bundle.items():
        if hasattr(sf, 'df'):
            data[key] = sf.df
        elif isinstance(sf, pd.DataFrame):
            data[key] = sf
    
    dashboard = NowcastDashboard()
    return dashboard.run_full_nowcast(data)