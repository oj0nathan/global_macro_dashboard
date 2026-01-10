# config.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
<<<<<<< HEAD
from enum import Enum

# -----------------------------
# Regime Framework
# -----------------------------
class GrowthRegime(Enum):
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    STABLE = "stable"

class InflationRegime(Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"

class LiquidityRegime(Enum):
    EASING = "easing"           # Real rates falling, credit spreads tight
    TIGHTENING = "tightening"   # Real rates rising, credit spreads widening
    NEUTRAL = "neutral"

class MacroRegime(Enum):
    """Quadrant framework from research"""
    GOLDILOCKS = "goldilocks"      # Growth ↑, Inflation ↓
    REFLATION = "reflation"        # Growth ↑, Inflation ↑
    STAGFLATION = "stagflation"    # Growth ↓, Inflation ↑
    DEFLATION = "deflation"        # Growth ↓, Inflation ↓

# -----------------------------
# Enhanced Series Specs
=======

# -----------------------------
# Specs
>>>>>>> 95bbab53f6b0c4f11a35206ef5578faa2ed53401
# -----------------------------
@dataclass(frozen=True)
class FredSeriesSpec:
    id: str
    label: str
<<<<<<< HEAD
    category: str
    freq: str
    unit: str
    chg_unit: str
    regime_weight: float = 1.0  # Weight in regime scoring
=======
    category: str              # Growth / Inflation / Liquidity / Rates / Credit / Macro
    freq: str                  # D / W / M / Q
    unit: str                  # index / rate / level
    chg_unit: str              # pct / bps / level
>>>>>>> 95bbab53f6b0c4f11a35206ef5578faa2ed53401
    notes: str = ""

@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    label: str
<<<<<<< HEAD
    category: str
    unit: str = "price"
    beta_benchmark: str = "SPY"  # For relative attribution

# -----------------------------
# FRED Series - Regime Weighted
# -----------------------------
FRED_SERIES: Dict[str, FredSeriesSpec] = {
    # === GROWTH COMPLEX ===
    # High-frequency monthly proxies
    "INDPRO": FredSeriesSpec("INDPRO", "Industrial Production", "Growth", "M", "index", "pct", 1.5),
    "RSAFS":  FredSeriesSpec("RSAFS",  "Retail Sales", "Growth", "M", "index", "pct", 1.2),
    "PAYEMS": FredSeriesSpec("PAYEMS", "Nonfarm Payrolls", "Growth", "M", "level", "level", 1.3),
    "UNRATE": FredSeriesSpec("UNRATE", "Unemployment Rate", "Growth", "M", "rate", "bps", 1.0),
    "AWHMAN": FredSeriesSpec("AWHMAN", "Avg Weekly Hours (Mfg)", "Growth", "M", "level", "level", 0.8),
    
    # PMI/ISM (leading indicators)
    "MANEMP": FredSeriesSpec("MANEMP", "Manufacturing Employment", "Growth", "M", "level", "level", 0.7),
    
    # === NOWCAST BENCHMARKS ===
    "GDPNOW": FredSeriesSpec("GDPNOW", "Atlanta Fed GDPNow", "Nowcast", "D", "rate", "level", 0.0,
                            "Real-time GDP nowcast from Atlanta Fed"),
    
    # === GDP COMPONENTS (Quarterly - for validation) ===
    "GDPC1":  FredSeriesSpec("GDPC1", "Real GDP", "GDP", "Q", "level", "pct", 0.0),
    "PCECC96": FredSeriesSpec("PCECC96", "Real PCE", "GDP", "Q", "level", "pct", 0.0),
    
    # === INFLATION COMPLEX ===
    "CPIAUCSL": FredSeriesSpec("CPIAUCSL", "CPI Headline (NSA)", "Inflation", "M", "index", "pct", 1.2),
    "CPILFESL": FredSeriesSpec("CPILFESL", "CPI Core", "Inflation", "M", "index", "pct", 1.5),
    "PCEPILFE": FredSeriesSpec("PCEPILFE", "Core PCE", "Inflation", "M", "index", "pct", 2.0),  # Fed's preferred
    "PPIACO":   FredSeriesSpec("PPIACO", "PPI All Commodities", "Inflation", "M", "index", "pct", 0.8),
    
    # Inflation expectations
    "T5YIE":  FredSeriesSpec("T5YIE",  "5Y Breakeven", "Inflation", "D", "rate", "bps", 1.2),
    "T10YIE": FredSeriesSpec("T10YIE", "10Y Breakeven", "Inflation", "D", "rate", "bps", 1.0),
    
    # === INFLATION COMPONENTS (for nowcast) ===
    "CUSR0000SAH1": FredSeriesSpec("CUSR0000SAH1", "CPI Shelter", "Inflation-Component", "M", "index", "pct", 0.0),
    "CUSR0000SETB01": FredSeriesSpec("CUSR0000SETB01", "CPI Gasoline", "Inflation-Component", "M", "index", "pct", 0.0),
    "CUSR0000SAF11": FredSeriesSpec("CUSR0000SAF11", "CPI Food at Home", "Inflation-Component", "M", "index", "pct", 0.0),
    
    # === LIQUIDITY / FINANCIAL CONDITIONS ===
    "NFCI":   FredSeriesSpec("NFCI", "Chicago Fed NFCI", "Liquidity", "W", "level", "level", 1.5),
    "NFCILEVERAGE": FredSeriesSpec("NFCILEVERAGE", "NFCI Leverage", "Liquidity", "W", "level", "level", 1.0),
    "M2SL":   FredSeriesSpec("M2SL", "M2 Money Stock", "Liquidity", "M", "index", "pct", 0.8),
    "WALCL":  FredSeriesSpec("WALCL", "Fed Balance Sheet", "Liquidity", "W", "level", "level", 1.2),
    
    # === RATES COMPLEX ===
    # Nominal yields
    "DFF":    FredSeriesSpec("DFF", "Fed Funds Effective", "Rates", "D", "rate", "bps", 1.0),
    "DGS3MO": FredSeriesSpec("DGS3MO", "3M Treasury", "Rates", "D", "rate", "bps", 1.0),
    "DGS2":   FredSeriesSpec("DGS2", "2Y Treasury", "Rates", "D", "rate", "bps", 1.5),
    "DGS10":  FredSeriesSpec("DGS10", "10Y Treasury", "Rates", "D", "rate", "bps", 1.5),
    "DGS30":  FredSeriesSpec("DGS30", "30Y Treasury", "Rates", "D", "rate", "bps", 1.0),
    
    # Real yields (CRITICAL for regime)
    "DFII5":  FredSeriesSpec("DFII5", "5Y Real Yield", "Rates", "D", "rate", "bps", 1.2),
    "DFII10": FredSeriesSpec("DFII10", "10Y Real Yield", "Rates", "D", "rate", "bps", 2.0),
    
    # === CREDIT COMPLEX ===
    "BAMLC0A0CM": FredSeriesSpec("BAMLC0A0CM", "IG Corp OAS", "Credit", "D", "rate", "bps", 1.5),
    "BAMLH0A0HYM2": FredSeriesSpec("BAMLH0A0HYM2", "HY Corp OAS", "Credit", "D", "rate", "bps", 2.0),
    "BAMLC0A4CBBB": FredSeriesSpec("BAMLC0A4CBBB", "BBB Corp OAS", "Credit", "D", "rate", "bps", 1.2),
    
    # === LABOR MARKET DEPTH ===
    "ICSA":   FredSeriesSpec("ICSA", "Initial Claims", "Growth", "W", "level", "level", 1.0),
    "CCSA":   FredSeriesSpec("CCSA", "Continued Claims", "Growth", "W", "level", "level", 0.8),
    
    # === SHADING ===
    "USREC": FredSeriesSpec("USREC", "NBER Recession", "Macro", "M", "level", "level", 0.0),
}

# Core series for regime engine
REGIME_GROWTH_KEYS = ["INDPRO", "RSAFS", "PAYEMS", "UNRATE", "AWHMAN"]
REGIME_INFLATION_KEYS = ["CPILFESL", "PCEPILFE", "T5YIE", "T10YIE"]
REGIME_LIQUIDITY_KEYS = ["NFCI", "DFII10", "BAMLH0A0HYM2", "DGS2"]

# Nowcast input series
NOWCAST_GDP_INPUTS = ["INDPRO", "RSAFS", "PAYEMS", "AWHMAN", "MANEMP"]
NOWCAST_INFLATION_INPUTS = ["CPIAUCSL", "CPILFESL", "PCEPILFE", "PPIACO"]

# Default load set
DEFAULT_FRED_IDS = list(FRED_SERIES.keys())

# -----------------------------
# Market Assets - Factor/Sector Framework
# -----------------------------
YF_ASSETS: Dict[str, AssetSpec] = {
    # === EQUITY INDICES ===
    "SPY": AssetSpec("SPY", "S&P 500", "Equity-Index"),
    "QQQ": AssetSpec("QQQ", "Nasdaq 100", "Equity-Index"),
    "IWM": AssetSpec("IWM", "Russell 2000", "Equity-Index"),
    "EFA": AssetSpec("EFA", "EAFE (Intl)", "Equity-Index"),
    
    # === SECTOR SPDR ===
    "XLY": AssetSpec("XLY", "Discretionary", "Equity-Sector", beta_benchmark="SPY"),
    "XLP": AssetSpec("XLP", "Staples", "Equity-Sector", beta_benchmark="SPY"),
    "XLE": AssetSpec("XLE", "Energy", "Equity-Sector", beta_benchmark="SPY"),
    "XLF": AssetSpec("XLF", "Financials", "Equity-Sector", beta_benchmark="SPY"),
    "XLV": AssetSpec("XLV", "Healthcare", "Equity-Sector", beta_benchmark="SPY"),
    "XLI": AssetSpec("XLI", "Industrials", "Equity-Sector", beta_benchmark="SPY"),
    "XLB": AssetSpec("XLB", "Materials", "Equity-Sector", beta_benchmark="SPY"),
    "XLRE": AssetSpec("XLRE", "Real Estate", "Equity-Sector", beta_benchmark="SPY"),
    "XLK": AssetSpec("XLK", "Technology", "Equity-Sector", beta_benchmark="SPY"),
    "XLC": AssetSpec("XLC", "Communication", "Equity-Sector", beta_benchmark="SPY"),
    "XLU": AssetSpec("XLU", "Utilities", "Equity-Sector", beta_benchmark="SPY"),
    
    # === FACTOR PROXIES ===
    # Research emphasizes: "factor leadership rotate"
    "MTUM": AssetSpec("MTUM", "Momentum Factor", "Equity-Factor"),
    "QUAL": AssetSpec("QUAL", "Quality Factor", "Equity-Factor"),
    "SIZE": AssetSpec("SIZE", "Size Factor", "Equity-Factor"),
    "VLUE": AssetSpec("VLUE", "Value Factor", "Equity-Factor"),
    "USMV": AssetSpec("USMV", "Low Vol Factor", "Equity-Factor"),
    
    # === RATES / CREDIT PROXIES ===
    "TLT": AssetSpec("TLT", "20Y+ Treasury", "Rates"),
    "IEF": AssetSpec("IEF", "7-10Y Treasury", "Rates"),
    "SHY": AssetSpec("SHY", "1-3Y Treasury", "Rates"),
    "LQD": AssetSpec("LQD", "IG Credit", "Credit"),
    "HYG": AssetSpec("HYG", "High Yield", "Credit"),
    "EMB": AssetSpec("EMB", "EM Debt", "Credit"),
    
    # === COMMODITIES ===
    "GLD": AssetSpec("GLD", "Gold", "Commodities"),
    "SLV": AssetSpec("SLV", "Silver", "Commodities"),
    "USO": AssetSpec("USO", "WTI Crude", "Commodities"),
    "UNG": AssetSpec("UNG", "Natural Gas", "Commodities"),
    "DBA": AssetSpec("DBA", "Agriculture", "Commodities"),
    
    # === FX / VOL ===
    "UUP": AssetSpec("UUP", "USD Index", "FX"),
    "VIX": AssetSpec("^VIX", "VIX", "Volatility"),
    "VVIX": AssetSpec("^VVIX", "Vol of Vol", "Volatility"),
}

DEFAULT_ASSET_KEYS = list(YF_ASSETS.keys())

# -----------------------------
# Transmission Pairs - Enhanced
# -----------------------------
TRANSMISSION_PAIRS: List[Tuple[str, str, str, str]] = [
    # Growth → Cyclicals
    ("RSAFS", "XLY", "XLP", "Retail Sales → Consumer Risk (XLY/XLP)"),
    ("INDPRO", "XLI", "SPY", "Industrial Production → Industrials/SPY"),
    ("PAYEMS", "IWM", "SPY", "Payrolls → Small Cap Risk (IWM/SPY)"),
    
    # Inflation → Real Assets
    ("CPILFESL", "GLD", "TLT", "Core CPI → Gold/Bonds"),
    ("T10YIE", "XLE", "SPY", "Breakevens → Energy/SPY"),
    
    # Liquidity → Risk Assets
    ("DFII10", "QQQ", "SPY", "Real Yields → Tech/SPY"),
    ("BAMLH0A0HYM2", "HYG", "LQD", "HY Spreads → Credit Risk"),
]

# -----------------------------
# Curve Shapes for Regime Detection
# Research: "yield curve is a direct reflection of the macro regime"
# -----------------------------
CURVE_PAIRS = [
    ("DGS2", "DGS10", "2s10s"),
    ("DGS2", "DGS30", "2s30s"),
    ("DGS3MO", "DGS10", "3m10s"),
]

# -----------------------------
# Nowcast Model Weights
# Based on GDP component contributions and empirical fit
# -----------------------------
GDP_NOWCAST_WEIGHTS = {
    # Series: (weight, GDP_component, transform)
    # Weights roughly based on GDP composition and leading properties
    "INDPRO": (0.15, "Investment", "mom_annualized"),   # Industrial production
    "RSAFS":  (0.35, "Consumption", "mom_annualized"),  # Retail sales (PCE proxy)
    "PAYEMS": (0.25, "Labor", "mom_annualized"),        # Employment momentum
    "AWHMAN": (0.10, "Labor", "level"),                 # Hours worked (intensity)
    "MANEMP": (0.15, "Manufacturing", "mom_annualized"), # Manufacturing employment
}

INFLATION_NOWCAST_WEIGHTS = {
    # For CPI nowcast
    "CUSR0000SAH1": (0.33, "Shelter"),      # ~33% of CPI
    "CUSR0000SETB01": (0.05, "Energy"),     # Gasoline
    "CUSR0000SAF11": (0.08, "Food"),        # Food at home
    # Core services ex-shelter estimated from wage growth
}

# Sector ETFs
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Factor ETFs
FACTOR_ETFS = {
    "MTUM": "Momentum",
    "VLUE": "Value",
    "QUAL": "Quality",
    "SIZE": "Size",
    "USMV": "Low Volatility",
    "SPHB": "High Beta",
}
=======
    category: str              # Equity / Rates / FX / Commodities
    unit: str = "price"

# -----------------------------
# FRED series universe
# NOTE: For monthly series (M), we will shift dates to month-end in data.py
# -----------------------------
FRED_SERIES: Dict[str, FredSeriesSpec] = {
    # Growth
    "INDPRO": FredSeriesSpec("INDPRO", "Industrial Production", "Growth", "M", "index", "pct"),
    "RSAFS":  FredSeriesSpec("RSAFS",  "Retail Sales",          "Growth", "M", "index", "pct"),
    "UNRATE": FredSeriesSpec("UNRATE", "Unemployment Rate",     "Growth", "M", "rate",  "bps"),

    # Inflation
    "CPIAUCSL": FredSeriesSpec("CPIAUCSL", "CPI (Headline, NSA)", "Inflation", "M", "index", "pct"),
    "PCEPILFE": FredSeriesSpec("PCEPILFE", "Core PCE",            "Inflation", "M", "index", "pct"),
    "T10YIE":   FredSeriesSpec("T10YIE",   "10Y Breakeven Inflation", "Inflation", "D", "rate", "bps"),

    # Liquidity / conditions
    "NFCI": FredSeriesSpec("NFCI", "Chicago Fed NFCI", "Liquidity", "W", "level", "level"),
    "M2SL": FredSeriesSpec("M2SL", "M2 Money Stock",   "Liquidity", "M", "index", "pct"),

    # Rates / credit
    "DGS2":   FredSeriesSpec("DGS2",   "UST 2Y Yield",      "Rates",  "D", "rate", "bps"),
    "DGS10":  FredSeriesSpec("DGS10",  "UST 10Y Yield",     "Rates",  "D", "rate", "bps"),
    "DFII10": FredSeriesSpec("DFII10", "UST 10Y Real Yield","Rates",  "D", "rate", "bps"),
    "BAMLH0A0HYM2": FredSeriesSpec("BAMLH0A0HYM2", "High Yield OAS", "Credit", "D", "rate", "bps"),

    # Macro shading
    "USREC": FredSeriesSpec("USREC", "NBER Recession Indicator", "Macro", "M", "level", "level"),
}

DEFAULT_FRED_IDS: List[str] = [
    "INDPRO", "CPIAUCSL", "NFCI",
    "RSAFS", "UNRATE", "PCEPILFE",
    "M2SL", "T10YIE",
    "DGS2", "DGS10", "DFII10",
    "BAMLH0A0HYM2",
    "USREC",
]

# -----------------------------
# yfinance assets used for overlays / proxies
# -----------------------------
YF_ASSETS: Dict[str, AssetSpec] = {
    "SPY": AssetSpec("SPY", "S&P 500 (SPY)", "Equity"),
    "QQQ": AssetSpec("QQQ", "Nasdaq 100 (QQQ)", "Equity"),
    "XLI": AssetSpec("XLI", "Industrials (XLI)", "Equity"),
    "XLE": AssetSpec("XLE", "Energy (XLE)", "Equity"),
    "XLY": AssetSpec("XLY", "Cons Discretionary (XLY)", "Equity"),
    "XLP": AssetSpec("XLP", "Cons Staples (XLP)", "Equity"),
    "XRT": AssetSpec("XRT", "Retail (XRT)", "Equity"),
    "GLD": AssetSpec("GLD", "Gold (GLD)", "Commodities"),
    "WTI": AssetSpec("CL=F", "WTI Crude (front)", "Commodities"),
    "DXY": AssetSpec("DX-Y.NYB", "DXY (ICE)", "FX"),
    "VIX": AssetSpec("^VIX", "VIX", "FX"),
}

DEFAULT_ASSET_KEYS: List[str] = ["SPY", "QQQ", "XLI", "XLE", "XLY", "XLP", "XRT", "GLD", "WTI", "DXY", "VIX"]

# -----------------------------
# Transmission pairs (macro vs relative market proxy)
# (macro_key, numerator_asset, denominator_asset, title)
# -----------------------------
TRANSMISSION_PAIRS: List[Tuple[str, str, str, str]] = [
    ("RSAFS",  "XRT", "SPY", "Retail Sales vs Retail (relative)"),
    ("INDPRO", "XLI", "SPY", "Industrial Production vs Industrials (relative)"),
    ("UNRATE", "XLY", "XLP", "Consumer Risk Appetite (XLY/XLP)"),
]
>>>>>>> 95bbab53f6b0c4f11a35206ef5578faa2ed53401
