# config.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# Specs
# -----------------------------
@dataclass(frozen=True)
class FredSeriesSpec:
    id: str
    label: str
    category: str              # Growth / Inflation / Liquidity / Rates / Credit / Macro
    freq: str                  # D / W / M / Q
    unit: str                  # index / rate / level
    chg_unit: str              # pct / bps / level
    notes: str = ""

@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    label: str
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