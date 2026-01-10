# Top-Down Macro Strategy Dashboard (GIP + Transmission)

A Streamlit dashboard that turns **raw macro + market data** into a consistent set of **levels, momentum, and “what’s driving what”** views.

It’s designed to feel like a lightweight “macro terminal”:
- **GIP regime** snapshot (Growth / Inflation / Liquidity)
- **Transmission** from macro → sectors (relative performance + rolling beta)
- **Rates drivers** (real yields, curve, credit, USD, vol)
- **Commodities** (macro proxies like real yields ↔ gold)
- **Scoreboard** that standardizes everything into comparable metrics

> **Important:** This is a **visual + analytical monitoring tool**, not a trading system. It does not handle execution, position sizing, or portfolio constraints.

---

## What you’ll see

### Tabs
- **GIP Regime**
  - Growth proxy (e.g., Industrial Production YoY)
  - Inflation proxy (e.g., CPI YoY)
  - Liquidity / financial conditions (e.g., Chicago Fed NFCI)
  - “Current regime” classification (baseline quadrant logic)

- **Sector Transmission**
  - Macro series vs sector *relative* performance
  - “Risk appetite” ratios (e.g., **XLY/XLP**)
  - **Rolling beta (monthly)** to show conditional sensitivity over time

- **Rates Drivers**
  - Real yields vs duration equity (e.g., DFII10 vs QQQ)
  - Yield curve (10Y–2Y)
  - Credit spreads, DXY, VIX (month-end aligned)

- **Commodities**
  - Real yields vs gold (DFII10 vs GLD)
  - Oil vs energy equities *relative* (WTI vs XLE/SPY)

- **Scoreboard**
  - A standardized table: **level**, **1-period change**, **3-period change**, **YoY**, **z-score**, **percentile**, latest date

---

## Data Sources
- **FRED (fredapi)** for macro and rates series
- **yfinance** for ETFs / proxies (QQQ, GLD, XLE, SPY, XLY, XLP, etc.)

All series definitions live in `config.py` (FRED IDs + Yahoo tickers + labels + categories + frequency).

---

## Installation & Usage

### 1) Clone + create environment
```bash
git clone <your_repo_url>
cd global_macro
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure your FRED API key

**Option A (Streamlit secrets, recommended):**
Create `.streamlit/secrets.toml`:

```toml
FRED_API_KEY = "YOUR_KEY_HERE"
```

**Option B (environment variable):**

```bash
# Windows (PowerShell)
$env:FRED_API_KEY="YOUR_KEY_HERE"
# Mac/Linux
export FRED_API_KEY="YOUR_KEY_HERE"
```

### 4) Run locally

```bash
streamlit run app.py
```

---

## How the dashboard calculates everything

### 1) Frequency-aware transforms (core idea)

Every series is stored with:

* a time index
* a frequency label (**D/W/M/Q**)
* a “unit” label (level, rate, index, price)

Because “3 months” means different things depending on frequency, the code uses **frequency-aware period mapping**:

| Frequency     | 1-period | ~3-period | YoY |
| ------------- | -------: | --------: | --: |
| Monthly (M)   |        1 |         3 |  12 |
| Weekly (W)    |        4 |        13 |  52 |
| Daily (D)     |       21 |        63 | 252 |
| Quarterly (Q) |        1 |         2 |   4 |

This prevents common mistakes like treating “3 months” as “3 observations” for daily data.

---

### 2) Momentum transforms (Chg_1, Chg_3, YoY)

For each series (X_t), the dashboard calculates:

* **Chg_1 (%):**
  $$\text{Chg}*{1} = 100\left(\frac{X_t}{X*{t-k_1}} - 1\right)$$

* **Chg_3 (%):**
  $$\text{Chg}*{3} = 100\left(\frac{X_t}{X*{t-k_3}} - 1\right)$$

* **YoY (%):**
  $$\text{YoY} = 100\left(\frac{X_t}{X_{t-k_y}} - 1\right)$$

Where (k_1, k_3, k_y) depend on frequency (daily/weekly/monthly/quarterly).

> Notes:
>
> * For **rate series** (e.g., yields), it’s often more interpretable to track **bps change**:
>   $$\Delta_{bps} = 100 \cdot (X_t - X_{t-k})$$
>   Your scoreboard labels this in `Unit(Chg/YoY)` so the PM knows whether it’s **%** or **bps**.

---

### 3) Normalized overlays (index-to-100)

To compare two level series on one chart, the dashboard uses a base index:

$$X^{(norm)}*t = 100 \cdot \frac{X_t}{X*{t_0}}$$

Where (t_0) is the first timestamp after the selected start date.

---

### 4) “Relative” series (to remove broad beta)

For sector transmission, the dashboard often uses *relative performance* (ratio form):

$$\text{Rel}_{A/B,t} = 100 \cdot \frac{A_t}{B_t}$$

Examples:

* **XLE/SPY** = energy vs broad market (equity “belief” about oil exposure)
* **XLY/XLP** = discretionary vs staples (consumer risk appetite)

> Why: this strips out broad market direction and makes “tilt” visible.

---

### 5) Month-end alignment (avoids false precision)

Macro is mostly monthly. ETFs are daily. To avoid misleading point-by-point overlap, market series are often resampled to **month-end**:

$$P^{(ME)}*t = P*{\text{last trading day of month}}$$

The “Transmission” and “Rates/Commodities” pages emphasize **month-end aligned** comparisons.

---

### 6) Rolling correlation (optional)

If enabled, the dashboard computes rolling correlation on aligned returns:

$$\rho_{t,w} = \text{Corr}\left(r^{(A)}*{t-w+1:t}, r^{(B)}*{t-w+1:t}\right)$$

Where (w) is the rolling window in months (e.g., 24).

---

### 7) Rolling beta (monthly)

Rolling beta answers: *“How sensitive is asset/relative-return (Y) to factor (X) right now?”*

Compute aligned monthly returns:
$$r^X_t = \ln\left(\frac{X_t}{X_{t-1}}\right), \quad r^Y_t = \ln\left(\frac{Y_t}{Y_{t-1}}\right)$$

Then run a rolling regression:
$$r^Y_t = \alpha + \beta r^X_t + \epsilon_t$$

Beta is:
$$\beta_{t,w} = \frac{\text{Cov}(r^Y_{t-w+1:t}, r^X_{t-w+1:t})}{\text{Var}(r^X_{t-w+1:t})}$$

> This is primarily used in the **Transmission** tab (macro → sector relative), but it’s also valid wherever you have two aligned return series.

---

### 8) Scoreboard standardization (Z-score + percentile)

To make series comparable, the dashboard computes rolling z-scores (on YoY or chosen transform):

$$z_t = \frac{X_t - \mu}{\sigma}$$

Where (\mu) and (\sigma) are computed over a lookback window (e.g., 5–10y).

Percentile is computed as the empirical rank of (X_t) within the historical window.

---

### 9) Shaded regions (what they represent)

The shaded vertical bands represent **NBER recession periods** (when available), typically pulled from a recession indicator series (e.g., USREC on FRED).

Logic:

* If recession flag = 1, shade the corresponding dates on all charts.
* Purpose: provides historical context for how indicators and markets behave in stress regimes.

---

## Why these metrics 

This dashboard is built around a simple PM workflow:

1. **Level** answers “where are we?”
2. **Momentum (3m, YoY)** answers “what’s changing at the margin?”
3. **Relative + beta** answers “what is the market rewarding / penalizing, and what’s driving it?”
4. **Scoreboard** forces consistency and reduces narrative bias.

Macro trading is rarely about the *level* alone; it’s usually about the *direction and the rate of change* in growth/inflation/liquidity and how those shifts transmit into pricing.

---

## Repo structure (typical)

```
global_macro/
  app.py
  data.py
  config.py
  requirements.txt
  .streamlit/
    secrets.toml   # not committed
```

---

## Disclaimer

This project is for educational and research purposes. It is not financial advice and should not be used for live trading without extensive validation.

```
::contentReference[oaicite:0]{index=0}
```
