# 📈 Stock Market Performance & Portfolio Risk Analysis

> **Finance & Investment Analytics Project** | Python · Pandas · Matplotlib · Seaborn · NumPy

---

## 📌 Project Overview

Understanding stock performance and portfolio risk is central to investment decision-making. This project analyzes **3,600 OHLCV records** across **10 stocks from 7 sectors** over a 2-year period (2022–2023), computing industry-standard financial metrics — total return, volatility, Sharpe ratio, max drawdown — and presenting them in a **dark-themed executive dashboard** ready for portfolio review.

---

## 🎯 Business Questions Answered

| # | Question |
|---|----------|
| 1 | Which stocks delivered the highest total return? |
| 2 | Which stocks carried the most risk (volatility)? |
| 3 | Which stocks had the best risk-adjusted return (Sharpe Ratio)? |
| 4 | What was the worst loss an investor could have faced (Max Drawdown)? |
| 5 | Are sectors correlated — does diversification actually help? |
| 6 | How did an equal-weight portfolio perform vs individual stocks? |

---

## 📊 Key Findings

- **AMZN** delivered the highest total return (+32.4%) while **PFE** lagged most
- **TSLA** had the highest annualised volatility (55%+) — nearly 3× the portfolio average
- **MSFT** posted the best Sharpe Ratio (> 1.0), meaning best return per unit of risk
- **Technology stocks** are highly correlated (r > 0.75), reducing diversification benefit
- The **equal-weight portfolio** outperformed 6 of 10 individual stocks on a risk-adjusted basis
- **Healthcare stocks** (JNJ, PFE) showed the lowest drawdowns — ideal defensive allocation

---

## 🗂️ Project Structure

```
stock_market_analysis/
│
├── data/
│   ├── generate_data.py       # Synthetic OHLCV data generator (GBM model)
│   └── stock_prices.csv       # Raw dataset (3,600 rows)
│
├── outputs/
│   ├── stock_market_dashboard.png   # 7-panel dark-theme dashboard
│   └── stock_summary.csv            # Per-stock KPI summary table
│
├── analysis.py                # Full analysis + visualisation pipeline
└── README.md
```

---

## 📐 Financial Metrics Computed

| Metric | Formula / Definition |
|--------|----------------------|
| **Daily Return** | `(Close_t - Close_{t-1}) / Close_{t-1}` |
| **Log Return** | `ln(Close_t / Close_{t-1})` |
| **Annualised Volatility** | `std(daily_return) × √252 × 100` |
| **Sharpe Ratio** | `(mean_return - rf) / std(return) × √252` · rf = 4.5% |
| **Max Drawdown** | `min((price - rolling_max) / rolling_max)` |
| **30-Day Moving Average** | Rolling 30-day window on close price |
| **Cumulative Return** | `∏(1 + daily_return)` |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** — OHLCV processing, pivot tables, rolling statistics
- **NumPy** — Geometric Brownian Motion data generation, financial formulas
- **Matplotlib** — Dark-theme 7-panel dashboard, custom KPI banners
- **Seaborn** — Correlation heatmap

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/stock-market-risk-analysis.git
cd stock-market-risk-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn

# Generate dataset
python data/generate_data.py

# Run full analysis
python analysis.py
```

---

## 📈 Dashboard Panels

The script generates a 7-panel dark dashboard (`outputs/stock_market_dashboard.png`):

1. **Cumulative Price Performance** — All 10 stocks normalised to base 100
2. **Total Return Bar Chart** — Winner vs loser comparison
3. **Sharpe Ratio** — Risk-adjusted return with Sharpe=1 benchmark line
4. **Annualised Volatility** — Risk ranking with sector average line
5. **Return Correlation Matrix** — Heatmap showing diversification potential
6. **Portfolio vs Best/Worst Stock** — Equal-weight vs individual extremes
7. **Maximum Drawdown** — Worst-case loss per stock

---

## 💡 Investment Insights / Recommendations

1. **MSFT & AMZN** offer strong risk-adjusted returns — core portfolio candidates
2. **TSLA** is a high-risk/high-reward bet — limit allocation to <5% for conservative portfolios
3. **JNJ & WMT** provide defensive stability — useful during market downturns
4. **Finance stocks (JPM, GS)** are loosely correlated with tech — add diversification value
5. **Equal-weight diversification** reduces max drawdown by ~40% vs holding a single stock

---

## ⚠️ Disclaimer

> This project uses **synthetically generated data** for educational and portfolio purposes only. It does **not** constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## 📬 Contact

Created by **[Your Name]**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
