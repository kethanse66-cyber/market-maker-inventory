# 🌍 Cross-Sectional Market-Neutral Statistical Arbitrage Engine
### Global Factor Model | Russell 1000 + STOXX 600 + Nikkei 225 | 2010–2024

<div align="center">

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Progress](https://img.shields.io/badge/Progress-Month%201%20of%209-orange)

*Built from Chitradurga, Karnataka, India — targeting global quant research roles*

</div>

---

## 📌 Project Overview

This repository documents the complete 9-month journey of building an **institutional-grade cross-sectional factor model** on global developed market equities — from Python beginner to published quant researcher.

**Universe:** ~2,000 stocks across Russell 1000 (US) + STOXX 600 (Europe) + Nikkei 225 (Japan)  
**Period:** 2010 – 2024 (14 years of out-of-sample history)  
**Factors:** Momentum, Value, Quality, Low Volatility, Short-term Reversal  
**Portfolio:** Long-short, dollar-neutral, sector-neutral, beta-neutralized  
**Validation:** Walk-forward (5 splits), Stress tested (4 regimes: 2008, 2015, 2020, 2022)  

---

## 🗂️ Repository Structure

```
global-quant-factor-model/
├── data/
│   ├── fetch_data.py           # Download US, Europe, Asia stock data
│   ├── fetch_fundamentals.py   # P/E, P/B, ROE from SimFin API
│   └── fetch_fx.py             # FX rates from FRED API
├── factors/
│   ├── momentum.py             # 12-1 month cross-sectional momentum
│   ├── value.py                # Earnings yield, Book-to-Price
│   ├── quality.py              # ROE, Debt/Equity, earnings stability
│   ├── low_volatility.py       # Realized volatility, beta
│   └── reversal.py             # 1-month short-term reversal
├── backtest/
│   ├── engine.py               # Clean backtesting engine — zero lookahead bias
│   ├── ic_analysis.py          # Information Coefficient calculations
│   └── walk_forward.py         # Walk-forward validation framework
├── portfolio/
│   ├── optimizer.py            # CVXPY optimization — dollar+sector neutral
│   ├── risk_engine.py          # Beta neutralization, VaR, CVaR
│   └── transaction_costs.py    # 7-10 bps cost modeling
├── analysis/
│   ├── stress_test.py          # 2008, 2020, 2022, 2015 regimes
│   ├── attribution.py          # Factor contribution analysis
│   └── capacity.py             # Strategy capacity analysis
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── paper/
│   └── research_paper.pdf      # 12-15 page academic research paper
└── README.md
```

---

## 📊 The 5 Alpha Factors

| Factor | Signal | Formula | IC Target |
|--------|--------|---------|-----------|
| **Momentum** | 12-1 Month Return | Return(t-12, t-1) | 0.05–0.09 |
| **Value** | Earnings Yield | EPS / Price | 0.03–0.06 |
| **Quality** | Return on Equity | Net Income / Equity | 0.03–0.06 |
| **Low Volatility** | Realized Volatility | Std(returns, 252d) | 0.03–0.07 |
| **Reversal** | 1-Month Return | Return(t-1, t) | 0.02–0.05 |

---

## 🎯 Target Performance Metrics

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Annual Return (Net) | 8–12% | 12–18% | 20%+ |
| Sharpe Ratio | 0.6–0.8 | 0.8–1.2 | 1.5+ |
| Max Drawdown | < -30% | < -20% | < -12% |
| Average IC | 0.03–0.05 | 0.05–0.08 | 0.08+ |
| Market Correlation | < 0.4 | < 0.25 | < 0.1 |
| Win Rate (monthly) | 51–53% | 54–57% | 58%+ |

---

## 🗓️ 9-Month Build Journey

This project is being built and documented in real-time. Each month represents a new milestone.

---

### ✅ Month 1 | Feb 20 – Mar 20 | PYTHON FOUNDATIONS
> **Status: 🔄 In Progress**

**Goal:** Go from Python beginner to finance-ready in 30 days

**What I am building:**
- Python core — variables, loops, functions, lists, dictionaries
- Pandas + NumPy — data manipulation, slicing, filtering, aggregation
- yFinance integration — download real stock data
- Statistics in Python — mean, std, correlation, distributions on real stocks

**Milestone:** Can calculate returns, volatility, Sharpe ratio, correlation matrix from memory

**Key learnings this month:**
- [ ] Python core syntax complete
- [ ] Pandas DataFrame operations
- [ ] First stock data downloaded and plotted
- [ ] Returns and volatility calculated on real data

---

### ⏳ Month 2 | Mar 20 – Apr 20 | FINANCE BRAIN
> **Status: Upcoming**

**Goal:** Understand financial theory + build first backtest

**What I will build:**
- Backtesting fundamentals — understanding lookahead bias
- Factor investing theory — Momentum, Value, Quality, Low Vol
- Portfolio mathematics — Sharpe, Sortino, Max Drawdown, CAGR, IC
- Mini Backtest #1 — simple momentum strategy

**Milestone:** First real backtest running — momentum working on US stocks

---

### ⏳ Month 3 | Apr 20 – May 20 | BACKTESTING ENGINE
> **Status: Upcoming**

**Goal:** Build clean, professional backtesting engine

**What I will build:**
- BacktestEngine class with zero lookahead bias
- Cross-sectional ranking framework
- Momentum factor (12-1 month) with IC testing
- Value factor (earnings yield, book-to-price) with IC testing

**Milestone:** Clean engine + Momentum + Value factors with IC analysis

---

### ⏳ Month 4 | May 20 – Jun 20 | FACTOR RESEARCHER
> **Status: Upcoming**

**Goal:** Complete multi-factor research — 5 factors with full IC analysis

**What I will build:**
- Quality factor — ROE, debt/equity, earnings stability
- Low Volatility factor — realized vol (252-day), beta
- Short-term Reversal factor — 1-month mean reversion
- Rolling IC analysis for all 5 factors
- Factor correlation matrix

**Milestone:** 5 factors built, IC calculated, rolling IC charts complete

---

### ⏳ Month 5 | Jun 20 – Jul 20 | PORTFOLIO CONSTRUCTOR
> **Status: Upcoming**

**Goal:** Institutional portfolio construction + Global market expansion

**What I will build:**
- Long-short market neutral portfolio with CVXPY optimization
- Dollar neutral + Sector neutral constraints
- Transaction cost modeling (7–10 basis points)
- European stocks (STOXX 600 via yfinance)
- Asian stocks (Nikkei 225 via yfinance)
- Currency adjustment — all returns converted to USD via FRED FX data

**Milestone:** Full global long-short portfolio: US + Europe + Asia, dollar+sector neutral

---

### ⏳ Month 6 | Jul 20 – Aug 20 | RISK MANAGER
> **Status: Upcoming**

**Goal:** Complete risk engine — stress testing, beta neutralization, walk-forward

**What I will build:**
- Beta neutralization — remove market exposure
- Walk-forward validation across 5 out-of-sample periods
- Stress test: 2008 Global Financial Crisis
- Stress test: 2020 COVID Crash
- Stress test: 2022 Rate Hike Bear Market
- Stress test: 2015 China Shock

**Milestone:** Full risk engine, walk-forward validated, stress tested across 4 regimes

---

### ⏳ Month 7 | Aug 20 – Sep 20 | RESEARCH PAPER + DASHBOARD
> **Status: Upcoming**

**Goal:** Document everything professionally — research paper + live dashboard

**What I will build:**
- 12–15 page academic research paper
- Failure analysis section — most critical part
- Streamlit interactive dashboard
- Publish paper on SSRN

**Milestone:** SSRN paper published, dashboard live, GitHub repository complete

---

### ⏳ Month 8 | Sep 20 – Oct 20 | INTERVIEW PREPARATION
> **Status: Upcoming**

**Goal:** Convert preparation into job applications

**What I will do:**
- Polish all code — clean, documented, professional
- Factor attribution analysis complete
- 50+ job applications sent
- Mock interviews recorded and reviewed

**Milestone:** 50+ applications sent, first interview requests

---

### ⏳ Month 9 | Oct 20 – Nov 30 | CLOSE THE DEAL
> **Status: Upcoming**

**Goal:** 100+ applications, active interview pipeline, first offers

**What I will do:**
- 100+ total applications sent
- Active interview rounds at multiple firms
- Final GitHub cleanup
- Visa process begins

**Milestone:** Job offer in hand. Life changed.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core programming language |
| **Pandas + NumPy** | Data manipulation and numerical computing |
| **yFinance** | Stock price data (US, Europe, Asia) |
| **SimFin API** | Fundamental data (EPS, Book Value, ROE) |
| **FRED API** | Macro data (risk-free rate, FX rates) |
| **CVXPY** | Portfolio optimization with constraints |
| **Matplotlib + Seaborn** | Visualization |
| **Streamlit** | Interactive dashboard |
| **Kenneth French Library** | Factor validation and benchmarking |

---

## 📚 Research Foundation

### Books
| Book | Author | Purpose |
|------|--------|---------|
| A Practical Guide to Quantitative Finance Interviews | Xinfeng Zhou | Interview preparation |
| Advances in Financial Machine Learning | Marcos Lopez de Prado | Backtesting methodology |
| Quantitative Equity Portfolio Management | Chincarini & Kim | Factor model framework |
| Your Complete Guide to Factor-Based Investing | Berkin & Swedroe | Factor intuition |
| Active Portfolio Management | Grinold & Kahn | IC, IR, portfolio theory |

### Key Academic Papers
| Paper | Authors | Year |
|-------|---------|------|
| Common Risk Factors in the Returns on Stocks and Bonds | Fama & French | 1993 |
| Returns to Buying Winners and Selling Losers | Jegadeesh & Titman | 1993 |
| Value and Momentum Everywhere | Asness, Moskowitz & Pedersen | 2013 |
| Betting Against Beta | Frazzini & Pedersen | 2014 |
| Quality Minus Junk | Asness, Frazzini & Pedersen | 2019 |
| A Five-Factor Asset Pricing Model | Fama & French | 2015 |

---

## 📈 Progress Tracker

| Month | Technical | English | Career | Status |
|-------|-----------|---------|--------|--------|
| 1 | Python Foundations | Grammar Basics | LinkedIn Built | 🔄 In Progress |
| 2 | Backtesting Engine | Reading Habit | First LinkedIn Post | ⏳ |
| 3 | 5 Factors + IC | Speaking Practice | Networking Starts | ⏳ |
| 4 | Portfolio Optimization | Writing Practice | 5 Connections/Week | ⏳ |
| 5 | Global Markets | Pronunciation | Resume Draft | ⏳ |
| 6 | Risk Engine | Interview English | Mock Interviews | ⏳ |
| 7 | Research Paper | Academic Writing | SSRN Paper Live | ⏳ |
| 8 | Polish + Attribution | Presentation | 50+ Applications | ⏳ |
| 9 | Final Documentation | Negotiation English | Offers Incoming | ⏳ |

---

## 🌐 Target Markets

This project targets quant research roles in:

| Location | Why |
|----------|-----|
| 🇦🇪 **Dubai, UAE** | Zero income tax, fastest visa, growing quant ecosystem |
| 🇸🇬 **Singapore** | Major Asian financial hub, strong quant demand |
| 🇬🇧 **London, UK** | Global quant capital, Man AHL, Winton, hedge fund cluster |
| 🇺🇸 **USA** | Citadel, Two Sigma, DE Shaw, AQR |

---

## 📝 Research Paper (Month 7)

The research paper will be published on **SSRN** and will follow this structure:

1. Abstract
2. Introduction — Why factor investing, why global markets
3. Literature Review — Fama-French, AQR, Jegadeesh-Titman
4. Data & Universe — Stocks, period, sources, survivorship bias handling
5. Factor Construction — Exact methodology for all 5 factors
6. Empirical Results — IC tables, rolling IC charts, factor returns
7. Portfolio Construction — Optimization methodology, constraints
8. Risk Analysis — Stress test results, drawdown analysis
9. **Failure Analysis** — When and why strategy fails *(most important section)*
10. Conclusion — Summary, limitations, future research

> *"Any junior can show good backtest numbers. Only someone who truly understands the work can explain when and why it fails."*

---

## ⚠️ Known Limitations (Intellectual Honesty)

This project uses free data sources which come with limitations I am fully aware of:

- **Survivorship bias:** yFinance does not provide point-in-time constituent lists. This may inflate backtest returns. This limitation is explicitly acknowledged and discussed in the research paper.
- **Data quality:** Free APIs may have occasional data errors. Cleaning steps are documented in the code.
- **Market impact:** Transaction cost model uses flat 7–10 bps. Actual market impact for large orders would be higher.
- **Fundamental data lag:** SimFin data may not perfectly replicate point-in-time reporting. Earnings announcement dates are approximated.

These limitations do not invalidate the research — they define its scope and are clearly disclosed.

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/yourusername/global-quant-factor-model.git
cd global-quant-factor-model

# Install dependencies
pip install -r requirements.txt

# Fetch data
python data/fetch_data.py

# Run backtest
python backtest/engine.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 📬 Contact & Connect

**Currently building:** Cross-Sectional Global Factor Model  
**Location:** Chitradurga, Karnataka, India  
**Target:** Quant Research roles in Dubai / Singapore / London / USA  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile)
[![SSRN](https://img.shields.io/badge/SSRN-Research%20Paper-red)](https://ssrn.com)
[![Email](https://img.shields.io/badge/Email-Contact-green)](mailto:youremail@gmail.com)

---

## 📅 Journey Started

**February 20, 2025** — Day 1. Python beginner. Chitradurga, Karnataka, India.  
**November 30, 2025** — Target completion date.

*1,750+ hours. 9 months. 4 target countries. Built from scratch.*

---

<div align="center">

*"The stars have written the destination. You do the walking."*

**Updated monthly as the project progresses.**

</div>
