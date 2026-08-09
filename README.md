# SPY ETF Stock Price Forecasting & Algorithmic Trading Backtesting
## Project Overview
An end-to-end quantitative finance pipeline built with Python, predicting next-day closing prices of SPY (S&P 500 ETF) via 3 machine learning models:
1. Logistic Regression (binary classification for bullish/bearish signals)
2. SVR Support Vector Regression (direct continuous price prediction)
3. LSTM Recurrent Neural Network (time-series deep learning forecasting)

After model evaluation with RMSE and R² metrics, we build a long-only trading strategy based on SVR predicted returns, and implement lightweight manual backtesting to calculate core risk-return indicators: total cumulative return, annualized Sharpe ratio, maximum drawdown.

**Key Features**
- Historical US ETF data download via AkShare
- Technical feature engineering: 10-day & 50-day moving averages
- Strict chronological train/test split (85% train / 15% test, no shuffle to avoid data leakage)
- MinMax normalization for regression and deep learning inputs
- Cross-model prediction visualization
- Full error handling for network, TensorFlow and training exceptions
- Pure manual backtesting (removed backtrader to eliminate environment compatibility errors)
- No Chinese font dependency, compatible with Windows / macOS / Linux / Kaggle

**Project Demo**
Kaggle Interactive Notebook
https://www.kaggle.com/code/jenniferxfl/stock-price-forecasting-algorithmic-backtesting

**SPY ETF Price Forecast**
![Static Preview of stock-price-forecasting](./output/SPY_ETF_Price_Forecast.png)


# Stock Price Forecasting & Algorithmic Backtesting

A modular Python framework for forecasting stock prices and evaluating trading strategies via algorithmic backtesting. Originally developed on [Kaggle](https://www.kaggle.com/code/jenniferxfl/stock-price-forecasting-algorithmic-backtesting), ported to a production-ready repository structure.

## Features

- **Data Ingestion**: Yahoo Finance API (`yfinance`) with local CSV caching
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Forecasting Models**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - LSTM (Deep Learning)
  - Prophet (Meta's time-series model)
- **Backtesting Engine**:
  - Custom strategy class with long/short/cash positions
  - Performance metrics: Sharpe ratio, max drawdown, CAGR, win rate
  - Equity curve visualization
- **Walk-forward validation** for realistic out-of-sample testing

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/stock-price-forecasting-backtesting.git
cd stock-price-forecasting-backtesting
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter lab notebooks/Stock_Price_Forecasting_and_Algorithmic_Backtesting.ipynb
```

### 3. Or Use as a Library

```python
from src.data_loader import load_stock_data
from src.models import LSTMForecaster
from src.backtest import Backtester

# Load data
df = load_stock_data("AAPL", start="2018-01-01", end="2024-12-31")

# Train & predict
model = LSTMForecaster(lookback=60)
model.fit(df["Close"])
predictions = model.predict(df["Close"])

# Backtest a simple MA crossover strategy
bt = Backtester(df, strategy="ma_crossover", fast_window=20, slow_window=50)
results = bt.run()
bt.plot_equity_curve()
print(bt.summary())
```

## Project Structure

| Path | Description |
|------|-------------|
| `notebooks/` | Main Jupyter notebook (Kaggle-equivalent workflow) |
| `src/data_loader.py` | Data fetching & caching from yfinance |
| `src/features.py` | Technical indicator computation |
| `src/models.py` | All forecasting model classes |
| `src/backtest.py` | Strategy engine & performance metrics |
| `src/visualize.py` | Plotting utilities |
| `config.yaml` | Default hyperparameters & ticker list |

## Backtesting Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative % return over period |
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | % of profitable trades |
| CAGR | Compound Annual Growth Rate |
| Volatility | Annualized standard deviation |

## Example Results

> *Example equity curve comparing Buy & Hold vs. MA Crossover vs. LSTM Signal strategy*

![Equity Curve](results/equity_curve_example.png)

| Strategy | Total Return | Sharpe | Max DD | Win Rate |
|----------|-------------|--------|--------|----------|
| Buy & Hold | 187% | 1.12 | -34% | — |
| MA Crossover | 203% | 1.34 | -22% | 58% |
| LSTM Signals | 241% | 1.51 | -19% | 62% |

*(Results are illustrative; actual performance depends on ticker, date range, and transaction costs.)*

## Requirements

See `requirements.txt`. Core dependencies:
- Python ≥ 3.9
- numpy, pandas, scikit-learn
- yfinance, ta
- tensorflow / keras (for LSTM)
- prophet (optional, for Facebook Prophet forecasts)
- matplotlib, seaborn

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past performance does not guarantee future results. Always do your own due diligence before making investment decisions.

## License

MIT License — feel free to fork, modify, and use in your own projects.
