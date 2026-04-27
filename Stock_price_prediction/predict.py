"""
Stock Price Predictor — Linear Regression
==========================================
Predicts the next day's closing price of a stock (default: AAPL)
using Open, High, Low, and Volume as features.

Usage:
    python src/predict.py
    python src/predict.py --ticker MSFT --start 2020-01-01 --end 2024-01-01
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    print(f"\n📥 Downloading {ticker} data from {start} to {end} ...")
    stock = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if stock.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol or date range.")
    print(f"   ✅ Loaded {len(stock):,} trading days.\n")
    return stock


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def prepare_features(stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select relevant columns and create the target variable.

    Target  → next day's closing price  (shift by -1)
    Features → Open, High, Low, Volume
    """
    data = stock[["Open", "High", "Low", "Volume", "Close"]].copy()

    # Next-day closing price as target
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)

    X = data[["Open", "High", "Low", "Volume"]]
    y = data["Target"]
    return X, y


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (time-series safe)
# ─────────────────────────────────────────────
def time_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.80):
    """Chronological 80/20 split — no data leakage."""
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(X_train, y_train) -> LinearRegression:
    """Fit a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("🤖 Model trained successfully.")
    print(f"   Coefficients : {dict(zip(X_train.columns, model.coef_.round(6)))}")
    print(f"   Intercept    : {model.intercept_:.4f}\n")
    return model


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and R² metrics."""
    metrics = {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }
    print("📊 Evaluation Metrics:")
    print(f"   MAE  : ${metrics['MAE']:.4f}")
    print(f"   RMSE : ${metrics['RMSE']:.4f}")
    print(f"   R²   : {metrics['R2']:.4f}\n")
    return metrics


# ─────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────
def plot_results(y_test, predictions, ticker: str, save_path: str = None):
    """Plot actual vs predicted closing prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values,  label="Actual Price",    color="#1f77b4", linewidth=1.5)
    plt.plot(predictions,    label="Predicted Price", color="#ff7f0e", linewidth=1.5, linestyle="--")
    plt.title(f"{ticker} — Actual vs Predicted Closing Price", fontsize=15, fontweight="bold")
    plt.xlabel("Trading Days (Test Set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"📈 Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
def run(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    stock          = load_data(ticker, start, end)
    X, y           = prepare_features(stock)
    X_train, X_test, y_train, y_test = time_split(X, y)
    model          = train_model(X_train, y_train)
    predictions    = model.predict(X_test)
    metrics        = evaluate(y_test, predictions)
    plot_results(y_test, predictions, ticker, save_path=f"results/{ticker}_prediction.png")
    return metrics


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Next-Day Price Predictor")
    parser.add_argument("--ticker", default="AAPL",       help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start",  default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-01-01", help="End date   YYYY-MM-DD")
    args = parser.parse_args()
    run(ticker=args.ticker, start=args.start, end=args.end)
