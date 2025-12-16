# live_trading/analyze_live_results.py
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os


LOG_PATH = "logs/live/kalshi_live_trades.csv"
OUT_DIR = "reports/figs"
INITIAL_CASH = 10_000.0  # same as KalshiEnvConfig.initial_cash


def load_live_trades(log_path=LOG_PATH):
    timestamps = []
    tickers = []
    actions = []
    sides = []
    prices = []
    cash_snapshots = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # we logged these columns earlier:
            # timestamp, event_ticker, market_ticker, btc_price_proxy,
            # yes_price, no_price, action, side, count, price, portfolio_cash
            timestamps.append(datetime.fromisoformat(row["timestamp"]))
            tickers.append(row["ticker"])
            actions.append(row["action"])
            sides.append(row["side"])
            prices.append(float(row["price"]))
            cash_snapshots.append(float(row["portfolio_value"]))

    return {
        "timestamps": np.array(timestamps),
        "tickers": np.array(tickers),
        "actions": np.array(actions),
        "sides": np.array(sides),
        "prices": np.array(prices),
        "cash": np.array(cash_snapshots),
    }


def compute_equity_series(cash_array):
    """
    For now we approximate equity by portfolio_cash snapshots.
    If you later log full portfolio value (cash + mark-to-market),
    replace this with that field instead.
    """
    return cash_array


def main():
    if not os.path.exists(LOG_PATH):
        print(f"No live log found at {LOG_PATH}")
        return

    data = load_live_trades(LOG_PATH)
    ts = data["timestamps"]
    cash = data["cash"]

    if len(cash) == 0:
        print("No trades logged yet.")
        return

    equity = compute_equity_series(cash)
    pnl = equity - INITIAL_CASH
    returns = pnl / INITIAL_CASH

    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret if std_ret > 1e-8 else np.nan

    print("\n=== LIVE TRADING RESULTS ===")
    print(f"Number of logged trades: {len(equity)}")
    print(f"Final equity: {equity[-1]:.2f} $")
    print(f"Total PnL: {pnl[-1]:.2f} $")
    print(f"Mean PnL per trade snapshot: {mean_pnl:.4f} $")
    print(f"Std PnL: {std_pnl:.4f} $")
    print(f"Mean return: {mean_ret:.6f}")
    print(f"Std return: {std_ret:.6f}")
    print(f"Approx. Sharpe (per snapshot): {sharpe:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Plot equity over time
    plt.figure(figsize=(10, 5))
    plt.plot(ts, equity, marker="o", linestyle="-")
    plt.axhline(INITIAL_CASH, linestyle="--", linewidth=1, label="Initial cash")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("Live Demo Trading: Equity Over Time")
    plt.legend()
    plt.tight_layout()
    equity_path = os.path.join(OUT_DIR, "live_equity_curve.png")
    plt.savefig(equity_path)
    plt.close()

    # Histogram of per-step changes in cash
    deltas = np.diff(equity)
    plt.figure(figsize=(8, 5))
    plt.hist(deltas, bins=20)
    plt.xlabel("ΔCash between logged points ($)")
    plt.ylabel("Frequency")
    plt.title("Live Demo Trading: Distribution of Cash Changes")
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, "live_pnl_histogram.png")
    plt.savefig(hist_path)
    plt.close()

    print("\nSaved figures:")
    print("  -", equity_path)
    print("  -", hist_path)


if __name__ == "__main__":
    main()
