# agent/analyze_live_results.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TRADES_CSV = "logs/live/kalshi_live_trades.csv"
RESOLUTIONS_CSV = "logs/live/kalshi_resolutions.csv"
OUTCOMES_CSV = "logs/live/live_trade_outcomes.csv"
FIG_DIR = "reports/figs"


def _ensure_dirs():
    os.makedirs(os.path.dirname(OUTCOMES_CSV), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    return df


def compute_trade_outcomes(trades: pd.DataFrame, resolutions: pd.DataFrame) -> pd.DataFrame:
    """
    Approx PnL model (per contract):
      - If you BUY YES at price p (prob in [0,1]):
          payout = 1 if result == "yes" else 0
          pnl_per_contract = payout - p
      - If you BUY NO at price p_no:
          payout = 1 if result == "no" else 0
          pnl_per_contract = payout - p_no

    We use 'limit_price_prob' as entry price (best available in your logs).
    """
    # Normalize timestamps
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    resolutions["timestamp"] = pd.to_datetime(resolutions["timestamp"], utc=True, errors="coerce")

    # Normalize result/status columns
    # resolutions has: ticker, status, result, expiration_time (or effective time in newer version)
    resolutions["result"] = resolutions["result"].astype(str).str.lower()

    # Join on ticker (many trades can exist per ticker; keep the first trade unless you want FIFO)
    # We'll keep ALL trades and attach the same resolution result to each trade of that ticker.
    merged = trades.merge(
        resolutions[["ticker", "result", "status", "expiration_time"]].drop_duplicates("ticker"),
        on="ticker",
        how="left",
        suffixes=("", "_res"),
    )

    # Ensure numeric
    merged["count"] = pd.to_numeric(merged.get("count", 1), errors="coerce").fillna(1).astype(int)
    merged["limit_price_prob"] = pd.to_numeric(merged["limit_price_prob"], errors="coerce")
    merged["portfolio_cash"] = pd.to_numeric(merged.get("portfolio_cash", np.nan), errors="coerce")

    # Determine side & entry price
    merged["side"] = merged["side"].astype(str).str.lower()
    merged["result"] = merged["result"].astype(str).str.lower()

    # payout
    merged["payout_per_contract"] = np.where(
        (merged["side"] == "yes") & (merged["result"] == "yes"),
        1.0,
        np.where(
            (merged["side"] == "no") & (merged["result"] == "no"),
            1.0,
            np.where(merged["result"].isin(["yes", "no"]), 0.0, np.nan),
        ),
    )

    merged["pnl_per_contract"] = merged["payout_per_contract"] - merged["limit_price_prob"]
    merged["pnl"] = merged["pnl_per_contract"] * merged["count"]

    merged["resolved"] = merged["result"].isin(["yes", "no"])
    merged["won"] = np.where(merged["resolved"], merged["pnl_per_contract"] > 0, np.nan)

    # Day buckets
    merged["trade_day"] = merged["timestamp"].dt.floor("D")

    # Order columns nicely
    cols = [
        "timestamp",
        "trade_day",
        "ticker",
        "action_id",
        "side",
        "count",
        "limit_price_prob",
        "result",
        "status",
        "payout_per_contract",
        "pnl_per_contract",
        "pnl",
        "resolved",
        "won",
    ]
    existing = [c for c in cols if c in merged.columns]
    return merged[existing].sort_values("timestamp")


def print_summary(outcomes: pd.DataFrame):
    total = len(outcomes)
    resolved = outcomes["resolved"].sum() if "resolved" in outcomes else 0
    unresolved = total - resolved

    print("\n=== LIVE SUMMARY ===")
    print(f"Trades logged: {total}")
    print(f"Resolved trades: {resolved}")
    print(f"Unresolved trades: {unresolved}")

    if resolved == 0:
        print("No resolved trades yet — run longer or ensure resolution logging includes status='finalized'.")
        return

    res = outcomes[outcomes["resolved"]].copy()
    total_pnl = float(res["pnl"].sum())
    mean_pnl = float(res["pnl"].mean())
    win_rate = float((res["pnl"] > 0).mean())

    print(f"\nTotal PnL (approx): {total_pnl:.4f}")
    print(f"Mean PnL per trade: {mean_pnl:.4f}")
    print(f"Win rate: {win_rate*100:.1f}%")

    # by side
    for side in ["yes", "no"]:
        s = res[res["side"] == side]
        if len(s) == 0:
            continue
        print(f"\nSide={side.upper()}: n={len(s)} | mean pnl={s['pnl'].mean():.4f} | win rate={(s['pnl']>0).mean()*100:.1f}%")

    # daily
    daily = res.groupby("trade_day")["pnl"].sum()
    print("\nDaily PnL:")
    for day, val in daily.items():
        print(f"  {day.date()}: {val:.4f}")


def make_plots(outcomes: pd.DataFrame):
    resolved = outcomes[outcomes["resolved"]].copy()
    if len(resolved) == 0:
        return

    # cumulative pnl by time
    resolved = resolved.sort_values("timestamp")
    resolved["cum_pnl"] = resolved["pnl"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(resolved["timestamp"], resolved["cum_pnl"])
    plt.title("Live Cumulative PnL (approx)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "live_cum_pnl.png"))
    plt.close()

    # daily pnl bar
    daily = resolved.groupby("trade_day")["pnl"].sum().reset_index()
    plt.figure(figsize=(10, 5))
    plt.bar(daily["trade_day"].astype(str), daily["pnl"])
    plt.title("Live Daily PnL (approx)")
    plt.xlabel("Day")
    plt.ylabel("PnL ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "live_daily_pnl.png"))
    plt.close()

    # pnl histogram
    plt.figure(figsize=(8, 4))
    plt.hist(resolved["pnl"], bins=20)
    plt.title("Live Trade PnL Histogram (approx)")
    plt.xlabel("PnL ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "live_trade_pnl_hist.png"))
    plt.close()

    # win rate over time (rolling)
    resolved["win"] = (resolved["pnl"] > 0).astype(int)
    window = min(20, len(resolved))
    if window >= 5:
        resolved["roll_win"] = resolved["win"].rolling(window).mean()
        plt.figure(figsize=(10, 4))
        plt.plot(resolved["timestamp"], resolved["roll_win"])
        plt.ylim(0, 1)
        plt.title(f"Rolling Win Rate (window={window})")
        plt.xlabel("Time")
        plt.ylabel("Win rate")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "live_rolling_winrate.png"))
        plt.close()


def main():
    _ensure_dirs()

    trades = _load_csv(TRADES_CSV)
    resolutions = _load_csv(RESOLUTIONS_CSV)

    outcomes = compute_trade_outcomes(trades, resolutions)
    outcomes.to_csv(OUTCOMES_CSV, index=False)

    print_summary(outcomes)
    make_plots(outcomes)

    print("\nSaved:")
    print(f"  {OUTCOMES_CSV}")
    print(f"  {os.path.join(FIG_DIR, 'live_cum_pnl.png')}")
    print(f"  {os.path.join(FIG_DIR, 'live_daily_pnl.png')}")
    print(f"  {os.path.join(FIG_DIR, 'live_trade_pnl_hist.png')}")
    print(f"  {os.path.join(FIG_DIR, 'live_rolling_winrate.png')}")


if __name__ == "__main__":
    main()
