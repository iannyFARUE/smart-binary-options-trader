# env/data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple


# env/data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple


def load_raw_btc_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw BTC candles from Kaggle-style CSV.

    Assumes:
      - 'Timestamp' column = Unix time in SECONDS (start time of 60s window)
      - Columns: Open, High, Low, Close, Volume, ... (case-sensitive from Kaggle)

    Returns a DataFrame indexed by proper datetime, with columns:
      - open, high, low, close, Volume (and any others left as-is)
    """
    df = pd.read_csv(csv_path)

    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' column not found in {df.columns}")

    # Convert Unix seconds to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", origin="unix")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df = df.set_index("Timestamp")

    # Normalize OHLC names to lowercase (what the rest of the code expects)
    rename_map = {}
    for col in df.columns:
        if col.lower() == "open":
            rename_map[col] = "open"
        elif col.lower() == "high":
            rename_map[col] = "high"
        elif col.lower() == "low":
            rename_map[col] = "low"
        elif col.lower() == "close":
            rename_map[col] = "close"

    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}, got {df.columns}")

    # We keep Volume and others if present; they can be useful later
    return df



def build_hourly_kalshi_like_events(
    df: pd.DataFrame,
    trading_hours: Tuple[int, ...] = tuple(range(9, 25)),
) -> pd.DataFrame:
    """
    Given high-frequency BTC OHLC data indexed by datetime,
    build a Kalshi-like hourly event dataset.

    For each day D and hour H in trading_hours:
        - price_at_hour_open  = hourly close at D H:00
        - price_at_hour_close = hourly close at D H+1:00
        - threshold T_H = price_at_hour_open
        - outcome = 1 if price_at_hour_close > T_H else 0
        - yes_price, no_price simulated from short-term momentum

    Returns DataFrame with:
        day, hour, btc_price, yes_price, no_price, outcome
    """
    if "close" not in df.columns:
        raise ValueError("Expected 'close' column in df.")

    # 1) Hourly close series
    hourly_close = df["close"].resample("1h").last().dropna()
    hourly_df = hourly_close.to_frame("close")
    hourly_df["date"] = hourly_df.index.date
    hourly_df["hour"] = hourly_df.index.hour

    rows = []

    for day, group in hourly_df.groupby("date"):
        group = group.sort_index()

        for H in trading_hours:
            ts_H = pd.Timestamp(day) + pd.Timedelta(hours=H)
            ts_H1 = ts_H + pd.Timedelta(hours=1)

            if ts_H not in hourly_df.index or ts_H1 not in hourly_df.index:
                continue

            price_at_hour_open = float(hourly_df.loc[ts_H, "close"])
            price_at_hour_close = float(hourly_df.loc[ts_H1, "close"])

            # Threshold & outcome
            T_H = price_at_hour_open
            outcome = int(price_at_hour_close > T_H)

            # Momentum: use original high-frequency df for 30min lookback
            lookback_minutes = 30
            start_lb = ts_H - pd.Timedelta(minutes=lookback_minutes)
            lb_slice = df.loc[start_lb:ts_H]
            if lb_slice.empty:
                momentum = 0.0
            else:
                price_lb = float(lb_slice["close"].iloc[0])
                momentum = (price_at_hour_open - price_lb) / price_lb

            alpha = 5.0
            p_yes = 0.5 + alpha * momentum
            p_yes = float(np.clip(p_yes, 0.05, 0.95))
            p_no = 1.0 - p_yes

            rows.append(
                {
                    "day": day,       # actual date
                    "hour": H,
                    "btc_price": price_at_hour_open,
                    "yes_price": p_yes,
                    "no_price": p_no,
                    "outcome": outcome,
                }
            )

    events_df = pd.DataFrame(rows)

    if events_df.empty:
        raise ValueError(
            "No hourly events were constructed. "
            "Check timestamp parsing and that your data covers the trading_hours range (e.g., 9..24)."
        )

    # Make a numeric day ID for the env
    events_df = events_df.sort_values(["day", "hour"]).reset_index(drop=True)

    # Create numeric day representation
    events_df["day_id"] = events_df["day"].astype("category").cat.codes

    # Replace old 'day' with integer day
    events_df = events_df.drop(columns=["day"])
    events_df = events_df.rename(columns={"day_id": "day"})

    # Final ordering
    events_df = events_df[["day", "hour", "btc_price", "yes_price", "no_price", "outcome"]]

    return events_df
