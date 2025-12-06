# env/data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple


def load_raw_btc_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw BTC candles from CSV.

    Expected columns (you may adjust to match your dataset):
        - timestamp or date/time columns
        - open, high, low, close

    We'll:
        - parse timestamp to pandas datetime
        - set to index
        - resample to 1-minute or 5-minute if needed
    """
    df = pd.read_csv(csv_path)

    # Try to infer datetime column name
    # Adjust this to match your actual CSV (e.g. 'datetime', 'Timestamp', 'date')
    datetime_col_candidates = ["timestamp", "date", "datetime", "time"]
    dt_col = None
    for c in datetime_col_candidates:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        raise ValueError("No datetime-like column found in CSV.")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    # Set index for resampling convenience
    df = df.set_index(dt_col)

    # Try to standardize candle columns
    # Adjust these names if your CSV uses 'Open', 'High', etc.
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "open":
            rename_map[col] = "open"
        elif lc == "high":
            rename_map[col] = "high"
        elif lc == "low":
            rename_map[col] = "low"
        elif lc == "close":
            rename_map[col] = "close"
    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}, got {df.columns}.")

    # If data is higher frequency, you can resample here.
    # Example: ensure 1-minute candles with last price as close:
    # df = df.resample("1min").agg({"open": "first", "high": "max",
    #                               "low": "min", "close": "last"}).dropna()

    return df


def build_hourly_kalshi_like_events(
    df: pd.DataFrame,
    trading_hours: Tuple[int, ...] = tuple(range(9, 25)),
) -> pd.DataFrame:
    """
    Given minute-level BTC OHLCV (or similar) indexed by datetime,
    construct a DataFrame with one row per (day, hour) for Kalshi-like events.

    For each day D and hour H in trading_hours:
        - price_at_hour_open  = close at D H:00
        - price_at_hour_close = close at D H+1:00
        - threshold T_H = price_at_hour_open
        - outcome = 1 if price_at_hour_close > T_H else 0
        - yes_price, no_price simulated from short-term momentum

    Returns columns:
        day, hour, btc_price, yes_price, no_price, outcome
    """

    # Ensure close price exists
    if "close" not in df.columns:
        raise ValueError("Expected 'close' column in df.")

    # We'll work with a copy
    data = df.copy()

    # Add helper columns for day/hour
    data["date"] = data.index.date
    data["hour"] = data.index.hour

    # We'll store rows here
    rows = []

    # Group by date (day)
    for day, day_df in data.groupby("date"):
        # Make sure day_df is sorted
        day_df = day_df.sort_index()

        for H in trading_hours:
            # Event: "BTC > T_H at H+1:00"
            # Need price at H:00 and H+1:00
            try:
                # Last close before or at H:00
                hour_open_ts = day_df.between_time(f"{H:02d}:00", f"{H:02d}:59")
                if hour_open_ts.empty:
                    continue
                price_at_hour_open = float(hour_open_ts["close"].iloc[0])

                # Last close before or at H+1:00
                H_next = H + 1
                if H_next > 23:
                    # If day ends at 23:59, you may skip the last event
                    # or you can roll into next day; for simplicity, skip.
                    continue
                hour_close_ts = day_df.between_time(f"{H_next:02d}:00", f"{H_next:02d}:59")
                if hour_close_ts.empty:
                    continue
                price_at_hour_close = float(hour_close_ts["close"].iloc[-1])
            except Exception:
                continue

            # Define threshold and outcome
            T_H = price_at_hour_open
            outcome = int(price_at_hour_close > T_H)

            # Compute short-term momentum: compare current price to price 30 minutes ago
            # Find 30 minutes window before H:00
            lookback_minutes = 30
            start_lb = hour_open_ts.index[0] - pd.Timedelta(minutes=lookback_minutes)
            lb_slice = day_df.loc[start_lb : hour_open_ts.index[0]]
            if lb_slice.empty:
                # fallback: no momentum info
                momentum = 0.0
            else:
                price_lb = float(lb_slice["close"].iloc[0])
                momentum = (price_at_hour_open - price_lb) / price_lb

            # Simulate yes_price as 0.5 + alpha * momentum (clipped)
            alpha = 5.0
            p_yes = 0.5 + alpha * momentum
            p_yes = float(np.clip(p_yes, 0.05, 0.95))
            p_no = 1.0 - p_yes

            rows.append(
                {
                    "day": day,  # date object, we can keep as is for now
                    "hour": H,
                    "btc_price": price_at_hour_open,
                    "yes_price": p_yes,
                    "no_price": p_no,
                    "outcome": outcome,
                }
            )

    events_df = pd.DataFrame(rows)

    # Convert day to something consistent (e.g. string or int)
    # For env, it's often easier to use int index
    events_df = events_df.sort_values(["day", "hour"]).reset_index(drop=True)
    events_df["day_id"] = events_df["day"].astype("category").cat.codes

    # You can either return with both 'day' and 'day_id', or only 'day_id'
    events_df = events_df.rename(columns={"day_id": "day"})
    events_df = events_df[["day", "hour", "btc_price", "yes_price", "no_price", "outcome"]]

    return events_df
