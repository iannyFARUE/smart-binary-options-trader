# live_trading/live_agent_runner.py
import os
import time
import csv
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO

from live_trading.kalshi_api import KalshiClient
from env.kalshi_env import KalshiEnvConfig


# ---------- helper: build observation vector for live trading ----------

def build_live_observation(
    market: dict,
    portfolio: dict,
    config: KalshiEnvConfig,
    btc_price: float,
    btc_mean: float,
    btc_std: float,
):
    """
    Map live Kalshi BTC market + portfolio to the observation format used by PPO.

    In the training env, obs was:
      [btc_norm, yes_price, no_price, hour_sin, hour_cos,
       pos_yes, pos_no, cash_norm]

    Here we'll approximate:
      - pos_yes/pos_no from portfolio exposures in this market
      - cash from portfolio 'cash' or 'available_funds'
      - hour from market expiration or current time
    """
    # 1) BTC normalized
    btc_norm = (btc_price - btc_mean) / (btc_std if btc_std > 0 else 1.0)

    # 2) YES/NO prices from the market orderbook/last trade
    yes_price = float(market.get("yes_ask") or market.get("yes_price") or 0.5)
    no_price = float(market.get("no_ask") or market.get("no_price") or (1.0 - yes_price))

    # 3) Time encoding: use market expiration hour in UTC
    #    or fallback to current hour
    expiry = market.get("expiration_time")
    if expiry:
        # assuming ISO 8601 string
        try:
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            hour = exp_dt.hour
        except Exception:
            hour = datetime.now(timezone.utc).hour
    else:
        hour = datetime.now(timezone.utc).hour

    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)

    # 4) Positions in this market from portfolio
    #    For simplicity, we set pos_yes = net yes contracts, pos_no = net no contracts
    pos_yes = 0.0
    pos_no = 0.0
    positions = portfolio.get("positions", [])
    ticker = market["ticker"]
    for pos in positions:
        if pos.get("ticker") == ticker:
            # Example fields: side ("yes" / "no"), count
            side = pos.get("side")
            count = float(pos.get("count", 0))
            if side == "yes":
                pos_yes += count
            elif side == "no":
                pos_no += count

    # 5) Cash normalized
    cash = float(portfolio.get("cash", config.initial_cash))
    cash_norm = cash / config.initial_cash

    obs = np.array(
        [
            btc_norm,
            yes_price,
            no_price,
            hour_sin,
            hour_cos,
            pos_yes,
            pos_no,
            cash_norm,
        ],
        dtype=np.float32,
    )

    return obs


# ---------- mapping PPO action -> Kalshi order side ----------

def action_to_side_action(action: int):
    """
    Map PPO action (0..4) to (side, action) for Kalshi.

    0: do nothing
    1: buy  YES -> side="yes", action="buy"
    2: sell YES -> side="yes", action="sell"
    3: buy  NO  -> side="no",  action="buy"
    4: sell NO  -> side="no",  action="sell"
    """
    if action == 1:
        return "yes", "buy"
    elif action == 2:
        return "yes", "sell"
    elif action == 3:
        return "no", "buy"
    elif action == 4:
        return "no", "sell"
    else:
        return "", ""  # do nothing


# ---------- live loop ----------

def main():
    # 1. Load PPO model
    model_path = "agent/models/ppo_kalshi_realdata.zip"
    model = PPO.load(model_path)

    # 2. Set up Kalshi client
    client = KalshiClient()

    # 3. Stats from training data for BTC normalization
    #    For now, we hardcode; ideally compute from training events_df
    #    You can later load these from a config or serialized file.
    btc_mean = 20000.0
    btc_std = 10000.0

    config = KalshiEnvConfig()

    # 4. Live trading parameters
    contracts_per_trade = 1
    poll_interval_sec = 60  # how often to check markets
    log_path = "logs/live/kalshi_live_trades.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create log file with header if not exists
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "ticker",
                    "btc_price",
                    "yes_price",
                    "no_price",
                    "action",
                    "side",
                    "count",
                    "price",
                    "portfolio_value",
                    "user_id",
                    "order_id"
                ]
            )

    print("[live] Starting live demo loop (demo environment, no real money!)")

    while True:
        try:
            # 1) Fetch BTC hourly markets (adjust filter to match actual ticker symbol)
            markets_resp = client.get_markets(series_ticker="KXBTC",status="open",filter_liquid=True)
            btc_markets = markets_resp.get("markets",[])

            # Filter for relevant BTC hourly threshold markets, adjust this filter
            # btc_markets = [
            #     m for m in markets
            #     if "BTC" in m.get("underlying", "") or "BTC" in m.get("title", "")
            # ]

            if not btc_markets:
                print("[live] No BTC markets found, sleeping...")
                time.sleep(poll_interval_sec)
                continue

            print(btc_markets[:2])

            # For now, just pick the next expiring BTC market
            btc_markets.sort(key=lambda m: m.get("expiration_time", ""))
            market = btc_markets[0]
            ticker = market["ticker"]

            # --- Normalize yes/no prices from Kalshi (0..100 cents) to 0..1 probabilities ---
            def norm_price_raw(raw, default):
                """
                Convert raw Kalshi price (cents or 0-1) to float in [0,1].
                If raw is None, use default.
                """
                if raw is None:
                    return default

                if isinstance(raw, str):
                    raw = float(raw)

                raw = float(raw)

                # If > 1.0, treat as cents (0..100)
                if raw > 1.0:
                    return raw / 100.0

                return raw


            yes_raw = market.get("yes_ask", None) or market.get("yes_bid", None)
            no_raw  = market.get("no_ask", None) or market.get("no_bid", None)

            # First try to normalize from the book; if both are missing, fall back to 0.5 / 0.5
            yes_price = norm_price_raw(yes_raw, default=0.5)
            no_price  = norm_price_raw(no_raw, default=(1.0 - yes_price))

            # Optional: clamp a bit to avoid exactly 0 or 1
            yes_price = max(0.01, min(0.99, yes_price))
            no_price  = max(0.01, min(0.99, no_price))

            print("[DEBUG] normalized yes_price=", yes_price, "no_price=", no_price)
            # crude proxy: treat 0.5 as at-the-money, but we don't know exact threshold here.
            # For now, just treat btc_price as a dummy variable consistent with training scale:
            btc_price = btc_mean + (yes_price - 0.5) * 2 * btc_std  # hacky but consistent scale

            # 3) Get portfolio info
            portfolio = client.get_portfolio()
            # You might want to compute portfolio_value from cash + positions; here we log cash only
            portfolio_value = float(portfolio.get("cash", config.initial_cash))

            # 4) Build observation
            obs = build_live_observation(
                market=market,
                portfolio=portfolio,
                config=config,
                btc_price=btc_price,
                btc_mean=btc_mean,
                btc_std=btc_std,
            )

            # 5) Ask PPO for action
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            side, act = action_to_side_action(action)
            if side == "" or act == "":
                print(f"[live] Action {action} -> do nothing this cycle.")
                time.sleep(poll_interval_sec)
                continue

            # 6) Decide order price: we can place at current ask for simplicity
                # Pick execution price
            if side == "yes":
                price = yes_price
            else:
                price = no_price

            print(
                f"[live] Placing order: event={ticker}, market={ticker}, "
                f"side={side}, action={act}, count={contracts_per_trade}, price={price:.3f}"
            )

            order_resp = client.create_order(
                ticker=ticker,
                side=side,
                action=act,
                count=contracts_per_trade,
                price=price,
            )
            print("[live] Order response:", order_resp)
            print(order_resp.keys())

            # 8) Log to CSV
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        ticker,
                        btc_price,
                        yes_price,
                        no_price,
                        order_resp["order"]["action"],
                        side,
                        contracts_per_trade,
                        price,
                        portfolio_value,
                        order_resp["order"]["user_id"],
                        order_resp["order"]["order_id"]
                    ]
                )

            # 9) Sleep until next poll
            time.sleep(poll_interval_sec)

        except KeyboardInterrupt:
            print("\n[live] Stopping live loop (KeyboardInterrupt).")
            break
        except Exception as e:
            print("[live] ERROR:", e)
            time.sleep(poll_interval_sec)


if __name__ == "__main__":
    main()
