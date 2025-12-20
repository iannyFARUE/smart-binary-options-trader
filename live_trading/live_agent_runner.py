import os
import time
import csv
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO

from live_trading.kalshi_api import KalshiClient
from env.kalshi_env import KalshiEnvConfig


# ---------------------------
# Helpers
# ---------------------------

def norm_price_raw(raw, default=0.5):
    if raw is None:
        return default
    try:
        raw = float(raw)
    except Exception:
        return default
    return raw / 100.0 if raw > 1.0 else raw


def extract_market_obj(resp: dict) -> dict:
    print(resp)
    return resp.get("market", resp)


def parse_iso_z(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def extract_times(m: dict):
    close_time = parse_iso_z(m.get("close_time"))
    expected_exp = parse_iso_z(m.get("expected_expiration_time"))
    expiration = parse_iso_z(m.get("expiration_time"))
    effective = close_time or expected_exp or expiration
    return effective, close_time, expected_exp, expiration


def is_resolved(status: str, result: str) -> bool:
    if result in ("yes", "no"):
        return True
    if status in ("finalized", "resolved", "settled", "closed"):
        return True
    return False


def extract_portfolio_cash(portfolio_resp: dict, fallback: float) -> float:
    """
    Kalshi demo often returns:
      {"balance": 240273, ...}  # cents
    or nested dicts in other modes.

    Return cash in DOLLARS.
    """
    # Direct cash if present
    if isinstance(portfolio_resp, dict) and portfolio_resp.get("cash") is not None:
        return float(portfolio_resp["cash"])

    # If balance is a number -> treat as cents
    bal = portfolio_resp.get("balance") if isinstance(portfolio_resp, dict) else None
    if isinstance(bal, (int, float)):
        return float(bal) / 100.0

    # If balance is a dict -> try common keys
    if isinstance(bal, dict):
        for k in ("available_cash", "cash", "balance", "available_funds", "funds"):
            v = bal.get(k)
            if v is None:
                continue
            # if nested numeric assume cents unless it’s clearly already dollars
            if isinstance(v, (int, float)):
                # heuristic: big numbers are cents
                return float(v) / 100.0 if v > 1000 else float(v)
            # if it's a string like "2402.73"
            try:
                fv = float(v)
                return fv / 100.0 if fv > 1000 else fv
            except Exception:
                pass

    return float(fallback)


def market_is_tradeable(market: dict) -> bool:
    """
    Detect empty Kalshi demo books.
    """
    yes_bid = market.get("yes_bid", 0)
    yes_ask = market.get("yes_ask", 0)
    no_bid = market.get("no_bid", 0)
    no_ask = market.get("no_ask", 0)

    # dead-book pattern seen in demo
    if yes_bid == 0 and yes_ask == 0 and no_bid == 100 and no_ask == 100:
        return False

    has_yes = 0 < yes_bid < yes_ask < 100
    has_no = 0 < no_bid < no_ask < 100
    return has_yes or has_no


def choose_fill_price(side: str, market: dict, fallback: float) -> float:
    if side == "yes":
        raw = market.get("yes_ask") or market.get("yes_bid")
    else:
        raw = market.get("no_ask") or market.get("no_bid")

    p = norm_price_raw(raw, default=fallback)
    return float(max(0.01, min(0.99, p)))


def build_live_observation(market: dict, cash: float, config: KalshiEnvConfig):
    yes_raw = market.get("yes_ask") or market.get("yes_bid")
    no_raw = market.get("no_ask") or market.get("no_bid")

    yes_p = norm_price_raw(yes_raw, default=0.5)
    no_p = norm_price_raw(no_raw, default=(1.0 - yes_p))

    yes_p = float(max(0.01, min(0.99, yes_p)))
    no_p = float(max(0.01, min(0.99, no_p)))

    effective_time, _, _, _ = extract_times(market)
    hour = effective_time.hour if effective_time else datetime.now(timezone.utc).hour

    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = float(np.sin(hour_rad))
    hour_cos = float(np.cos(hour_rad))

    obs = np.array(
        [
            0.0,  # btc_norm (placeholder)
            0.0,  # btc_ret
            0.0,  # btc_vol
            yes_p,
            no_p,
            hour_sin,
            hour_cos,
            0.0,  # inventory placeholder
            cash / config.initial_cash,
        ],
        dtype=np.float32,
    )
    return obs, yes_p, no_p


def action_to_order(action: int):
    if action == 1:
        return "yes", "buy"
    if action == 2:
        return "no", "buy"
    return "", ""


# ---------------------------
# Main
# ---------------------------

def main():
    model_path = "agent/models/ppo_kalshi_realdata.zip"

    trade_log = "logs/live/kalshi_live_trades.csv"
    resolution_log = "logs/live/kalshi_resolutions.csv"

    os.makedirs("logs/live", exist_ok=True)

    model = PPO.load(model_path)
    client = KalshiClient()
    config = KalshiEnvConfig()

    contracts_per_trade = 1
    poll_interval_sec = 60

    if not os.path.exists(trade_log):
        with open(trade_log, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp",
                "ticker",
                "action_id",
                "side",
                "count",
                "entry_price_prob",
                "is_paper_fill",
                "paper_reason",
            ])

    if not os.path.exists(resolution_log):
        with open(resolution_log, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp",
                "ticker",
                "status",
                "result",
            ])

    watchlist = set()
    resolved = set()

    print("[live] Starting live demo loop (with paper-fill fallback).")

    while True:
        try:
            # ---- resolution scan ----
            for tkr in list(watchlist):
                if tkr in resolved:
                    continue

                m = extract_market_obj(client.get_market(tkr))
                
                status = m.get("status")
                result = m.get("result")

                if is_resolved(status, result):
                    print(f"[live] RESOLVED {tkr}: {result}")
                    with open(resolution_log, "a", newline="") as f:
                        csv.writer(f).writerow([
                            datetime.now(timezone.utc).isoformat(),
                            tkr,
                            status,
                            result,
                        ])
                    resolved.add(tkr)

            # ---- get BTC markets ----
            markets = client.get_markets(series_ticker="KXBTC", status="open").get("markets", [])

            if not markets:
                time.sleep(poll_interval_sec)
                continue

            markets.sort(key=lambda m: extract_times(m)[0] or datetime.max.replace(tzinfo=timezone.utc))
            market = markets[0]
            ticker = market["ticker"]
           
            portfolio = client.get_portfolio()
            cash = extract_portfolio_cash(portfolio, config.initial_cash)
            

            obs, yes_p, no_p = build_live_observation(market, cash, config)
            action_id, _ = model.predict(obs, deterministic=True)
            action_id = int(action_id)

            side, act = action_to_order(action_id)
            if side == "":
                print("[live] HOLD")
                time.sleep(poll_interval_sec)
                continue

            limit_price = yes_p if side == "yes" else no_p
            limit_price = float(max(0.01, min(0.99, limit_price)))

            is_liquid = market_is_tradeable(market)
            is_paper = not is_liquid
            reason = "illiquid_demo_book" if is_paper else ""

            if not is_paper:
                try:
                    client.create_order(
                        ticker=ticker,
                        side=side,
                        action="buy",
                        count=contracts_per_trade,
                        price=limit_price,
                    )
                    print(f"[live] LIVE BUY {side.upper()} {ticker} @ {limit_price:.3f}")
                except Exception as e:
                    print("[live] LIVE FAILED → paperfill", e)
                    is_paper = True
                    reason = "live_failed"

            if is_paper:
                limit_price = choose_fill_price(side, market, limit_price)
                print(f"[live] PAPERFILL BUY {side.upper()} {ticker} @ {limit_price:.3f}")

            with open(trade_log, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).isoformat(),
                    ticker,
                    action_id,
                    side,
                    contracts_per_trade,
                    limit_price,
                    int(is_paper),
                    reason,
                ])

            watchlist.add(ticker)
            time.sleep(poll_interval_sec)

        except KeyboardInterrupt:
            print("[live] Stopped.")
            break
        except Exception as e:
            print("[live] ERROR:", e)
            time.sleep(poll_interval_sec)


if __name__ == "__main__":
    main()
