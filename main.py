import json
import os
from pathlib import Path

from live_trading.kalshi_api import Environment, KalshiClient


def resolve_environment() -> Environment:
    """Map KALSHI_ENV env var to the API enum."""
    env_name = os.getenv("KALSHI_ENV", "demo").lower()
    return Environment.PROD if env_name.startswith("prod") else Environment.DEMO


def main():
    # Use a ticker prefix so we can discover the actual market tickers first.
    ticker_prefix = os.getenv("KALSHI_TICKER_PREFIX", "BTC")

    client = KalshiClient(environment=resolve_environment())
    dump_dir = Path("logs/dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # 1) List markets that match the prefix (e.g., BTC hourly thresholds)
    markets_resp = client.get_portfolio()
    print(f"Balance is {markets_resp}")

if __name__ == "__main__":
    main()
