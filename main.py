import json
import os
from pathlib import Path

from live_trading.kalshi_api import Environment, KalshiClient
from agent.train_agent import train
from scripts.market_orders import create_orders


def main():
    create_orders()


if __name__ == "__main__":
    main()
