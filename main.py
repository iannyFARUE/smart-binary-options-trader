from scripts.balance import get_balance
from agent.train_agent import train
from live_trading.kalshi_api import KalshiClient
from pathlib import Path
import json
def main():
    client = KalshiClient()
    file_path = Path("logs/dumps/sample_market_response.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    market_res = client.get_markets(ticker="BTC")
    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(market_res,indent=4))
    

if __name__ == "__main__":
    main()
