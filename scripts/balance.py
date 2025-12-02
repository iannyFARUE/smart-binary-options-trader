import os
from dotenv import load_dotenv
from kalshi_python import Configuration, KalshiClient

def get_balance():
    load_dotenv()

    API_KEY_ID = os.environ["KALSHI_API_KEY_ID"]
    PRIVATE_KEY_PATH = os.environ["KALSHI_PRIVATE_KEY_PATH"]

    with open(PRIVATE_KEY_PATH, "r") as f:
        private_key = f.read()

    config = Configuration(host = "https://demo-api.kalshi.co/trade-api/v2",
                        )
    
    config.api_key_id=API_KEY_ID
    config.private_key_pem=private_key

    client = KalshiClient(config)

    balance = client.get_balance()
    print(f"Raw balcne response: {balance}")
    print(f"Cash balance: ${balance.balance / 100:.2f}")