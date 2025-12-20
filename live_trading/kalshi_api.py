# live_trading/kalshi_api.py
import os
import datetime
import json
from typing import Any, Dict, Optional
from enum import Enum

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
import uuid
from typing import Any, Dict, Optional, List

class Environment(Enum):
    DEMO = "demo"
    PROD = "prod"

class KalshiClient:
    """
    Minimal Kalshi demo API client.

    Auth: RSA signature over "<timestamp><method><path>" with private key.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        environment: Environment = Environment.DEMO
    ):
        load_dotenv()
        self.environment = environment
        self.base_url = base_url or os.getenv(
            "KALSHI_DEMO_BASE_URL",
            "https://demo-api.kalshi.co/trade-api/v2",
        )
        self.key_id = key_id or os.environ["KALSHI_API_KEY_ID"]
        self.private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")

        if not self.key_id:
            raise ValueError("KALSHI_KEY_ID not set in env and not provided.")
        if not self.private_key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH not set in env and not provided.")
        
        if self.environment not in {Environment.DEMO, Environment.PROD}:
            raise ValueError("Invalid environment specified")

        with open(self.private_key_path, "rb") as f:
            self._private_key = load_pem_private_key(f.read(), password=None,backend=default_backend())

    # ---------- low-level helpers ----------

    def _sign(self, message: bytes) -> str:
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        # Kalshi expects base64 string
        import base64

        return base64.b64encode(signature).decode("utf-8")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send signed request to Kalshi demo API.
        path: e.g. "/markets"
        """
        url = self.base_url + path
        method_upper = method.upper()
        path_without_query = path.split("?")[0]

        timestamp_ms = str(int(datetime.datetime.now().timestamp() * 1000))
        signing_payload = f"{timestamp_ms}{method_upper}{path_without_query}".encode("utf-8")
        signature = self._sign(signing_payload)
        print(method)

        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

        if method_upper == "GET":
            resp = requests.get(url, headers=headers, params=params, timeout=10)
        else:
            data = json.dumps(body or {})
            resp = requests.post(url, headers=headers, data=data, timeout=10)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Kalshi API error {resp.status_code}: {resp.text}"
            )

        return resp.json()
    
    @staticmethod
    def filter_liquid_markets(
        markets: List[Dict[str, Any]], 
        min_liquidity: float = 0,
        min_spread_pct: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Filter markets to only those with real liquidity.
        
        Args:
            markets: List of market dicts
            min_liquidity: Minimum liquidity in dollars (default 0)
            min_spread_pct: Maximum spread percentage (default 50%)
        
        Returns:
            Filtered list of markets
        """
        liquid_markets = []
        
        for market in markets:
            if not KalshiClient.has_liquidity(market, min_spread_pct):
                continue
            
            # Optional: filter by minimum liquidity amount
            liquidity = float(market.get("liquidity_dollars", "0"))
            if liquidity < min_liquidity:
                continue
            
            liquid_markets.append(market)
        
        return liquid_markets

    # ---------- high-level API helpers ----------
    def get_market(self, ticker: str) -> Dict[str, Any]:
        """
        GET /markets/{ticker}
        """
        print("here ",)
        return self._request("GET", f"/trade-api/v2/markets/{ticker}")
    
    @staticmethod
    def has_liquidity(market: Dict[str, Any], min_spread_pct: float = 50.0) -> bool:
        """
        Check if a market has tradeable liquidity.
        
        Args:
            market: Market data dict
            min_spread_pct: Maximum spread percentage to consider liquid (default 50%)
        
        Returns:
            True if market has real liquidity
        """
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 0)
        no_bid = market.get("no_bid", 0)
        no_ask = market.get("no_ask", 0)
        
        # Filter out empty order books
        if yes_bid == 0 and yes_ask == 0 and no_bid == 100 and no_ask == 100:
            return False
        
        # Check if there's a valid spread on either side
        has_yes_liquidity = 0 < yes_bid < yes_ask < 100
        has_no_liquidity = 0 < no_bid < no_ask < 100
        
        # Optional: check spread width
        if has_yes_liquidity:
            yes_spread = (yes_ask - yes_bid) / yes_ask * 100
            if yes_spread > min_spread_pct:
                has_yes_liquidity = False
        
        if has_no_liquidity:
            no_spread = (no_ask - no_bid) / no_ask * 100
            if no_spread > min_spread_pct:
                has_no_liquidity = False
        
        return has_yes_liquidity or has_no_liquidity
    
    @staticmethod
    def get_best_prices(market: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """
        Extract best bid/ask prices from a market.
        
        Returns:
            Dict with 'yes_bid', 'yes_ask', 'no_bid', 'no_ask' or None if not available
        """
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 0)
        no_bid = market.get("no_bid", 0)
        no_ask = market.get("no_ask", 0)
        
        return {
            "yes_bid": yes_bid if 0 < yes_bid < 100 else None,
            "yes_ask": yes_ask if 0 < yes_ask < 100 else None,
            "no_bid": no_bid if 0 < no_bid < 100 else None,
            "no_ask": no_ask if 0 < no_ask < 100 else None,
        }

    def get_portfolio(self) -> Dict[str, Any]:
        """
        GET /portfolio
        """
        return self._request("GET", "/trade-api/v2/portfolio/balance")

    def get_orders(self, **params) -> Dict[str, Any]:
        """
        GET /orders
        """
        return self._request("GET", "/trade-api/v2/portfolio/orders", params=params)

    def create_order(
        self,
        ticker: str,
        side: str,          # "yes" or "no"
        action: str,        # "buy" or "sell"
        count: int,
        price: float,       # in [0,1] or [0,100]
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST /portfolio/orders

        side   : "yes" or "no"
        action : "buy" or "sell"
        count  : number of contracts
        price  : float in [0,1] (prob) or [0,100] (cents)

        Kalshi expects:
          - exactly ONE of yes_price or no_price
          - price in CENTS (int), with 1 <= price <= 99
        """

        # 1) Normalize price to cents
        if price <= 1.0:
            price_cents = int(round(price * 100))
        else:
            price_cents = int(round(price))

        # 2) Clamp to [1, 99] since 0 and 100 are invalid
        price_cents = max(1, min(99, price_cents))

        body: Dict[str, Any] = {
            "ticker": ticker,
            "side": side,            # "yes" | "no"
            "action": action,        # "buy" | "sell"
            "count": count,
            "type": "limit",
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }

        # 3) Set ONLY the correct price field based on side
        if side == "yes":
            body["yes_price"] = price_cents
        elif side == "no":
            body["no_price"] = price_cents
        else:
            raise ValueError(f"Invalid side: {side}, expected 'yes' or 'no'")

        # base_url already has /trade-api/v2, so path is just /portfolio/orders
        return self._request("POST", "/trade-api/v2/portfolio/orders", body=body)
    
    def get_markets(self, filter_liquid: bool = False, **params) -> Dict[str, Any]:
        """
        GET /markets
        Use filters like:
        - ticker: "BTCH"
        - status: "open"
        - filter_liquid: if True, only return markets with real liquidity
        """
        result = self._request("GET", "/trade-api/v2/markets", params=params)
        
        if filter_liquid and "markets" in result:
            result["markets"] = self.filter_liquid_markets(result["markets"])
        
        return result