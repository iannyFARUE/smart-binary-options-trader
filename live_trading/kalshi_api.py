# live_trading/kalshi_api.py
import os
import time
import json
from typing import Any, Dict, Optional

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from dotenv import load_dotenv


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
    ):
        load_dotenv()
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

        with open(self.private_key_path, "rb") as f:
            self._private_key = load_pem_private_key(f.read(), password=None)

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

        timestamp_ms = str(int(time.time() * 1000))
        signing_payload = (timestamp_ms + method_upper + path).encode("utf-8")
        signature = self._sign(signing_payload)

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

    # ---------- high-level API helpers ----------

    def get_markets(self, **params) -> Dict[str, Any]:
        """
        GET /markets
        Use filters like:
          - ticker: "BTCH"
          - status: "open"
        """
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """
        GET /markets/{ticker}
        """
        return self._request("GET", f"/markets/{ticker}")

    def get_portfolio(self) -> Dict[str, Any]:
        """
        GET /portfolio
        """
        return self._request("GET", "/portfolio")

    def get_orders(self, **params) -> Dict[str, Any]:
        """
        GET /orders
        """
        return self._request("GET", "/orders", params=params)

    def create_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price: float,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST /orders
        side: "yes_buy", "yes_sell", "no_buy", "no_sell"
        price: in [0, 1] for Kalshi (probability) or 0-100 depending on API,
               but demo uses 0-1 probabilities in JSON.
        """
        body = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "price": price,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id

        return self._request("POST", "/orders", body=body)
