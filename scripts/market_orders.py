from live_trading.kalshi_api import KalshiClient


def create_orders():
    client = KalshiClient()
    # Post a limit order to BUY at 45 cents
    client.create_order(
        ticker="KXBTC-25DEC1417-B90500",
        side="yes",
        action="buy",
        count=10,
        price=45
    )

    # Post a limit order to SELL at 55 cents  
    client.create_order(
        ticker="KXBTC-25DEC1417-B90500",
        side="yes",
        action="sell",
        count=10,
        price=55
    )