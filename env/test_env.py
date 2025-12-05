import pandas as pd
from env.kalshi_env import KalshiEnvConfig, KalshiBTCHourlyEnv

rows = []
for day in [1,2]:
    for hour in [9,10,11]:
        btc = 60_000 + (day * 10) + (hour - 9) * 5
        yes_price = 0.5
        no_price = 0.5
        outcome = int(btc % 2 == 0)
        rows.append(
            {
                "day":day,
                "hour": hour,
                "btc_price":btc,
                "yes_price":yes_price,
                "no_price": no_price,
                "outcome":outcome
            }
        )

df = pd.DataFrame(rows)
print(df.head())

