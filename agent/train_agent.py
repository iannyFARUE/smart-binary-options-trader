import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from env.data_loader import load_raw_btc_data, build_hourly_kalshi_like_events


from env.kalshi_env import KalshiBTCHourlyEnv, KalshiEnvConfig


def make_dummy_data():
    # Same dummy logic as test_env, but more days
    rows = []
    for day in range(1, 31):  # 30 days
        for hour in range(9, 12):  # keep small at first: 9,10,11
            btc = 60_000 + (day * 10) + (hour - 9) * 5
            yes_price = 0.5
            no_price = 0.5
            # arbitrary outcome: even price => YES wins
            outcome = int(btc % 2 == 0)
            rows.append(
                {
                    "day": day,
                    "hour": hour,
                    "btc_price": btc,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def train():
        # 1. Load raw BTC minute data
    raw_path = "data/raw/btcusd_minute.csv"  # adjust to your actual file path
    print("[train] Loading raw BTC data...")
    raw_df = load_raw_btc_data(raw_path)

    # 2. Build Kalshi-like hourly events
    print("[train] Building hourly kalshi-like events...")
    events_df = build_hourly_kalshi_like_events(raw_df)
    print("Events df head:")
    print(events_df.head())
    config = KalshiEnvConfig()

    env = KalshiBTCHourlyEnv(events_df, config)

    # Optional: set up logging dir for SB3
    log_dir = "./logs/ppo_kalshi/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,        # small for toy example
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
    )
    model.set_logger(new_logger)

    # Train for a small number of steps just to see if it runs
    model.learn(total_timesteps=10_000_000)

    model_path = "./agent/models/ppo_kalshi_realdata.zip"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[train] Saved model to {model_path}")
