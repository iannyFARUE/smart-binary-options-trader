import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


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
    data = make_dummy_data()
    config = KalshiEnvConfig()

    env = KalshiBTCHourlyEnv(data, config)

    # Optional: set up logging dir for SB3
    log_dir = "./logs/ppo_kalshi/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=64,        # small for toy example
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
    )
    model.set_logger(new_logger)

    # Train for a small number of steps just to see if it runs
    model.learn(total_timesteps=10_000)

    model_path = "./agent/models/ppo_kalshi_dummy.zip"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")
