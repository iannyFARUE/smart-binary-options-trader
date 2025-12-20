# agent/backtest_agent.py
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from env.data_loader import load_raw_btc_data, build_hourly_kalshi_like_events
from env.kalshi_env import KalshiEnvConfig, KalshiBTCHourlyEnv
import os


def backtest(
    model_path="agent/models/ppo_kalshi_realdata.zip",
    csv_path="data/raw/btcusd_minute.csv",
    n_episodes=5,
):
    # 1. Load data & build events
    raw_df = load_raw_btc_data(csv_path)
    events_df = build_hourly_kalshi_like_events(raw_df)

    config = KalshiEnvConfig()
    env = KalshiBTCHourlyEnv(events_df, config)

    # 2. Load trained PPO
    model = PPO.load(model_path)

    daily_pnls = []
    all_equity = []

    action_counts = np.zeros(5, dtype=int)  # 0..4

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_equity = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1

            obs, reward, done, truncated, info = env.step(action)
            # portfolio_value is in info based on our env
            ep_equity.append(info["portfolio_value"])

        all_equity.append(ep_equity)
        daily_pnls.append(ep_equity[-1] - config.initial_cash)

    daily_pnls = np.array(daily_pnls)
    mean_pnl = daily_pnls.mean()
    std_pnl = daily_pnls.std()
    sharpe = mean_pnl / std_pnl if std_pnl > 1e-8 else np.nan

    print("\n=== BACKTEST RESULTS (PPO) ===")
    print(f"Episodes (days): {n_episodes}")
    print(f"Mean daily PnL: {mean_pnl:.4f}")
    print(f"Std daily PnL: {std_pnl:.4f}")
    print(f"Approx. Sharpe (per day): {sharpe:.3f}")
    print("\nAction counts (0..4):", action_counts)
    print("  0 = do nothing")
    print("  1 = buy YES")
    print("  2 = sell YES")
    print("  3 = buy NO")
    print("  4 = sell NO")

    # Plot one sample equity curve
    if all_equity:
        eq = all_equity[0]
        plt.figure(figsize=(8, 4))
        plt.plot(eq)
        plt.axhline(config.initial_cash, linestyle="--", linewidth=1, label="Initial cash")
        plt.title("Sample Equity Curve (Episode 1)")
        plt.xlabel("Time steps (hourly events)")
        plt.ylabel("Portfolio value")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_path = os.path.dirname(os.path.abspath(__file__))
    backtest()
