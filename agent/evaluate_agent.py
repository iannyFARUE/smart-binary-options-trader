import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from env.kalshi_env import KalshiBTCHourlyEnv, KalshiEnvConfig
from agent.baselines import baseline_flat, baseline_momentum, baseline_random


def evaluate_policy(env, model, episodes=200):
    rewards = []
    pnls = []

    for _ in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False
        total_reward = 0
        start_pv = info.get("portfolio_value", env.config.initial_cash)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        final_pv = info["portfolio_value"]
        pnl = final_pv - start_pv

        rewards.append(total_reward)
        pnls.append(pnl)

    return np.array(rewards), np.array(pnls)


def evaluate_baseline(env, policy_fn, episodes=200, random=False):
    rewards = []
    pnls = []

    for _ in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False
        total_reward = 0
        start_pv = info.get("portfolio_value", env.config.initial_cash)

        while not done:
            if random:
                action = policy_fn(obs, env.action_space)
            else:
                action = policy_fn(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        final_pv = info["portfolio_value"]
        pnl = final_pv - start_pv

        rewards.append(total_reward)
        pnls.append(pnl)

    return np.array(rewards), np.array(pnls)


def main():
    # Load data & environment
    from env.data_loader import load_raw_btc_data, build_hourly_kalshi_like_events

    raw_path = "data/raw/btc_minute.csv"
    raw_df = load_raw_btc_data(raw_path)
    events_df = build_hourly_kalshi_like_events(raw_df)

    env = KalshiBTCHourlyEnv(events_df, KalshiEnvConfig())

    # Load trained PPO model
    model = PPO.load("agent/models/ppo_kalshi_realdata.zip")

    print("Evaluating PPO...")
    rl_rewards, rl_pnls = evaluate_policy(env, model)

    print("Evaluating flat baseline...")
    flat_rewards, flat_pnls = evaluate_baseline(env, baseline_flat)

    print("Evaluating momentum baseline...")
    mom_rewards, mom_pnls = evaluate_baseline(env, baseline_momentum)

    print("Evaluating random baseline...")
    rand_rewards, rand_pnls = evaluate_baseline(env, baseline_random, random=True)

    print("\nRESULTS:")
    print("PPO mean daily PnL:", rl_pnls.mean())
    print("Flat mean daily PnL:", flat_pnls.mean())
    print("Momentum mean daily PnL:", mom_pnls.mean())
    print("Random mean daily PnL:", rand_pnls.mean())


if __name__ == "__main__":
    main()
