import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO

from env.kalshi_env import KalshiBTCHourlyEnv, KalshiEnvConfig
from env.data_loader import load_raw_btc_data, build_hourly_kalshi_like_events
from agent.baselines import baseline_flat, baseline_momentum, baseline_random


# ---------- evaluation helpers ----------

def evaluate_policy(env, model, episodes=200):
    rewards = []
    pnls = []

    for _ in tqdm(range(episodes), desc="PPO episodes"):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
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

    for _ in tqdm(range(episodes), desc=f"{policy_fn.__name__} episodes"):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
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


# ---------- metrics helpers ----------

def compute_metrics(pnls, initial_cash):
    """
    pnls: array of daily PnL (in dollars)
    returns: dict of mean_pnl, std_pnl, sharpe, max_drawdown, etc.
    """
    pnls = np.asarray(pnls)
    daily_returns = pnls / initial_cash

    mean_pnl = pnls.mean()
    std_pnl = pnls.std()
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    sharpe = mean_ret / std_ret if std_ret > 1e-8 else np.nan

    # Build cumulative equity curve for max drawdown
    equity = initial_cash + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak)
    max_drawdown = drawdown.min()  # negative number

    return {
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "equity_curve": equity,
    }


# ---------- plotting helpers ----------

def plot_equity_curves(results_dict, initial_cash, out_path):
    """
    results_dict: name -> metrics dict from compute_metrics
    """
    plt.figure(figsize=(10, 6))
    for name, metrics in results_dict.items():
        equity = metrics["equity_curve"]
        plt.plot(equity, label=name)

    plt.axhline(initial_cash, linestyle="--", linewidth=1, label="Initial cash")
    plt.xlabel("Episode (day)")
    plt.ylabel("Equity ($)")
    plt.title("Equity Curves (Cumulative PnL over Episodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pnl_histograms(pnl_dict, out_path):
    """
    pnl_dict: name -> pnls (array)
    """
    plt.figure(figsize=(10, 6))
    for name, pnls in pnl_dict.items():
        plt.hist(pnls, bins=30, alpha=0.5, label=name)

    plt.xlabel("Daily PnL ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Daily PnL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_boxplot(pnl_dict, out_path):
    names = list(pnl_dict.keys())
    data = [np.asarray(pnl_dict[name]) for name in names]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=names, showfliers=False)
    plt.ylabel("Daily PnL ($)")
    plt.title("Daily PnL by Strategy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- main ----------

def main():
    # 1. Load data & environment
    raw_path = "data/raw/btcusd_minute.csv"
    raw_df = load_raw_btc_data(raw_path)
    events_df = build_hourly_kalshi_like_events(raw_df)

    config = KalshiEnvConfig()
    env = KalshiBTCHourlyEnv(events_df, config)
    initial_cash = config.initial_cash

    # 2. Load trained PPO model
    model_path = "agent/models/ppo_kalshi_realdata.zip"
    model = PPO.load(model_path)

    episodes = 200  # you can increase if it's fast enough

    # 3. Evaluate all strategies
    print("Evaluating PPO...")
    rl_rewards, rl_pnls = evaluate_policy(env, model, episodes=episodes)

    print("Evaluating flat baseline...")
    flat_rewards, flat_pnls = evaluate_baseline(env, baseline_flat, episodes=episodes)

    print("Evaluating momentum baseline...")
    mom_rewards, mom_pnls = evaluate_baseline(env, baseline_momentum, episodes=episodes)

    print("Evaluating random baseline...")
    rand_rewards, rand_pnls = evaluate_baseline(
        env, baseline_random, episodes=episodes, random=True
    )

    # 4. Compute metrics
    rl_metrics = compute_metrics(rl_pnls, initial_cash)
    flat_metrics = compute_metrics(flat_pnls, initial_cash)
    mom_metrics = compute_metrics(mom_pnls, initial_cash)
    rand_metrics = compute_metrics(rand_pnls, initial_cash)

    results = {
        "PPO": rl_metrics,
        "Flat": flat_metrics,
        "Momentum": mom_metrics,
        "Random": rand_metrics,
    }
    pnl_dict = {
        "PPO": rl_pnls,
        "Flat": flat_pnls,
        "Momentum": mom_pnls,
        "Random": rand_pnls,
    }

    # 5. Print metrics (for your report)
    print("\n=== Metrics (per day) ===")
    for name, m in results.items():
        print(f"\n{name}:")
        print(f"  mean PnL       : {m['mean_pnl']:.4f} $")
        print(f"  std  PnL       : {m['std_pnl']:.4f} $")
        print(f"  mean return    : {m['mean_ret']:.6f}")
        print(f"  std  return    : {m['std_ret']:.6f}")
        print(f"  Sharpe (daily) : {m['sharpe']:.3f}")
        print(f"  max drawdown   : {m['max_drawdown']:.2f} $")

    # 6. Make plots
    os.makedirs("reports/figs", exist_ok=True)

    plot_equity_curves(
        results,
        initial_cash,
        out_path="reports/figs/equity_curves.png",
    )
    plot_pnl_histograms(
        pnl_dict,
        out_path="reports/figs/pnl_histograms.png",
    )
    plot_boxplot(
        pnl_dict,
        out_path="reports/figs/pnl_boxplot.png",
    )

    print("\nSaved plots to reports/figs/:")
    print("  - equity_curves.png")
    print("  - pnl_histograms.png")
    print("  - pnl_boxplot.png")


if __name__ == "__main__":
    main()
