# agent/backtest_agent.py
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import load_raw_btc_data, build_hourly_kalshi_like_events
from env.kalshi_env import KalshiEnvConfig, KalshiBTCHourlyEnv


FIG_DIR = "reports/figs"


def make_env(events_df, config):
    return KalshiBTCHourlyEnv(events_df, config)


def split_by_day(events_df, train_frac=0.85):
    unique_days = sorted(events_df["day"].unique())
    n_days = len(unique_days)
    split = int(train_frac * n_days)
    train_days = set(unique_days[:split])
    eval_days = set(unique_days[split:])

    train_df = events_df[events_df["day"].isin(train_days)].copy()
    eval_df = events_df[events_df["day"].isin(eval_days)].copy()

    return train_df, eval_df, len(train_days), len(eval_days)


def compute_max_drawdown(equity_curve):
    """Max drawdown from an equity curve (list/array)."""
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return np.nan
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())  # negative number


def summarize_results(name, pnls, drawdowns):
    pnls = np.asarray(pnls, dtype=float)
    drawdowns = np.asarray(drawdowns, dtype=float)

    mean = float(np.mean(pnls))
    std = float(np.std(pnls))
    sharpe = mean / std if std > 1e-8 else np.nan
    win_rate = float(np.mean(pnls > 0.0))
    mdd = float(np.mean(drawdowns)) if drawdowns.size else np.nan

    print(f"\n=== {name} ===")
    print(f"Mean daily PnL: {mean:.4f}")
    print(f"Std daily PnL:  {std:.4f}")
    print(f"Sharpe (per day): {sharpe:.3f}")
    print(f"Win rate: {win_rate*100:.1f}%")
    print(f"Avg max drawdown (episode): {mdd*100:.2f}%")  # drawdown is negative

    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_mdd": mdd,
    }


def run_episode_with_policy(model, vec_env):
    """
    Run one day (episode). Returns:
      equity_curve, actions, final_pnl, max_drawdown
    """
    obs = vec_env.reset()
    done = False
    equity_curve = []
    actions = []

    last_info = None
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        last_info = info
        actions.append(int(action[0]))
        equity_curve.append(float(last_info[0].get("portfolio_value", np.nan)))

    init_cash = vec_env.get_attr("config")[0].initial_cash
    final_pnl = equity_curve[-1] - init_cash if equity_curve else np.nan
    mdd = compute_max_drawdown(equity_curve)
    return equity_curve, actions, float(final_pnl), float(mdd)


def run_episode_hold(vec_env):
    """
    Baseline HOLD: always action 0.
    """
    obs = vec_env.reset()
    done = False
    equity_curve = []
    actions = []
    last_info = None

    while not done:
        action = np.array([0], dtype=int)
        obs, reward, done, info = vec_env.step(action)
        last_info = info
        actions.append(0)
        equity_curve.append(float(last_info[0].get("portfolio_value", np.nan)))

    init_cash = vec_env.get_attr("config")[0].initial_cash
    final_pnl = equity_curve[-1] - init_cash if equity_curve else np.nan
    mdd = compute_max_drawdown(equity_curve)
    return equity_curve, actions, float(final_pnl), float(mdd)


def run_episode_random(vec_env, rng):
    """
    Baseline RANDOM: random action in {0,1,2}.
    """
    obs = vec_env.reset()
    done = False
    equity_curve = []
    actions = []
    last_info = None

    while not done:
        action = np.array([int(rng.integers(0, 3))], dtype=int)
        obs, reward, done, info = vec_env.step(action)
        last_info = info
        actions.append(int(action[0]))
        equity_curve.append(float(last_info[0].get("portfolio_value", np.nan)))

    init_cash = vec_env.get_attr("config")[0].initial_cash
    final_pnl = equity_curve[-1] - init_cash if equity_curve else np.nan
    mdd = compute_max_drawdown(equity_curve)
    return equity_curve, actions, float(final_pnl), float(mdd)


def run_episode_momentum(vec_env, threshold=0.0):
    """
    Simple MOMENTUM baseline:
      - If btc_ret > threshold => BUY YES (action 1)
      - If btc_ret < -threshold => BUY NO (action 2)
      - Else HOLD (action 0)

    Note: btc_ret is in observation index 1 (see kalshi_env.py obs order).
    """
    obs = vec_env.reset()
    done = False
    equity_curve = []
    actions = []
    last_info = None

    while not done:
        # obs is shape (1, obs_dim)
        btc_ret = float(obs[0][1])

        if btc_ret > threshold:
            action = np.array([1], dtype=int)
        elif btc_ret < -threshold:
            action = np.array([2], dtype=int)
        else:
            action = np.array([0], dtype=int)

        obs, reward, done, info = vec_env.step(action)
        last_info = info
        actions.append(int(action[0]))
        equity_curve.append(float(last_info[0].get("portfolio_value", np.nan)))

    init_cash = vec_env.get_attr("config")[0].initial_cash
    final_pnl = equity_curve[-1] - init_cash if equity_curve else np.nan
    mdd = compute_max_drawdown(equity_curve)
    return equity_curve, actions, float(final_pnl), float(mdd)


def pad_and_average(curves):
    if not curves:
        return np.array([])
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = np.asarray(c, dtype=float)
    return np.nanmean(padded, axis=0)


def plot_action_distribution(action_counts, title, outpath):
    labels = ["HOLD (0)", "BUY YES (1)", "BUY NO (2)"]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, action_counts.astype(int))
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_pnl_histogram(pnls_dict, outpath):
    """
    pnls_dict: {name: pnls_array}
    """
    plt.figure(figsize=(10, 5))
    for name, pnls in pnls_dict.items():
        plt.hist(pnls, bins=20, alpha=0.6, label=name)
    plt.title("Daily PnL Histogram (per episode/day)")
    plt.xlabel("PnL ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_equity_curves(sample_curves, avg_curves, init_cash, outpath_prefix):
    # Sample curves
    plt.figure(figsize=(10, 5))
    for name, curve in sample_curves.items():
        plt.plot(curve, label=f"{name} (sample)")
    plt.axhline(init_cash, linestyle="--", linewidth=1, label="Initial cash")
    plt.title("Sample Equity Curves (Episode 1)")
    plt.xlabel("Time step (hourly market)")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_prefix + "_sample.png")
    plt.close()

    # Average curves
    plt.figure(figsize=(10, 5))
    for name, curve in avg_curves.items():
        plt.plot(curve, label=f"{name} (avg)")
    plt.axhline(init_cash, linestyle="--", linewidth=1, label="Initial cash")
    plt.title("Average Equity Curves (Across Episodes)")
    plt.xlabel("Time step (hourly market)")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_prefix + "_avg.png")
    plt.close()


def backtest(
    model_path="agent/models/ppo_kalshi_realdata.zip",
    vecnorm_path="agent/models/vecnormalize.pkl",
    csv_path="data/raw/btcusd_minute.csv",
    n_episodes=30,
    use_eval_split=True,
    momentum_threshold=0.0,
):
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) Load data & build events
    raw_df = load_raw_btc_data(csv_path)
    events_df = build_hourly_kalshi_like_events(raw_df)

    # 2) Split days
    train_df, eval_df, n_train, n_eval = split_by_day(events_df, train_frac=0.85)
    chosen_df = eval_df if use_eval_split else train_df
    print(f"[backtest] Days: train={n_train} | eval={n_eval} | using={'EVAL' if use_eval_split else 'TRAIN'}")

    # 3) Config MUST match training
    config = KalshiEnvConfig(
        initial_cash=10_000.0,
        max_position_per_market=5,
        trade_size=1,
        fee_per_contract=0.002,
        slippage_cents=1,
        reward_scale=100.0,
        turnover_penalty=0.001,
        inventory_penalty=0.0005,
        lookback=5,
    )

    # 4) Vec env + normalization
    vec_env = DummyVecEnv([lambda: make_env(chosen_df, config)])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"[backtest] Loaded VecNormalize: {vecnorm_path}")
    else:
        print(f"[backtest] WARNING: VecNormalize stats not found at {vecnorm_path}.")
        print("[backtest] Results may be misleading without normalization.")

    init_cash = config.initial_cash

    # 5) Load PPO
    model = PPO.load(model_path)
    print(f"[backtest] Loaded model: {model_path}")

    # ---- PPO runs ----
    ppo_pnls, ppo_mdds, ppo_curves, ppo_actions_all = [], [], [], []
    action_counts = np.zeros(3, dtype=int)

    for _ in range(n_episodes):
        curve, acts, pnl, mdd = run_episode_with_policy(model, vec_env)
        ppo_curves.append(curve)
        ppo_actions_all.append(acts)
        ppo_pnls.append(pnl)
        ppo_mdds.append(mdd)
        for a in acts:
            action_counts[a] += 1

    # ---- Baselines ----
    rng = np.random.default_rng(123)

    hold_pnls, hold_mdds, hold_curves = [], [], []
    rand_pnls, rand_mdds, rand_curves = [], [], []
    mom_pnls, mom_mdds, mom_curves = [], [], []

    for _ in range(n_episodes):
        c, a, pnl, mdd = run_episode_hold(vec_env)
        hold_curves.append(c); hold_pnls.append(pnl); hold_mdds.append(mdd)

        c, a, pnl, mdd = run_episode_random(vec_env, rng)
        rand_curves.append(c); rand_pnls.append(pnl); rand_mdds.append(mdd)

        c, a, pnl, mdd = run_episode_momentum(vec_env, threshold=momentum_threshold)
        mom_curves.append(c); mom_pnls.append(pnl); mom_mdds.append(mdd)

    # ---- Summaries ----
    ppo_stats = summarize_results("PPO", ppo_pnls, ppo_mdds)
    hold_stats = summarize_results("HOLD", hold_pnls, hold_mdds)
    rand_stats = summarize_results("RANDOM", rand_pnls, rand_mdds)
    mom_stats = summarize_results(f"MOMENTUM (thr={momentum_threshold})", mom_pnls, mom_mdds)

    # ---- Plots ----
    plot_action_distribution(
        action_counts,
        title="PPO Action Distribution (All Steps)",
        outpath=os.path.join(FIG_DIR, "action_distribution_ppo.png"),
    )

    plot_pnl_histogram(
        {
            "PPO": np.asarray(ppo_pnls),
            "HOLD": np.asarray(hold_pnls),
            "RANDOM": np.asarray(rand_pnls),
            "MOMENTUM": np.asarray(mom_pnls),
        },
        outpath=os.path.join(FIG_DIR, "pnl_histogram.png"),
    )

    sample_curves = {
        "PPO": ppo_curves[0] if ppo_curves else [],
        "HOLD": hold_curves[0] if hold_curves else [],
        "RANDOM": rand_curves[0] if rand_curves else [],
        "MOMENTUM": mom_curves[0] if mom_curves else [],
    }
    avg_curves = {
        "PPO": pad_and_average(ppo_curves),
        "HOLD": pad_and_average(hold_curves),
        "RANDOM": pad_and_average(rand_curves),
        "MOMENTUM": pad_and_average(mom_curves),
    }

    plot_equity_curves(
        sample_curves=sample_curves,
        avg_curves=avg_curves,
        init_cash=init_cash,
        outpath_prefix=os.path.join(FIG_DIR, "equity_curves"),
    )

    # Print where saved
    print("\nSaved figures to:")
    print(f"  - {os.path.join(FIG_DIR, 'action_distribution_ppo.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'pnl_histogram.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'equity_curves_sample.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'equity_curves_avg.png')}")


if __name__ == "__main__":
    backtest(
        model_path="agent/models/ppo_kalshi_realdata.zip",
        vecnorm_path="agent/models/vecnormalize.pkl",
        csv_path="data/raw/btcusd_minute.csv",
        n_episodes=30,
        use_eval_split=True,
        momentum_threshold=0.0,  # try 0.0005 later if you want less trading
    )
