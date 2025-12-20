import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import load_raw_btc_data, build_hourly_kalshi_like_events
from env.kalshi_env import KalshiBTCHourlyEnv, KalshiEnvConfig


def train():
    # 1) Load raw BTC minute data
    raw_path = "data/raw/btcusd_minute.csv"  # adjust
    print("[train] Loading raw BTC data...")
    raw_df = load_raw_btc_data(raw_path)

    # 2) Build Kalshi-like hourly events
    print("[train] Building hourly kalshi-like events...")
    events_df = build_hourly_kalshi_like_events(raw_df)

    # 3) Train / eval split by day
    # events_df has integer day index (0..N-1)
    unique_days = sorted(events_df["day"].unique())
    n_days = len(unique_days)
    split = int(0.85 * n_days)
    train_days = set(unique_days[:split])
    eval_days = set(unique_days[split:])

    train_df = events_df[events_df["day"].isin(train_days)].copy()
    eval_df = events_df[events_df["day"].isin(eval_days)].copy()

    print(f"[train] Days total={n_days} | train={len(train_days)} | eval={len(eval_days)}")

    # 4) Env config (tune these later)
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

    # 5) Wrap env for SB3 (VecNormalize helps a lot)
    def make_train_env():
        return KalshiBTCHourlyEnv(train_df, config)

    vec_env = DummyVecEnv([make_train_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 6) Logging
    log_dir = "./logs/ppo_kalshi/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # 7) PPO hyperparameters (more “real” than toy defaults)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,          # bigger rollout
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.005,        # slightly lower entropy than before
        clip_range=0.2,
        n_epochs=10,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
    )
    model.set_logger(new_logger)

    # 8) Train
    total_steps = 300_000   # start smaller; increase later once stable
    print(f"[train] Training PPO for {total_steps} steps...")
    model.learn(total_timesteps=total_steps)

    # 9) Save model + VecNormalize stats (CRITICAL!)
    model_path = "./agent/models/ppo_kalshi_realdata.zip"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    vecnorm_path = "./agent/models/vecnormalize.pkl"
    vec_env.save(vecnorm_path)

    print(f"[train] Saved model to {model_path}")
    print(f"[train] Saved VecNormalize stats to {vecnorm_path}")

    # Optional: quick evaluation loop (sanity)
    print("[train] Quick eval sanity check...")
    quick_eval(eval_df, config, model_path, vecnorm_path)


def quick_eval(eval_df, config, model_path, vecnorm_path, n_episodes=5):
    """
    Minimal evaluation after training.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_eval_env():
        return KalshiBTCHourlyEnv(eval_df, config)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path)

    pnls = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        last_info = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            last_info = info
        # last_info is a list with one dict (because vec env)
        if last_info:
            pv = last_info[0].get("portfolio_value", config.initial_cash)
            pnls.append(pv - config.initial_cash)

    pnls = np.array(pnls)
    print(f"[eval] mean pnl={pnls.mean():.4f} | std={pnls.std():.4f} | n={len(pnls)}")


if __name__ == "__main__":
    train()
