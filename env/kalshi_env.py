from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Config
# -----------------------------

@dataclass
class KalshiEnvConfig:
    initial_cash: float = 10_000.0

    # Trading constraints
    max_position_per_market: int = 5          # max contracts you can hold in a single hourly market
    trade_size: int = 1                       # contracts per action (keep it simple for PPO)

    # Frictions (realism)
    fee_per_contract: float = 0.002           # $0.002 per contract
    slippage_cents: int = 1                   # 1 cent worse than observed price

    # Reward shaping
    reward_scale: float = 100.0               # amplify reward for PPO stability
    turnover_penalty: float = 0.001           # penalty per contract traded
    inventory_penalty: float = 0.0005         # penalty * (inventory^2) per step (discourage big exposure)

    # Feature engineering
    lookback: int = 5                         # rolling window for volatility


# -----------------------------
# Feature utilities
# -----------------------------

def _ensure_features(events_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Ensures btc_ret and btc_vol exist.
    Assumes events_df has: day, hour, btc_price
    """
    df = events_df.copy()

    # basic sanity
    for col in ["day", "hour", "btc_price", "yes_price", "no_price", "outcome"]:
        if col not in df.columns:
            raise ValueError(f"events_df missing required column: {col}")

    # sort for consistent rolling calcs
    df = df.sort_values(["day", "hour"]).reset_index(drop=True)

    if "btc_ret" not in df.columns:
        df["btc_ret"] = df.groupby("day")["btc_price"].pct_change().fillna(0.0)

    if "btc_vol" not in df.columns:
        # rolling std of returns per day
        df["btc_vol"] = (
            df.groupby("day")["btc_ret"]
              .rolling(lookback, min_periods=1)
              .std()
              .reset_index(level=0, drop=True)
              .fillna(0.0)
        )

    # clamp market prices to [0,1]
    df["yes_price"] = df["yes_price"].clip(0.0, 1.0)
    df["no_price"] = df["no_price"].clip(0.0, 1.0)

    # outcome must be 0/1
    df["outcome"] = df["outcome"].astype(int).clip(0, 1)

    return df


def _hour_sin_cos(hour: int) -> Tuple[float, float]:
    hr = int(hour) % 24
    rad = 2 * np.pi * hr / 24.0
    return float(np.sin(rad)), float(np.cos(rad))


# -----------------------------
# Environment
# -----------------------------

class KalshiBTCHourlyEnv(gym.Env):
    """
    Realistic-ish backtest env for Kalshi-style hourly BTC threshold contracts.

    IMPORTANT DESIGN:
    - Each step is ONE hourly market (one contract).
    - Any positions taken in that market are settled immediately at the end of the step
      using the market's outcome (0/1). This matches the reality that a 10AM market
      resolves at 10AM and doesn't carry to 11AM.

    Actions (Discrete):
      0 = HOLD
      1 = BUY YES (trade_size contracts)
      2 = BUY NO  (trade_size contracts)

    State includes:
      btc_norm, btc_ret, btc_vol, yes_price, no_price, hour_sin, hour_cos,
      inventory (contracts in current market), cash_norm

    Reward:
      reward = (delta_equity - penalties) * reward_scale
      - turnover penalty discourages churn
      - inventory penalty discourages maxing size constantly
    """

    metadata = {"render_modes": []}

    def __init__(self, events_df: pd.DataFrame, config: Optional[KalshiEnvConfig] = None):
        super().__init__()
        self.config = config or KalshiEnvConfig()

        self.raw_data = _ensure_features(events_df, lookback=self.config.lookback)

        # unique days
        self.days = sorted(self.raw_data["day"].unique().tolist())

        # Precompute BTC normalization stats (global)
        self.btc_mean = float(self.raw_data["btc_price"].mean())
        self.btc_std = float(self.raw_data["btc_price"].std()) if float(self.raw_data["btc_price"].std()) > 0 else 1.0

        # Action space: HOLD, BUY YES, BUY NO
        self.action_space = spaces.Discrete(3)

        # Observation space (10 dims)
        # [btc_norm, btc_ret, btc_vol, yes_p, no_p, hour_sin, hour_cos, inv, cash_norm]
        low = np.array([-10, -1, 0, 0, 0, -1, -1, 0, 0], dtype=np.float32)
        high = np.array([10, 1, 10, 1, 1, 1, 1, self.config.max_position_per_market, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state
        self.day_idx: int = 0
        self.t: int = 0
        self.day_events: pd.DataFrame = pd.DataFrame()
        self.current_row: Optional[pd.Series] = None

        # Portfolio state
        self.cash: float = self.config.initial_cash
        self.inventory: int = 0  # contracts held in current market (resets each step)

        # RNG
        self.np_random = np.random.default_rng()

    # -------------
    # Gym API
    # -------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # choose a random day to reduce overfitting
        self.day_idx = int(self.np_random.integers(0, len(self.days)))
        day = self.days[self.day_idx]

        self.day_events = (
            self.raw_data[self.raw_data["day"] == day]
            .sort_values("hour")
            .reset_index(drop=True)
        )

        self.t = 0
        self.cash = self.config.initial_cash
        self.inventory = 0

        self.current_row = self.day_events.iloc[self.t]
        obs = self._get_obs()

        info = {
            "day": day,
            "t": self.t,
            "cash": self.cash,
            "portfolio_value": self.cash,
        }
        return obs, info

    def step(self, action: int):
        assert self.current_row is not None, "Call reset() before step()."

        row = self.current_row

        yes_p = float(row["yes_price"])  # 0..1 ($)
        no_p = float(row["no_price"])    # 0..1 ($)
        outcome = int(row["outcome"])    # 0/1

        prev_equity = self.cash  # no carry positions; equity starts as cash each step

        # --- execute trade ---
        traded_contracts = 0
        if action == 1:  # BUY YES
            can_buy = self.inventory + self.config.trade_size <= self.config.max_position_per_market
            if can_buy:
                # pay worse price due to slippage
                exec_price = min(1.0, yes_p + self.config.slippage_cents / 100.0)
                cost = self.config.trade_size * exec_price + self.config.trade_size * self.config.fee_per_contract
                if self.cash >= cost:
                    self.cash -= cost
                    self.inventory += self.config.trade_size
                    traded_contracts = self.config.trade_size

        elif action == 2:  # BUY NO
            can_buy = self.inventory + self.config.trade_size <= self.config.max_position_per_market
            if can_buy:
                exec_price = min(1.0, no_p + self.config.slippage_cents / 100.0)
                cost = self.config.trade_size * exec_price + self.config.trade_size * self.config.fee_per_contract
                if self.cash >= cost:
                    self.cash -= cost
                    self.inventory += self.config.trade_size
                    traded_contracts = self.config.trade_size

        # HOLD (0): nothing

        # --- settle this hourly market immediately ---
        # If agent bought YES: payoff = 1 if outcome==1 else 0
        # If agent bought NO : payoff = 1 if outcome==0 else 0
        payoff = 0.0
        if action == 1:  # YES contracts
            payoff = float(self.inventory) * float(outcome)
        elif action == 2:  # NO contracts
            payoff = float(self.inventory) * float(1 - outcome)

        self.cash += payoff

        # inventory resets each market
        inv_after = self.inventory
        self.inventory = 0

        # --- reward ---
        equity = self.cash
        pnl = equity - prev_equity  # realized PnL from this market

        penalty_turnover = self.config.turnover_penalty * abs(traded_contracts)
        penalty_inventory = self.config.inventory_penalty * float(inv_after ** 2)

        reward = (pnl - penalty_turnover - penalty_inventory) * self.config.reward_scale

        # --- advance time ---
        self.t += 1
        done = self.t >= len(self.day_events)

        if not done:
            self.current_row = self.day_events.iloc[self.t]
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "day": self.days[self.day_idx],
            "t": self.t,
            "cash": self.cash,
            "portfolio_value": self.cash,
            "pnl": pnl,
            "traded_contracts": traded_contracts,
            "inv_before_settle": inv_after,
            "outcome": outcome,
            "yes_price": yes_p,
            "no_price": no_p,
        }

        return obs, float(reward), bool(done), False, info

    # -------------
    # Internals
    # -------------

    def _get_obs(self) -> np.ndarray:
        row = self.current_row
        assert row is not None

        btc_price = float(row["btc_price"])
        btc_ret = float(row["btc_ret"])
        btc_vol = float(row["btc_vol"])

        yes_p = float(row["yes_price"])
        no_p = float(row["no_price"])

        hour = int(row["hour"])
        hour_sin, hour_cos = _hour_sin_cos(hour)

        btc_norm = (btc_price - self.btc_mean) / self.btc_std
        cash_norm = self.cash / self.config.initial_cash

        # inventory is always 0 at observation time in this env design
        # (since each market is one step). That’s OK; we keep the field for clarity.
        inv = 0.0

        obs = np.array(
            [
                btc_norm,
                btc_ret,
                btc_vol,
                yes_p,
                no_p,
                hour_sin,
                hour_cos,
                inv,
                cash_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def render(self):
        # optional: implement later
        pass
