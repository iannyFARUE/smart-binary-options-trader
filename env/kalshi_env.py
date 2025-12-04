import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class KalshiEnvConfig:
    initial_cash: float = 10_000.0
    trading_hours: Tuple[int, ...] = tuple(range(9, 25))  # 9..24 inclusive
    max_yes_position: int = 100
    max_no_position: int = 100
    fee_per_contract: float = 0.002  # e.g. 0.2% per contract notional
    spread_bps: float = 0.01         # 1% spread
    obs_normalize_cash: bool = True


class KalshiBTCHourlyEnv(gym.Env):
    """
    Gymnasium-style environment simulating Kalshi-like hourly BTC threshold markets.

    One episode = one trading day.
    One step     = one hourly contract (9:00, 10:00, ..., 24:00).

    DataFrame expected columns:
        - day         : episode identifier (int or date)
        - hour        : integer trading hour (9..24)
        - btc_price   : float
        - yes_price   : float in [0, 1]
        - no_price    : float in [0, 1]
        - outcome     : 1 if BTC > threshold at that hour, else 0

    Prices are in probability space (0..1). Reward is change in portfolio value.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[KalshiEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        assert {"day", "hour", "btc_price", "yes_price", "no_price", "outcome"}.issubset(
            data.columns
        ), "DataFrame missing required columns."

        self.raw_data = data.copy().reset_index(drop=True)
        self.config = config or KalshiEnvConfig()
        self.render_mode = render_mode

        # Precompute per-day groups for fast reset
        self.days = sorted(self.raw_data["day"].unique())
        self.day_to_idx = {d: i for i, d in enumerate(self.days)}

        # Compute stats for normalization
        self.btc_mean = self.raw_data["btc_price"].mean()
        self.btc_std = self.raw_data["btc_price"].std() or 1.0

        # Define action & observation spaces
        self.action_space = spaces.Discrete(5)  # 0..4, as defined above

        # Observation: [btc_norm, yes, no, hour_sin, hour_cos, pos_yes, pos_no, cash_norm]
        obs_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal state
        self.current_day: Optional[int] = None
        self.day_data: Optional[pd.DataFrame] = None
        self.t: int = 0  # index within the trading_hours sequence

        self.cash: float = self.config.initial_cash
        self.position_yes: int = 0
        self.position_no: int = 0
        self.portfolio_value: float = self.config.initial_cash

    # ------------- core gym methods -------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        # Choose random day if not specified
        if options and "day" in options:
            day = options["day"]
        else:
            day = self.np_random.choice(self.days)

        self.current_day = day
        self.day_data = (
            self.raw_data[self.raw_data["day"] == day]
            .sort_values("hour")
            .reset_index(drop=True)
        )

        # Filter only trading hours we care about
        self.day_data = self.day_data[
            self.day_data["hour"].isin(self.config.trading_hours)
        ].reset_index(drop=True)

        # Reset time index and portfolio
        self.t = 0
        self.cash = self.config.initial_cash
        self.position_yes = 0
        self.position_no = 0
        self.portfolio_value = self._compute_portfolio_value()

        obs = self._get_observation()
        info = {"day": self.current_day, "t": self.t}
        return obs, info

    def step(self, action: int):
        assert self.day_data is not None, "Environment not reset."

        # Get current row (market info for this hour)
        row = self.day_data.iloc[self.t]
        yes_price = float(row["yes_price"])
        no_price = float(row["no_price"])

        # Apply action (update positions & cash)
        self._apply_action(action, yes_price, no_price)

        # Move to next time step
        prev_portfolio_value = self.portfolio_value

        # At resolution time (end of hour), contracts settle:
        # For simplicity: we assume this hour resolves immediately after trading.
        outcome = int(row["outcome"])
        self._settle_contracts(outcome, yes_price, no_price)

        self.portfolio_value = self._compute_portfolio_value()

        # Reward: change in portfolio value relative to initial capital
        reward = (self.portfolio_value - prev_portfolio_value) / self.config.initial_cash

        self.t += 1
        terminated = self.t >= len(self.day_data)  # end of day
        truncated = False  # no truncation logic yet

        if not terminated:
            obs = self._get_observation()
        else:
            obs = self._get_observation()  # could also return final obs

        info = {
            "day": self.current_day,
            "t": self.t,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_yes": self.position_yes,
            "position_no": self.position_no,
            "outcome": outcome,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Day {self.current_day}, step {self.t}, "
            f"cash={self.cash:.2f}, YES={self.position_yes}, "
            f"NO={self.position_no}, portfolio={self.portfolio_value:.2f}"
        )

    # ------------- internal helpers -------------

    def _get_observation(self) -> np.ndarray:
        """Builds current observation vector."""
        assert self.day_data is not None, "Env not initialized."
        t_idx = min(self.t, len(self.day_data) - 1)
        row = self.day_data.iloc[t_idx]

        btc_price = float(row["btc_price"])
        yes_price = float(row["yes_price"])
        no_price = float(row["no_price"])
        hour = int(row["hour"])

        btc_norm = (btc_price - self.btc_mean) / self.btc_std

        # Time-of-day encoding
        hour_rad = 2 * np.pi * hour / 24.0
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # Positions & cash
        pos_yes = float(self.position_yes)
        pos_no = float(self.position_no)
        cash_norm = (
            self.cash / self.config.initial_cash
            if self.config.obs_normalize_cash
            else self.cash
        )

        obs = np.array(
            [
                btc_norm,
                yes_price,
                no_price,
                hour_sin,
                hour_cos,
                pos_yes,
                pos_no,
                cash_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_portfolio_value(self) -> float:
        """Current cash + mark-to-market of open positions."""
        if self.day_data is None:
            return self.cash

        t_idx = min(self.t, len(self.day_data) - 1)
        row = self.day_data.iloc[t_idx]
        yes_price = float(row["yes_price"])
        no_price = float(row["no_price"])

        # Mark-to-market using current mid prices
        mtm_yes = self.position_yes * yes_price
        mtm_no = self.position_no * no_price
        return self.cash + mtm_yes + mtm_no

    def _apply_action(self, action: int, yes_price: float, no_price: float):
        """
        Update positions & cash given the chosen action.
        We apply a simple spread & fee model:

        - buy price = mid * (1 + spread/2)
        - sell price = mid * (1 - spread/2)
        - fee_per_contract charged on notional each trade
        """
        spread = self.config.spread_bps
        buy_yes_price = yes_price * (1.0 + spread / 2.0)
        sell_yes_price = yes_price * (1.0 - spread / 2.0)
        buy_no_price = no_price * (1.0 + spread / 2.0)
        sell_no_price = no_price * (1.0 - spread / 2.0)

        fee = self.config.fee_per_contract

        if action == 1:  # Buy 1 YES
            if self.position_yes < self.config.max_yes_position:
                cost = buy_yes_price + fee
                self.cash -= cost
                self.position_yes += 1

        elif action == 2:  # Sell 1 YES
            if self.position_yes > -self.config.max_yes_position:
                proceeds = sell_yes_price - fee
                self.cash += proceeds
                self.position_yes -= 1

        elif action == 3:  # Buy 1 NO
            if self.position_no < self.config.max_no_position:
                cost = buy_no_price + fee
                self.cash -= cost
                self.position_no += 1

        elif action == 4:  # Sell 1 NO
            if self.position_no > -self.config.max_no_position:
                proceeds = sell_no_price - fee
                self.cash += proceeds
                self.position_no -= 1

        # action == 0 → do nothing

    def _settle_contracts(self, outcome: int, yes_price: float, no_price: float):
        """
        Settle contracts at hour resolution.

        Simplification: we assume all YES/NO positions are for the current hour
        and are fully settled at the end of this step.

        YES pays outcome * 1
        NO  pays (1 - outcome) * 1
        """
        # Payout per contract
        yes_payout = float(outcome)         # 1 if outcome=1 else 0
        no_payout = float(1 - outcome)      # 1 if outcome=0 else 0

        # Cash in payouts
        self.cash += self.position_yes * yes_payout
        self.cash += self.position_no * no_payout

        # Clear positions for next hour
        self.position_yes = 0
        self.position_no = 0
