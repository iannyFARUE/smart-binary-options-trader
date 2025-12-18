import os, sys
import sys, os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from stable_baselines3 import PPO

from live_trading.kalshi_api import KalshiClient
from env.kalshi_env import KalshiEnvConfig


# ---------- Config & globals ----------

INITIAL_CASH = 10_000.0  # should match KalshiEnvConfig.initial_cash
LIVE_LOG_PATH = "logs/live/kalshi_live_trades.csv"


# ---------- Helpers ----------

def norm_price_raw(raw, default=0.5):
    """
    Convert raw Kalshi price (cents or 0-1) -> float in [0,1].
    If raw is None or 0, use default.
    """
    if raw is None:
        return default

    if isinstance(raw, str):
        raw = float(raw)

    raw = float(raw)

    if raw <= 0:
        return default

    # If > 1.0, treat as cents (0..100)
    if raw > 1.0:
        return raw / 100.0

    return raw


def build_obs_for_market(market, portfolio_cash, config, btc_mean=20000.0, btc_std=10000.0):
    """
    Build a state vector compatible with the PPO agent, using a simplified
    version of the live runner's logic.
    """
    # Normalized prices
    yes_raw = market.get("yes_ask", None) or market.get("yes_bid", None)
    no_raw = market.get("no_ask", None) or market.get("no_bid", None)

    yes_price = norm_price_raw(yes_raw, default=0.5)
    no_price = norm_price_raw(no_raw, default=(1.0 - yes_price))

    # Clamp a bit to avoid exactly 0 or 1
    yes_price = max(0.01, min(0.99, yes_price))
    no_price = max(0.01, min(0.99, no_price))

    # Approximate BTC proxy from yes_price
    btc_price_proxy = btc_mean + (yes_price - 0.5) * 2 * btc_std
    btc_norm = (btc_price_proxy - btc_mean) / (btc_std if btc_std > 0 else 1.0)

    # Time features from expiration
    expiry = market.get("expiration_time")
    if expiry:
        try:
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            hour = exp_dt.hour
        except Exception:
            hour = datetime.now(timezone.utc).hour
    else:
        hour = datetime.now(timezone.utc).hour

    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)

    # Positions (we keep 0 here; advanced version could derive from portfolio)
    pos_yes = 0.0
    pos_no = 0.0

    cash_norm = portfolio_cash / config.initial_cash

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

    return obs, yes_price, no_price, btc_price_proxy


def action_to_side_action(action: int):
    """
    Map PPO discrete action to human-readable side/action.
    0: do nothing
    1: buy  YES -> side="yes", action="buy"
    2: sell YES -> side="yes", action="sell"
    3: buy  NO  -> side="no",  action="buy"
    4: sell NO  -> side="no",  action="sell"
    """
    if action == 1:
        return "yes", "buy"
    elif action == 2:
        return "yes", "sell"
    elif action == 3:
        return "no", "buy"
    elif action == 4:
        return "no", "sell"
    else:
        return "", ""


@st.cache_resource
def load_model():
    model_path = "agent/models/ppo_kalshi_realdata.zip"
    return PPO.load(model_path)


@st.cache_resource
def get_kalshi_client():
    return KalshiClient()


def load_live_trades(log_path=LIVE_LOG_PATH):
    if not os.path.exists(log_path):
        return pd.DataFrame()

    df = pd.read_csv(log_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------- Streamlit app ----------

def main():
    st.set_page_config(page_title="RL Kalshi BTC Dashboard", layout="wide")

    st.title("📈 RL Bitcoin Kalshi Demo Dashboard")
    st.caption("Demo account · RL agent with PPO · Kalshi BTC hourly thresholds")

    # Sidebar
    st.sidebar.header("Settings & Status")
    refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 15)

    btc_series_ticker = os.getenv("KALSHI_BTC_SERIES_TICKER", "KXBTC")
    st.sidebar.write(f"BTC series ticker: `{btc_series_ticker}`")

    # Load core components
    client = get_kalshi_client()
    config = KalshiEnvConfig()
    model = load_model()

    live_df = load_live_trades()
    print(live_df)

    tab_overview, tab_markets, tab_trades = st.tabs(
        ["📊 Overview", "📈 Markets & Agent", "📜 Trades"]
    )



    # Portfolio section
    with tab_overview:
        st.subheader("Portfolio Overview")

        # Portfolio & equity
        try:
            portfolio = client.get_portfolio()
            cash = float(portfolio.get("cash", INITIAL_CASH))
        except Exception as e:
            st.error(f"Error fetching portfolio: {e}")
            portfolio = {}
            cash = INITIAL_CASH

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio cash (demo)", f"${cash:,.2f}")
        with col2:
            st.metric("Initial cash", f"${INITIAL_CASH:,.2f}")
        with col3:
            if not live_df.empty and "portfolio_cash" in live_df.columns:
                last_equity = live_df["portfolio_cash"].iloc[-1]
                delta = last_equity - INITIAL_CASH
                st.metric(
                    "Live equity (approx.)",
                    f"${last_equity:,.2f}",
                    f"{delta:,.2f} vs start",
                )
            else:
                st.metric("Live equity (approx.)", "N/A")

        st.markdown("### Equity Curve (Live Demo)")

        if not live_df.empty and "portfolio_value" in live_df.columns:
            df_plot = (
                live_df[["timestamp", "portfolio_value"]]
                .sort_values("timestamp")
                .set_index("timestamp")
            )
            st.line_chart(df_plot, height=250)
        else:
            st.info("No live equity data yet. Let the live agent run a bit longer.")


    # Fetch BTC events with nested markets
    with tab_markets:
        st.subheader("BTC Hourly Markets")

        btc_series_ticker = os.getenv("KALSHI_BTC_SERIES_TICKER", "KXBTC")
        st.caption(f"Using BTC series ticker: `{btc_series_ticker}`")

        # Fetch BTC events with nested markets
        try:
            markets_resp = client.get_markets(
                series_ticker=btc_series_ticker,
                status="open",
                with_nested_markets=True,
                limit=50,
                filter_liquid=True
            )
            markets = markets_resp.get("markets", [])
        except Exception as e:
            st.error(f"Error fetching BTC markets: {e}")
            markets = []

        markets_rows = []
        for m in markets:
            ev_ticker = m.get("event_ticker")
            ev_title = m.get("title")
            ticker = m.get("ticker")
            expiry = m.get("expiration_time")
            yes_raw = m.get("yes_ask", None) or m.get("yes_bid", None)
            no_raw = m.get("no_ask", None) or m.get("no_bid", None)
            yes_p = norm_price_raw(yes_raw, default=0.5)
            no_p = norm_price_raw(no_raw, default=(1.0 - yes_p))

            markets_rows.append(
                {
                    "event_ticker": ev_ticker,
                    "event_title": ev_title,
                    "market_ticker": ticker,
                    "expiration_time": expiry,
                    "YES price (est)": round(yes_p, 3),
                    "NO price (est)": round(no_p, 3),
                }
            )

        if markets_rows:
            markets_df = pd.DataFrame(markets_rows)
            markets_df = markets_df.sort_values("expiration_time")
            st.dataframe(
                markets_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open BTC markets found at the moment.")

        st.markdown("---")
        st.subheader("🤖 RL Agent Recommendation")

        if markets_rows:
            # pick the next expiring market
            next_market_row = markets_df.iloc[0]
            next_market_ticker = next_market_row["market_ticker"]

            # find the corresponding market JSON
            selected_market = None
            for ev in events:
                for m in ev.get("markets", []):
                    if m.get("ticker") == next_market_ticker:
                        selected_market = m
                        break
                if selected_market:
                    break

            if selected_market is not None:
                obs, yes_p, no_p, btc_proxy = build_obs_for_market(
                    selected_market, portfolio_cash=cash, config=config
                )

                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                side, act = action_to_side_action(action)

                st.write(f"**Next expiring market:** `{next_market_ticker}`")
                st.write(f"- Approx. BTC proxy: `{btc_proxy:.2f}`")
                st.write(f"- YES price ≈ `{yes_p:.2f}`, NO price ≈ `{no_p:.2f}`")

                if side == "" or act == "":
                    st.warning("Agent recommendation: **do nothing** for this market.")
                else:
                    nice_side = "YES" if side == "yes" else "NO"
                    nice_action = "BUY" if act == "buy" else "SELL"

                    bg_color = "#16a34a" if act == "buy" else "#dc2626"  # green / red

                    st.markdown(
                        f"""
                        <div style="
                            padding: 1rem;
                            border-radius: 0.75rem;
                            background-color: {bg_color};
                            color: white;
                            font-weight: 600;
                            font-size: 1.05rem;
                        ">
                            Agent recommendation: {nice_action} {nice_side}
                            (1 contract, demo)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Could not match next market in event list.")
        else:
            st.info("No markets to evaluate recommendation on.")


    # Recent trades table
    with tab_trades:
        st.subheader("Recent Live Trades")

        if live_df.empty:
            st.info("No live trades logged yet. Let the live agent run for a while.")
        else:
            live_df_sorted = live_df.sort_values("timestamp", ascending=False)

            # basic stats
            st.markdown("### Quick Stats")
            last_equity = live_df_sorted["portfolio_value"].iloc[0]
            delta = last_equity - INITIAL_CASH
            n_trades = len(live_df_sorted)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Last equity snapshot", f"${last_equity:,.2f}")
            with col_b:
                st.metric("PnL vs start", f"${delta:,.2f}")
            with col_c:
                st.metric("Logged trade events", f"{n_trades}")

            st.markdown("### Trade Log (latest 50)")
            st.dataframe(
                live_df_sorted.head(50),
                use_container_width=True,
                hide_index=True,
            )


    # Auto-refresh using Streamlit's built-in rerun trick
    st.sidebar.markdown("---")
    st.sidebar.caption("This page auto-refreshes based on the selected interval.")
    # ---------- Auto-refresh ----------
    import time

    last_run = st.session_state.get("last_run", 0)
    now = time.time()

    if now - last_run > refresh_seconds:
        st.session_state["last_run"] = now
        st.rerun()



if __name__ == "__main__":
    main()
