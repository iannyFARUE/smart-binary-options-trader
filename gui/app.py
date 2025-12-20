# gui/app.py
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

# -----------------------
# Ensure project root is importable
# -----------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from live_trading.kalshi_api import KalshiClient
from env.kalshi_env import KalshiEnvConfig

# -----------------------
# Paths
# -----------------------
LIVE_TRADES_CSV = "logs/live/kalshi_live_trades.csv"
RESOLUTIONS_CSV = "logs/live/kalshi_resolutions.csv"

DEFAULT_SERIES_TICKER = os.getenv("KALSHI_BTC_SERIES_TICKER", "KXBTC")


# -----------------------
# Helpers
# -----------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def norm_price_raw(raw, default=0.5) -> float:
    """Convert cents (0..100) OR prob (0..1) -> prob in [0,1]."""
    if raw is None:
        return default
    try:
        v = float(raw)
    except Exception:
        return default
    if v <= 0:
        return default
    return v / 100.0 if v > 1.0 else v


def parse_iso_z(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def effective_market_time(market: dict):
    """Prefer close_time/expected_expiration_time over expiration_time for 'when it matters'."""
    close_time = parse_iso_z(market.get("close_time"))
    expected_exp = parse_iso_z(market.get("expected_expiration_time"))
    expiration = parse_iso_z(market.get("expiration_time"))
    return close_time or expected_exp or expiration


def extract_portfolio_cash(portfolio_resp: dict, fallback: float) -> float:
    """
    Kalshi demo often returns:
      {"balance": 240273, ...}  # cents
    Return cash in DOLLARS.
    """
    if not isinstance(portfolio_resp, dict):
        return float(fallback)

    if portfolio_resp.get("cash") is not None:
        return float(portfolio_resp["cash"])

    bal = portfolio_resp.get("balance", None)
    if isinstance(bal, (int, float)):
        return float(bal) / 100.0  # cents -> dollars

    if isinstance(bal, dict):
        for k in ("available_cash", "cash", "balance", "available_funds", "funds"):
            v = bal.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
                return fv / 100.0 if fv > 1000 else fv
            except Exception:
                pass

    return float(fallback)


def mode_badge(is_paper_fill):
    try:
        return "🟣 PAPER" if int(is_paper_fill) == 1 else "🟢 LIVE"
    except Exception:
        return "—"


def action_label(action_id: int) -> str:
    # NEW action space: 0 HOLD, 1 BUY YES, 2 BUY NO
    if action_id == 0:
        return "HOLD"
    if action_id == 1:
        return "BUY YES"
    if action_id == 2:
        return "BUY NO"
    return str(action_id)


def build_obs_for_market_v9(market: dict, cash: float, config: KalshiEnvConfig):
    """
    9-dim observation to match the newer live runner:
      [btc_norm, btc_ret, btc_vol, yes_p, no_p, hour_sin, hour_cos, inv, cash_norm]
    Live GUI doesn't have BTC feed; placeholders for first 3 + inv.
    """
    yes_raw = market.get("yes_ask") or market.get("yes_bid")
    no_raw = market.get("no_ask") or market.get("no_bid")

    yes_p = norm_price_raw(yes_raw, default=0.5)
    no_p = norm_price_raw(no_raw, default=(1.0 - yes_p))

    yes_p = float(max(0.01, min(0.99, yes_p)))
    no_p = float(max(0.01, min(0.99, no_p)))

    eff = effective_market_time(market)
    hour = eff.hour if eff else datetime.now(timezone.utc).hour

    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = float(np.sin(hour_rad))
    hour_cos = float(np.cos(hour_rad))

    btc_norm = 0.0
    btc_ret = 0.0
    btc_vol = 0.0
    inv = 0.0
    cash_norm = float(cash / config.initial_cash)

    obs = np.array(
        [btc_norm, btc_ret, btc_vol, yes_p, no_p, hour_sin, hour_cos, inv, cash_norm],
        dtype=np.float32,
    )
    return obs, yes_p, no_p, eff


@st.cache_resource
def get_client():
    return KalshiClient()


@st.cache_resource
def get_model():
    # Adjust if you rename your model
    return PPO.load("agent/models/ppo_kalshi_realdata.zip")


# -----------------------
# App
# -----------------------
def main():
    st.set_page_config(page_title="Kalshi RL BTC Demo", layout="wide")

    st.title("📈 Kalshi RL BTC Demo Dashboard")
    st.caption("Demo account • PPO agent • Live runner logs • LIVE vs PAPER badge")

    # Sidebar controls
    st.sidebar.header("Controls")
    refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 15)
    series_ticker = st.sidebar.text_input("Series ticker", value=DEFAULT_SERIES_TICKER)

    # Load core
    client = get_client()
    config = KalshiEnvConfig()
    model = get_model()

    # Load logs
    trades_df = load_csv(LIVE_TRADES_CSV)
    resolutions_df = load_csv(RESOLUTIONS_CSV)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Markets & Agent", "📜 Trades"])

    # -----------------------
    # Overview
    # -----------------------
    with tab1:
        st.subheader("Portfolio")

        try:
            portfolio = client.get_portfolio()
            cash = extract_portfolio_cash(portfolio, fallback=config.initial_cash)
        except Exception as e:
            st.error(f"Portfolio fetch error: {e}")
            cash = float(config.initial_cash)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cash (demo)", f"${cash:,.2f}")
        c2.metric("Initial cash", f"${float(config.initial_cash):,.2f}")
        c3.metric("BTC series ticker", series_ticker)
        c4.metric("Trades logged", f"{len(trades_df):,}")

        st.markdown("### LIVE vs PAPER split")
        if not trades_df.empty and "is_paper_fill" in trades_df.columns:
            is_paper = pd.to_numeric(trades_df["is_paper_fill"], errors="coerce").fillna(0).astype(int)
            paper = int((is_paper == 1).sum())
            live = int((is_paper == 0).sum())
            st.write(f"🟢 LIVE: **{live}**   •   🟣 PAPER: **{paper}**")
        else:
            st.info("No paper-fill column found yet (or no trades).")

        st.markdown("### Latest resolutions (outcome evidence)")
        if resolutions_df.empty:
            st.info("No resolutions logged yet. (The live runner logs this when markets finalize.)")
        else:
            st.dataframe(
                resolutions_df.sort_values("timestamp", ascending=False).head(25),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Recent trades (with badges)")
        if trades_df.empty:
            st.info("No trades logged yet. Run your live agent to create logs.")
        else:
            show = trades_df.sort_values("timestamp", ascending=False).head(25).copy()
            if "is_paper_fill" in show.columns:
                show["mode"] = show["is_paper_fill"].apply(mode_badge)
            else:
                show["mode"] = "—"
            if "action_id" in show.columns:
                show["action"] = show["action_id"].apply(lambda x: action_label(int(x)) if pd.notna(x) else "")
            preferred = [
                "timestamp", "mode", "ticker", "action", "side", "count",
                "entry_price_prob", "paper_reason",
            ]
            cols = [c for c in preferred if c in show.columns] + [c for c in show.columns if c not in preferred]
            st.dataframe(show[cols], use_container_width=True, hide_index=True)

    # -----------------------
    # Markets & Agent
    # -----------------------
    with tab2:
        st.subheader("Open BTC markets")

        try:
            resp = client.get_markets(series_ticker=series_ticker, status="open")
            markets = resp.get("markets", [])
            if not isinstance(markets, list):
                markets = []
        except Exception as e:
            st.error(f"Market fetch error: {e}")
            markets = []

        if not markets:
            st.info("No open BTC markets found right now.")
        else:
            rows = []
            for m in markets:
                eff = effective_market_time(m)
                yes_p = norm_price_raw(m.get("yes_ask") or m.get("yes_bid"), 0.5)
                no_p = norm_price_raw(m.get("no_ask") or m.get("no_bid"), 1.0 - yes_p)
                rows.append({
                    "ticker": m.get("ticker"),
                    "effective_time": eff.isoformat() if eff else "",
                    "yes_prob": round(float(max(0.01, min(0.99, yes_p))), 3),
                    "no_prob": round(float(max(0.01, min(0.99, no_p))), 3),
                    "status": m.get("status", ""),
                    "liquidity_dollars": m.get("liquidity_dollars", ""),
                    "title": m.get("title", ""),
                })

            markets_df = pd.DataFrame(rows).sort_values("effective_time")
            st.dataframe(markets_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🤖 Agent recommendation (next market)")

            next_ticker = markets_df.iloc[0]["ticker"]
            selected = None
            for m in markets:
                if m.get("ticker") == next_ticker:
                    selected = m
                    break

            try:
                portfolio = client.get_portfolio()
                cash = extract_portfolio_cash(portfolio, fallback=config.initial_cash)
            except Exception:
                cash = float(config.initial_cash)

            if selected:
                obs, yes_p, no_p, eff = build_obs_for_market_v9(selected, cash, config)
                try:
                    action_id, _ = model.predict(obs, deterministic=True)
                    action_id = int(action_id)
                    st.write(f"**Market:** `{next_ticker}`")
                    st.write(f"**Effective time:** `{eff.isoformat() if eff else ''}`")
                    st.write(f"YES ≈ `{yes_p:.2f}` • NO ≈ `{no_p:.2f}`")
                    st.success(f"Recommendation: **{action_label(action_id)}**")
                except Exception as e:
                    st.error(
                        "Model prediction failed (likely observation shape mismatch). "
                        f"Error: {e}"
                    )
                    st.info(
                        "If this happens, your PPO model may have been trained on a different observation size. "
                        "Tell me the error and I’ll align the GUI obs builder to match your model."
                    )

    # -----------------------
    # Trades
    # -----------------------
    with tab3:
        st.subheader("Trades (LIVE vs PAPER badge)")

        if trades_df.empty:
            st.info("No trades logged yet.")
        else:
            df = trades_df.sort_values("timestamp", ascending=False).copy()

            only_live = st.checkbox("Show only LIVE trades", value=False)
            only_paper = st.checkbox("Show only PAPER trades", value=False)

            if "is_paper_fill" in df.columns:
                df["mode"] = df["is_paper_fill"].apply(mode_badge)
                is_paper = pd.to_numeric(df["is_paper_fill"], errors="coerce").fillna(0).astype(int)

                if only_live and not only_paper:
                    df = df[is_paper == 0]
                if only_paper and not only_live:
                    df = df[is_paper == 1]
            else:
                df["mode"] = "—"

            if "action_id" in df.columns:
                df["action"] = df["action_id"].apply(lambda x: action_label(int(x)) if pd.notna(x) else "")

            preferred = [
                "timestamp", "mode", "ticker", "action", "side", "count",
                "entry_price_prob", "paper_reason",
            ]
            cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
            st.dataframe(df[cols].head(300), use_container_width=True, hide_index=True)

    # Auto-refresh
    st.sidebar.markdown("---")
    st.sidebar.caption("Auto-refresh is ON.")
    last_run = st.session_state.get("last_run", 0.0)
    now = time.time()
    if now - last_run > refresh_seconds:
        st.session_state["last_run"] = now
        st.rerun()


if __name__ == "__main__":
    main()
