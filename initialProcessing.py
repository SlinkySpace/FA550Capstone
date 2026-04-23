import os
import numpy as np
import pandas as pd

# ============================================================
# FA 550 Capstone Prototype
# BTC Futures Microstructure - Unified 1-Second Analysis Table
# ============================================================

# -----------------------------
# Base paths
# -----------------------------
BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
DATA_DIR = os.path.join(BASE, "data", "live", "processed")
OUTPUT_DIR = os.path.join(BASE, "data", "live", "derived")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TBBO_PATH = os.path.join(DATA_DIR, "tbbo_continuous.parquet")
TRADES_PATH = os.path.join(DATA_DIR, "trades_continuous.parquet")
OHLCV_1S_PATH = os.path.join(DATA_DIR, "ohlcv-1s_continuous.parquet")  # optional validation/context
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "btc_microstructure_1s_mvp.parquet")

# -----------------------------
# Settings
# -----------------------------
FWD_HORIZON_SEC = 5
RV_WINDOW_SEC = 60
EVENT_QUANTILE = 0.95


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert to timezone-aware datetime if needed."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_tbbo(path: str) -> pd.DataFrame:
    print(f"Loading TBBO: {path}")
    df = pd.read_parquet(path)

    # Bring index back if needed
    df = df.reset_index(drop=False)

    keep_cols = [
        "ts_event",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "symbol",
    ]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    if "ts_event" not in df.columns:
        raise ValueError("TBBO file must contain ts_event.")

    df["ts_event"] = safe_to_datetime(df["ts_event"])
    df = df.dropna(subset=["ts_event"]).copy()

    for col in ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values("ts_event").reset_index(drop=True)

    # Floor to 1 second
    df["ts_1s"] = df["ts_event"].dt.floor("1s")

    # Last quote in each second
    tbbo_1s = (
        df.groupby("ts_1s", as_index=False)
        .agg(
            symbol=("symbol", "last"),
            best_bid=("bid_px_00", "last"),
            best_ask=("ask_px_00", "last"),
            bid_size=("bid_sz_00", "last"),
            ask_size=("ask_sz_00", "last"),
        )
        .sort_values("ts_1s")
        .reset_index(drop=True)
    )

    return tbbo_1s


def load_trades(path: str) -> pd.DataFrame:
    print(f"Loading trades: {path}")
    df = pd.read_parquet(path)

    # Bring index back if needed
    df = df.reset_index(drop=False)

    keep_cols = [
        "ts_event",
        "price",
        "size",
        "symbol",
    ]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    if "ts_event" not in df.columns:
        raise ValueError("Trades file must contain ts_event.")

    df["ts_event"] = safe_to_datetime(df["ts_event"])
    df = df.dropna(subset=["ts_event"]).copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)

    df = df.sort_values("ts_event").reset_index(drop=True)

    # Floor to 1 second
    df["ts_1s"] = df["ts_event"].dt.floor("1s")

    trades_1s = (
        df.groupby("ts_1s", as_index=False)
        .agg(
            trade_symbol=("symbol", "last"),
            trade_count=("price", "size"),
            trade_volume=("size", "sum"),
            last_trade_price=("price", "last"),
        )
        .sort_values("ts_1s")
        .reset_index(drop=True)
    )

    return trades_1s


def load_ohlcv_1s(path: str) -> pd.DataFrame:
    """Optional validation/context only."""
    if not os.path.exists(path):
        print("OHLCV-1s file not found. Skipping.")
        return pd.DataFrame()

    print(f"Loading OHLCV-1s: {path}")
    df = pd.read_parquet(path)
    df = df.reset_index(drop=False)

    if "ts_event" not in df.columns:
        print("OHLCV-1s does not have ts_event as a column after reset_index. Skipping.")
        return pd.DataFrame()

    keep_cols = ["ts_event", "close", "volume", "symbol"]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    df["ts_event"] = safe_to_datetime(df["ts_event"])
    df = df.dropna(subset=["ts_event"]).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)

    df = df.sort_values("ts_event").reset_index(drop=True)

    df = df.rename(
        columns={
            "ts_event": "timestamp",
            "close": "ohlcv_close_1s",
            "volume": "ohlcv_volume_1s",
            "symbol": "ohlcv_symbol",
        }
    )
    return df


def build_full_grid(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="1s", tz="UTC")
    return pd.DataFrame({"timestamp": full_index})


def assign_vol_regime(rv_series: pd.Series) -> pd.Series:
    """
    Assign low / medium / high volatility regimes using approximate terciles.
    Returns an object dtype Series so string labels and missing values can coexist.
    """
    valid = rv_series.dropna()
    regime = pd.Series(index=rv_series.index, dtype="object")

    if valid.empty:
        regime[:] = pd.NA
        return regime

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    # Fallback if quantiles collapse
    if pd.isna(q1) or pd.isna(q2) or q1 == q2:
        median_val = valid.median()
        regime.loc[rv_series <= median_val] = "low"
        regime.loc[rv_series > median_val] = "high"
        return regime

    regime.loc[rv_series <= q1] = "low"
    regime.loc[(rv_series > q1) & (rv_series <= q2)] = "medium"
    regime.loc[rv_series > q2] = "high"

    return regime


def main():
    # -----------------------------
    # Load inputs
    # -----------------------------
    tbbo_1s = load_tbbo(TBBO_PATH)
    trades_1s = load_trades(TRADES_PATH)
    ohlcv_1s = load_ohlcv_1s(OHLCV_1S_PATH)

    # -----------------------------
    # Build full timestamp grid
    # -----------------------------
    min_ts = min(tbbo_1s["ts_1s"].min(), trades_1s["ts_1s"].min())
    max_ts = max(tbbo_1s["ts_1s"].max(), trades_1s["ts_1s"].max())
    df = build_full_grid(min_ts, max_ts)

    # -----------------------------
    # Merge 1-second TBBO and trades
    # -----------------------------
    df = df.merge(tbbo_1s.rename(columns={"ts_1s": "timestamp"}), on="timestamp", how="left")
    df = df.merge(trades_1s.rename(columns={"ts_1s": "timestamp"}), on="timestamp", how="left")

    if not ohlcv_1s.empty:
        df = df.merge(ohlcv_1s, on="timestamp", how="left")

    # -----------------------------
    # Forward-fill quote state
    # -----------------------------
    # Since TBBO may be attached to trade/event times, not all market quote updates,
    # we only carry forward last known top-of-book state.
    quote_cols = ["symbol", "best_bid", "best_ask", "bid_size", "ask_size"]
    for col in quote_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Fallback symbol from trades if needed
    if "trade_symbol" in df.columns:
        df["symbol"] = df["symbol"].fillna(df["trade_symbol"])

    # Missing trade activity becomes zero
    for col in ["trade_count", "trade_volume"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # -----------------------------
    # Basic quote sanity checks
    # -----------------------------
    bad_quote_mask = (
        df["best_bid"].isna()
        | df["best_ask"].isna()
        | (df["best_bid"] <= 0)
        | (df["best_ask"] <= 0)
        | (df["best_ask"] < df["best_bid"])
    )

    df["best_bid_clean"] = df["best_bid"]
    df["best_ask_clean"] = df["best_ask"]
    df.loc[bad_quote_mask, ["best_bid_clean", "best_ask_clean"]] = np.nan

    # Clean bad sizes if needed
    for col in ["bid_size", "ask_size"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # -----------------------------
    # Core features
    # -----------------------------
    df["mid_price"] = (df["best_bid_clean"] + df["best_ask_clean"]) / 2.0
    df["spread"] = df["best_ask_clean"] - df["best_bid_clean"]

    df["spread_bps"] = np.where(
        (df["mid_price"] > 0) & df["spread"].notna(),
        (df["spread"] / df["mid_price"]) * 10000.0,
        np.nan,
    )

    df["trade_intensity"] = df["trade_count"]

    df["top_depth"] = df["bid_size"] + df["ask_size"]
    depth_denom = df["bid_size"] + df["ask_size"]
    df["imbalance"] = np.where(
        (depth_denom > 0) & depth_denom.notna(),
        (df["bid_size"] - df["ask_size"]) / depth_denom,
        np.nan,
    )

    # -----------------------------
    # Contract rollover-safe returns
    # -----------------------------
    df["symbol_prev"] = df["symbol"].shift(1)
    df["symbol_fwd_5s"] = df["symbol"].shift(-FWD_HORIZON_SEC)

    same_as_prev = df["symbol"].eq(df["symbol_prev"])
    same_as_fwd = df["symbol"].eq(df["symbol_fwd_5s"])

    prev_mid = df["mid_price"].shift(1)
    valid_logret = (
        same_as_prev
        & df["mid_price"].notna()
        & prev_mid.notna()
        & (df["mid_price"] > 0)
        & (prev_mid > 0)
    )
    df["log_return_1s"] = np.where(
        valid_logret,
        np.log(df["mid_price"] / prev_mid),
        np.nan,
    )

    fwd_mid = df["mid_price"].shift(-FWD_HORIZON_SEC)
    valid_fwd = (
        same_as_fwd
        & df["mid_price"].notna()
        & fwd_mid.notna()
        & (df["mid_price"] > 0)
        & (fwd_mid > 0)
    )
    df["fwd_return_5s"] = np.where(
        valid_fwd,
        (fwd_mid - df["mid_price"]) / df["mid_price"],
        np.nan,
    )
    df["abs_fwd_return_5s"] = df["fwd_return_5s"].abs()

    # -----------------------------
    # Rolling realized volatility
    # -----------------------------
    df["rv_60s"] = (
        df["log_return_1s"]
        .rolling(window=RV_WINDOW_SEC, min_periods=max(10, RV_WINDOW_SEC // 3))
        .std()
    )

    # -----------------------------
    # Volatility regimes
    # -----------------------------
    df["vol_regime"] = assign_vol_regime(df["rv_60s"])
    df["vol_regime"] = pd.Categorical(
        df["vol_regime"],
        categories=["low", "medium", "high"],
        ordered=True,
    )

    # -----------------------------
    # Event flag
    # -----------------------------
    valid_abs_fwd = df["abs_fwd_return_5s"].dropna()
    if len(valid_abs_fwd) > 0:
        move_threshold = valid_abs_fwd.quantile(EVENT_QUANTILE)
        df["event_large_move"] = np.where(
            df["abs_fwd_return_5s"] >= move_threshold, 1, 0
        )
    else:
        move_threshold = np.nan
        df["event_large_move"] = 0

    # -----------------------------
    # Optional OHLCV validation
    # -----------------------------
    if "ohlcv_close_1s" in df.columns:
        df["mid_vs_ohlcv_close_diff"] = df["mid_price"] - df["ohlcv_close_1s"]
    else:
        df["mid_vs_ohlcv_close_diff"] = np.nan

    # -----------------------------
    # Trim rows before first valid quote state
    # -----------------------------
    first_valid_mid_idx = df["mid_price"].first_valid_index()
    if first_valid_mid_idx is not None:
        df = df.loc[first_valid_mid_idx:].copy()

    # -----------------------------
    # Final MVP column set
    # -----------------------------
    final_cols = [
        "timestamp",
        "symbol",
        "best_bid",
        "best_ask",
        "bid_size",
        "ask_size",
        "trade_count",
        "trade_volume",
        "last_trade_price",
        "mid_price",
        "spread",
        "spread_bps",
        "trade_intensity",
        "log_return_1s",
        "fwd_return_5s",
        "abs_fwd_return_5s",
        "rv_60s",
        "top_depth",
        "imbalance",
        "vol_regime",
        "event_large_move",
        "ohlcv_close_1s" if "ohlcv_close_1s" in df.columns else None,
        "ohlcv_volume_1s" if "ohlcv_volume_1s" in df.columns else None,
        "mid_vs_ohlcv_close_diff",
    ]
    final_cols = [c for c in final_cols if c is not None and c in df.columns]
    out = df[final_cols].copy()

    # Save parquet
    out.to_parquet(OUTPUT_PATH, index=False)

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n" + "=" * 80)
    print("SANITY CHECK SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns)}")
    print(f"Move threshold (95th pct of abs_fwd_return_5s): {move_threshold}")

    if "spread" in out.columns:
        neg_spreads = (out["spread"] < 0).sum(skipna=True)
        print(f"Negative spread rows after cleaning: {neg_spreads}")

    if "symbol" in out.columns:
        print("Symbol counts:")
        print(out["symbol"].value_counts(dropna=False).head(10))

    print("\nHead:")
    print(out.head())

    print("\nTail:")
    print(out.tail())

    print("\nTop null counts:")
    print(out.isnull().sum().sort_values(ascending=False).head(20))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()