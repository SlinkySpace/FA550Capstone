import os
import numpy as np
import pandas as pd

# ============================================================
# FA 550 Capstone Prototype
# Build active-analysis BTC microstructure dataset
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "microstructure_1s_dashboard.parquet")
OUTPUT_DIR = os.path.join(BASE, "data", "live", "derived")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "microstructure_1s_active.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FWD_HORIZON_SEC = 5
RECENT_WINDOW_SEC = 5
RV_WINDOW_SEC = 60
EVENT_QUANTILE = 0.95
REGIME_ORDER = ["low", "medium", "high"]


def assign_vol_regime(rv_series: pd.Series) -> pd.Series:
    """
    Assign low / medium / high volatility regimes using terciles
    on the filtered active-analysis dataset.
    """
    valid = rv_series.dropna()
    regime = pd.Series(index=rv_series.index, dtype="object")

    if valid.empty:
        regime[:] = pd.NA
        return regime

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    if pd.isna(q1) or pd.isna(q2) or q1 == q2:
        median_val = valid.median()
        regime.loc[rv_series <= median_val] = "low"
        regime.loc[rv_series > median_val] = "high"
        return regime

    regime.loc[rv_series <= q1] = "low"
    regime.loc[(rv_series > q1) & (rv_series <= q2)] = "medium"
    regime.loc[rv_series > q2] = "high"
    return regime


def print_regime_summary(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    summary = (
        df.groupby("vol_regime", dropna=False)
        .agg(
            row_count=("timestamp", "size"),
            mean_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            mean_trade_count=("trade_count", "mean"),
            median_trade_count=("trade_count", "median"),
            zero_trade_share=("trade_count", lambda x: (x == 0).mean()),
            stale_share=("is_stale_quote", "mean"),
            mean_rv_60s=("rv_60s", "mean"),
            median_rv_60s=("rv_60s", "median"),
            mean_abs_fwd_return_5s=("abs_fwd_return_5s", "mean"),
            median_abs_fwd_return_5s=("abs_fwd_return_5s", "median"),
        )
        .reindex(REGIME_ORDER)
    )
    print(summary)


def main():
    print(f"Loading dashboard-ready file:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    # ------------------------------------------------------------
    # Basic cleanup
    # ------------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Make sure key numeric fields are numeric
    numeric_cols = [
        "best_bid",
        "best_ask",
        "bid_size",
        "ask_size",
        "trade_count",
        "trade_volume",
        "mid_price",
        "spread",
        "spread_bps",
        "trade_intensity",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------
    # Activity indicators
    # ------------------------------------------------------------
    df["is_active"] = (df["trade_count"] > 0).astype(int)

    # At least one trade in past 5 seconds (including current row)
    past_recent = (
        df["is_active"]
        .rolling(window=RECENT_WINDOW_SEC + 1, min_periods=1)
        .max()
        .fillna(0)
        .astype(int)
    )

    # At least one trade in next 5 seconds (including current row)
    future_recent = (
        df["is_active"][::-1]
        .rolling(window=RECENT_WINDOW_SEC + 1, min_periods=1)
        .max()[::-1]
        .fillna(0)
        .astype(int)
    )

    df["is_active_or_recent"] = ((past_recent == 1) | (future_recent == 1)).astype(int)

    # ------------------------------------------------------------
    # Filter to active-analysis dataset
    # ------------------------------------------------------------
    original_rows = len(df)
    active_df = df.loc[df["is_active_or_recent"] == 1].copy()
    active_rows = len(active_df)
    retained_pct = 100 * active_rows / original_rows if original_rows > 0 else np.nan

    active_df = active_df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------
    # Recompute returns on filtered dataset
    # Make sure returns do not bridge across symbol changes
    # ------------------------------------------------------------
    active_df["symbol_prev"] = active_df["symbol"].shift(1)
    active_df["symbol_fwd_5s"] = active_df["symbol"].shift(-FWD_HORIZON_SEC)

    same_as_prev = active_df["symbol"].eq(active_df["symbol_prev"])
    same_as_fwd = active_df["symbol"].eq(active_df["symbol_fwd_5s"])

    prev_mid = active_df["mid_price"].shift(1)
    valid_logret = (
        same_as_prev
        & active_df["mid_price"].notna()
        & prev_mid.notna()
        & (active_df["mid_price"] > 0)
        & (prev_mid > 0)
    )

    active_df["log_return_1s"] = np.where(
        valid_logret,
        np.log(active_df["mid_price"] / prev_mid),
        np.nan,
    )

    fwd_mid = active_df["mid_price"].shift(-FWD_HORIZON_SEC)
    valid_fwd = (
        same_as_fwd
        & active_df["mid_price"].notna()
        & fwd_mid.notna()
        & (active_df["mid_price"] > 0)
        & (fwd_mid > 0)
    )

    active_df["fwd_return_5s"] = np.where(
        valid_fwd,
        (fwd_mid - active_df["mid_price"]) / active_df["mid_price"],
        np.nan,
    )
    active_df["abs_fwd_return_5s"] = active_df["fwd_return_5s"].abs()

    # ------------------------------------------------------------
    # Recompute rolling realized volatility on filtered dataset
    # ------------------------------------------------------------
    active_df["rv_60s"] = (
        active_df["log_return_1s"]
        .rolling(window=RV_WINDOW_SEC, min_periods=max(10, RV_WINDOW_SEC // 3))
        .std()
    )

    # ------------------------------------------------------------
    # Recompute stale indicator on filtered dataset
    # stale if mid_price unchanged from previous filtered row and no trade now
    # ------------------------------------------------------------
    prev_mid_active = active_df["mid_price"].shift(1)
    active_df["is_stale_quote"] = np.where(
        active_df["mid_price"].eq(prev_mid_active) & (active_df["trade_count"] == 0),
        1,
        0,
    )

    # ------------------------------------------------------------
    # Recompute volatility regimes on filtered dataset only
    # ------------------------------------------------------------
    active_df["vol_regime"] = assign_vol_regime(active_df["rv_60s"])
    active_df["vol_regime"] = pd.Categorical(
        active_df["vol_regime"],
        categories=REGIME_ORDER,
        ordered=True,
    )

    # ------------------------------------------------------------
    # Recompute event_large_move using filtered dataset only
    # ------------------------------------------------------------
    valid_abs_fwd = active_df["abs_fwd_return_5s"].dropna()
    if len(valid_abs_fwd) > 0:
        move_threshold = valid_abs_fwd.quantile(EVENT_QUANTILE)
        active_df["event_large_move"] = np.where(
            active_df["abs_fwd_return_5s"] >= move_threshold,
            1,
            0,
        )
    else:
        move_threshold = np.nan
        active_df["event_large_move"] = 0

    # ------------------------------------------------------------
    # Recompute event_start and event_id
    # ------------------------------------------------------------
    prev_event = active_df["event_large_move"].shift(1, fill_value=0)
    active_df["event_start"] = (
        (active_df["event_large_move"] == 1) & (prev_event != 1)
    ).astype(int)

    active_df["event_id"] = active_df["event_start"].cumsum()
    active_df.loc[active_df["event_large_move"] == 0, "event_id"] = np.nan
    active_df["event_id"] = active_df["event_id"].astype("Int64")

    # ------------------------------------------------------------
    # Final column set
    # ------------------------------------------------------------
    desired_cols = [
        "timestamp",
        "symbol",
        "best_bid",
        "best_ask",
        "bid_size",
        "ask_size",
        "trade_count",
        "trade_volume",
        "mid_price",
        "spread",
        "spread_bps",
        "trade_intensity",
        "is_active",
        "is_active_or_recent",
        "log_return_1s",
        "fwd_return_5s",
        "abs_fwd_return_5s",
        "rv_60s",
        "vol_regime",
        "is_stale_quote",
        "event_large_move",
        "event_start",
        "event_id",
    ]
    final_cols = [c for c in desired_cols if c in active_df.columns]
    out = active_df[final_cols].copy()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    out.to_parquet(OUTPUT_PATH, index=False)

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ACTIVE ANALYSIS DATASET SUMMARY")
    print("=" * 100)
    print(f"Rows before filtering: {original_rows:,}")
    print(f"Rows after filtering:  {active_rows:,}")
    print(f"Percentage retained:   {retained_pct:.2f}%")
    print(f"event_large_move == 1 count: {int((out['event_large_move'] == 1).sum()):,}")
    print(f"event_start == 1 count:      {int((out['event_start'] == 1).sum()):,}")
    print(f"Large-move threshold (95th pct abs_fwd_return_5s): {move_threshold}")

    print_regime_summary(out, "SUMMARY BY NEW vol_regime")

    zero_trade_share = (
        out.groupby("vol_regime")["trade_count"]
        .apply(lambda x: (x == 0).mean())
        .reindex(REGIME_ORDER)
    )
    print("\nZero-trade share by new vol_regime:")
    print(zero_trade_share)

    stale_share = (
        out.groupby("vol_regime")["is_stale_quote"]
        .mean()
        .reindex(REGIME_ORDER)
    )
    print("\nStale-row share by new vol_regime:")
    print(stale_share)

    print("\nTop null counts:")
    print(out.isnull().sum().sort_values(ascending=False).head(20))

    print(f"\nSaved active-analysis file to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()