import os
import pandas as pd

# ============================================================
# FA 550 Capstone Prototype
# Build compact event-level summary table
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "event_windows_large_move_30s.parquet")
OUTPUT_DIR = os.path.join(BASE, "data", "live", "derived")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "event_level_summary_large_move_30s.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIME_ORDER = ["low", "medium", "high"]


def main():
    print(f"Loading event-window dataset:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["vol_regime"] = df["vol_regime"].astype("string")

    df = df.dropna(subset=["event_id", "relative_second"]).copy()
    df = df.sort_values(["event_id", "relative_second"]).reset_index(drop=True)

    # Event-second rows
    event_sec = df.loc[df["relative_second"] == 0].copy()

    event_core = event_sec[
        [
            "event_id",
            "event_time",
            "symbol",
            "vol_regime",
            "mid_price",
            "spread",
            "trade_count",
            "rv_60s",
            "abs_fwd_return_5s",
        ]
    ].copy()

    event_core = event_core.rename(
        columns={
            "mid_price": "event_mid_price",
            "spread": "event_spread",
            "trade_count": "event_trade_count",
            "rv_60s": "event_rv_60s",
            "abs_fwd_return_5s": "event_abs_fwd_return_5s",
        }
    )

    # Pre-event summaries: -30 to -1
    pre_df = df.loc[(df["relative_second"] >= -30) & (df["relative_second"] <= -1)].copy()
    pre_summary = (
        pre_df.groupby("event_id", as_index=False)
        .agg(
            pre_avg_spread=("spread", "mean"),
            pre_avg_trade_count=("trade_count", "mean"),
            pre_avg_mid_price_rel=("mid_price_rel", "mean"),
        )
    )

    # Post-event summaries: 1 to 30
    post_df = df.loc[(df["relative_second"] >= 1) & (df["relative_second"] <= 30)].copy()
    post_summary = (
        post_df.groupby("event_id", as_index=False)
        .agg(
            post_avg_spread=("spread", "mean"),
            post_avg_trade_count=("trade_count", "mean"),
            post_avg_mid_price_rel=("mid_price_rel", "mean"),
        )
    )

    # Merge together
    summary = event_core.merge(pre_summary, on="event_id", how="left")
    summary = summary.merge(post_summary, on="event_id", how="left")

    # Save
    summary.to_parquet(OUTPUT_PATH, index=False)

    # Summary stats by vol_regime
    regime_summary = (
        summary.groupby("vol_regime")
        .agg(
            event_count=("event_id", "size"),
            mean_event_spread=("event_spread", "mean"),
            median_event_spread=("event_spread", "median"),
            mean_event_trade_count=("event_trade_count", "mean"),
            median_event_trade_count=("event_trade_count", "median"),
            mean_event_abs_fwd_return_5s=("event_abs_fwd_return_5s", "mean"),
            median_event_abs_fwd_return_5s=("event_abs_fwd_return_5s", "median"),
            mean_pre_avg_spread=("pre_avg_spread", "mean"),
            mean_post_avg_spread=("post_avg_spread", "mean"),
            mean_pre_avg_trade_count=("pre_avg_trade_count", "mean"),
            mean_post_avg_trade_count=("post_avg_trade_count", "mean"),
        )
        .reindex(REGIME_ORDER)
    )

    print("\n" + "=" * 80)
    print("EVENT-LEVEL SUMMARY")
    print("=" * 80)
    print(f"Number of events: {summary['event_id'].nunique():,}")

    print("\nSummary statistics by vol_regime:")
    print(regime_summary)

    print("\nHead of event-level summary:")
    print(summary.head())

    print(f"\nSaved event-level summary to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()