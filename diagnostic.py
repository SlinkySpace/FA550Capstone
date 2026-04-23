import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FA 550 Capstone Prototype
# Regime + Trade Activity Diagnostics
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "microstructure_1s_dashboard.parquet")
FIG_DIR = os.path.join(BASE, "outputs", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

REGIME_ORDER = ["low", "medium", "high"]


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
            mean_rv_60s=("rv_60s", "mean"),
            median_rv_60s=("rv_60s", "median"),
            mean_abs_fwd_return_5s=("abs_fwd_return_5s", "mean"),
            median_abs_fwd_return_5s=("abs_fwd_return_5s", "median"),
        )
        .reindex(REGIME_ORDER)
    )

    print(summary)
    print()


def save_histogram(series: pd.Series, title: str, xlabel: str, filename: str, bins=100, log_y=False):
    plot_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(plot_series) == 0:
        print(f"Skipping {filename} because there is no valid data.")
        return None

    plt.figure(figsize=(10, 5))
    plt.hist(plot_series, bins=bins)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_boxplot_by_regime(df: pd.DataFrame, value_col: str, title: str, ylabel: str, filename: str):
    plot_df = df[[value_col, "vol_regime"]].dropna().copy()

    data = []
    labels = []
    for reg in REGIME_ORDER:
        vals = plot_df.loc[plot_df["vol_regime"] == reg, value_col].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(reg)

    if len(data) == 0:
        print(f"Skipping {filename} because there is no valid data.")
        return None

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel("Volatility Regime")
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_zero_trade_bar(df: pd.DataFrame, filename: str):
    plot_df = df[["trade_count", "vol_regime"]].dropna().copy()
    zero_trade_pct = (
        plot_df.groupby("vol_regime")["trade_count"]
        .apply(lambda x: 100 * (x == 0).mean())
        .reindex(REGIME_ORDER)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(zero_trade_pct.index.astype(str), zero_trade_pct.values)
    plt.title("Percentage of Zero-Trade Rows by Volatility Regime")
    plt.xlabel("Volatility Regime")
    plt.ylabel("Percent of Rows with trade_count == 0")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def print_stale_summary(df: pd.DataFrame) -> None:
    overall_stale_share = df["is_stale_quote"].mean()

    stale_by_regime = (
        df.groupby("vol_regime")["is_stale_quote"]
        .mean()
        .reindex(REGIME_ORDER)
    )

    print("\n" + "=" * 100)
    print("STALE QUOTE SUMMARY")
    print("=" * 100)
    print(f"Overall stale-row share: {overall_stale_share:.4%}")
    print("\nStale-row share by vol_regime:")
    print(stale_by_regime)
    print()


def main():
    print(f"Loading dashboard-ready parquet:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    # Basic cleanup
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep vol_regime as readable strings
    df["vol_regime"] = df["vol_regime"].astype("string")

    # Numeric cleanup
    numeric_cols = [
        "mid_price",
        "spread",
        "trade_count",
        "rv_60s",
        "abs_fwd_return_5s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n" + "=" * 100)
    print("DATASET OVERVIEW")
    print("=" * 100)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nTop null counts:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    # ------------------------------------------------------------
    # Regime summaries - all rows
    # ------------------------------------------------------------
    print_regime_summary(df, "REGIME SUMMARY - ALL ROWS")

    # ------------------------------------------------------------
    # Stale quote indicator
    # is_stale_quote = 1 if mid_price unchanged from previous second
    # and trade_count == 0, else 0
    # ------------------------------------------------------------
    prev_mid = df["mid_price"].shift(1)
    same_mid_as_prev = df["mid_price"].eq(prev_mid)

    df["is_stale_quote"] = np.where(
        same_mid_as_prev & (df["trade_count"] == 0),
        1,
        0,
    )

    print_stale_summary(df)

    # ------------------------------------------------------------
    # Regime summaries - active rows only
    # ------------------------------------------------------------
    active_df = df.loc[df["trade_count"] > 0].copy()
    print_regime_summary(active_df, "REGIME SUMMARY - ACTIVE ROWS ONLY (trade_count > 0)")

    # ------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------
    saved_files = []

    # Histogram of rv_60s
    saved = save_histogram(
        series=df["rv_60s"],
        title="Histogram of Rolling Realized Volatility (rv_60s)",
        xlabel="rv_60s",
        filename="diag_hist_rv_60s.png",
        bins=100,
        log_y=False,
    )
    if saved:
        saved_files.append(saved)

    # Histogram of trade_count
    saved = save_histogram(
        series=df["trade_count"],
        title="Histogram of Trade Count per Second",
        xlabel="trade_count",
        filename="diag_hist_trade_count.png",
        bins=100,
        log_y=True,
    )
    if saved:
        saved_files.append(saved)

    # Boxplot of trade_count by vol_regime
    saved = save_boxplot_by_regime(
        df=df,
        value_col="trade_count",
        title="Trade Count by Volatility Regime",
        ylabel="trade_count",
        filename="diag_box_trade_count_by_regime.png",
    )
    if saved:
        saved_files.append(saved)

    # Bar chart of zero-trade percentage by vol_regime
    saved = save_zero_trade_bar(
        df=df,
        filename="diag_bar_zero_trade_pct_by_regime.png",
    )
    if saved:
        saved_files.append(saved)

    # Boxplot of abs_fwd_return_5s by vol_regime
    saved = save_boxplot_by_regime(
        df=df,
        value_col="abs_fwd_return_5s",
        title="Absolute 5-Second Forward Return by Volatility Regime",
        ylabel="abs_fwd_return_5s",
        filename="diag_box_abs_fwd_return_5s_by_regime.png",
    )
    if saved:
        saved_files.append(saved)

    # ------------------------------------------------------------
    # Quick comparison tables for all vs active rows
    # ------------------------------------------------------------
    all_zero_trade = (
        df.groupby("vol_regime")["trade_count"]
        .apply(lambda x: (x == 0).mean())
        .reindex(REGIME_ORDER)
    )

    active_counts = (
        active_df.groupby("vol_regime")["timestamp"]
        .size()
        .reindex(REGIME_ORDER)
    )

    print("\n" + "=" * 100)
    print("QUICK COMPARISON HELPERS")
    print("=" * 100)
    print("Zero-trade share by regime (all rows):")
    print(all_zero_trade)
    print("\nActive row counts by regime:")
    print(active_counts)

    print("\n" + "=" * 100)
    print("FIGURES SAVED")
    print("=" * 100)
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()