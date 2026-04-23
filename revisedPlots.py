import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FA 550 Capstone Prototype
# First prototype charts from active-analysis dataset
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "microstructure_1s_active.parquet")
FIG_DIR = os.path.join(BASE, "outputs", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

REGIME_ORDER = ["low", "medium", "high"]
MAX_TS_POINTS = 200_000


def downsample_for_plot(df: pd.DataFrame, max_points: int = MAX_TS_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def save_timeseries_plot(df, y_col, title, ylabel, filename):
    plot_df = df[["timestamp", y_col]].dropna().sort_values("timestamp").copy()
    plot_df = downsample_for_plot(plot_df)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["timestamp"], plot_df[y_col], linewidth=0.8)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_boxplot_by_regime(df, value_col, title, ylabel, filename):
    plot_df = df[[value_col, "vol_regime"]].dropna().copy()

    data = []
    labels = []
    for reg in REGIME_ORDER:
        vals = plot_df.loc[plot_df["vol_regime"] == reg, value_col].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(reg)

    if len(data) == 0:
        print(f"Skipping {filename} because no valid data was found.")
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


def save_tradecount_bar_by_regime(df, filename):
    plot_df = df[["trade_count", "vol_regime"]].dropna().copy()
    means = plot_df.groupby("vol_regime")["trade_count"].mean().reindex(REGIME_ORDER)

    plt.figure(figsize=(8, 5))
    plt.bar(means.index.astype(str), means.values)
    plt.title("Mean Trade Count by Volatility Regime")
    plt.xlabel("Volatility Regime")
    plt.ylabel("Mean Trade Count")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    print(f"Loading active dataset:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep vol_regime readable
    df["vol_regime"] = df["vol_regime"].astype("string")

    saved_files = []

    # 1. Time-series of mid_price
    saved_files.append(
        save_timeseries_plot(
            df=df,
            y_col="mid_price",
            title="BTC Futures Mid-Price Over Time (Active Dataset)",
            ylabel="Mid-Price",
            filename="active_mid_price_timeseries.png",
        )
    )

    # 2. Time-series of spread
    saved_files.append(
        save_timeseries_plot(
            df=df,
            y_col="spread",
            title="BTC Futures Spread Over Time (Active Dataset)",
            ylabel="Spread",
            filename="active_spread_timeseries.png",
        )
    )

    # 3. Time-series of rv_60s
    saved_files.append(
        save_timeseries_plot(
            df=df,
            y_col="rv_60s",
            title="BTC Futures Rolling Realized Volatility (60s) Over Time (Active Dataset)",
            ylabel="RV 60s",
            filename="active_rv60s_timeseries.png",
        )
    )

    # 4. Boxplot of spread by vol_regime
    saved = save_boxplot_by_regime(
        df=df,
        value_col="spread",
        title="Spread by Volatility Regime (Active Dataset)",
        ylabel="Spread",
        filename="active_spread_by_regime_boxplot.png",
    )
    if saved:
        saved_files.append(saved)

    # 5. Bar chart of trade_count by vol_regime
    saved_files.append(
        save_tradecount_bar_by_regime(
            df=df,
            filename="active_trade_count_by_regime_bar.png",
        )
    )

    # 6. Boxplot of abs_fwd_return_5s by vol_regime
    saved = save_boxplot_by_regime(
        df=df,
        value_col="abs_fwd_return_5s",
        title="Absolute 5-Second Forward Return by Volatility Regime (Active Dataset)",
        ylabel="abs_fwd_return_5s",
        filename="active_abs_fwd_return_5s_by_regime_boxplot.png",
    )
    if saved:
        saved_files.append(saved)

    # Summary stats
    spread_stats = (
        df[["spread", "vol_regime"]]
        .dropna()
        .groupby("vol_regime")["spread"]
        .agg(["mean", "median"])
        .reindex(REGIME_ORDER)
    )

    trade_count_stats = (
        df[["trade_count", "vol_regime"]]
        .dropna()
        .groupby("vol_regime")["trade_count"]
        .agg(["mean", "median"])
        .reindex(REGIME_ORDER)
    )

    abs_fwd_stats = (
        df[["abs_fwd_return_5s", "vol_regime"]]
        .dropna()
        .groupby("vol_regime")["abs_fwd_return_5s"]
        .agg(["mean", "median"])
        .reindex(REGIME_ORDER)
    )

    print("\n" + "=" * 80)
    print("SUMMARY STATS BY VOL REGIME")
    print("=" * 80)

    print("\nMean and median spread by vol_regime:")
    print(spread_stats)

    print("\nMean and median trade_count by vol_regime:")
    print(trade_count_stats)

    print("\nMean and median abs_fwd_return_5s by vol_regime:")
    print(abs_fwd_stats)

    print("\n" + "=" * 80)
    print("FIGURES SAVED")
    print("=" * 80)
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()