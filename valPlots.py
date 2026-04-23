import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FA 550 Capstone Prototype
# First validation plots from dashboard-ready dataset
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "microstructure_1s_dashboard.parquet")
FIG_DIR = os.path.join(BASE, "outputs", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# If the dataset is very large, sample every Nth point for time-series plotting
MAX_TS_POINTS = 200_000


def downsample_for_plot(df: pd.DataFrame, max_points: int = MAX_TS_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def save_timeseries_plot(df, x_col, y_col, title, ylabel, filename):
    plot_df = df[[x_col, y_col]].dropna().sort_values(x_col).copy()
    plot_df = downsample_for_plot(plot_df)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df[x_col], plot_df[y_col], linewidth=0.8)
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

    order = ["low", "medium", "high"]
    data = [plot_df.loc[plot_df["vol_regime"] == reg, value_col].values for reg in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=order, showfliers=False)
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
    order = ["low", "medium", "high"]
    means = plot_df.groupby("vol_regime")["trade_count"].mean().reindex(order)

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
    print(f"Loading dashboard-ready parquet:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep vol_regime clean for plotting
    df["vol_regime"] = df["vol_regime"].astype("string")

    saved_files = []

    # 1. Time-series plot of mid_price
    saved_files.append(
        save_timeseries_plot(
            df=df,
            x_col="timestamp",
            y_col="mid_price",
            title="BTC Futures Mid-Price Over Time",
            ylabel="Mid-Price",
            filename="mid_price_timeseries.png",
        )
    )

    # 2. Time-series plot of spread
    saved_files.append(
        save_timeseries_plot(
            df=df,
            x_col="timestamp",
            y_col="spread",
            title="BTC Futures Spread Over Time",
            ylabel="Spread",
            filename="spread_timeseries.png",
        )
    )

    # 3. Time-series plot of rv_60s
    saved_files.append(
        save_timeseries_plot(
            df=df,
            x_col="timestamp",
            y_col="rv_60s",
            title="BTC Futures Rolling Realized Volatility (60s) Over Time",
            ylabel="RV 60s",
            filename="rv60s_timeseries.png",
        )
    )

    # 4. Boxplot of spread by vol_regime
    saved_files.append(
        save_boxplot_by_regime(
            df=df,
            value_col="spread",
            title="Spread by Volatility Regime",
            ylabel="Spread",
            filename="spread_by_vol_regime_boxplot.png",
        )
    )

    # 5. Bar comparison of trade_count by vol_regime
    saved_files.append(
        save_tradecount_bar_by_regime(
            df=df,
            filename="trade_count_by_vol_regime_bar.png",
        )
    )

    # Summary stats
    spread_means = (
        df[["spread", "vol_regime"]]
        .dropna()
        .groupby("vol_regime")["spread"]
        .mean()
        .sort_index()
    )

    trade_count_means = (
        df[["trade_count", "vol_regime"]]
        .dropna()
        .groupby("vol_regime")["trade_count"]
        .mean()
        .sort_index()
    )

    print("\n" + "=" * 80)
    print("FIGURES SAVED")
    print("=" * 80)
    for path in saved_files:
        print(path)

    print("\nMean spread by regime:")
    print(spread_means)

    print("\nMean trade_count by regime:")
    print(trade_count_means)


if __name__ == "__main__":
    main()