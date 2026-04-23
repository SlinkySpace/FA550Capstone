import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FA 550 Capstone Prototype
# Event-window plots from large-move event dataset
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "event_windows_large_move_30s.parquet")
FIG_DIR = os.path.join(BASE, "outputs", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

REGIME_ORDER = ["low", "medium", "high"]


def save_single_line_plot(df, x_col, y_col, title, ylabel, filename):
    plot_df = (
        df.groupby(x_col, as_index=False)[y_col]
        .mean()
        .sort_values(x_col)
    )

    plt.figure(figsize=(9, 5))
    plt.plot(plot_df[x_col], plot_df[y_col], linewidth=1.5)
    plt.axvline(0, linewidth=1.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Relative Second")
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_regime_split_plot(df, value_col, title, ylabel, filename):
    plt.figure(figsize=(9, 5))

    for regime in REGIME_ORDER:
        reg_df = df.loc[df["vol_regime"] == regime].copy()
        if reg_df.empty:
            continue

        plot_df = (
            reg_df.groupby("relative_second", as_index=False)[value_col]
            .mean()
            .sort_values("relative_second")
        )

        plt.plot(
            plot_df["relative_second"],
            plot_df[value_col],
            linewidth=1.5,
            label=regime,
        )

    plt.axvline(0, linewidth=1.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Relative Second")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    print(f"Loading event-window dataset:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["vol_regime"] = df["vol_regime"].astype("string")

    df = df.dropna(subset=["event_id", "relative_second", "timestamp"]).copy()
    df = df.sort_values(["event_id", "relative_second"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Centering checks
    # ------------------------------------------------------------
    event_zero_counts = df.loc[df["relative_second"] == 0].groupby("event_id").size()
    bad_center_events = event_zero_counts[event_zero_counts != 1]

    all_event_ids = df["event_id"].dropna().unique()
    centered_event_ids = df.loc[df["relative_second"] == 0, "event_id"].dropna().unique()

    print("\n" + "=" * 80)
    print("EVENT WINDOW CENTERING CHECK")
    print("=" * 80)
    print(f"Unique event_ids in dataset: {len(all_event_ids):,}")
    print(f"Unique event_ids with relative_second == 0 row: {len(centered_event_ids):,}")
    print(f"Events with bad number of center rows: {len(bad_center_events):,}")

    # ------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------
    unique_events = df["event_id"].nunique()
    avg_rows_per_event = df.groupby("event_id").size().mean()

    event_second_df = df.loc[df["relative_second"] == 0].copy()

    spread_by_regime = (
        event_second_df.groupby("vol_regime")["spread"]
        .mean()
        .reindex(REGIME_ORDER)
    )

    trade_count_by_regime = (
        event_second_df.groupby("vol_regime")["trade_count"]
        .mean()
        .reindex(REGIME_ORDER)
    )

    abs_fwd_by_regime = (
        event_second_df.groupby("vol_regime")["abs_fwd_return_5s"]
        .mean()
        .reindex(REGIME_ORDER)
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    saved_files = []

    saved_files.append(
        save_single_line_plot(
            df=df,
            x_col="relative_second",
            y_col="mid_price_rel",
            title="Average Mid-Price Change Around Large-Move Events",
            ylabel="Average mid_price_rel",
            filename="event_avg_mid_price_rel_all.png",
        )
    )

    saved_files.append(
        save_single_line_plot(
            df=df,
            x_col="relative_second",
            y_col="spread_rel",
            title="Average Spread Change Around Large-Move Events",
            ylabel="Average spread_rel",
            filename="event_avg_spread_rel_all.png",
        )
    )

    saved_files.append(
        save_single_line_plot(
            df=df,
            x_col="relative_second",
            y_col="trade_count",
            title="Average Trade Count Around Large-Move Events",
            ylabel="Average trade_count",
            filename="event_avg_trade_count_all.png",
        )
    )

    saved_files.append(
        save_single_line_plot(
            df=df,
            x_col="relative_second",
            y_col="rv_60s",
            title="Average Rolling Volatility Around Large-Move Events",
            ylabel="Average rv_60s",
            filename="event_avg_rv60s_all.png",
        )
    )

    saved_files.append(
        save_regime_split_plot(
            df=df,
            value_col="mid_price_rel",
            title="Average Mid-Price Change Around Large-Move Events by Volatility Regime",
            ylabel="Average mid_price_rel",
            filename="event_avg_mid_price_rel_by_regime.png",
        )
    )

    saved_files.append(
        save_regime_split_plot(
            df=df,
            value_col="spread_rel",
            title="Average Spread Change Around Large-Move Events by Volatility Regime",
            ylabel="Average spread_rel",
            filename="event_avg_spread_rel_by_regime.png",
        )
    )

    # ------------------------------------------------------------
    # Print outputs
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVENT WINDOW SUMMARY")
    print("=" * 80)
    print(f"Number of unique events: {unique_events:,}")
    print(f"Average number of rows per event window: {avg_rows_per_event:.2f}")

    print("\nMean event-second spread by vol_regime:")
    print(spread_by_regime)

    print("\nMean event-second trade_count by vol_regime:")
    print(trade_count_by_regime)

    print("\nMean event-second abs_fwd_return_5s by vol_regime:")
    print(abs_fwd_by_regime)

    print("\n" + "=" * 80)
    print("FIGURES SAVED")
    print("=" * 80)
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()