import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FA 550 Capstone Prototype
# Final prototype figures for presentation/demo
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
ACTIVE_PATH = os.path.join(BASE, "data", "live", "derived", "microstructure_1s_active.parquet")
EVENT_PATH = os.path.join(BASE, "data", "live", "derived", "event_windows_large_move_30s.parquet")
FIG_DIR = os.path.join(BASE, "outputs", "final_figures")

os.makedirs(FIG_DIR, exist_ok=True)

REGIME_ORDER = ["low", "medium", "high"]
FIGSIZE_REGIME = (8, 5)
FIGSIZE_EVENT = (9, 5)
DPI = 200


def clip_series_by_quantile(series: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """Light clipping for readability only."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def save_spread_by_regime(df: pd.DataFrame) -> str:
    plot_df = df[["spread", "vol_regime"]].dropna().copy()

    data = []
    labels = []
    for reg in REGIME_ORDER:
        vals = plot_df.loc[plot_df["vol_regime"] == reg, "spread"]
        if len(vals) > 0:
            vals = clip_series_by_quantile(vals, 0.01, 0.99)
            data.append(vals.values)
            labels.append(reg.capitalize())

    plt.figure(figsize=FIGSIZE_REGIME)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Spread by Volatility Regime")
    plt.xlabel("Volatility Regime")
    plt.ylabel("Spread")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "final_spread_by_vol_regime.png")
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    return save_path


def save_trade_count_by_regime(df: pd.DataFrame) -> str:
    plot_df = df[["trade_count", "vol_regime"]].dropna().copy()

    summary = (
        plot_df.groupby("vol_regime")["trade_count"]
        .mean()
        .reindex(REGIME_ORDER)
    )

    plt.figure(figsize=FIGSIZE_REGIME)
    plt.bar(summary.index.str.capitalize(), summary.values)
    plt.title("Average Trade Count by Volatility Regime")
    plt.xlabel("Volatility Regime")
    plt.ylabel("Average Trade Count per Second")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "final_trade_count_by_vol_regime.png")
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    return save_path


def save_abs_return_by_regime(df: pd.DataFrame) -> str:
    plot_df = df[["abs_fwd_return_5s", "vol_regime"]].dropna().copy()

    data = []
    labels = []
    for reg in REGIME_ORDER:
        vals = plot_df.loc[plot_df["vol_regime"] == reg, "abs_fwd_return_5s"]
        if len(vals) > 0:
            vals = clip_series_by_quantile(vals, 0.01, 0.99)
            data.append(vals.values)
            labels.append(reg.capitalize())

    plt.figure(figsize=FIGSIZE_REGIME)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Absolute 5-Second Forward Return by Volatility Regime")
    plt.xlabel("Volatility Regime")
    plt.ylabel("Absolute 5-Second Forward Return")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "final_abs_fwd_return_by_vol_regime.png")
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    return save_path


def save_event_midprice_rel_by_regime(event_df: pd.DataFrame) -> str:
    plot_df = event_df[["relative_second", "mid_price_rel", "vol_regime"]].dropna().copy()

    plt.figure(figsize=FIGSIZE_EVENT)

    for reg in REGIME_ORDER:
        reg_df = plot_df.loc[plot_df["vol_regime"] == reg].copy()
        if reg_df.empty:
            continue

        avg_line = (
            reg_df.groupby("relative_second", as_index=False)["mid_price_rel"]
            .mean()
            .sort_values("relative_second")
        )

        plt.plot(
            avg_line["relative_second"],
            avg_line["mid_price_rel"],
            linewidth=1.8,
            label=reg.capitalize(),
        )

    plt.axvline(0, linewidth=1.0, linestyle="--")
    plt.title("Average Mid-Price Change Around Large-Move Events")
    plt.xlabel("Relative Second")
    plt.ylabel("Average Mid-Price Change from Event Second")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "final_event_midprice_rel_by_vol_regime.png")
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    return save_path


def main():
    print(f"Loading active dataset:\n{ACTIVE_PATH}")
    active_df = pd.read_parquet(ACTIVE_PATH)

    print(f"Loading event-window dataset:\n{EVENT_PATH}")
    event_df = pd.read_parquet(EVENT_PATH)

    active_df["vol_regime"] = active_df["vol_regime"].astype("string")
    event_df["vol_regime"] = event_df["vol_regime"].astype("string")

    saved_files = []
    saved_files.append(save_spread_by_regime(active_df))
    saved_files.append(save_trade_count_by_regime(active_df))
    saved_files.append(save_abs_return_by_regime(active_df))
    saved_files.append(save_event_midprice_rel_by_regime(event_df))

    print("\n" + "=" * 80)
    print("FINAL FIGURES SAVED")
    print("=" * 80)
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()