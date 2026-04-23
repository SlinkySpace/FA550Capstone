import os
import pandas as pd

# ============================================================
# FA 550 Capstone Prototype
# Build event-window dataset from active-analysis file
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "microstructure_1s_active.parquet")
OUTPUT_DIR = os.path.join(BASE, "data", "live", "derived")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "event_windows_large_move_30s.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_BEFORE = 30
WINDOW_AFTER = 30


def main():
    print(f"Loading active dataset:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clean key columns
    df["event_start"] = pd.to_numeric(df["event_start"], errors="coerce").fillna(0).astype(int)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["vol_regime"] = df["vol_regime"].astype("string")

    total_event_start_rows = int((df["event_start"] == 1).sum())

    event_rows = df.loc[df["event_start"] == 1].copy()
    event_rows = event_rows.reset_index(drop=False).rename(columns={"index": "event_index"})

    all_windows = []
    valid_windows_retained = 0

    for _, event in event_rows.iterrows():
        center_idx = int(event["event_index"])
        start_idx = center_idx - WINDOW_BEFORE
        end_idx = center_idx + WINDOW_AFTER

        # Need full symmetric window
        if start_idx < 0 or end_idx >= len(df):
            continue

        window = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)

        # Keep only windows entirely within one symbol
        event_symbol = event["symbol"]
        if window["symbol"].nunique() != 1 or window["symbol"].iloc[0] != event_symbol:
            continue

        # Confirm the center row still lines up with the event
        center_row = window.iloc[WINDOW_BEFORE]
        if center_row["timestamp"] != event["timestamp"]:
            continue

        event_mid = center_row["mid_price"]
        event_spread = center_row["spread"]

        window["event_id"] = int(event["event_id"]) if pd.notna(event["event_id"]) else None
        window["event_time"] = event["timestamp"]
        window["relative_second"] = range(-WINDOW_BEFORE, WINDOW_AFTER + 1)

        # Keep only needed columns
        window = window[
            [
                "event_id",
                "event_time",
                "relative_second",
                "timestamp",
                "symbol",
                "mid_price",
                "spread",
                "trade_count",
                "rv_60s",
                "abs_fwd_return_5s",
                "vol_regime",
            ]
        ].copy()

        # Normalized versions for plotting
        window["mid_price_rel"] = window["mid_price"] - event_mid
        window["spread_rel"] = window["spread"] - event_spread

        all_windows.append(window)
        valid_windows_retained += 1

    if len(all_windows) > 0:
        event_window_df = pd.concat(all_windows, ignore_index=True)
    else:
        event_window_df = pd.DataFrame(
            columns=[
                "event_id",
                "event_time",
                "relative_second",
                "timestamp",
                "symbol",
                "mid_price",
                "spread",
                "trade_count",
                "rv_60s",
                "abs_fwd_return_5s",
                "vol_regime",
                "mid_price_rel",
                "spread_rel",
            ]
        )

    event_window_df.to_parquet(OUTPUT_PATH, index=False)

    # Sanity checks
    event_symbol_dist = (
        event_window_df[["event_id", "symbol"]]
        .drop_duplicates()
        .groupby("symbol")
        .size()
        .sort_values(ascending=False)
        if len(event_window_df) > 0
        else pd.Series(dtype="int64")
    )

    event_regime_dist = (
        event_window_df.loc[event_window_df["relative_second"] == 0, ["event_id", "vol_regime"]]
        .drop_duplicates()
        .groupby("vol_regime")
        .size()
        .sort_values(ascending=False)
        if len(event_window_df) > 0
        else pd.Series(dtype="int64")
    )

    print("\n" + "=" * 80)
    print("EVENT WINDOW DATASET SUMMARY")
    print("=" * 80)
    print(f"Total event_start rows: {total_event_start_rows:,}")
    print(f"Valid windows retained: {valid_windows_retained:,}")

    print("\nDistribution of events by symbol:")
    print(event_symbol_dist)

    print("\nDistribution of events by vol_regime:")
    print(event_regime_dist)

    print("\nHead of extracted event-window dataframe:")
    print(event_window_df.head())

    print(f"\nSaved event-window dataset to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()