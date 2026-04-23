import os
import numpy as np
import pandas as pd

# ============================================================
# FA 550 Capstone Prototype
# Prepare dashboard-ready BTC microstructure dataset
# ============================================================

BASE = r"C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi"
INPUT_PATH = os.path.join(BASE, "data", "live", "derived", "btc_microstructure_1s_mvp.parquet")
OUTPUT_DIR = os.path.join(BASE, "data", "live", "derived")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "microstructure_1s_dashboard.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CORE_COLS = [
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
    "log_return_1s",
    "fwd_return_5s",
    "abs_fwd_return_5s",
    "rv_60s",
    "vol_regime",
    "event_large_move",
]


def main():
    print(f"Loading input file:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[CORE_COLS].copy()

    # Basic cleanup
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Make sure event_large_move is clean integer 0/1
    df["event_large_move"] = pd.to_numeric(df["event_large_move"], errors="coerce").fillna(0)
    df["event_large_move"] = (df["event_large_move"] > 0).astype(int)

    # event_start = 1 when current row is an event and previous row is not
    prev_event = df["event_large_move"].shift(1, fill_value=0)
    df["event_start"] = ((df["event_large_move"] == 1) & (prev_event != 1)).astype(int)

    # event_id increments at each new event_start, but only for event rows
    df["event_id"] = df["event_start"].cumsum()
    df.loc[df["event_large_move"] == 0, "event_id"] = np.nan

    # Optional: make event_id integer-like where present
    df["event_id"] = df["event_id"].astype("Int64")

    # Save
    df.to_parquet(OUTPUT_PATH, index=False)

    # Sanity checks
    print("\n" + "=" * 80)
    print("DASHBOARD DATASET SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"event_large_move == 1 count: {int((df['event_large_move'] == 1).sum()):,}")
    print(f"event_start == 1 count: {int((df['event_start'] == 1).sum()):,}")

    print("\nTop null counts:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    print(f"\nSaved dashboard-ready file to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()