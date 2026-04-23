from pathlib import Path
from datetime import timedelta

import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# FA 550 Capstone Prototype
# BTC Futures Microstructure Dashboard - Responsive Version
# Run with: streamlit run app.py
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "live" / "derived"

ACTIVE_PATH = DATA_DIR / "microstructure_1s_active.parquet"
EVENT_WINDOWS_PATH = DATA_DIR / "event_windows_large_move_30s.parquet"
EVENT_SUMMARY_PATH = DATA_DIR / "event_level_summary_large_move_30s.parquet"

REGIME_ORDER = ["low", "medium", "high"]

st.set_page_config(
    page_title="BTC Futures Microstructure Prototype",
    layout="wide",
)

# -----------------------------
# Performance settings
# -----------------------------
MAX_OVERVIEW_POINTS = 50000
MAX_BOXPOINTS_PER_REGIME = 5000

# -----------------------------
# Plotly config
# -----------------------------
PLOTLY_CONFIG = {
    "displayModeBar": True,
}

# ============================================================
# Data loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_active_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "vol_regime" in df.columns:
        df["vol_regime"] = df["vol_regime"].astype("string")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")

    return df


@st.cache_data(show_spinner=False)
def load_event_windows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "event_time"]).copy()
    df = df.sort_values(["event_id", "relative_second"]).reset_index(drop=True)

    if "vol_regime" in df.columns:
        df["vol_regime"] = df["vol_regime"].astype("string")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")

    return df


@st.cache_data(show_spinner=False)
def load_event_summary_if_available(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        if "vol_regime" in df.columns:
            df["vol_regime"] = df["vol_regime"].astype("string")
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype("string")
        return df
    return pd.DataFrame()


# ============================================================
# Helpers
# ============================================================

def get_default_date_range(df: pd.DataFrame, days_back: int = 7):
    max_date = df["timestamp"].dt.date.max()
    min_date = df["timestamp"].dt.date.min()
    start_date = max(min_date, max_date - timedelta(days=days_back))
    return start_date, max_date


def filter_active_df(
    df: pd.DataFrame,
    start_date,
    end_date,
    selected_regimes,
    selected_symbols,
) -> pd.DataFrame:
    out = df[
        (df["timestamp"].dt.date >= start_date) &
        (df["timestamp"].dt.date <= end_date)
    ].copy()

    if selected_regimes:
        out = out[out["vol_regime"].isin(selected_regimes)]

    if selected_symbols:
        out = out[out["symbol"].isin(selected_symbols)]

    return out.reset_index(drop=True)


def filter_event_df(
    df: pd.DataFrame,
    start_date,
    end_date,
    selected_symbols,
    selected_event_regimes,
) -> pd.DataFrame:
    out = df[
        (df["event_time"].dt.date >= start_date) &
        (df["event_time"].dt.date <= end_date)
    ].copy()

    if selected_symbols:
        out = out[out["symbol"].isin(selected_symbols)]

    if selected_event_regimes:
        out = out[out["vol_regime"].isin(selected_event_regimes)]

    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_overview_display_df(
    df: pd.DataFrame,
    start_date_str: str,
    end_date_str: str,
    regimes_key: tuple,
    symbols_key: tuple,
    max_points: int = MAX_OVERVIEW_POINTS,
) -> pd.DataFrame:
    """
    Preserve the same filtered active data, but downsample rows for plotting only.
    """
    if df.empty:
        return pd.DataFrame()

    plot_df = df[["timestamp", "mid_price", "spread", "rv_60s"]].dropna(how="all").copy()
    n = len(plot_df)

    if n <= max_points:
        return plot_df

    step = max(1, n // max_points)
    return plot_df.iloc[::step].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_sampled_regime_box_df(
    df: pd.DataFrame,
    value_col: str,
    start_date_str: str,
    end_date_str: str,
    regimes_key: tuple,
    symbols_key: tuple,
    max_per_regime: int = MAX_BOXPOINTS_PER_REGIME,
) -> pd.DataFrame:
    """
    Keep boxplots, but sample rows per regime for speed.
    """
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()

    temp = df[["vol_regime", value_col]].dropna().copy()
    if temp.empty:
        return pd.DataFrame()

    parts = []
    for regime in REGIME_ORDER:
        reg_df = temp[temp["vol_regime"] == regime].copy()
        if reg_df.empty:
            continue
        if len(reg_df) > max_per_regime:
            reg_df = reg_df.sample(n=max_per_regime, random_state=42)
        parts.append(reg_df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["vol_regime"] = pd.Categorical(out["vol_regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values("vol_regime").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_trade_bar_data(
    df: pd.DataFrame,
    start_date_str: str,
    end_date_str: str,
    regimes_key: tuple,
    symbols_key: tuple,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df[["vol_regime", "trade_count"]]
        .dropna()
        .groupby("vol_regime", as_index=False)["trade_count"]
        .mean()
    )
    out["vol_regime"] = pd.Categorical(out["vol_regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values("vol_regime").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_event_grouped_data(
    df: pd.DataFrame,
    start_date_str: str,
    end_date_str: str,
    symbols_key: tuple,
    event_regimes_key: tuple,
) -> dict:
    if df.empty:
        return {
            "mid_price_rel": pd.DataFrame(),
            "spread_rel": pd.DataFrame(),
        }

    out = {}
    for col in ["mid_price_rel", "spread_rel"]:
        grouped = (
            df[["relative_second", "vol_regime", col]]
            .dropna()
            .groupby(["relative_second", "vol_regime"], as_index=False)[col]
            .mean()
        )
        grouped["vol_regime"] = pd.Categorical(grouped["vol_regime"], categories=REGIME_ORDER, ordered=True)
        grouped = grouped.sort_values(["vol_regime", "relative_second"]).reset_index(drop=True)
        out[col] = grouped

    return out


def make_time_series(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    if df.empty or y_col not in df.columns:
        return None

    plot_df = df[["timestamp", y_col]].dropna().copy()
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="timestamp",
        y=y_col,
        title=title,
        labels={"timestamp": "Timestamp", y_col: y_label},
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_regime_box(df: pd.DataFrame, value_col: str, title: str, y_label: str):
    if df.empty or value_col not in df.columns:
        return None

    fig = px.box(
        df,
        x="vol_regime",
        y=value_col,
        title=title,
        labels={"vol_regime": "Volatility Regime", value_col: y_label},
        category_orders={"vol_regime": REGIME_ORDER},
        points=False,
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_regime_bar(df: pd.DataFrame, value_col: str, title: str, y_label: str):
    if df.empty or value_col not in df.columns:
        return None

    fig = px.bar(
        df,
        x="vol_regime",
        y=value_col,
        title=title,
        labels={"vol_regime": "Volatility Regime", value_col: y_label},
        category_orders={"vol_regime": REGIME_ORDER},
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_event_line(grouped_df: pd.DataFrame, value_col: str, title: str, y_label: str):
    if grouped_df.empty or value_col not in grouped_df.columns:
        return None

    fig = px.line(
        grouped_df,
        x="relative_second",
        y=value_col,
        color="vol_regime",
        title=title,
        labels={
            "relative_second": "Relative Second",
            value_col: y_label,
            "vol_regime": "Volatility Regime",
        },
        category_orders={"vol_regime": REGIME_ORDER},
    )
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ============================================================
# Load data
# ============================================================

st.title("BTC Futures Microstructure Prototype")
st.write(
    "This dashboard is the first working prototype for the BTC futures microstructure phase of the capstone. "
    "It uses the active-market filtered dataset as the main analytical source because the original full 1-second sample "
    "was dominated by inactive and stale periods. The current views focus on market overview, volatility-regime comparisons, "
    "and event-window behavior around large-move events."
)

try:
    active_df = load_active_data(ACTIVE_PATH)
except Exception as e:
    st.error(f"Could not load active dataset:\n{e}")
    st.stop()

try:
    event_df = load_event_windows(EVENT_WINDOWS_PATH)
except Exception as e:
    st.error(f"Could not load event-window dataset:\n{e}")
    st.stop()

event_summary_df = load_event_summary_if_available(EVENT_SUMMARY_PATH)

if active_df.empty:
    st.error("The active dataset loaded successfully but is empty.")
    st.stop()

# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.header("Filters")

min_date = active_df["timestamp"].dt.date.min()
max_date = active_df["timestamp"].dt.date.max()
default_start_date, default_end_date = get_default_date_range(active_df, days_back=7)

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start_date, default_end_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start_date, default_end_date

available_regimes = [r for r in REGIME_ORDER if r in active_df["vol_regime"].dropna().unique().tolist()]
selected_regimes = st.sidebar.multiselect(
    "Volatility regime",
    options=available_regimes,
    default=available_regimes,
)

available_symbols = sorted(active_df["symbol"].dropna().unique().tolist())
default_symbols = available_symbols[:1] if available_symbols else []
selected_symbols = st.sidebar.multiselect(
    "Symbol",
    options=available_symbols,
    default=default_symbols,
)

available_event_regimes = [r for r in REGIME_ORDER if r in event_df["vol_regime"].dropna().unique().tolist()] if not event_df.empty else []
selected_event_regimes = st.sidebar.multiselect(
    "Event-window regime filter",
    options=available_event_regimes,
    default=available_event_regimes,
)

# ============================================================
# Apply filters
# ============================================================

filtered_active = filter_active_df(
    active_df,
    start_date=start_date,
    end_date=end_date,
    selected_regimes=selected_regimes,
    selected_symbols=selected_symbols,
)

filtered_events = filter_event_df(
    event_df,
    start_date=start_date,
    end_date=end_date,
    selected_symbols=selected_symbols,
    selected_event_regimes=selected_event_regimes,
)

if filtered_active.empty:
    st.warning("No active-market rows match the current filters. Please widen the date range or adjust the filters.")
    st.stop()

# ============================================================
# Build lighter display data
# ============================================================

start_date_str = str(start_date)
end_date_str = str(end_date)
regimes_key = tuple(selected_regimes)
symbols_key = tuple(selected_symbols)
event_regimes_key = tuple(selected_event_regimes)

overview_display_df = build_overview_display_df(
    filtered_active,
    start_date_str=start_date_str,
    end_date_str=end_date_str,
    regimes_key=regimes_key,
    symbols_key=symbols_key,
)

spread_box_df = build_sampled_regime_box_df(
    filtered_active,
    value_col="spread",
    start_date_str=start_date_str,
    end_date_str=end_date_str,
    regimes_key=regimes_key,
    symbols_key=symbols_key,
)

absret_box_df = build_sampled_regime_box_df(
    filtered_active,
    value_col="abs_fwd_return_5s",
    start_date_str=start_date_str,
    end_date_str=end_date_str,
    regimes_key=regimes_key,
    symbols_key=symbols_key,
)

trade_bar_df = build_trade_bar_data(
    filtered_active,
    start_date_str=start_date_str,
    end_date_str=end_date_str,
    regimes_key=regimes_key,
    symbols_key=symbols_key,
)

event_grouped = build_event_grouped_data(
    filtered_events,
    start_date_str=start_date_str,
    end_date_str=end_date_str,
    symbols_key=symbols_key,
    event_regimes_key=event_regimes_key,
)

# ============================================================
# KPI section
# ============================================================

if "event_start" in filtered_active.columns:
    num_events_kpi = int((filtered_active["event_start"] == 1).sum())
elif not filtered_events.empty and "event_id" in filtered_events.columns:
    num_events_kpi = int(filtered_events["event_id"].nunique())
else:
    num_events_kpi = 0

st.subheader("Summary Metrics")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Filtered active rows", f"{len(filtered_active):,}")
with k2:
    st.metric("Large-move events", f"{num_events_kpi:,}")
with k3:
    st.metric("Selected date range", f"{start_date} to {end_date}")
with k4:
    st.metric("Symbols selected", f"{filtered_active['symbol'].nunique():,}")

# ============================================================
# Market overview
# ============================================================

st.subheader("Market Overview")
st.caption("Overview charts use a downsampled display version of the filtered active dataset.")

c1, c2, c3 = st.columns(3)

with c1:
    fig = make_time_series(overview_display_df, "mid_price", "Mid-Price Over Time", "Mid-Price")
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with c2:
    fig = make_time_series(overview_display_df, "spread", "Spread Over Time", "Spread")
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with c3:
    fig = make_time_series(overview_display_df, "rv_60s", "Rolling Realized Volatility Over Time", "RV 60s")
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ============================================================
# Regime comparisons
# ============================================================

st.subheader("Regime Comparisons")
st.caption("Distribution views are preserved, but boxplots use capped samples per regime for speed.")

r1, r2, r3 = st.columns(3)

with r1:
    fig = make_regime_box(spread_box_df, "spread", "Spread by Volatility Regime", "Spread")
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with r2:
    fig = make_regime_bar(trade_bar_df, "trade_count", "Average Trade Count by Volatility Regime", "Average Trade Count")
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with r3:
    fig = make_regime_box(
        absret_box_df,
        "abs_fwd_return_5s",
        "Absolute 5-Second Forward Return by Volatility Regime",
        "Absolute 5s Forward Return",
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ============================================================
# Event-window analysis
# ============================================================

st.subheader("Event-Window Analysis")
st.write(
    "These views summarize average behavior around large-move events using grouped event-window averages. "
    "The event second is relative second 0."
)

if filtered_events.empty:
    st.warning("No event-window rows match the current filters.")
else:
    e1, e2 = st.columns(2)

    with e1:
        fig = make_event_line(
            event_grouped["mid_price_rel"],
            "mid_price_rel",
            "Average Mid-Price Change Around Large-Move Events",
            "Average Mid-Price Change",
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with e2:
        fig = make_event_line(
            event_grouped["spread_rel"],
            "spread_rel",
            "Average Spread Change Around Large-Move Events",
            "Average Spread Change",
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ============================================================
# Snapshot
# ============================================================

st.subheader("Filtered Data Snapshot")
snapshot_cols = ["timestamp", "symbol", "mid_price", "spread", "trade_count", "rv_60s", "vol_regime"]
snapshot_cols = [c for c in snapshot_cols if c in filtered_active.columns]
st.dataframe(filtered_active[snapshot_cols].head(20), use_container_width=True)

st.caption(
    "This version keeps the same prototype logic and views, but reduces plotting load by using display downsampling "
    "for overview charts and capped per-regime samples for boxplots."
)