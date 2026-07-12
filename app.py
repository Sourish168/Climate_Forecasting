"""
Kolkata Weather Forecast — Streamlit App
Targets  : daily Tmax, Tmin, Rain Probability
Models   : SARIMA-X + XGBoost residual ensemble
"""

import warnings
warnings.filterwarnings("ignore")

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kolkata Weather Forecast",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e8eaf0;
}
.stApp { background: #0b0f1a; }
section[data-testid="stSidebar"] { background: #111827; }

.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f1f3d 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f0f9ff;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.city-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}
.metric-card {
    background: linear-gradient(145deg, #111827, #1a2236);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(56,189,248,0.3); }
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
}
.hot  { color: #fb923c; }
.cool { color: #38bdf8; }
.rain { color: #818cf8; }
.green{ color: #34d399; }

.day-card {
    background: linear-gradient(160deg, #111827 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s;
}
.day-card:hover {
    border-color: rgba(56,189,248,0.35);
    transform: translateY(-2px);
}
.day-card .day-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.6rem;
}
.day-card .day-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.day-card .day-max {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fb923c;
}
.day-card .day-min {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #38bdf8;
}
.day-card .rain-bar-wrap {
    margin-top: 0.6rem;
    background: rgba(129,140,248,0.1);
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
}
.day-card .rain-bar {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #818cf8);
    border-radius: 6px;
}
.day-card .rain-pct {
    font-size: 0.7rem;
    color: #818cf8;
    margin-top: 0.25rem;
    font-weight: 500;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #475569;
    margin: 1.5rem 0 0.75rem 0;
    border-left: 3px solid #38bdf8;
    padding-left: 0.75rem;
}
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.status-ok {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #34d399;
    font-size: 0.85rem;
}
.status-warn {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #fbbf24;
    font-size: 0.85rem;
}
.status-err {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #f87171;
    font-size: 0.85rem;
}
.acc-badge {
    display: inline-block;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 8px;
    padding: 3px 10px;
    color: #34d399;
    font-size: 0.78rem;
    font-weight: 500;
    margin-left: 8px;
}
.train-log {
    background: #0b0f1a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem;
    font-family: 'DM Sans', monospace;
    font-size: 0.82rem;
    color: #64748b;
    max-height: 180px;
    overflow-y: auto;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LAT, LON   = 22.5726, 88.3639
TIMEZONE   = "Asia/Kolkata"
START_DATE = "2020-01-01"
FORECAST_DAYS = 7
# ── FASTER BACKTEST: only 14 days, every-other-day stride ──
BACKTEST_DAYS   = 14
BACKTEST_STRIDE = 2   # evaluate every 2nd day to halve SARIMA refits

SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)

XGB_PARAMS = dict(
    n_estimators=200,       # reduced from 300 – still accurate, much faster
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)

EXOG_COLS = [
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "windspeed_10m_max", "shortwave_radiation_sum",
]
TARGETS = {"max": "temperature_2m_max", "min": "temperature_2m_min"}


# ─────────────────────────────────────────────
# COMPAT HELPER  (fixes pandas ≥ 2.2 deprecation)
# ─────────────────────────────────────────────
def _ffill(df_or_series):
    """Forward-fill – works on both old and new pandas."""
    if hasattr(df_or_series, "ffill"):
        return df_or_series.ffill()
    return df_or_series.fillna(method="ffill")   # pandas < 1.5 fallback


def _bfill(df_or_series):
    """Back-fill – works on both old and new pandas."""
    if hasattr(df_or_series, "bfill"):
        return df_or_series.bfill()
    return df_or_series.fillna(method="bfill")


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def fetch_historical(log_fn=None) -> pd.DataFrame:
    import pytz

    ist = pytz.timezone("Asia/Kolkata")

    # Open-Meteo historical archive can lag, so avoid requesting today.
    end_date = (
        pd.Timestamp.now(tz=ist).normalize() - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if log_fn:
        log_fn(f"📡 Fetching historical data up to {end_date} …")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
            "sunrise",
            "sunset",
        ],
        "timezone": TIMEZONE,
    }

    r = requests.get(url, params=params, timeout=90)

    if r.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo archive error {r.status_code}: {r.text}"
        )

    data = r.json().get("daily", {})
    if not data:
        raise ValueError("Open-Meteo archive returned no daily data.")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["time"])
    df.set_index("date", inplace=True)
    df.drop(columns=["time"], inplace=True, errors="ignore")

    # Open-Meteo uses the newer names. Rename to your app's existing column names.
    df.rename(
        columns={
            "wind_speed_10m_max": "windspeed_10m_max",
            "wind_gusts_10m_max": "windgusts_10m_max",
        },
        inplace=True,
    )

    # Compute day length
    if "sunrise" in df.columns and "sunset" in df.columns:
        df["sunrise_dt"] = pd.to_datetime(df["sunrise"], errors="coerce")
        df["sunset_dt"] = pd.to_datetime(df["sunset"], errors="coerce")

        df["day_length_h"] = (
            df["sunset_dt"] - df["sunrise_dt"]
        ).dt.total_seconds() / 3600

        df.drop(
            columns=["sunrise", "sunset", "sunrise_dt", "sunset_dt"],
            inplace=True,
            errors="ignore",
        )

    # Ensure all EXOG_COLS are present and numeric
    for col in EXOG_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.ffill().bfill()

    if log_fn:
        log_fn(
            f"✅ {len(df)} days loaded "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )

    return df

def fetch_future(log_fn=None) -> pd.DataFrame:
    if log_fn:
        log_fn("📡 Fetching 7-day NWP forecast from Open-Meteo …")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "windspeed_10m_max", "shortwave_radiation_sum",
            "apparent_temperature_max", "apparent_temperature_min",
            "precipitation_hours", "windgusts_10m_max",
            "et0_fao_evapotranspiration", "precipitation_probability_max",
            "sunrise", "sunset",
        ],
        "forecast_days": 16,
        "timezone": TIMEZONE,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    raw = r.json()
    if "daily" not in raw:
        raise ValueError(f"Open-Meteo API error: {raw}")

    df = pd.DataFrame(raw["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df.set_index("date", inplace=True)
    df.drop(columns=["time"], inplace=True, errors="ignore")

    if log_fn:
        log_fn(f"   API returned {len(df)} rows: {df.index[0].date()} → {df.index[-1].date()}")

    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    today_ist = pd.Timestamp.now(tz=ist).normalize().tz_localize(None)

    df_future = df[df.index > today_ist].copy()
    if len(df_future) == 0:
        if log_fn:
            log_fn("   ⚠️ Timezone filter returned 0 rows, falling back to iloc[1:]")
        df_future = df.iloc[1:].copy()

    df_future = df_future.iloc[:FORECAST_DAYS]

    if len(df_future) == 0:
        raise ValueError(
            f"fetch_future yielded 0 rows after all fallbacks. "
            f"Raw API dates: {list(df.index.date)}"
        )

    if "sunrise" in df_future.columns and "sunset" in df_future.columns:
        df_future["sunrise_dt"] = pd.to_datetime(df_future["sunrise"])
        df_future["sunset_dt"]  = pd.to_datetime(df_future["sunset"])
        df_future["day_length_h"] = (
            (df_future["sunset_dt"] - df_future["sunrise_dt"]).dt.total_seconds() / 3600
        )
        df_future.drop(
            columns=["sunrise", "sunset", "sunrise_dt", "sunset_dt"],
            inplace=True, errors="ignore"
        )

    for col in EXOG_COLS:
        if col not in df_future.columns:
            df_future[col] = np.nan
        df_future[col] = pd.to_numeric(df_future[col], errors="coerce")

    df_future = _ffill(df_future)
    df_future = _bfill(df_future)

    if log_fn:
        log_fn(
            f"✅ {len(df_future)} future days fetched "
            f"({df_future.index[0].date()} → {df_future.index[-1].date()})"
        )
    return df_future


def fetch_today_conditions() -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "current": [
            "temperature_2m", "apparent_temperature", "precipitation",
            "precipitation_probability", "weathercode", "windspeed_10m",
            "relativehumidity_2m",
        ],
        "timezone": TIMEZONE,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        return r.json().get("current", {})
    except Exception:
        return {}


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dayofyear"]  = df.index.dayofyear
    df["month"]      = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    for k in [1, 2, 3]:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365)

    t_max = df.get("temperature_2m_max", pd.Series(dtype=float))
    t_min = df.get("temperature_2m_min", pd.Series(dtype=float))
    if not t_max.empty and not t_min.empty:
        df["temp_range"] = t_max - t_min

    for tc in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]:
        if tc not in df.columns:
            continue
        for lag in range(1, 15):
            df[f"{tc}_lag{lag}"] = df[tc].shift(lag)

    for col in ["precipitation_sum", "shortwave_radiation_sum", "windspeed_10m_max"]:
        if col not in df.columns:
            continue
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for w in [3, 7, 14, 30]:
        for tc in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]:
            if tc not in df.columns:
                continue
            df[f"{tc}_roll{w}_mean"] = df[tc].rolling(w, min_periods=1).mean()
            df[f"{tc}_roll{w}_std"]  = df[tc].rolling(w, min_periods=1).std()
            df[f"{tc}_roll{w}_min"]  = df[tc].rolling(w, min_periods=1).min()
            df[f"{tc}_roll{w}_max"]  = df[tc].rolling(w, min_periods=1).max()

        if "precipitation_sum" in df.columns:
            df[f"precip_roll{w}_sum"] = df["precipitation_sum"].rolling(w, min_periods=1).sum()
        if "shortwave_radiation_sum" in df.columns:
            df[f"rad_roll{w}_mean"] = df["shortwave_radiation_sum"].rolling(w, min_periods=1).mean()

    for tc in ["temperature_2m_max", "temperature_2m_min"]:
        if tc not in df.columns:
            continue
        df[f"{tc}_delta1"] = df[tc].diff(1)
        df[f"{tc}_delta7"] = df[tc].diff(7)

    return df


# ─────────────────────────────────────────────
# MODEL FITTING
# ─────────────────────────────────────────────
def _get_feature_cols(df: pd.DataFrame, target_key: str) -> list:
    exclude = set(TARGETS.values())
    exclude.update([f"sarima_fitted_{k}" for k in TARGETS])
    exclude.update([f"residual_{k}" for k in TARGETS])
    if "precipitation_probability_max" in df.columns:
        exclude.add("precipitation_probability_max")

    cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def fit_pipeline(df_feat: pd.DataFrame, target_key: str, log_fn=None):
    target_col = TARGETS[target_key]
    exog_cols  = [c for c in EXOG_COLS if c != target_col and c in df_feat.columns]

    sarima_df = df_feat[[target_col] + exog_cols].dropna()
    if len(sarima_df) < 60:
        raise ValueError(
            f"Not enough clean rows for SARIMA ({len(sarima_df)}). Check data fetch."
        )

    if log_fn:
        log_fn(f"🔧 Fitting SARIMA for T{target_key.upper()} on {len(sarima_df)} rows …")

    sarima = SARIMAX(
        sarima_df[target_col],
        exog=sarima_df[exog_cols],
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_model = sarima.fit(disp=False, maxiter=200)

    df_out = df_feat.copy()
    df_out[f"sarima_fitted_{target_key}"] = np.nan
    df_out.loc[sarima_df.index, f"sarima_fitted_{target_key}"] = sarima_model.fittedvalues.values
    df_out[f"residual_{target_key}"] = (
        df_out[target_col] - df_out[f"sarima_fitted_{target_key}"]
    )

    feature_cols = _get_feature_cols(df_out, target_key)

    xgb_df = df_out[feature_cols + [f"residual_{target_key}"]].dropna()
    if len(xgb_df) < 30:
        raise ValueError(
            f"Not enough clean rows for XGB ({len(xgb_df)}) after residual calc."
        )

    if log_fn:
        log_fn(
            f"🌲 Fitting XGBoost for T{target_key.upper()} "
            f"({len(feature_cols)} features, {len(xgb_df)} rows) …"
        )

    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(xgb_df[feature_cols], xgb_df[f"residual_{target_key}"])

    if log_fn:
        log_fn(f"✅ T{target_key.upper()} pipeline ready.")

    return sarima_model, xgb, feature_cols, df_out


# ─────────────────────────────────────────────
# FAST BACK-TEST  (apply already-fitted model)
# ─────────────────────────────────────────────
def run_backtest(
    df_feat: pd.DataFrame,
    target_key: str,
    sarima_model,          # ← re-use the already-fitted model
    xgb_model,
    feature_cols: list,
    log_fn=None,
):
    """
    Fast back-test: apply the already-fitted SARIMA + XGBoost to the last
    BACKTEST_DAYS of training data (strided). No re-fitting – ~10x faster.
    """
    target_col = TARGETS[target_key]
    exog_cols  = [c for c in EXOG_COLS if c != target_col and c in df_feat.columns]

    if log_fn:
        log_fn(
            f"📊 Back-testing T{target_key.upper()} "
            f"(last {BACKTEST_DAYS} days, stride {BACKTEST_STRIDE}) …"
        )

    test_slice = df_feat.iloc[-BACKTEST_DAYS::BACKTEST_STRIDE]
    records = []

    for date_val, row in test_slice.iterrows():
        # Skip if target is missing
        if pd.isna(row.get(target_col)):
            continue

        try:
            # ── SARIMA one-step in-sample fitted value ──────────────────
            # Use the model's fittedvalues if the date was in the training window
            sarima_col = f"sarima_fitted_{target_key}"
            if sarima_col in df_feat.columns and not pd.isna(df_feat.at[date_val, sarima_col]):
                sp = float(df_feat.at[date_val, sarima_col])
            else:
                # Fallback: forecast from the point just before date_val
                idx_pos = df_feat.index.get_loc(date_val)
                if idx_pos == 0:
                    continue
                train_end = df_feat.index[idx_pos - 1]
                train_slice = df_feat.loc[:train_end, [target_col] + exog_cols].dropna()
                if len(train_slice) < 90:
                    continue
                sm_tmp = SARIMAX(
                    train_slice[target_col],
                    exog=train_slice[exog_cols],
                    order=SARIMA_ORDER,
                    seasonal_order=SARIMA_SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=100)
                fexog = df_feat.loc[[date_val], exog_cols].fillna(0)
                sp = float(sm_tmp.forecast(steps=1, exog=fexog).iloc[0])

            # ── XGBoost residual ─────────────────────────────────────────
            valid_fc = [c for c in feature_cols if c in df_feat.columns]
            test_row = df_feat.loc[[date_val], valid_fc].fillna(0)
            if test_row.shape[1] == 0:
                records.append({"date": date_val, "actual": row[target_col], "predicted": sp})
                continue

            rp = float(xgb_model.predict(test_row)[0])
            records.append({
                "date": date_val,
                "actual": float(row[target_col]),
                "predicted": sp + rp,
            })

        except Exception:
            continue

    if len(records) == 0:
        if log_fn:
            log_fn(f"   ⚠️ No valid back-test records for T{target_key.upper()}.")
        return pd.DataFrame(columns=["actual", "predicted"]), float("nan"), float("nan")

    bt   = pd.DataFrame(records).set_index("date")
    mae  = mean_absolute_error(bt["actual"], bt["predicted"])
    rmse = float(np.sqrt(mean_squared_error(bt["actual"], bt["predicted"])))

    if log_fn:
        log_fn(f"   MAE={mae:.2f}°C  RMSE={rmse:.2f}°C")

    return bt, mae, rmse


# ─────────────────────────────────────────────
# RECURSIVE FORECAST
# ─────────────────────────────────────────────
def recursive_forecast(
    df_feat: pd.DataFrame,
    future_df: pd.DataFrame,
    sarima_model,
    xgb_model,
    feature_cols: list,
    target_key: str,
) -> tuple:
    target_col = TARGETS[target_key]
    exog_cols  = [c for c in EXOG_COLS if c != target_col and c in future_df.columns]

    n_steps = len(future_df)
    if n_steps == 0:
        return np.array([]), np.array([])

    # ── SARIMA forecast ──────────────────────────────────────────────────
    # FIX: replace deprecated fillna(method=...) with ffill()
    sarima_exog = future_df[exog_cols].ffill().fillna(0)
    sarima_preds = sarima_model.forecast(steps=n_steps, exog=sarima_exog).values.flatten()

    # ── XGBoost residual forecast (recursive) ────────────────────────────
    current_row = df_feat.iloc[[-1]].copy()
    valid_feature_cols = [c for c in feature_cols if c in current_row.columns]
    residual_preds = []

    for i, next_date in enumerate(future_df.index):
        exog_row = future_df.iloc[i]

        for col in future_df.columns:
            if col in current_row.columns:
                current_row[col] = exog_row[col]

        current_row.index = [next_date]
        current_row["dayofyear"]  = next_date.dayofyear
        current_row["month"]      = next_date.month
        current_row["weekofyear"] = int(next_date.isocalendar()[1])

        for k in [1, 2, 3]:
            current_row[f"sin_{k}"] = np.sin(2 * np.pi * k * next_date.dayofyear / 365)
            current_row[f"cos_{k}"] = np.cos(2 * np.pi * k * next_date.dayofyear / 365)

        if "temperature_2m_max" in current_row.columns and "temperature_2m_min" in current_row.columns:
            current_row["temp_range"] = (
                current_row["temperature_2m_max"] - current_row["temperature_2m_min"]
            )

        X_input = current_row.reindex(columns=feature_cols, fill_value=0).fillna(0)

        if X_input.shape[0] == 0 or X_input.shape[1] == 0:
            residual_preds.append(0.0)
        else:
            res_pred = float(xgb_model.predict(X_input)[0])
            residual_preds.append(res_pred)

        for lag in range(14, 1, -1):
            lag_col  = f"{target_col}_lag{lag}"
            prev_col = f"{target_col}_lag{lag - 1}"
            if lag_col in current_row.columns and prev_col in current_row.columns:
                current_row[lag_col] = current_row[prev_col].values[0]

        if f"{target_col}_lag1" in current_row.columns:
            current_row[f"{target_col}_lag1"] = sarima_preds[i]

    return sarima_preds, np.array(residual_preds)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def weather_icon(tmax, rain_pct, precip_mm):
    if rain_pct >= 80 or precip_mm >= 10:
        return "⛈️" if rain_pct >= 80 else "🌧️"
    if rain_pct >= 50 or precip_mm >= 2:
        return "🌦️"
    if rain_pct >= 30:
        return "🌂"
    if tmax >= 38:
        return "🥵"
    if tmax >= 32:
        return "☀️"
    return "⛅"


def wmo_desc(code):
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog",
        51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Rain showers", 81: "Moderate showers", 82: "Violent showers",
        95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
    }
    try:
        return mapping.get(int(code), "—")
    except (TypeError, ValueError):
        return "—"


# ─────────────────────────────────────────────
# TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────
def run_full_training(log_container):
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_container.markdown(
            "<div class='train-log'>" +
            "<br>".join(log_lines[-12:]) +
            "</div>",
            unsafe_allow_html=True,
        )

    log("🚀 Starting full training pipeline …")

    # ── 1. Fetch historical data
    df_raw = fetch_historical(log)

    # ── 2. Feature engineering
    log("⚙️  Engineering features …")
    df_feat = make_features(df_raw)
    df_feat = df_feat.dropna(subset=list(TARGETS.values()))
    log(f"   {df_feat.shape[1]} features × {len(df_feat)} days (after target dropna)")

    # ── 3. Fit models + fast backtest (re-uses fitted models)
    results = {}
    metrics = {}
    for tk in TARGETS:
        sm, xgb, fcols, df_res = fit_pipeline(df_feat, tk, log)
        bt, mae, rmse = run_backtest(df_feat, tk, sm, xgb, fcols, log)
        results[tk] = dict(sarima=sm, xgb=xgb, features=fcols, df=df_res)
        metrics[tk] = dict(mae=mae, rmse=rmse)

    # ── 4. Fetch future NWP data
    log("📡 Fetching future NWP data …")
    future_raw = fetch_future(log)
    log(
        f"   {len(future_raw)} future rows: "
        f"{future_raw.index[0].date()} → {future_raw.index[-1].date()}"
    )

    # ── 5. Feature-engineer future rows
    combined      = pd.concat([df_raw, future_raw])
    combined      = combined[~combined.index.duplicated(keep="last")]
    combined_feat = make_features(combined)
    future_df     = combined_feat.reindex(future_raw.index).iloc[:FORECAST_DAYS]
    log(f"   future_df shape after feature engineering: {future_df.shape}")

    future_df = _ffill(future_df)
    future_df = _bfill(future_df)
    future_df = future_df.fillna(0)

    _zero = pd.Series(0.0, index=future_raw.index[:FORECAST_DAYS])
    future_rain = (
        future_raw["precipitation_probability_max"]
        if "precipitation_probability_max" in future_raw.columns
        else _zero
    )
    future_precip = (
        future_raw["precipitation_sum"]
        if "precipitation_sum" in future_raw.columns
        else _zero
    )
    future_rain   = future_rain.fillna(0).reindex(future_df.index, fill_value=0)
    future_precip = future_precip.fillna(0).reindex(future_df.index, fill_value=0)

    # ── 6. Generate forecasts
    rows = []
    for tk in TARGETS:
        r = results[tk]
        sp, rp = recursive_forecast(
            r["df"], future_df, r["sarima"], r["xgb"], r["features"], tk
        )
        final = sp + rp
        for i, dt in enumerate(future_df.index[:FORECAST_DAYS]):
            rows.append({
                "date": dt, "target": tk,
                "sarima": round(float(sp[i]), 2) if i < len(sp) else 0.0,
                "xgb_residual": round(float(rp[i]), 2) if i < len(rp) else 0.0,
                "final": round(float(final[i]), 2) if i < len(final) else 0.0,
            })

    forecast_df = pd.DataFrame(rows)
    pivot = forecast_df.pivot(index="date", columns="target", values=["sarima", "xgb_residual", "final"])
    pivot.columns = [f"{v}_{k}" for v, k in pivot.columns]
    pivot["rain_pct"]  = future_rain.values[:FORECAST_DAYS]
    pivot["precip_mm"] = future_precip.values[:FORECAST_DAYS]

    log("✅ All done! Forecast ready.")
    return pivot, metrics, df_raw, df_feat, results


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="app-header">
        <div class="city-badge">📍 Kolkata, West Bengal</div>
        <h1>🌦️ Sourish's Weather Intelligence</h1>
        <p>Made by Sourish Dey</p>
    </div>
    """, unsafe_allow_html=True)

    today_data = fetch_today_conditions()
    now_temp   = today_data.get("temperature_2m", "—")
    now_feel   = today_data.get("apparent_temperature", "—")
    now_rain   = today_data.get("precipitation_probability", 0) or 0
    now_precip = today_data.get("precipitation", 0) or 0
    now_wind   = today_data.get("windspeed_10m", "—")
    now_humid  = today_data.get("relativehumidity_2m", "—")
    now_code   = today_data.get("weathercode", 0)

    st.markdown("<div class='section-title'>Current Conditions</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Temperature</div>
            <div class="value hot">{now_temp}°C</div>
            <div class="sub">Feels like {now_feel}°C</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        rain_color = "rain" if now_rain > 40 else "green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Rain Chance</div>
            <div class="value {rain_color}">{now_rain}%</div>
            <div class="sub">{now_precip} mm today</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Wind Speed</div>
            <div class="value cool">{now_wind} km/h</div>
            <div class="sub">Surface · 10 m</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Humidity</div>
            <div class="value">{now_humid}%</div>
            <div class="sub">Relative humidity</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Conditions</div>
            <div class="value" style="font-size:1.4rem">{wmo_desc(now_code)}</div>
            <div class="sub">{datetime.now().strftime('%a %d %b, %H:%M IST')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, col_status = st.columns([2, 5])
    with col_btn:
        train_btn = st.button("🔄  Train Model & Refresh Forecast", use_container_width=True)
    with col_status:
        if "trained_at" in st.session_state:
            mae_max = st.session_state.get("mae_max")
            mae_min = st.session_state.get("mae_min")
            mae_max_str = f"{mae_max:.2f}°C" if mae_max and not np.isnan(mae_max) else "N/A"
            mae_min_str = f"{mae_min:.2f}°C" if mae_min and not np.isnan(mae_min) else "N/A"
            st.markdown(
                f"<div class='status-ok'>✅ Model trained · {st.session_state['trained_at']} "
                f"· MAX MAE <b>{mae_max_str}</b> "
                f"· MIN MAE <b>{mae_min_str}</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='status-warn'>⚠️ Model not trained yet — "
                "click the button to train and generate forecast.</div>",
                unsafe_allow_html=True,
            )

    if train_btn:
        log_ph = st.empty()
        log_ph.markdown("<div class='train-log'>Initialising …</div>", unsafe_allow_html=True)
        try:
            t0 = time.time()
            pivot, metrics, df_raw, df_feat, results = run_full_training(log_ph)
            elapsed = time.time() - t0
            st.session_state["pivot"]      = pivot
            st.session_state["metrics"]    = metrics
            st.session_state["df_raw"]     = df_raw
            st.session_state["df_feat"]    = df_feat
            st.session_state["results"]    = results
            st.session_state["trained_at"] = datetime.now().strftime("%d %b %Y %H:%M IST")
            st.session_state["mae_max"]    = metrics["max"]["mae"]
            st.session_state["mae_min"]    = metrics["min"]["mae"]
            log_ph.markdown(
                f"<div class='status-ok'>✅ Training complete in {elapsed:.0f}s</div>",
                unsafe_allow_html=True,
            )
            st.rerun()
        except Exception as e:
            import traceback
            log_ph.markdown(
                f"<div class='status-err'>❌ Error: {e}<br>"
                f"<pre style='font-size:0.7rem;color:#f87171'>"
                f"{traceback.format_exc()[-1200:]}"
                f"</pre></div>",
                unsafe_allow_html=True,
            )

    if "pivot" not in st.session_state:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#334155;'>
            <div style='font-size:3rem'>🌦️</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;margin-top:0.5rem;'>
                Train the model above to see the 7-day forecast
            </div>
        </div>""", unsafe_allow_html=True)
        return

    pivot   = st.session_state["pivot"]
    df_raw  = st.session_state["df_raw"]
    metrics = st.session_state["metrics"]

    st.markdown("<div class='section-title'>7-Day Forecast</div>", unsafe_allow_html=True)
    cols = st.columns(FORECAST_DAYS)
    for i, (dt, row) in enumerate(pivot.iterrows()):
        tmax     = row.get("final_max", row.get("sarima_max", 30))
        tmin     = row.get("final_min", row.get("sarima_min", 22))
        rain_pct = int(row.get("rain_pct", 0) or 0)
        precip   = row.get("precip_mm", 0) or 0
        icon     = weather_icon(tmax, rain_pct, precip)
        day_name = pd.Timestamp(dt).strftime("%a")
        day_date = pd.Timestamp(dt).strftime("%d %b")

        with cols[i]:
            st.markdown(f"""
            <div class="day-card">
                <div class="day-name">{day_name}<br>
                    <span style='font-size:0.65rem;color:#334155'>{day_date}</span>
                </div>
                <div class="day-icon">{icon}</div>
                <div class="day-max">{tmax:.1f}°</div>
                <div class="day-min">{tmin:.1f}°</div>
                <div class="rain-bar-wrap">
                    <div class="rain-bar" style="width:{rain_pct}%"></div>
                </div>
                <div class="rain-pct">💧 {rain_pct}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["📈 Temperature Forecast", "🌧️ Rain Probability", "🔬 Model Performance"]
    )

    with tab1:
        fig = make_subplots(rows=1, cols=1)
        dates = [pd.Timestamp(d) for d in pivot.index]

        hist_tail = df_raw.tail(30)
        fig.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail["temperature_2m_max"],
            name="Hist Tmax",
            line=dict(color="#fb923c", width=1.5, dash="dot"),
            opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail["temperature_2m_min"],
            name="Hist Tmin",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
            opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=list(pivot["final_max"]) + list(pivot["final_min"])[::-1],
            fill="toself",
            fillcolor="rgba(251,146,60,0.07)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="Forecast band",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=pivot["final_max"],
            name="Forecast Tmax",
            mode="lines+markers",
            line=dict(color="#fb923c", width=2.5),
            marker=dict(size=7, color="#fb923c"),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=pivot["final_min"],
            name="Forecast Tmin",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=7, color="#38bdf8"),
        ))
        fig.update_layout(
            plot_bgcolor="#0b0f1a", paper_bgcolor="#0b0f1a",
            font=dict(color="#94a3b8", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, title="°C"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dates       = [pd.Timestamp(d) for d in pivot.index]
        rain_vals   = list(pivot["rain_pct"])
        precip_vals = list(pivot["precip_mm"])
        day_labels  = [pd.Timestamp(d).strftime("%a %d") for d in pivot.index]

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        colors = [
            "#6366f1" if v >= 50 else "#818cf8" if v >= 30 else "#334155"
            for v in rain_vals
        ]
        fig2.add_trace(go.Bar(
            x=day_labels, y=rain_vals,
            name="Rain probability %",
            marker_color=colors, opacity=0.85,
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=day_labels, y=precip_vals,
            name="Precipitation mm",
            mode="lines+markers",
            line=dict(color="#34d399", width=2),
            marker=dict(size=7),
        ), secondary_y=True)
        fig2.add_hline(
            y=50, line_dash="dot", line_color="rgba(129,140,248,0.4)",
            annotation_text="50% threshold", annotation_font_color="#818cf8",
        )
        fig2.update_layout(
            plot_bgcolor="#0b0f1a", paper_bgcolor="#0b0f1a",
            font=dict(color="#94a3b8", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.04)",
                title="Rain chance %", range=[0, 105],
            ),
            yaxis2=dict(title="Precipitation mm", gridcolor="rgba(255,255,255,0)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown(
            "<div class='section-title'>Back-test Results (last 14 days, strided)</div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        with m1:
            mae_max  = metrics["max"]["mae"]
            rmse_max = metrics["max"]["rmse"]
            mae_str  = f"{mae_max:.2f}°C" if not (isinstance(mae_max, float) and np.isnan(mae_max)) else "N/A"
            rmse_str = f"{rmse_max:.2f}" if not (isinstance(rmse_max, float) and np.isnan(rmse_max)) else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Tmax · MAE / RMSE</div>
                <div class="value hot">{mae_str}
                    <span class="acc-badge">RMSE {rmse_str}</span>
                </div>
                <div class="sub">Mean absolute error on held-out days</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            mae_min  = metrics["min"]["mae"]
            rmse_min = metrics["min"]["rmse"]
            mae_str  = f"{mae_min:.2f}°C" if not (isinstance(mae_min, float) and np.isnan(mae_min)) else "N/A"
            rmse_str = f"{rmse_min:.2f}" if not (isinstance(rmse_min, float) and np.isnan(rmse_min)) else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Tmin · MAE / RMSE</div>
                <div class="value cool">{mae_str}
                    <span class="acc-badge">RMSE {rmse_str}</span>
                </div>
                <div class="sub">Mean absolute error on held-out days</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fi_col1, fi_col2 = st.columns(2)
        for col_ui, tk, color in [(fi_col1, "max", "#fb923c"), (fi_col2, "min", "#38bdf8")]:
            with col_ui:
                r  = st.session_state["results"][tk]
                fi = pd.Series(
                    r["xgb"].feature_importances_,
                    index=r["features"],
                ).sort_values(ascending=False).head(12)

                fig_fi = go.Figure(go.Bar(
                    x=fi.values[::-1],
                    y=fi.index[::-1],
                    orientation="h",
                    marker_color=color,
                    opacity=0.8,
                ))
                fig_fi.update_layout(
                    title=f"Top Features — T{tk.upper()}",
                    plot_bgcolor="#0b0f1a", paper_bgcolor="#0b0f1a",
                    font=dict(color="#94a3b8", family="DM Sans", size=11),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0)"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=380,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        "<div class='section-title'>Historical Temperature Explorer</div>",
        unsafe_allow_html=True,
    )
    yr_min = int(df_raw.index.year.min())
    yr_max = int(df_raw.index.year.max())
    sel_years = st.slider("Select year range", yr_min, yr_max, (max(yr_min, yr_max - 1), yr_max))

    hist_sel = df_raw[
        (df_raw.index.year >= sel_years[0]) & (df_raw.index.year <= sel_years[1])
    ]

    fig_hist = go.Figure()
    if "temperature_2m_max" in hist_sel.columns:
        fig_hist.add_trace(go.Scatter(
            x=hist_sel.index,
            y=hist_sel["temperature_2m_max"].rolling(7, min_periods=1).mean(),
            name="Tmax (7d avg)",
            line=dict(color="#fb923c", width=1.8),
        ))
    if "temperature_2m_min" in hist_sel.columns:
        fig_hist.add_trace(go.Scatter(
            x=hist_sel.index,
            y=hist_sel["temperature_2m_min"].rolling(7, min_periods=1).mean(),
            name="Tmin (7d avg)",
            line=dict(color="#38bdf8", width=1.8),
        ))
    if "precipitation_sum" in hist_sel.columns:
        fig_hist.add_trace(go.Bar(
            x=hist_sel.index,
            y=hist_sel["precipitation_sum"],
            name="Precipitation mm",
            marker_color="#6366f1",
            opacity=0.3,
            yaxis="y2",
        ))
    fig_hist.update_layout(
        plot_bgcolor="#0b0f1a", paper_bgcolor="#0b0f1a",
        font=dict(color="#94a3b8", family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="°C"),
        yaxis2=dict(
            title="Precip mm", overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("📋 Raw Forecast Data"):
        display = pivot.copy()
        display.index = [pd.Timestamp(d).strftime("%A, %d %b %Y") for d in display.index]
        display.columns = [c.replace("_", " ").title() for c in display.columns]
        st.dataframe(display.style.format("{:.2f}"), use_container_width=True)

    st.markdown("""
    <div style='text-align:center;color:#1e293b;font-size:0.75rem;margin-top:2rem;'>
        Data: Open-Meteo Archive &amp; NWP · Model: SARIMA-X + XGBoost · Kolkata (22.57°N 88.36°E)
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()