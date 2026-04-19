# =============================================================
# models/supply_demand.py
# ChronosLab — Urban Supply/Demand Forecasting Pipeline
# Architecture: ETS baseline → Dual-Gated DL (GBR + BiLSTM)
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, Bidirectional,
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers

tf.get_logger().setLevel("ERROR")


# ── NYC Taxi demo dataset ──────────────────────────────────────

_NYC_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/"
    "data/realKnownCause/nyc_taxi.csv"
)

# Three pre-defined slices used by the demo route
NYC_SCENARIOS = {
    "weekday_24h": ("2014-11-18 23:00:00", 24),   # Tuesday
    "weekend_24h": ("2014-11-22 23:00:00", 24),   # Saturdaya
    "mon_wed_72h": ("2014-11-19 23:00:00", 72),   # Mon-Wed
}


def _load_nyc_taxi() -> pd.Series:
    """Download and cache the NYC Taxi hourly demand series."""
    if not hasattr(_load_nyc_taxi, "_cache"):
        df = pd.read_csv(_NYC_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        _load_nyc_taxi._cache = df["value"].resample("h").sum()
    return _load_nyc_taxi._cache


def get_nyc_slice(end_date: str, n_points: int = 3_000) -> pd.Series:
    return _load_nyc_taxi().loc[:end_date].iloc[-n_points:]


# =============================================================
# MODULE 0 — PREPROCESSING
# =============================================================

def _detect_frequency(ts: pd.Series) -> int:
    """
    Infer ETS seasonal period.
    Hourly data → 168  (weekly seasonality)
    Daily  data → 7
    Monthly    → 12
    """
    print("Detecting frequency")
    if hasattr(ts.index, "freq") and ts.index.freq is not None:
        freq = str(ts.index.freq)
        if "H" in freq or "h" in freq:
            return 168
        if "D" in freq:
            return 7
        if "M" in freq:
            return 12

    n = len(ts)
    if n > 1_000:
        return 168
    if n > 200:
        return 7
    return 12


def _winsorize(ts: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    return ts.clip(ts.quantile(lower), ts.quantile(upper))


def _time_features(index) -> np.ndarray:
    """Build (N, 10) time-feature matrix from DatetimeIndex or integer index."""
    
    if hasattr(index, "hour"):
        hour = np.array(index.hour, dtype=float)
        dow  = np.array(index.dayofweek, dtype=float)
    else:
        n    = len(index)
        hour = (np.arange(n) % 24).astype(float)
        dow  = ((np.arange(n) // 24) % 7).astype(float)

    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),
        np.cos(2 * np.pi * dow  / 7),
        ((hour >= 7)  & (hour <= 9)).astype(float),   # morning peak
        ((hour >= 17) & (hour <= 19)).astype(float),  # evening peak
        ((hour >= 22) | (hour <= 5)).astype(float),   # overnight
        (dow >= 5).astype(float),                      # weekend
        hour / 23.0,
        dow  / 6.0,
    ])


# =============================================================
# MODULE 1 — ETS BASELINE
# =============================================================

def _safe_forecast(preds) -> np.ndarray:
    arr = np.array(preds, dtype=float)
    if np.any(~np.isfinite(arr)):
        finite = arr[np.isfinite(arr)]
        fill   = float(np.median(finite)) if len(finite) > 0 else 0.0
        arr    = np.where(np.isfinite(arr), arr, fill)
    return arr


def _get_ets_forecast(
    train_log: pd.Series,
    test_log: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    print(f"   [ETS] period={period} | train_n={len(train_log)}")
    model     = ExponentialSmoothing(
        train_log, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=period,
    ).fit(optimized=True)
    stat_preds = pd.Series(_safe_forecast(model.forecast(len(test_log))), index=test_log.index)
    residuals  = pd.Series(model.resid.values, index=train_log.index).fillna(0)
    print(f"   [ETS] Residual std={residuals.std():.6f}")
    return stat_preds, residuals


# =============================================================
# MODULE 2 — DUAL GATE
# =============================================================

def _check_alpha_breakout(residuals: pd.Series, thresh_ratio: float = 0.05) -> bool:
    print("Checking gate 1")
    var    = float(np.var(residuals))
    signal = float(np.var(residuals.cumsum()))
    thresh = thresh_ratio * max(signal, 1e-4)
    print(f"   [GATE 1] Variance={var:.6f} | AdaptiveThresh={thresh:.6f}")
    triggered = var > thresh
    print(f"   [GATE 1] >>> {'TRIGGERED ✓' if triggered else 'NOT triggered — ETS sufficient'}")
    return triggered

def _gated_alpha(
    val_corr: float,
    base_alpha: float = 0.40,
    min_corr: float = 0.05,
) -> tuple[float, float]:
    print("gate 2")
    if np.isnan(val_corr) or val_corr <= min_corr:
        return 0.0, float(val_corr) if not np.isnan(val_corr) else 0.0
    alpha = float(np.clip(base_alpha * val_corr, 0.05, base_alpha))
    return alpha, float(val_corr)


# =============================================================
# MODULE 3A — GBR CORRECTOR
# =============================================================

def _gbr_features(
    res_vals: np.ndarray,
    tf_arr: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(res_vals)):
        lag   = res_vals[i - window:i]
        stats = [np.mean(lag), np.std(lag), np.min(lag),
                 np.max(lag), np.median(lag), lag[-1], lag[-2], lag[-3]]
        X.append(np.concatenate([lag, stats, tf_arr[i]]))
        y.append(res_vals[i])
    return np.array(X), np.array(y)


def _run_gbr(
    residuals: pd.Series,
    horizon: int,
    test_index,
    window: int = 24,
) -> tuple[np.ndarray, float]:
    print("   --- GBR CORRECTOR ---")
    smoothed = residuals.rolling(window=3, min_periods=1).mean()
    tf_arr   = _time_features(smoothed.index.append(test_index))

    scaler = RobustScaler()
    res_sc = scaler.fit_transform(smoothed.values.reshape(-1, 1)).flatten()

    X, y = _gbr_features(res_sc, tf_arr, window)
    gbr  = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gbr.fit(X, y)
    print(f"       GBR trained on {len(X)} samples.")
    
    # Calculate in-sample validation correlation for gating
    train_preds = gbr.predict(X)
    val_corr = np.corrcoef(y, train_preds)[0, 1]

    history = list(res_sc)
    corrs   = []
    for t in range(horizon):
        lag   = np.array(history[-window:])
        stats = [np.mean(lag), np.std(lag), np.min(lag),
                 np.max(lag), np.median(lag), lag[-1], lag[-2], lag[-3]]
        p_sc  = gbr.predict([np.concatenate([lag, stats, tf_arr[len(smoothed) + t]])])[0]
        corrs.append(scaler.inverse_transform([[p_sc]])[0, 0])
        history.append(p_sc)   # no lookahead

    return np.array(corrs), val_corr


# =============================================================
# MODULE 3B — BiLSTM CORRECTOR
# =============================================================

def _build_bilstm(window: int, n_tf: int = 10) -> Model:
    """
    Full architecture from original notebook:
    - L2 regularisation on both LSTM layers
    - Two Dense layers in the time-feature stream
    - Merge hidden Dense + Dropout before output
    """
    res_in = Input(shape=(window, 1), name="residuals")
    x = Bidirectional(LSTM(64, activation="tanh", return_sequences=True,
                           kernel_regularizer=regularizers.l2(0.005)))(res_in)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, activation="tanh",
                           kernel_regularizer=regularizers.l2(0.005)))(x)
    x = Dropout(0.2)(x)

    tf_in = Input(shape=(n_tf,), name="time_features")
    t = Dense(32, activation="relu")(tf_in)
    t = Dense(16, activation="relu")(t)

    m   = Concatenate()([x, t])
    m   = Dense(32, activation="relu")(m)
    m   = Dropout(0.1)(m)
    out = Dense(1)(m)

    model = Model([res_in, tf_in], out)
    model.compile(optimizer=Adam(0.001), loss="huber")
    return model


def _run_lstm(
    residuals: pd.Series,
    horizon: int,
    test_index,
    window: int = 24,
    epochs: int = 120,
) -> tuple[np.ndarray, float]:
    print("   --- BiLSTM CORRECTOR ---")
    smoothed = residuals.rolling(window=3, min_periods=1).mean()
    tf_arr   = _time_features(smoothed.index.append(test_index))

    scaler = RobustScaler()
    res_sc = scaler.fit_transform(smoothed.values.reshape(-1, 1)).flatten()

    X_r, X_t, y = [], [], []
    for i in range(window, len(res_sc)):
        X_r.append(res_sc[i - window:i])
        X_t.append(tf_arr[i])
        y.append(res_sc[i])
    X_r = np.array(X_r).reshape(-1, window, 1)
    X_t = np.array(X_t)
    y   = np.array(y)

    print(f"       BiLSTM training on {len(X_r)} sequences (max {epochs} epochs)...")
    model = _build_bilstm(window)
    val_n = max(1, int(len(X_r) * 0.1))
    model.fit(
        [X_r[:-val_n], X_t[:-val_n]], y[:-val_n],
        validation_data=([X_r[-val_n:], X_t[-val_n:]], y[-val_n:]),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-5, verbose=0),
        ],
    )
    
    # Calculate validation correlation for gating
    val_preds = model.predict([X_r[-val_n:], X_t[-val_n:]], verbose=0).flatten()
    val_corr = np.corrcoef(y[-val_n:], val_preds)[0, 1]

    history = list(res_sc)
    corrs   = []
    for t in range(horizon):
        win  = np.array(history[-window:]).reshape(1, window, 1)
        t_f  = tf_arr[len(smoothed) + t].reshape(1, -1)
        p_sc = model.predict([win, t_f], verbose=0)[0, 0]
        corrs.append(scaler.inverse_transform([[p_sc]])[0, 0])
        history.append(p_sc)

    return np.array(corrs), val_corr


# =============================================================
# MASTER PIPELINE
# =============================================================

def chronoslab_urban_forecast(
    ts: pd.Series,
    test_hours: int = 72,
    period: int | None = None,
) -> dict:
    """
    Full pipeline: ETS baseline → dual-gated DL correction.

    Returns a dict with Series values under keys:
        actual, stat, final, stat_mape, final_mape,
        stat_mae, final_mae, gbr_alpha, lstm_alpha,
        gbr_corr, lstm_corr, is_hybrid
    """
    if period is None:
        period = _detect_frequency(ts)

    ts_log = np.log1p(_winsorize(ts))
    train_log, test_log = ts_log.iloc[:-test_hours], ts_log.iloc[-test_hours:]
    horizon = len(test_log)

    print(f"\n{'='*58}")
    print(f"  ChronosLab Urban | horizon={horizon}H | period={period}")
    print(f"{'='*58}")

    stat_preds_log, residuals_log = _get_ets_forecast(train_log, test_log, period)

    final_preds_log = stat_preds_log.copy()
    report    = dict(gbr_alpha=0.0, gbr_corr=0.0, lstm_alpha=0.0, lstm_corr=0.0)
    is_hybrid = False

    if _check_alpha_breakout(residuals_log, thresh_ratio=0.01): # <--- FIXED PARAMETER NAME
        gbr_corr_future, gbr_val_corr = _run_gbr(residuals_log,  horizon, test_log.index)
        lstm_corr_future, lstm_val_corr = _run_lstm(residuals_log, horizon, test_log.index)

        gbr_alpha,  gbr_c  = _gated_alpha(gbr_val_corr, base_alpha=0.60)
        lstm_alpha, lstm_c = _gated_alpha(lstm_val_corr, base_alpha=0.80)

        print(f"\n   [GATE 2 BLEND]")
        print(f"   GBR  corr={gbr_c:.3f} → α={gbr_alpha:.3f} ({'BLOCKED' if gbr_alpha == 0 else 'APPLIED ✓'})")
        print(f"   LSTM corr={lstm_c:.3f} → α={lstm_alpha:.3f} ({'BLOCKED' if lstm_alpha == 0 else 'APPLIED ✓'})")

        combined        = _safe_forecast(stat_preds_log.values + gbr_alpha * gbr_corr_future + lstm_alpha * lstm_corr_future)
        final_preds_log = pd.Series(combined, index=test_log.index)
        report          = dict(gbr_alpha=gbr_alpha, gbr_corr=gbr_c,
                               lstm_alpha=lstm_alpha, lstm_corr=lstm_c)
        is_hybrid = True

    actual = np.expm1(test_log)
    stat   = pd.Series(np.expm1(_safe_forecast(stat_preds_log.values)),  index=test_log.index)
    final  = pd.Series(np.expm1(_safe_forecast(final_preds_log.values)), index=test_log.index)

    mask        = actual > 10
    stat_mae    = mean_absolute_error(actual, stat)
    final_mae   = mean_absolute_error(actual, final)
    stat_mape   = np.mean(np.abs((actual[mask] - stat[mask])  / actual[mask])) * 100
    final_mape  = np.mean(np.abs((actual[mask] - final[mask]) / actual[mask])) * 100
    improvement = stat_mape - final_mape

    print(f"\n{'='*58}")
    print(f"  Status      : {'SUCCESS ✓' if improvement >= 0 else 'CAUTION ⚠'}")
    print(f"  Hybrid      : {is_hybrid}")
    print(f"  GBR α={report['gbr_alpha']:.3f} | LSTM α={report['lstm_alpha']:.3f}")
    print(f"  Improvement : {improvement:.2f}pp")
    print(f"  Stat  MAPE  : {stat_mape:.2f}%  | Final MAPE : {final_mape:.2f}%")
    print(f"  Stat  MAE   : {stat_mae:.2f}    | Final MAE  : {final_mae:.2f}")
    print(f"{'='*58}\n")

    return dict(
        actual=actual, stat=stat, final=final,
        stat_mape=float(stat_mape),   final_mape=float(final_mape),
        stat_mae=float(stat_mae),     final_mae=float(final_mae),
        is_hybrid=is_hybrid,
        **{k: float(v) for k, v in report.items()},
    )


# =============================================================
# PUBLIC WRAPPER  (called by FastAPI routes)
# =============================================================

def run_supply_demand(ts: pd.Series, test_hours: int = 72, period: int | None = None) -> dict:
    """
    JSON-serialisable wrapper over chronoslab_urban_forecast.
    pd.Series values are converted to plain lists.
    """
    r = chronoslab_urban_forecast(ts, test_hours=test_hours, period=period)
    return {
        "timestamps":     [str(t) for t in r["actual"].index],
        "actual":         r["actual"].tolist(),
        "stat":           r["stat"].tolist(),
        "final":          r["final"].tolist(),
        "stat_mape":      r["stat_mape"],
        "final_mape":     r["final_mape"],
        "stat_mae":       r["stat_mae"],
        "final_mae":      r["final_mae"],
        "improvement_pp": round(r["stat_mape"] - r["final_mape"], 4),
        "is_hybrid":      r["is_hybrid"],
        "gbr_alpha":      r["gbr_alpha"],
        "lstm_alpha":     r["lstm_alpha"],
        "gbr_corr":       r["gbr_corr"],
        "lstm_corr":      r["lstm_corr"],
    }