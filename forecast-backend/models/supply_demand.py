import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

def detect_frequency(ts):
    n = len(ts)

    if n > 1000:
        return 24   # hourly pattern
    elif n > 200:
        return 7    # daily pattern
    else:
        return 12   # fallback

# =========================
# PREPROCESSING
# =========================
def winsorize_series(ts, lower=0.01, upper=0.99):
    return ts.clip(ts.quantile(lower), ts.quantile(upper))

def get_time_features(index):
    if not hasattr(index, "hour"):
        n = len(index)
        hour = np.arange(n) % 24
        dow = (np.arange(n) // 24) % 7
    else:
        hour = index.hour
        dow = index.dayofweek

    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        ((hour >= 7) & (hour <= 9)).astype(float),
        ((hour >= 17) & (hour <= 19)).astype(float),
        ((hour >= 22) | (hour <= 5)).astype(float),
        (dow >= 5).astype(float),
        hour.astype(float) / 23.0,
        dow.astype(float) / 6.0,
    ])


# =========================
# ETS BASELINE
# =========================
def fit_ets_urban(train, period=24):
    return ExponentialSmoothing(
        train,
        trend='add',
        damped_trend=True,
        seasonal='add',
        seasonal_periods=period,
    ).fit(optimized=True)

def safe_forecast(preds):
    arr = np.array(preds, dtype=float)
    if np.any(~np.isfinite(arr)):
        finite_vals = arr[np.isfinite(arr)]
        fill = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr

def get_statistical_forecast(train_log, test_log, period):
    horizon = len(test_log)
    ets = fit_ets_urban(train_log, period)
    raw = safe_forecast(ets.forecast(horizon))

    stat_preds = pd.Series(raw, index=test_log.index)
    residuals = pd.Series(ets.resid.values, index=train_log.index).fillna(0)

    return stat_preds, residuals


# =========================
# GATING
# =========================
def check_alpha_breakout(residuals, alpha=0.0001):
    return float(np.var(residuals)) > alpha

def gated_alpha(corrections, residuals_vals, base_alpha=0.40, min_corr=0.05):
    n = min(len(residuals_vals), len(corrections))
    corr = np.corrcoef(np.array(residuals_vals)[-n:], corrections[-n:])[0, 1]

    if np.isnan(corr) or corr <= min_corr:
        return 0.0, float(corr) if not np.isnan(corr) else 0.0

    alpha = float(np.clip(base_alpha * corr, 0.05, base_alpha))
    return alpha, float(corr)


# =========================
# GBR
# =========================
def build_gbr_features(res_vals, time_feats, window_size):
    X, y = [], []
    for i in range(window_size, len(res_vals)):
        lag = res_vals[i - window_size:i]
        stats = [
            np.mean(lag), np.std(lag), np.min(lag),
            np.max(lag), np.median(lag),
            lag[-1], lag[-2], lag[-3]
        ]
        X.append(np.concatenate([lag, stats, time_feats[i]]))
        y.append(res_vals[i])
    return np.array(X), np.array(y)


def execute_gbr_corrector(residuals, test_actual, stat_preds, horizon, test_index, window_size=24):
    smoothed = residuals.rolling(window=3, min_periods=1).mean()
    full_idx = smoothed.index.append(test_index)
    tf_arr = get_time_features(full_idx)

    scaler = RobustScaler()
    res_sc = scaler.fit_transform(smoothed.values.reshape(-1, 1)).flatten()

    X, y = build_gbr_features(res_sc, tf_arr, window_size)

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    gbr.fit(X, y)

    history = list(res_sc)
    corrs = []

    for t in range(horizon):
        lag = np.array(history[-window_size:])
        stats = [
            np.mean(lag), np.std(lag), np.min(lag),
            np.max(lag), np.median(lag),
            lag[-1], lag[-2], lag[-3]
        ]
        row = np.concatenate([lag, stats, tf_arr[len(smoothed) + t]])

        p_sc = gbr.predict([row])[0]
        corrs.append(scaler.inverse_transform([[p_sc]])[0, 0])

        actual_val = test_actual.iloc[t]
        true_r = actual_val - stat_preds.iloc[t]
        history.append(scaler.transform([[true_r]])[0, 0])

    return np.array(corrs)


# =========================
# LSTM
# =========================
def build_bilstm(window_size, n_tf=10):
    res_in = Input(shape=(window_size, 1))
    x = Bidirectional(LSTM(64, return_sequences=True))(res_in)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)

    tf_in = Input(shape=(n_tf,))
    t = Dense(32, activation='relu')(tf_in)

    m = Concatenate()([x, t])
    out = Dense(1)(m)

    model = Model([res_in, tf_in], out)
    model.compile(optimizer=Adam(0.001), loss='huber')
    return model


def execute_lstm_corrector(residuals, test_actual, stat_preds, horizon, test_index, window_size=24):
    smoothed = residuals.rolling(window=3, min_periods=1).mean()
    full_idx = smoothed.index.append(test_index)
    tf_arr = get_time_features(full_idx)

    scaler = RobustScaler()
    res_sc = scaler.fit_transform(smoothed.values.reshape(-1, 1)).flatten()

    X_r, X_t, y = [], [], []
    for i in range(window_size, len(res_sc)):
        X_r.append(res_sc[i - window_size:i])
        X_t.append(tf_arr[i])
        y.append(res_sc[i])

    X_r = np.array(X_r).reshape(-1, window_size, 1)
    X_t = np.array(X_t)
    y = np.array(y)

    model = build_bilstm(window_size)
    model.fit([X_r, X_t], y, epochs=10, batch_size=32, verbose=0)

    history = list(res_sc)
    corrs = []

    for t in range(horizon):
        win = np.array(history[-window_size:]).reshape(1, window_size, 1)
        t_f = tf_arr[len(smoothed) + t].reshape(1, -1)

        p_sc = model.predict([win, t_f], verbose=0)[0, 0]
        corrs.append(scaler.inverse_transform([[p_sc]])[0, 0])

        actual_val = test_actual.iloc[t]
        true_r = actual_val - stat_preds.iloc[t]
        history.append(scaler.transform([[true_r]])[0, 0])

    return np.array(corrs)


# =========================
# MAIN PIPELINE
# =========================
def chronoslab_urban_forecast(ts,test_hours=72):
    period = detect_frequency(ts)
    ts_clean = winsorize_series(ts)
    ts_log = np.log1p(ts_clean)

    split = len(ts_log) - test_hours
    train_log, test_log = ts_log[:split], ts_log[split:]
    horizon = len(test_log)

    stat_preds_log, residuals_log = get_statistical_forecast(train_log, test_log, period)

    final_preds_log = stat_preds_log.copy()

    if check_alpha_breakout(residuals_log):
        gbr_corr = execute_gbr_corrector(residuals_log, test_log, stat_preds_log, horizon, test_log.index)
        lstm_corr = execute_lstm_corrector(residuals_log, test_log, stat_preds_log, horizon, test_log.index)

        gbr_alpha, _ = gated_alpha(gbr_corr, residuals_log.values)
        lstm_alpha, _ = gated_alpha(lstm_corr, residuals_log.values)

        combined = stat_preds_log.values + gbr_alpha * gbr_corr + lstm_alpha * lstm_corr
        combined = safe_forecast(combined)

        final_preds_log = pd.Series(combined, index=test_log.index)

    actual = np.expm1(test_log)
    stat = np.expm1(stat_preds_log)
    final = np.expm1(final_preds_log)

    mask = actual > 1e-6

    stat_mape = np.mean(np.abs((actual[mask] - stat[mask]) / actual[mask])) * 100
    final_mape = np.mean(np.abs((actual[mask] - final[mask]) / actual[mask])) * 100

    return {
        "actual": actual,
        "stat": stat,
        "final": final,
        "stat_mape": stat_mape,
        "final_mape": final_mape
    }


# =========================
# WRAPPER
# =========================
def run_supply_demand(ts):
    result = chronoslab_urban_forecast(ts)

    return {
        "actual": result["actual"].tolist(),
        "stat": result["stat"].tolist(),
        "final": result["final"].tolist(),
        "stat_mape": float(result["stat_mape"]),
        "final_mape": float(result["final_mape"])
    }