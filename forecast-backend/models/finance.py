import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

import ccxt
import time

tf.get_logger().setLevel('ERROR')

# =========================
# DATA
# =========================
def detect_frequency(ts):
    n = len(ts)

    if n > 1000:
        return 24   # hourly pattern
    elif n > 200:
        return 7    # daily pattern
    else:
        return 12   # fallback

def fetch_btc_ohlcv(symbol="BTC/USDT", timeframe="1h", n_candles=6500):
    exchange = ccxt.binance({"enableRateLimit": True})
    all_ohlcv = []
    since = exchange.milliseconds() - n_candles * 3600 * 1000
    limit = 1000

    while len(all_ohlcv) < n_candles:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.tail(n_candles)
    return df

def compute_log_returns(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()

# =========================
# ETS
# =========================
def fit_ets_mean(train, period=24):
    return ExponentialSmoothing(
        train, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=period
    ).fit(optimized=True)

def get_ets_forecast(train, test, period=24):
    model = fit_ets_mean(train, period)
    forecast = pd.Series(model.forecast(len(test)), index=test.index)
    residuals = pd.Series(model.resid.values, index=train.index).fillna(0)
    return forecast, residuals

# =========================
# GARCH
# =========================
def fit_garch(residuals):
    scaled = residuals * 100
    am = arch_model(scaled, vol="Garch", p=1, q=1, dist="studentst", rescale=False)
    res = am.fit(disp="off")
    return res

def get_garch_volatility(garch_result, residuals, horizon):
    cond_vol_train = garch_result.conditional_volatility / 100

    last_var = float(garch_result.conditional_volatility.iloc[-1]**2)
    omega = float(garch_result.params["omega"])
    alpha = float(garch_result.params["alpha[1]"])
    beta  = float(garch_result.params["beta[1]"])

    vols = []
    var_t = last_var
    for _ in range(horizon):
        var_t = omega + (alpha + beta) * var_t
        vols.append(np.sqrt(var_t) / 100)

    return np.array(cond_vol_train), np.array(vols)

def compute_standardized_residuals(residuals, cond_vol_train):
    sigma = np.clip(cond_vol_train, 1e-8, None)
    return residuals / sigma

# =========================
# FEATURES
# =========================
def get_time_features(index):
    hour = index.hour
    dow  = index.dayofweek
    return np.column_stack([
        np.sin(2*np.pi*hour/24),
        np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7),
        np.cos(2*np.pi*dow/7),
        hour/23.0,
        dow/6.0
    ])

def compute_vol_ratios(residuals, cond_vol):
    ratios = np.abs(residuals) / np.clip(cond_vol, 1e-8, None)
    return np.clip(ratios, 0.01, 20)

# =========================
# GATING
# =========================
def check_alpha_breakout(vol_ratios, threshold=0.10):
    return np.std(vol_ratios) > threshold

def gated_alpha(preds, actual, base_alpha=0.5):
    if len(preds) < 3:
        return 0, 0
    corr = np.corrcoef(preds, actual)[0,1]
    if np.isnan(corr) or corr < 0.05:
        return 0, corr
    return min(base_alpha*corr, base_alpha), corr

# =========================
# GBR
# =========================
def execute_gbr_corrector(residuals, cond_vol, horizon):
    ratios = compute_vol_ratios(residuals, cond_vol)

    X, y = [], []
    for i in range(24, len(ratios)):
        X.append(ratios[i-24:i])
        y.append(ratios[i])

    X = np.array(X)
    y = np.array(y)

    model = GradientBoostingRegressor(n_estimators=200)
    model.fit(X, y)

    last = ratios[-24:]
    preds = []
    for _ in range(horizon):
        p = model.predict([last])[0]
        preds.append(p)
        last = np.append(last[1:], p)

    return np.array(preds)

# =========================
# LSTM
# =========================
def build_lstm():
    inp = Input(shape=(24,1))
    x = LSTM(32)(inp)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def execute_lstm_corrector(residuals, cond_vol, horizon):
    ratios = compute_vol_ratios(residuals, cond_vol)

    X, y = [], []
    for i in range(24, len(ratios)):
        X.append(ratios[i-24:i])
        y.append(ratios[i])

    X = np.array(X).reshape(-1,24,1)
    y = np.array(y)

    model = build_lstm()
    model.fit(X, y, epochs=5, verbose=0)

    last = ratios[-24:]
    preds = []
    for _ in range(horizon):
        p = model.predict(last.reshape(1,24,1), verbose=0)[0,0]
        preds.append(p)
        last = np.append(last[1:], p)

    return np.array(preds)

# =========================
# FINAL PIPELINE
# =========================
def chronoslab_finance(ts=None, test_hours=5, n_candles=2000):

    if ts is None:
        df = fetch_btc_ohlcv(n_candles=n_candles)
        df = compute_log_returns(df)
        returns = df["log_return"]
    else:
        ts = ts.copy()
        ts = np.log(ts / ts.shift(1)).dropna()
        returns = ts
    
    if len(returns) <= test_hours + 10:
        raise ValueError("Dataset too small")
    split = len(returns) - test_hours
    train, test = returns[:split], returns[split:]
    horizon = len(test)

    # ETS
    ets_forecast, residuals = get_ets_forecast(train, test)

    # GARCH
    garch = fit_garch(residuals)
    cond_vol_train, garch_vol_test = get_garch_volatility(garch, residuals, horizon)

    # Gating
    ratios = compute_vol_ratios(residuals.values, cond_vol_train)

    gbr_alpha = lstm_alpha = 0
    gbr_preds = np.ones(horizon)
    lstm_preds = np.ones(horizon)

    if check_alpha_breakout(ratios):
        gbr_preds = execute_gbr_corrector(residuals.values, cond_vol_train, horizon)
        lstm_preds = execute_lstm_corrector(residuals.values, cond_vol_train, horizon)

        gbr_alpha, _ = gated_alpha(gbr_preds, ratios[-len(gbr_preds):])
        lstm_alpha, _ = gated_alpha(lstm_preds, ratios[-len(lstm_preds):])

    final_ratio = (1 - gbr_alpha - lstm_alpha) + gbr_alpha*gbr_preds + lstm_alpha*lstm_preds
    final_ratio = np.clip(final_ratio, 0.1, 5)
    corrected_vol = garch_vol_test * final_ratio

    return {
        "blend_ratio": final_ratio,
        "report": {
            "gbr_alpha": gbr_alpha,
            "lstm_alpha": lstm_alpha
        }
    }

# =========================
# API WRAPPER
# =========================
def run_finance(ts=None):
    result = chronoslab_finance(ts)

    return {
        "blend_ratio": result["blend_ratio"].tolist(),
        "report": result["report"]
    }