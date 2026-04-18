# =============================================================
# models/finance.py
# ChronosLab Finance — ETS → GARCH → Dual-Gated DL Vol Correction
# Target: Vol-correction ratio  |  Option B (volatility intervals)
#
# FIXES vs original:
#   1. test_hours default raised 5 → 24 (coverage metrics need n≥20)
#   2. chronoslab_finance / run_finance default test_hours=24
#   3. Pipeline prints a WARNING when test_hours < 20
#   4. compute_coverage_metrics adds n_samples + reliability flags
#   5. coverage_reliable flag gated on n≥20 in metrics dict
#   6. StrictMode double-call guard via _running sentinel (backend)
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from arch import arch_model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, Bidirectional,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

import ccxt
import time
import threading

tf.get_logger().setLevel("ERROR")

# ── Constants ────────────────────────────────────────────────
N_REGIME    = 7    # regime context features
N_TIME      = 10   # time-of-day features
EVAL_WINDOW = 400  # rolling evaluation window (origins)

# FIX 6: guard against React StrictMode double-invoke in dev
# If a request arrives while one is already running, the second is
# dropped rather than running the full pipeline twice.
_pipeline_lock = threading.Lock()
_pipeline_running = False

# Minimum test samples for coverage metrics to be statistically meaningful.
# Below this threshold metrics are computed but flagged as unreliable.
MIN_COVERAGE_SAMPLES = 20


# =============================================================
# MODULE 0 — DATA
# =============================================================

def fetch_btc_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h",
                    n_candles: int = 6500) -> pd.DataFrame:
    print(f"[DATA] Fetching {n_candles} × {timeframe} bars for {symbol} ...")
    exchange  = ccxt.binance({"enableRateLimit": True})
    all_ohlcv = []
    since     = exchange.milliseconds() - n_candles * 3600 * 1000
    limit     = 1000

    while len(all_ohlcv) < n_candles:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.tail(n_candles)
    print(f"[DATA] Loaded {len(df)} rows | {df.index[0]} → {df.index[-1]}")
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()


# =============================================================
# MODULE 1 — ETS MEAN MODEL
# =============================================================

def _fit_ets(train: pd.Series, period: int = 24) -> ExponentialSmoothing:
    return ExponentialSmoothing(
        train.values, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=period,
    ).fit(optimized=True)


def get_ets_forecast(train: pd.Series, test: pd.Series,
                     period: int = 24) -> tuple[pd.Series, pd.Series]:
    print(f"\n[ETS] Fitting on {len(train)} points | period={period}")
    model     = _fit_ets(train, period)
    forecast  = pd.Series(model.forecast(len(test)), index=test.index)
    residuals = pd.Series(model.resid, index=train.index).fillna(0)
    print(f"[ETS] Residual std={residuals.std():.6f}")
    return forecast, residuals


# =============================================================
# MODULE 2 — GARCH VOLATILITY
# =============================================================

def fit_garch(residuals: pd.Series, p: int = 1, q: int = 1,
              dist: str = "studentst"):
    print(f"\n[GARCH] Fitting GARCH({p},{q}) | dist={dist} | n={len(residuals)}")
    scaled = residuals * 100
    am     = arch_model(scaled, vol="Garch", p=p, q=q, dist=dist, rescale=False)
    res    = am.fit(disp="off", show_warning=False)
    print(f"[GARCH] AIC={res.aic:.2f} | BIC={res.bic:.2f}")
    print(f"[GARCH] omega={res.params['omega']:.5f} | "
          f"alpha={res.params['alpha[1]']:.4f} | beta={res.params['beta[1]']:.4f}")
    return res, scaled


def get_garch_volatility(garch_result, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    cond_vol_train = garch_result.conditional_volatility.values / 100

    last_var = float(garch_result.conditional_volatility.iloc[-1] ** 2)
    omega    = float(garch_result.params["omega"])
    alpha    = float(garch_result.params["alpha[1]"])
    beta     = float(garch_result.params["beta[1]"])

    vols, var_t = [], last_var
    for _ in range(horizon):
        var_t = omega + (alpha + beta) * var_t
        vols.append(np.sqrt(var_t) / 100)

    print(f"[GARCH] Forecast vol (next {horizon}h): "
          f"mean={np.mean(vols)*100:.4f}% | max={np.max(vols)*100:.4f}%")
    return cond_vol_train, np.array(vols)


def compute_standardized_residuals(residuals: pd.Series,
                                   cond_vol_train: np.ndarray) -> pd.Series:
    sigma = pd.Series(cond_vol_train, index=residuals.index).clip(lower=1e-8)
    z     = residuals / sigma
    print(f"\n[z_t] mean={z.mean():.4f} | std={z.std():.4f} | kurtosis={z.kurtosis():.4f}")
    return z


# =============================================================
# MODULE 3 — FEATURE ENGINEERING
# =============================================================

def get_time_features(index) -> np.ndarray:
    hour = np.array(index.hour, dtype=float)
    dow  = np.array(index.dayofweek, dtype=float)
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),
        np.cos(2 * np.pi * dow  / 7),
        ((hour >= 8)  & (hour <= 10)).astype(float),
        ((hour >= 14) & (hour <= 16)).astype(float),
        ((hour >= 20) & (hour <= 23)).astype(float),
        (dow >= 5).astype(float),
        hour / 23.0,
        dow  / 6.0,
    ])


def build_regime_features(returns_arr: np.ndarray,
                           garch_vol_arr: np.ndarray) -> np.ndarray:
    ret = pd.Series(returns_arr)
    vol = pd.Series(garch_vol_arr)

    rvol_24     = ret.rolling(24, min_periods=1).std()
    rvol_48     = ret.rolling(48, min_periods=1).std()
    mom_6       = ret.rolling(6,  min_periods=1).sum()
    mom_12      = ret.rolling(12, min_periods=1).sum()
    mom_24      = ret.rolling(24, min_periods=1).sum()
    vol_mean_48 = vol.rolling(48, min_periods=1).mean().clip(lower=1e-8)
    vol_ratio   = (vol / vol_mean_48).fillna(1.0)
    vol_of_vol  = vol.rolling(24, min_periods=1).std().fillna(0)

    return np.column_stack([
        rvol_24.values, rvol_48.values,
        mom_6.values, mom_12.values, mom_24.values,
        vol_ratio.values, vol_of_vol.values,
    ])


def compute_vol_ratios(res_raw: np.ndarray, cond_vol: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(res_raw) / np.clip(cond_vol, 1e-8, None), 0.01, 20.0)


# =============================================================
# MODULE 4 — DUAL GATE
# =============================================================

def check_alpha_breakout(vol_ratios: np.ndarray, alpha_thresh: float = 0.10) -> bool:
    ratio_std  = float(np.std(vol_ratios))
    ratio_mean = float(np.mean(vol_ratios))
    print(f"\n[ALPHA GATE] Vol-ratio mean={ratio_mean:.4f}  std={ratio_std:.4f}"
          f"  (threshold={alpha_thresh})")
    triggered = ratio_std > alpha_thresh
    print(f"[ALPHA GATE] >>> {'TRIGGERED: DL Phase active ✓' if triggered else 'NOT triggered. GARCH sufficient.'}")
    return triggered


def gated_alpha(val_preds: np.ndarray, val_actual: np.ndarray,
                base_alpha: float = 0.40, min_corr: float = 0.03) -> tuple[float, float]:
    n = min(len(val_preds), len(val_actual))
    if n < 3:
        return 0.0, 0.0
    corr = np.corrcoef(val_actual[:n], val_preds[:n])[0, 1]
    if np.isnan(corr) or corr <= min_corr:
        return 0.0, float(corr) if not np.isnan(corr) else 0.0
    alpha = float(np.clip(base_alpha * corr, 0.03, base_alpha))
    return alpha, float(corr)


# =============================================================
# MODULE 5A — GBR VOL CORRECTOR
# =============================================================

def execute_gbr_corrector(ets_residuals: pd.Series, returns_train: np.ndarray,
                           cond_vol_train: np.ndarray, garch_vol_test: np.ndarray,
                           horizon: int, test_index,
                           window_size: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"\n   --- GBR VOL CORRECTOR (target=vol_ratio, H={horizon}) ---")
    full_idx    = ets_residuals.index.append(test_index)
    tf_arr      = get_time_features(full_idx)
    ratio_train = compute_vol_ratios(ets_residuals.values, cond_vol_train)

    ratio_sc = RobustScaler()
    vol_sc   = RobustScaler()
    reg_sc   = RobustScaler()

    ratio_arr  = ratio_sc.fit_transform(ratio_train.reshape(-1, 1)).flatten()
    vol_arr    = vol_sc.fit_transform(cond_vol_train.reshape(-1, 1)).flatten()
    regime_raw = build_regime_features(returns_train, cond_vol_train)
    regime_arr = reg_sc.fit_transform(regime_raw)

    train_end = len(ratio_arr) - EVAL_WINDOW - horizon
    if train_end < window_size + horizon:
        train_end = len(ratio_arr) - horizon - 10

    X_list, ys, valid_idxs = [], [[] for _ in range(horizon)], []
    for i in range(window_size, len(ratio_arr)):
        if i + horizon >= len(ratio_train):
            continue
        lag   = ratio_arr[i - window_size:i]
        stats = [np.mean(lag), np.std(lag), np.min(lag),
                 np.max(lag), np.median(lag), lag[-1], lag[-2], lag[-3]]
        row   = np.concatenate([lag, stats, [vol_arr[i]], regime_arr[i], tf_arr[i]])
        X_list.append(row)
        valid_idxs.append(i)
        for k in range(horizon):
            ys[k].append(ratio_train[i + k])

    X_all      = np.array(X_list)
    valid_idxs = np.array(valid_idxs)
    train_mask = valid_idxs < train_end
    X_tr       = X_all[train_mask]

    gbr_models = []
    for k in range(horizon):
        y_k = np.array(ys[k])
        gbr = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=42 + k,
        )
        gbr.fit(X_tr, y_k[train_mask])
        gbr_models.append(gbr)
    print(f"       GBR trained {horizon} models on {X_tr.shape[0]} sequences.")

    eval_end  = min(train_end + EVAL_WINDOW, len(ratio_arr) - horizon)
    eval_mask = (valid_idxs >= train_end) & (valid_idxs < eval_end)
    X_eval    = X_all[eval_mask]
    idxs_eval = valid_idxs[eval_mask]

    all_val_preds, all_val_actual = [], []
    for ri, i in enumerate(idxs_eval):
        row = X_eval[ri]
        for k, gbr in enumerate(gbr_models):
            all_val_preds.append(gbr.predict([row])[0])
            all_val_actual.append(ratio_train[i + k])

    val_preds  = np.array(all_val_preds)
    val_actual = np.array(all_val_actual)
    corr_val   = float(np.corrcoef(val_actual, val_preds)[0, 1]) if len(val_preds) > 2 else 0.0
    print(f"       GBR rolling vol-ratio corr: {corr_val:.4f}  (n={len(val_preds)})")

    test_row    = X_all[-1]
    test_ratios = np.array([gbr_models[k].predict([test_row])[0] for k in range(horizon)])
    return test_ratios, val_preds, val_actual


# =============================================================
# MODULE 5B — BiLSTM VOL CORRECTOR
# =============================================================

def _build_bilstm_vol(window_size: int, n_static: int, horizon: int) -> Model:
    seq_in = Input(shape=(window_size, 2), name="seq")
    x = Bidirectional(LSTM(64, activation="tanh", return_sequences=True,
                           kernel_regularizer=regularizers.l2(0.005)))(seq_in)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, activation="tanh",
                           kernel_regularizer=regularizers.l2(0.005)))(x)
    x = Dropout(0.2)(x)

    static_in = Input(shape=(n_static,), name="static")
    s = Dense(32, activation="relu")(static_in)
    s = Dense(16, activation="relu")(s)

    m   = Concatenate()([x, s])
    m   = Dense(64, activation="relu")(m)
    m   = Dropout(0.1)(m)
    out = Dense(horizon, activation="softplus")(m)

    model = Model([seq_in, static_in], out)
    model.compile(optimizer=Adam(0.001), loss="huber")
    return model


def execute_lstm_corrector(ets_residuals: pd.Series, returns_train: np.ndarray,
                            cond_vol_train: np.ndarray, garch_vol_test: np.ndarray,
                            horizon: int, test_index,
                            window_size: int = 24,
                            epochs: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"\n   --- BiLSTM VOL CORRECTOR (target=vol_ratio, H={horizon}) ---")
    full_idx    = ets_residuals.index.append(test_index)
    tf_arr      = get_time_features(full_idx)
    ratio_train = compute_vol_ratios(ets_residuals.values, cond_vol_train)

    ratio_sc = RobustScaler()
    vol_sc   = RobustScaler()
    reg_sc   = RobustScaler()

    ratio_arr  = ratio_sc.fit_transform(ratio_train.reshape(-1, 1)).flatten()
    vol_arr    = vol_sc.fit_transform(cond_vol_train.reshape(-1, 1)).flatten()
    regime_raw = build_regime_features(returns_train, cond_vol_train)
    regime_arr = reg_sc.fit_transform(regime_raw)

    n_static  = N_REGIME + N_TIME
    train_end = len(ratio_arr) - EVAL_WINDOW - horizon
    if train_end < window_size + horizon:
        train_end = len(ratio_arr) - horizon - 10

    X_seq, X_static, y_ms = [], [], []
    for i in range(window_size, train_end):
        if i + horizon >= len(ratio_train):
            continue
        seq    = np.column_stack([ratio_arr[i-window_size:i],
                                   vol_arr[i-window_size:i]])
        static = np.concatenate([regime_arr[i], tf_arr[i]])
        X_seq.append(seq)
        X_static.append(static)
        y_ms.append(ratio_train[i:i+horizon])

    X_seq    = np.array(X_seq)
    X_static = np.array(X_static)
    y_ms     = np.array(y_ms)

    print(f"       BiLSTM training on {len(X_seq)} sequences (max {epochs} epochs)...")
    lstm = _build_bilstm_vol(window_size, n_static, horizon)
    vn   = max(1, int(len(X_seq) * 0.1))
    lstm.fit(
        [X_seq[:-vn], X_static[:-vn]], y_ms[:-vn],
        validation_data=([X_seq[-vn:], X_static[-vn:]], y_ms[-vn:]),
        epochs=epochs, batch_size=32, verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=12,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=6, min_lr=1e-5, verbose=0),
        ],
    )

    all_val_preds, all_val_actual = [], []
    eval_end = min(train_end + EVAL_WINDOW, len(ratio_arr) - horizon)
    for i in range(train_end, eval_end):
        if i < window_size or i + horizon >= len(ratio_train):
            continue
        seq    = np.column_stack([ratio_arr[i-window_size:i],
                                   vol_arr[i-window_size:i]]).reshape(1, window_size, 2)
        static = np.concatenate([regime_arr[i], tf_arr[i]]).reshape(1, -1)
        preds  = lstm.predict([seq, static], verbose=0)[0]
        for k in range(horizon):
            all_val_preds.append(float(preds[k]))
            all_val_actual.append(float(ratio_train[i + k]))

    val_preds  = np.array(all_val_preds)
    val_actual = np.array(all_val_actual)
    corr_val   = float(np.corrcoef(val_actual, val_preds)[0, 1]) if len(val_preds) > 2 else 0.0
    print(f"       BiLSTM rolling vol-ratio corr: {corr_val:.4f}  (n={len(val_preds)})")

    seq    = np.column_stack([ratio_arr[-window_size:],
                               vol_arr[-window_size:]]).reshape(1, window_size, 2)
    static = np.concatenate([regime_arr[-1], tf_arr[len(ets_residuals)]]).reshape(1, -1)
    return lstm.predict([seq, static], verbose=0)[0], val_preds, val_actual


# =============================================================
# MODULE 6 — RECONSTRUCTION
# =============================================================

def reconstruct_vol(garch_vol_test: np.ndarray, test_index,
                    alpha_gbr: float, alpha_lstm: float,
                    gbr_ratios: np.ndarray,
                    lstm_ratios: np.ndarray) -> tuple[pd.Series, pd.Series]:
    alpha_total = alpha_gbr + alpha_lstm
    if alpha_total > 0:
        blend_ratio = ((1 - alpha_total) * np.ones(len(garch_vol_test))
                       + alpha_gbr  * gbr_ratios
                       + alpha_lstm * lstm_ratios)
    else:
        blend_ratio = np.ones(len(garch_vol_test))

    blend_ratio     = np.clip(blend_ratio, 0.1, 5.0)
    corrected_sigma = garch_vol_test * blend_ratio
    return (pd.Series(corrected_sigma, index=test_index),
            pd.Series(blend_ratio,     index=test_index))


# =============================================================
# MODULE 7 — COVERAGE METRICS
# =============================================================

def compute_coverage_metrics(actual_returns, ets_forecast,
                              sigma_series, label: str = "") -> dict:
    """
    FIX: added n_samples and coverage_reliable flag.

    coverage_reliable=False when n < MIN_COVERAGE_SAMPLES (20).
    With n=5, P(100% cov_2sig | true=95%) = 79% — the metric is
    essentially random noise. Flag it so the frontend can warn the user
    rather than display a misleading "100% coverage" badge.
    """
    actual  = np.array(actual_returns)
    center  = np.array(ets_forecast)
    sigma   = np.array(sigma_series)
    abs_err = np.abs(actual - center)
    n       = len(actual)

    cov_1   = np.mean(abs_err <= sigma)     * 100
    cov_2   = np.mean(abs_err <= 2 * sigma) * 100
    vol_mae = np.mean(np.abs(sigma - abs_err)) * 1e4

    reliable = n >= MIN_COVERAGE_SAMPLES
    if not reliable:
        print(f"  [METRICS/{label}] ⚠ n={n} < {MIN_COVERAGE_SAMPLES} — "
              f"coverage stats unreliable (need test_hours≥{MIN_COVERAGE_SAMPLES})")

    return {
        "label":            label,
        "n_samples":        n,
        "coverage_reliable": reliable,
        "cov_1sig":         cov_1,
        "cov_2sig":         cov_2,
        "width_1sig":       np.mean(2 * sigma) * 1e4,
        "width_2sig":       np.mean(4 * sigma) * 1e4,
        "vol_mae_bps":      vol_mae,
        "calib_err_1":      abs(cov_1 - 68.27),
        "calib_err_2":      abs(cov_2 - 95.45),
    }


# =============================================================
# MASTER PIPELINE
# =============================================================

def chronoslab_finance(ts: pd.Series | None = None,
                       test_hours: int = 24,
                       n_candles: int = 6500) -> dict:
    """
    Full Option B pipeline: ETS → GARCH → Dual-Gated DL vol correction.

    FIX: test_hours default raised from 5 → 24.
    With test_hours=5 coverage metrics had P(100% | true=95%) = 79%,
    making them statistically meaningless. At test_hours=24 that drops
    to 29%, and at test_hours=48 to 8.6% — genuinely informative.

    Recommended values:
      test_hours=24   minimum for any meaningful coverage stat
      test_hours=48   good calibration check
      test_hours=168  robust 1-week evaluation
    """
    global _pipeline_running

    # FIX: double-call guard — React StrictMode fires callbacks twice in dev,
    # sending two POST requests. The second arrives while the first is still
    # running the BiLSTM. We detect this and return a 503-style error so the
    # frontend gets one clean response instead of two concurrent pipelines.
    with _pipeline_lock:
        if _pipeline_running:
            raise RuntimeError(
                "Pipeline already running — duplicate request detected "
                "(React StrictMode double-invoke). Retry in a moment."
            )
        _pipeline_running = True

    try:
        return _run_pipeline(ts, test_hours, n_candles)
    finally:
        with _pipeline_lock:
            _pipeline_running = False


def _run_pipeline(ts: pd.Series | None,
                  test_hours: int,
                  n_candles: int) -> dict:
    """Inner pipeline — called by chronoslab_finance after lock is acquired."""

    # FIX: warn when test_hours is too small for reliable coverage metrics
    if test_hours < MIN_COVERAGE_SAMPLES:
        print(f"\n  ⚠  WARNING: test_hours={test_hours} < {MIN_COVERAGE_SAMPLES}.")
        print(f"     Coverage metrics (cov_1sig, cov_2sig) will be flagged as")
        print(f"     unreliable. Use test_hours≥{MIN_COVERAGE_SAMPLES} for meaningful evaluation.")

    if ts is None:
        df      = fetch_btc_ohlcv(n_candles=n_candles)
        df      = compute_log_returns(df)
        returns = df["log_return"]
    else:
        ts      = ts.copy()
        returns = np.log(ts / ts.shift(1)).dropna()

    if len(returns) <= test_hours + 10:
        raise ValueError(f"Dataset too small: {len(returns)} rows, need > {test_hours + 10}")

    train_r = returns.iloc[:-test_hours]
    test_r  = returns.iloc[-test_hours:]
    horizon = len(test_r)

    print(f"\n{'='*60}")
    print(f"  ChronosLab Finance | horizon={horizon}H | train={len(train_r)}")
    print(f"  EVAL_WINDOW={EVAL_WINDOW} | coverage_reliable={horizon >= MIN_COVERAGE_SAMPLES}")
    print(f"{'='*60}")

    # Stage 1: ETS
    ets_forecast, ets_residuals = get_ets_forecast(train_r, test_r, period=24)

    # Stage 2: GARCH
    garch_result, _ = fit_garch(ets_residuals)
    cond_vol_train, garch_vol_test = get_garch_volatility(garch_result, horizon)
    z_t = compute_standardized_residuals(ets_residuals, cond_vol_train)

    # Stage 3: Vol ratios + dual gate
    vol_ratios_train = compute_vol_ratios(ets_residuals.values, cond_vol_train)
    garch_sigma_test = pd.Series(garch_vol_test, index=test_r.index)

    gbr_ratios  = np.ones(horizon)
    lstm_ratios = np.ones(horizon)
    gbr_vp = gbr_va = lstm_vp = lstm_va = np.array([1.0, 1.0])
    report  = dict(gbr_alpha=0.0, gbr_corr=0.0, lstm_alpha=0.0,
                   lstm_corr=0.0, is_hybrid=False)

    if check_alpha_breakout(vol_ratios_train, alpha_thresh=0.10):
        gbr_ratios, gbr_vp, gbr_va = execute_gbr_corrector(
            ets_residuals, train_r.values, cond_vol_train,
            garch_vol_test, horizon, test_r.index)

        lstm_ratios, lstm_vp, lstm_va = execute_lstm_corrector(
            ets_residuals, train_r.values, cond_vol_train,
            garch_vol_test, horizon, test_r.index)

        gbr_alpha,  gbr_c  = gated_alpha(gbr_vp,  gbr_va,  base_alpha=0.60)
        lstm_alpha, lstm_c = gated_alpha(lstm_vp, lstm_va, base_alpha=0.80)

        print(f"\n[GATED BLEND]")
        print(f"   GBR  roll_corr={gbr_c:.3f} → α={gbr_alpha:.3f}"
              f"  ({'BLOCKED' if gbr_alpha==0 else 'APPLIED ✓'})")
        print(f"   LSTM roll_corr={lstm_c:.3f} → α={lstm_alpha:.3f}"
              f"  ({'BLOCKED' if lstm_alpha==0 else 'APPLIED ✓'})")

        report = dict(gbr_alpha=gbr_alpha, gbr_corr=gbr_c,
                      lstm_alpha=lstm_alpha, lstm_corr=lstm_c, is_hybrid=True)

    # Stage 4: Reconstruct
    corrected_sigma, blend_ratio = reconstruct_vol(
        garch_vol_test, test_r.index,
        report["gbr_alpha"], report["lstm_alpha"],
        gbr_ratios, lstm_ratios,
    )

    m_pre  = compute_coverage_metrics(test_r, ets_forecast, garch_sigma_test,  "pre_dl")
    m_post = compute_coverage_metrics(test_r, ets_forecast, corrected_sigma,   "post_dl")

    print(f"\n{'='*60}")
    print(f"  VOL MAE  pre={m_pre['vol_mae_bps']:.2f} bps → post={m_post['vol_mae_bps']:.2f} bps")
    print(f"  ±2σ cov  pre={m_pre['cov_2sig']:.1f}%    → post={m_post['cov_2sig']:.1f}%"
          + ("  ⚠ unreliable (n<20)" if not m_post["coverage_reliable"] else ""))
    print(f"{'='*60}\n")

    return dict(
        actual=test_r,
        train_r=train_r,
        ets_forecast=ets_forecast,
        garch_sigma=garch_sigma_test,
        corrected_sigma=corrected_sigma,
        blend_ratio=blend_ratio,
        cond_vol_train=cond_vol_train,   # in-sample GARCH vol — needed by frontend
        metrics_pre=m_pre,
        metrics_post=m_post,
        report=report,
    )


# =============================================================
# PUBLIC WRAPPER  (called by FastAPI routes)
# =============================================================

def run_finance(ts: pd.Series | None = None,
                test_hours: int = 24,
                n_candles: int = 6500) -> dict:
    """
    FIX: test_hours default raised 5 → 24 here and in chronoslab_finance.
    """
    r = chronoslab_finance(ts, test_hours=test_hours, n_candles=n_candles)

    def to_py(x):
        """Convert numpy scalars to native Python types for JSON serialization."""
        if isinstance(x, dict):
            return {k: to_py(v) for k, v in x.items()}
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, bool):
            return bool(x)
        return x

    # In-sample GARCH vol — used by frontend for timeSeries + stats
    cond_vol_bps = r["cond_vol_train"] * 1e4
    train_index  = r["train_r"].index

    return {
        "timestamps":              [str(t) for t in r["actual"].index],
        "train_timestamps":        [str(t) for t in train_index],
        "garch_sigma_bps":         (r["garch_sigma"].values * 1e4).tolist(),
        "corrected_sigma_bps":     (r["corrected_sigma"].values * 1e4).tolist(),
        "cond_vol_train_bps":      cond_vol_bps.tolist(),
        "blend_ratio":             r["blend_ratio"].values.tolist(),
        "series_stats": {
            "count": int(len(cond_vol_bps) + len(r["actual"])),
            "mean":  float(np.mean(cond_vol_bps)),
            "std":   float(np.std(cond_vol_bps)),
            "min":   float(np.min(cond_vol_bps)),
            "max":   float(np.max(cond_vol_bps)),
        },
        "metrics_pre_dl":          to_py(r["metrics_pre"]),
        "metrics_post_dl":         to_py(r["metrics_post"]),
        "vol_mae_improvement_bps": round(
            float(r["metrics_pre"]["vol_mae_bps"]) -
            float(r["metrics_post"]["vol_mae_bps"]), 4),
        "report": to_py(r["report"]),
    }