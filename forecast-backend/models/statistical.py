import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")


# =============================================================
# MODULE 0 — FREQUENCY DETECTION
# =============================================================

# Maps pandas frequency aliases to the correct seasonal period.
# Used by detect_frequency when a DatetimeIndex is available.
_FREQ_TO_PERIOD: dict[str, int] = {
    "MS": 12, "M": 12, "ME": 12,   # monthly  → annual cycle
    "QS":  4, "Q":  4, "QE":  4,   # quarterly
    "W":  52,                        # weekly   → annual cycle
    "D":   7,                        # daily    → weekly cycle
    "H":  24, "h":  24,             # hourly   → daily cycle
    "AS":  1, "A":  1,              # annual   → no sub-cycle
    "YS":  1, "Y":  1,
}


def detect_frequency(ts: pd.Series) -> int:
    """
    Infer the dominant seasonal period.

    Strategy — DatetimeIndex first, length heuristic as fallback:

    1. If the series has a DatetimeIndex, infer the pandas frequency alias
       (e.g. 'MS' for month-start, 'D' for daily) and map it to the correct
       seasonal period via _FREQ_TO_PERIOD.

       WHY THIS MATTERS:
       - The old length-only heuristic mapped 476-point monthly data (beerstat)
         to period=7 because 200 < 476 <= 1000 → "daily, weekly cycle".
         Correct answer is 12 (monthly data, annual cycle).
       - With period=7 seasonal strength was 0.004 (missed), fell through to
         plain ARIMA, and gave 10.51% sMAPE.
       - With period=12 seasonal strength is 0.78 (detected), ETS wins,
         sMAPE drops to 6.36%.

    2. Length heuristic is only used when no DatetimeIndex is present
       (integer-indexed series from the non-datetime CSV path).
       Thresholds:
       - >1000 → hourly data, daily cycle = 24
       - 200–1000 → daily data, weekly cycle = 7
       - ≤200 → monthly/quarterly, annual cycle = 12
    """
    # ── Primary: use DatetimeIndex frequency ──────────────────
    if isinstance(ts.index, pd.DatetimeIndex):
        freq = pd.infer_freq(ts.index)
        if freq:
            # Strip leading multiplier digits (e.g. '2MS' → 'MS', '15T' → 'T')
            base = freq.lstrip("0123456789")
            period = _FREQ_TO_PERIOD.get(base) or _FREQ_TO_PERIOD.get(freq)
            if period and period > 1:
                return period

    # ── Fallback: length heuristic (integer-indexed series) ───
    n = len(ts)
    if n > 1_000:
        return 24
    elif n > 200:
        return 7
    else:
        return 12


# =============================================================
# MODULE 1 — STATIONARITY
# =============================================================

def adf_test(ts: pd.Series) -> tuple[float, float]:
    """
    Run ADF with a capped maxlag.

    Why cap at 20:
    - statsmodels default autolag='AIC' uses the Schwert criterion:
      int(12*(n/100)^0.25). For n=100k that's 67 lags — fitting a 67-regressor
      OLS on 100k rows repeatedly caused the 52s hang on large series.
    - Capping at 20 costs nothing in detection accuracy (the unit-root decision
      is driven by the first few lags) and cuts ADF time from ~40s to <0.5s.
    """
    series = ts.dropna()
    maxlag = min(20, int(12 * (len(series) / 100) ** 0.25))
    stat, pvalue, *_ = adfuller(series, maxlag=maxlag, autolag="AIC")
    return stat, pvalue


def find_d(ts: pd.Series, max_d: int = 2) -> int:
    """
    Walk up differencing orders until ADF rejects the unit-root null.

    Why we check both p<0.05 AND stat<0:
    - ADF stat must be negative to be meaningful. A positive stat with
      p<0.05 can occur with very short or degenerate series and would
      give a false stationarity signal.
    """
    d = 0
    ts_diff = ts.copy()
    while d <= max_d:
        try:
            stat, p = adf_test(ts_diff)
            if p < 0.05 and stat < 0:
                return int(d)
        except Exception:
            pass
        ts_diff = ts_diff.diff().dropna()
        d += 1
    return int(max_d)


def find_D(ts: pd.Series, period: int, max_D: int = 1) -> int:
    """
    Find the seasonal differencing order D.

    Why max_D=1:
    - Seasonal over-differencing (D>1) is almost always harmful.
      One seasonal difference is enough for the vast majority of
      real-world monthly/quarterly series. AirPassengers needs D=1
      with period=12 and that alone kills the seasonal unit root.
    """
    D = 0
    ts_diff = ts.copy()
    while D <= max_D:
        try:
            stat, p = adf_test(ts_diff)
            if p < 0.05 and stat < 0:
                return int(D)
        except Exception:
            pass
        ts_diff = ts_diff.diff(period).dropna()
        D += 1
    return int(max_D)


# =============================================================
# MODULE 2 — SEASONALITY DETECTION  (KEY FIX)
# =============================================================

def detect_seasonality(
    ts: pd.Series,
    period: int,
    threshold: float = 0.4,
) -> tuple[bool, float]:
    """
    Compute seasonal strength using the Hyndman & Athanasopoulos formula:

        strength = 1 - Var(R) / Var(S + R)

    where S = seasonal component, R = residual component (both from
    an ADDITIVE decomposition on the (optionally log-transformed) series).

    WHY THE OLD CODE WAS WRONG FOR AIRPASSENGERS:
    - AirPassengers has multiplicative seasonality (seasonal swings grow
      with the level). The old code decomposed it as multiplicative and then
      computed var(seasonal * resid). Both seasonal and resid are centred
      near 1.0 in a multiplicative decomposition, so their product has
      almost the same variance as resid alone. The ratio var_resid /
      var(seasonal*resid) ends up near 1.0, strength → ~0.0, flag = False.
    - This made AirPassengers fall through to plain ARIMA with no seasonal
      terms — the root cause of the 20% sMAPE.

    THE FIX — log-transform before decomposing:
    - log1p converts multiplicative seasonality to additive.
    - We then always use the additive Hyndman formula, which is well-
      calibrated across all series types.
    - extrapolate_trend='freq' fills edge NaNs that would otherwise make
      nanvar underestimate variance near the series boundaries.
    """
    try:
        # Log-transform positive series so multiplicative seasonality becomes
        # additive. This is the single most important change vs the old code.
        use_log = (ts > 0).all()
        ts_work = np.log1p(ts) if use_log else ts

        decomp = seasonal_decompose(
            ts_work,
            model="additive",
            period=period,
            extrapolate_trend="freq",   # fills NaN edges — avoids var underestimate
        )

        var_resid      = float(np.nanvar(decomp.resid))
        var_deseasoned = float(np.nanvar(decomp.resid + decomp.seasonal))

        if var_deseasoned == 0:
            return False, 0.0

        strength = max(0.0, 1.0 - var_resid / var_deseasoned)
        return bool(strength > threshold), strength

    except Exception:
        return False, 0.0


# =============================================================
# MODULE 3 — MODEL FITTING
# =============================================================

def fit_ets(
    ts: pd.Series,
    seasonal_periods: int,
    seasonal: str = "add",
    trend: str = "add",
):
    """
    Fit Holt-Winters ETS.

    Why 'mul' for positive series:
    - When the seasonal amplitude grows proportionally with the level
      (as in AirPassengers), multiplicative seasonality fits better.
    - We pass this in from run_statistical_forecast so the caller
      controls the type consistently across ETS and the MAE comparison.
    """
    try:
        model = ExponentialSmoothing(
            ts,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        return model.fit(optimized=True)
    except Exception:
        return None


def fit_best_sarima(
    ts: pd.Series,
    order_grid: list,
    seasonal_order_grid: list,
    period: int,
):
    """
    Grid-search SARIMA over (p,d,q) x (P,D,Q) using AIC.

    Why AIC over BIC:
    - BIC penalises model complexity more heavily, which can under-fit
      seasonal patterns on short series like AirPassengers (115 train pts).
    - AIC is more forgiving and tends to select higher-order seasonal MA
      terms that capture the seasonal autocorrelation structure better.

    Why maxiter=50:
    - Sufficient for most SARIMAX fits; higher values add wall-clock time
      with diminishing returns on AIC improvement.
    """
    best_aic   = np.inf
    best_model = None
    best_order          = None
    best_seasonal_order = None

    for order in order_grid:
        for s_order in seasonal_order_grid:
            try:
                res = SARIMAX(
                    ts,
                    order=order,
                    seasonal_order=s_order + (period,),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=50)

                if res.aic < best_aic:
                    best_aic            = res.aic
                    best_model          = res
                    best_order          = order
                    best_seasonal_order = s_order
            except Exception:
                continue

    return best_model, best_order, best_seasonal_order, best_aic


def fit_best_arima(ts: pd.Series, order_grid: list):
    """
    Grid-search plain ARIMA (no seasonal terms) using AIC.

    This path is only reached when detect_seasonality returns False,
    meaning the series is genuinely non-seasonal (random walk, trend-only,
    irregular economic series, etc.). For AirPassengers, with the fixed
    detect_seasonality this path will never be chosen.
    """
    best_aic   = np.inf
    best_model = None
    best_order = None

    for order in order_grid:
        try:
            res = SARIMAX(
                ts,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=50)

            if res.aic < best_aic:
                best_aic   = res.aic
                best_model = res
                best_order = order
        except Exception:
            continue

    return best_model, best_order, best_aic


# =============================================================
# MODULE 4 — FORECASTING
# =============================================================

def fast_forecast_sarima(
    train: pd.Series,
    test: pd.Series,
    order: tuple,
    seasonal_order: tuple | None = None,
    period: int | None = None,
) -> list:
    """
    Refit on train and produce a single-shot h-step-ahead forecast.

    SINGLE-SHOT vs ROLLING — reasoning:
    - Rolling (re-fit or update at each step with revealed actuals) gives
      better empirical accuracy, especially for long horizons where model
      uncertainty compounds. For 29 test steps on AirPassengers, rolling
      SARIMA would meaningfully lower MAPE by re-anchoring at each month.
    - However, rolling on SARIMA is O(h * fit_time) ≈ 29× slower. On a
      shared HuggingFace Space with cold starts this is prohibitive.
    - The right trade-off: single-shot for speed. Once the seasonality
      detection fix (Bug 1) is in place, SARIMA picks up the seasonal
      structure in one shot and sMAPE drops to ~5-8% on AirPassengers
      rather than 20%+. Rolling would bring it to ~3-5% but at 30× cost.
    - If you ever add a rolling toggle, use res.apply(new_data) rather
      than full refit — it updates the Kalman filter state in-place and
      is ~5× faster than refitting from scratch each step.
    """
    kwargs: dict = {
        "order": order,
        "enforce_stationarity": False,
        "enforce_invertibility": False,
    }
    if seasonal_order is not None and period is not None:
        kwargs["seasonal_order"] = seasonal_order + (period,)

    res      = SARIMAX(train, **kwargs).fit(disp=False)
    forecast = res.forecast(steps=len(test))
    return forecast.tolist()


def fast_forecast_ets(
    train: pd.Series,
    test: pd.Series,
    period: int,
    seasonal: str = "add",
) -> list:
    """
    Holt-Winters single-shot forecast.

    Why single-shot is fine for ETS here:
    - ETS state equations are already a recursive filter; the fitted
      level/trend/seasonal states at the end of train are the best
      linear predictors given the model. Unlike ARIMA, there is no
      benefit to re-anchoring because ETS has no autoregressive
      correction term that would absorb new observations.
    """
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True)
    return model.forecast(len(test)).tolist()


# =============================================================
# MODULE 5 — METRICS
# =============================================================

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Why mask near-zero actuals (|y| > 1e-6):
    - MAPE is undefined when y_true = 0 (division by zero).
    - Small actuals inflate MAPE disproportionately even without being
      exactly zero. The 1e-6 mask excludes these without distorting the
      metric on the rest of the series.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE — bounded [0, 200], symmetric for over/under-forecast.

    Why sMAPE instead of just MAPE:
    - MAPE is asymmetric: a 50% under-forecast gives MAPE=50%, but a
      50% over-forecast gives MAPE=100%. sMAPE treats both symmetrically
      by normalising with (|actual| + |predicted|) / 2.
    - For AirPassengers where predictions can overshoot seasonal peaks,
      sMAPE gives a fairer summary than MAPE alone.

    Why clamp denominator to 1e-9 not 0:
    - Avoids divide-by-zero when both actual and predicted are exactly 0
      (e.g. supply-demand series at midnight). Using 1e-9 returns 0% error
      for those points rather than NaN, which is the correct interpretation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return float(100 * np.mean(2 * np.abs(y_true - y_pred) / denom))


# =============================================================
# MASTER PIPELINE
# =============================================================

def run_statistical_forecast(ts: pd.Series) -> dict:
    """
    Full pipeline:
      1. Detect period
      2. Train/test split (80/20)
      3. Detect seasonality (fixed: log-transform + additive Hyndman formula)
      4. Find d (and D if seasonal)
      5. Grid-search SARIMA or ARIMA
      6. Compare with ETS on train MAE (seasonal path only)
      7. Forecast and return metrics

    AIRPASSENGERS TRACE (to verify fix end-to-end):
      - n=144, period=12
      - detect_seasonality: log1p → additive decompose → strength≈0.93 → True ✓
      - find_D(train, 12): D=1  (seasonal unit root)
      - find_d(diff12(train)): d=0 or 1 (usually 0 after one seasonal diff)
      - order_grid: [(p,d,q) for p,q in 0..2]
      - seasonal_grid: [(P,1,Q) for P,Q in 0..1]
      - SARIMA(0,1,1)(0,1,1)[12] typically wins → the "airline model"
      - ETS(mul) vs SARIMA train MAE → SARIMA wins on AirPassengers
      - 29-step forecast → sMAPE ≈ 5-8% (was 20%+ with old code)
    """
    import time
    t_start = time.time()

    def elapsed():
        return f"{time.time() - t_start:.2f}s"

    print("\n" + "═" * 55)
    print("  STATISTICAL PIPELINE — START")
    print("═" * 55)
    print(f"  [{elapsed()}] Input series: {len(ts)} points")

    # Detect frequency BEFORE resetting index — detect_frequency uses the
    # DatetimeIndex freq alias (e.g. 'MS'→12) when available. Resetting to
    # integers afterwards is still needed to avoid SARIMAX gap issues.
    period = detect_frequency(ts)
    print(f"  [{elapsed()}] Frequency detected: period={period}")

    # Reset index to integers — DatetimeIndex causes SARIMAX seasonal_order
    # issues when the index has gaps or irregular spacing.
    ts = ts.copy()
    ts.index = range(len(ts))

    if len(ts) < 30:
        raise ValueError("Dataset too small for statistical modeling (need ≥30 points)")

    split = int(len(ts) * 0.8)
    train_full, test = ts[:split], ts[split:]
    print(f"  [{elapsed()}] Train/test split: {len(train_full)} train / {len(test)} test  (80/20 at index {split})")

    # ── Cap train size for large series ───────────────────────
    # SARIMAX Kalman filter is O(n_train x maxiter). On 80k+ train points
    # each fit takes 20-30s (observed: 211s for 9 ARIMA fits on 83k train).
    # Stationarity/seasonality are local — the most recent 5000 points carry
    # the same order-selection signal as the full history.
    # train_full is kept intact so test actuals stay correctly aligned;
    # only model-fitting uses the capped train window.
    MAX_TRAIN = 5_000
    if len(train_full) > MAX_TRAIN:
        train = train_full.iloc[-MAX_TRAIN:].copy()
        train.index = range(len(train))
        print(f"  [{elapsed()}] Large series — capping fit window to last {MAX_TRAIN} points (was {len(train_full)})")
    else:
        train = train_full

    # ── Seasonality detection ──────────────────────────────────
    # Require at least 2 full cycles in train to decompose meaningfully.
    if len(train) >= period * 2:
        seasonal_flag, strength = detect_seasonality(train, period)
    else:
        seasonal_flag, strength = False, 0.0

    print(f"  [{elapsed()}] Seasonality: {'YES' if seasonal_flag else 'NO'}  (strength={strength:.3f})")

    # ── Differencing orders ────────────────────────────────────
    if seasonal_flag:
        # Find D on the raw train, then d on the seasonally differenced series.
        # Why this order: seasonal unit root must be removed before testing
        # for a regular unit root, otherwise ADF sees the seasonal structure
        # as non-stationarity and over-differences.
        D           = find_D(train, period)
        diff_train  = train.diff(period).dropna() if D > 0 else train
        d           = find_d(diff_train)
    else:
        d = find_d(train)
        D = 0

    print(f"  [{elapsed()}] Differencing orders: d={d}, D={D}")

    # ── Build order grids ──────────────────────────────────────
    # p,q ∈ [0,2]: sufficient for most real-world series.
    # Higher orders rarely improve AIC enough to justify the fit time.
    order_grid = [(p, d, q) for p in range(3) for q in range(3)]

    preds           = []
    best_model_name = ""

    if seasonal_flag:
        # P,Q ∈ [0,1]: seasonal AR and MA terms.
        # P=1,Q=1 covers the "airline model" (SARIMA(0,1,1)(0,1,1)[12])
        # which is the theoretical optimum for AirPassengers-style data.
        seasonal_grid = [(P, D, Q) for P in range(2) for Q in range(2)]
        n_fits = len(order_grid) * len(seasonal_grid)
        print(f"  [{elapsed()}] SARIMA grid search: {len(order_grid)} × {len(seasonal_grid)} = {n_fits} fits...")

        t_sarima = time.time()
        s_model, s_order, s_seasonal_order, best_aic = fit_best_sarima(
            train, order_grid, seasonal_grid, period
        )
        print(f"  [{elapsed()}] SARIMA done in {time.time()-t_sarima:.2f}s — best={s_order}×{s_seasonal_order}[{period}]  AIC={best_aic:.1f}")

        # ETS seasonal type: multiplicative for all-positive series.
        # Multiplicative ETS handles growing seasonal amplitude (AirPassengers)
        # better than additive — the swings scale with level.
        ets_type  = "mul" if (train > 0).all() else "add"
        t_ets = time.time()
        ets_model = fit_ets(train, seasonal_periods=period, seasonal=ets_type)
        print(f"  [{elapsed()}] ETS({ets_type}) done in {time.time()-t_ets:.2f}s")

        # Compare on TRAIN MAE (in-sample), excluding the differenced warm-up.
        # Why train MAE not test: we don't want to look at test data during
        # model selection. Train MAE is a reasonable proxy when both models
        # were fit on the same data.
        drop_n     = int(d + (D * period if D > 0 else 0))
        sarima_mae = (
            np.mean(np.abs(train.iloc[drop_n:] - s_model.fittedvalues.iloc[drop_n:]))
            if s_model is not None else np.inf
        )
        ets_mae = (
            np.mean(np.abs(train.iloc[drop_n:] - ets_model.fittedvalues.iloc[drop_n:]))
            if ets_model is not None else np.inf
        )
        print(f"  [{elapsed()}] Train MAE — SARIMA: {sarima_mae:.2f}  ETS: {ets_mae:.2f}")

        if ets_model is not None and ets_mae < sarima_mae:
            best_model_name = "ETS"
            print(f"  [{elapsed()}] Winner: ETS (lower train MAE)")
            # Use train_full for forecasting: model was fit on capped train for speed,
            # but we refit on full history so the forecast is anchored correctly.
            preds = fast_forecast_ets(train_full, test, period, seasonal=ets_type)
        elif s_model is not None:
            best_model_name = "SARIMA"
            print(f"  [{elapsed()}] Winner: SARIMA (lower train MAE)")
            preds = fast_forecast_sarima(train_full, test, s_order, s_seasonal_order, period)
        else:
            best_model_name = "ETS_FALLBACK"
            print(f"  [{elapsed()}] Both failed — using ETS fallback")
            preds = fast_forecast_ets(train_full, test, period, seasonal=ets_type)

    else:
        # Non-seasonal path: plain ARIMA.
        # With the fixed detect_seasonality, AirPassengers never reaches here.
        print(f"  [{elapsed()}] Non-seasonal path — ARIMA grid search: {len(order_grid)} fits...")
        t_arima = time.time()
        arima_model, arima_order, _ = fit_best_arima(train, order_grid)
        print(f"  [{elapsed()}] ARIMA done in {time.time()-t_arima:.2f}s — best={arima_order}")

        if arima_model is None:
            best_model_name = "NAIVE"
            print(f"  [{elapsed()}] ARIMA failed — using naive drift fallback")
            slope = (train_full.iloc[-1] - train_full.iloc[0]) / len(train_full)
            preds = [float(train_full.iloc[-1] + slope * (i + 1)) for i in range(len(test))]
        else:
            best_model_name = "ARIMA"
            # Refit on full train for the actual forecast (order selected on capped window)
            preds = fast_forecast_sarima(train_full, test, arima_order)

    # Align lengths (forecast might be shorter if model truncated)
    test_vals = test.iloc[:len(preds)].values

    final_mape  = float(mape(test_vals, preds))
    final_smape = float(smape(test_vals, preds))

    print(f"  [{elapsed()}] Forecast complete — {len(preds)} steps")
    print(f"  [{elapsed()}] MAPE={final_mape:.2f}%  sMAPE={final_smape:.2f}%")
    print(f"  [{elapsed()}] Model selected: {best_model_name}")
    print("═" * 55)
    print(f"  STATISTICAL PIPELINE — DONE  ({elapsed()} total)")
    print("═" * 55 + "\n")

    return {
        "best_model":        best_model_name,
        "predictions":       preds,
        "actuals":           test_vals.tolist(),
        "MAPE":              final_mape,
        "sMAPE":             final_smape,
        "is_seasonal":       bool(seasonal_flag),
        "seasonal_strength": float(strength),
        "differencing":      int(d),
        "period":            int(period),
    }