import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def detect_frequency(ts):
    n = len(ts)

    if n > 1000:
        return 24   # hourly pattern
    elif n > 200:
        return 7    # daily pattern
    else:
        return 12   # fallback

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------

def adf_test(ts):
    stat, pvalue, *_ = adfuller(ts.dropna())
    return stat, pvalue


def find_d(ts, max_d=2):
    d = 0
    ts_diff = ts.copy()
    while d <= max_d:
        try:
            stat, p = adf_test(ts_diff)
            if p < 0.05 and stat < 0:
                return int(d)
        except:
            pass
        ts_diff = ts_diff.diff().dropna()
        d += 1
    return int(max_d)


def find_D(ts, period, max_D=1):
    D = 0
    ts_diff = ts.copy()
    while D <= max_D:
        try:
            stat, p = adf_test(ts_diff)
            if p < 0.05 and stat < 0:
                return int(D)
        except:
            pass
        ts_diff = ts_diff.diff(period).dropna()
        D += 1
    return int(max_D)


def detect_seasonality(ts, period, threshold=0.4):
    try:
        m_type = 'multiplicative' if (ts > 0).all() else 'additive'
        decomposition = seasonal_decompose(ts, model=m_type, period=period)

        seasonal = decomposition.seasonal
        resid = decomposition.resid

        var_resid = np.nanvar(resid)
        var_seasonal_resid = np.nanvar(seasonal + resid) if m_type == 'additive' else np.nanvar(seasonal * resid)

        if var_seasonal_resid == 0:
            seasonal_strength = 0.0
        else:
            seasonal_strength = max(0.0, 1 - (var_resid / var_seasonal_resid))

        seasonal_flag = seasonal_strength > threshold
        return bool(seasonal_flag), float(seasonal_strength)
    except Exception:
        return False, 0.0


def fit_ets(ts, seasonal_periods, seasonal='add', trend='add'):
    try:
        model = ExponentialSmoothing(
            ts,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        return model.fit(optimized=True)
    except:
        return None


def fit_best_sarima(ts, order_grid, seasonal_order_grid, period):
    best_aic = np.inf
    best_model = None
    best_order = None
    best_seasonal_order = None

    for order in order_grid:
        for seasonal_order in seasonal_order_grid:
            try:
                model = SARIMAX(
                    ts,
                    order=order,
                    seasonal_order=seasonal_order + (period,),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False, maxiter=50)

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue

    return best_model, best_order, best_seasonal_order, best_aic


def fit_best_arima(ts, order_grid):
    best_aic = np.inf
    best_model = None
    best_order = None

    for order in order_grid:
        try:
            model = SARIMAX(
                ts,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False, maxiter=50)

            if results.aic < best_aic:
                best_aic = results.aic
                best_model = results
                best_order = order
        except:
            continue

    return best_model, best_order, best_aic


def rolling_forecast_sarima(train, test, order, seasonal_order=None, period=None):
    history = train.copy()
    preds = []

    for t in range(len(test)):
        kwargs = {'order': order, 'enforce_stationarity': False, 'enforce_invertibility': False}
        if seasonal_order is not None:
            kwargs['seasonal_order'] = seasonal_order + (period,)

        model = SARIMAX(history, **kwargs)
        res = model.fit(disp=False)
        forecast = res.forecast(steps=1)

        preds.append(float(forecast.iloc[0]))
        history = pd.concat([history, test.iloc[t:t+1]])

    return preds


def rolling_forecast_ets(train, test, period, seasonal='add'):
    preds = []
    for t in range(len(test)):
        model = ExponentialSmoothing(
            pd.concat([train, test[:t]]),
            trend='add',
            seasonal=seasonal,
            seasonal_periods=period,
            initialization_method="estimated"
        ).fit(optimized=True)

        preds.append(float(model.forecast(1).iloc[0]))

    return preds


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denom)


# ------------------------------
# MAIN FUNCTION
# ------------------------------

def run_statistical_forecast(ts):
    period = detect_frequency(ts)

    ts = ts.copy()
    ts.index = range(len(ts))
    if len(ts) < 30:
        raise ValueError("Dataset too small for statistical modeling")
    split = int(len(ts) * 0.8)
    train = ts[:split]
    test = ts[split:]

    if len(train) >= period * 2:
        seasonal_flag, strength = detect_seasonality(train, period)
    else:
        seasonal_flag, strength = False, 0.0

    if seasonal_flag:
        D = find_D(train, period)
        diff_train = train.diff(period * D).dropna() if D > 0 else train
        d = find_d(diff_train)
    else:
        d = find_d(train)
        D = 0

    p_grid = q_grid = range(0, 3)
    order_grid = [(i, d, j) for i in p_grid for j in q_grid]

    preds = []
    best_model_name = ""

    if seasonal_flag:
        P_grid = Q_grid = range(0, 2)
        seasonal_grid = [(i, D, j) for i in P_grid for j in Q_grid]

        s_model, s_order, s_seasonal_order, sarima_aic = fit_best_sarima(
            train, order_grid, seasonal_grid, period
        )

        ets_type = 'mul' if (train > 0).all() else 'add'
        ets_model = fit_ets(train, seasonal_periods=period, seasonal=ets_type)

        drop_n = int(d + (D * period if D > 0 else 0))

        sarima_mae = np.mean(np.abs(train.iloc[drop_n:] - s_model.fittedvalues.iloc[drop_n:])) if s_model else np.inf
        ets_mae = np.mean(np.abs(train.iloc[drop_n:] - ets_model.fittedvalues.iloc[drop_n:])) if ets_model else np.inf

        if ets_mae < sarima_mae and ets_model is not None:
            best_model_name = "ETS"
            preds = rolling_forecast_ets(train, test, period, seasonal=ets_type)
        else:
            if s_model is None:
                best_model_name = "ETS_FALLBACK"
                preds = rolling_forecast_ets(train, test, period, seasonal=ets_type)
            else:
                best_model_name = "SARIMA"
                preds = rolling_forecast_sarima(train, test, s_order, s_seasonal_order, period)
    else:
        arima_model, arima_order, _ = fit_best_arima(train, order_grid)
        best_model_name = "ARIMA"
        preds = rolling_forecast_sarima(train, test, arima_order)

    test_vals = test.iloc[:len(preds)].values

    return {
        "best_model": best_model_name,
        "predictions": preds,
        "MAPE": float(mape(test_vals, preds)),
        "sMAPE": float(smape(test_vals, preds)),
        "is_seasonal": bool(seasonal_flag),
        "seasonal_strength": float(strength)
    }