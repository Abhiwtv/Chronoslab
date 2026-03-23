import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Auto-detect if we should use multiplicative math
        m_type = 'multiplicative' if (ts > 0).all() else 'additive'
        decomposition = seasonal_decompose(ts, model=m_type, period=period)
        
        seasonal = decomposition.seasonal
        resid = decomposition.resid

        var_resid = np.nanvar(resid)
        var_seasonal_resid = np.nanvar(seasonal + resid) if m_type == 'additive' else np.nanvar(seasonal * resid)

        # Calculate strength, ensuring it doesn't break on weird variance math
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
            initialization_method="estimated" # <-- FIX: Prevents integer index crashes
        )
        res = model.fit(optimized=True)
        return res
    except Exception as e:
        print(f"ETS fitting failed: {e}")
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
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denom)

# ------------------------------
# MAIN FORECAST ROUTE
# ------------------------------

@app.post("/forecast")
async def forecast(file: UploadFile = File(...)):
    try:
        # Hardcoded period to prevent frontend form issues
        period = 12 

        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "Uploaded CSV is empty"})

        # Detect numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return JSONResponse(status_code=400, content={"error": "No numeric column found in CSV"})

        target_col = numeric_cols[0]
        ts = df[target_col].dropna().astype(float)
        
        # <-- FIX: Force a clean integer index so ETS doesn't crash on unstructured datasets
        ts.index = range(len(ts)) 
        print("\n" + "="*30)
        print(f"NEW CODE IS RUNNING!")
        print(f"Uploaded file has {len(ts)} rows.")
        print("="*30 + "\n")
        # Minimum dataset check
        if len(ts) < 10:
            return JSONResponse(status_code=400, content={
                "error": "Dataset too small. Need at least 10 numeric data points."
            })

        # Train-test split
        split = int(len(ts) * 0.8)
        train = ts[:split]
        test = ts[split:]

        # 1. Seasonality Detection
        if len(train) >= period * 2:
            seasonal_flag, strength = detect_seasonality(train, period)
        else:
            seasonal_flag, strength = False, 0.0

        # 2. Determine Differencing
        if seasonal_flag:
            D = find_D(train, period)
            diff_train = train.diff(period * D).dropna() if D > 0 else train
            d = find_d(diff_train)
        else:
            d = find_d(train)
            D = 0

        # Grids
        p_grid = q_grid = range(0, 3)
        order_grid = [(i, d, j) for i in p_grid for j in q_grid]
        
        best_model_name = ""
        best_order = None
        best_seasonal_order = None
        best_aic = np.inf
        preds = []

        # 3. Model Fitting Pipeline
        # 3. Model Fitting Pipeline
        if seasonal_flag:
            P_grid = Q_grid = range(0, 2)
            seasonal_grid = [(i, D, j) for i in P_grid for j in Q_grid]

            # Fit SARIMA
            s_model, s_order, s_seasonal_order, sarima_aic = fit_best_sarima(
                train, order_grid, seasonal_grid, period
            )

            # Fit ETS (Auto-switch to Multiplicative if data is strictly positive)
            ets_season_type = 'mul' if (train > 0).all() else 'add'
            ets_model = fit_ets(train, seasonal_periods=period, seasonal=ets_season_type, trend='add')

            # <-- THE FIX: Drop initial zeros caused by SARIMA differencing for a fair fight
            drop_n = int(d + (D * period if D > 0 else 0))

            if s_model is not None and len(train) > drop_n:
                sarima_mae = np.mean(np.abs(train.iloc[drop_n:] - s_model.fittedvalues.iloc[drop_n:]))
            else:
                sarima_mae = np.inf

            if ets_model is not None and len(train) > drop_n:
                ets_mae = np.mean(np.abs(train.iloc[drop_n:] - ets_model.fittedvalues.iloc[drop_n:]))
            else:
                ets_mae = np.inf

            # Now pick the TRUE winner
            if ets_mae < sarima_mae and ets_model is not None:
                best_model_name = f"ETS (Multiplicative)" if ets_season_type == 'mul' else "ETS (Additive)"
                best_aic = 0.0 
                best_order = (0, int(d), 0) 
                best_seasonal_order = (0, int(D), 0)
                preds = rolling_forecast_ets(train, test, period, seasonal=ets_season_type)
            else:
                best_model_name = "SARIMA"
                best_aic = float(sarima_aic)
                best_order = tuple(int(x) for x in s_order) if s_order else None
                best_seasonal_order = tuple(int(x) for x in s_seasonal_order) if s_seasonal_order else None
                preds = rolling_forecast_sarima(train, test, s_order, s_seasonal_order, period)
        else:
            arima_model, arima_order, arima_aic = fit_best_arima(train, order_grid)
            best_model_name = "ARIMA"
            best_aic = float(arima_aic)
            best_order = tuple(int(x) for x in arima_order) if arima_order else None
            preds = rolling_forecast_sarima(train, test, arima_order)

        # Calculate Error Metrics
        test_actuals = test.iloc[:len(preds)].values
        err_mape = mape(test_actuals, preds)
        err_smape = smape(test_actuals, preds)

        # --- THE TERMINAL SCORECARD ---
        print("\n" + "="*40)
        print("🏆 FORECAST PIPELINE SCORECARD 🏆")
        print("="*40)
        print(f"Is Seasonal?   : {bool(seasonal_flag)} (Strength: {strength:.2f})")
        print(f"Differencing   : d={d}, D={D}")
        print(f"Winning Model  : {best_model_name}")
        print(f"Model Order    : {best_order} x {best_seasonal_order}")
        print(f"Test Set sMAPE : {err_smape:.2f}%")
        print("="*40 + "\n")
        
        # 4. Construct Response Format...
        # 4. Construct Response Format
        return {
            "best_model": str(best_model_name),
            "order": list(best_order) if best_order else None,
            "differencing": int(d),
            "AIC": float(best_aic) if best_aic != np.inf else 0.0,
            "sMAPE": float(err_smape),
            "predictions": preds,
            "MAPE": float(err_mape),
            "is_seasonal": bool(seasonal_flag),
            "seasonal_strength": float(strength),
            "seasonal_order": list(best_seasonal_order) if best_seasonal_order else None,
            "seasonal_differencing": int(D)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})