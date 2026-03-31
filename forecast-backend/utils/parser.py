import pandas as pd
import numpy as np
import io

def process_csv(file_bytes, column=None):
    try:
        df = pd.read_csv(io.StringIO(file_bytes.decode("utf-8")))
    except:
        df = pd.read_csv(io.StringIO(file_bytes.decode("latin-1")))

    if df.shape[0] == 0:
        raise ValueError("Empty CSV file")

    # ---------- DATETIME DETECTION ----------
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors='coerce')
        if parsed.notna().sum() > 0.8 * len(df):
            df[col] = parsed
            df = df.sort_values(by=col)
            df.set_index(col, inplace=True)
            break

    # ---------- NUMERIC SELECTION ----------
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric column found")

    # Column selection support
    if column and column in numeric_df.columns:
        ts = numeric_df[column]
    else:
        ts = numeric_df.iloc[:, 0]
        
    # Force numeric
    ts = pd.to_numeric(ts, errors='coerce')
    
    # ---------- CLEANING & RESAMPLING ----------
    ts = ts.replace([np.inf, -np.inf], np.nan)

    if ts.isnull().all():
        raise ValueError("Column contains only invalid values")

    # IMPORTANT FIX: Preserve DatetimeIndex and resample to hourly
    # This ensures the 168-period ETS seasonality and DL time-features work correctly
    if isinstance(ts.index, pd.DatetimeIndex):
        ts = ts.resample('h').sum()

    # Interpolate to fill any gaps created by resampling or missing data
    ts = ts.interpolate(method="linear").bfill().ffill()

    if len(ts) < 20:
        raise ValueError("Dataset too small")

    if ts.std() == 0:
        raise ValueError("No variation in data")

    # REMOVED: ts.index = range(len(ts)) 
    # (This line was previously destroying the timestamps needed by the models)

    return ts