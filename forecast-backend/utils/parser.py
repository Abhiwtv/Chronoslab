import pandas as pd
import numpy as np
import io


# Frequencies that are finer than hourly and should be resampled down to 'h'
_SUB_HOURLY = {"T", "min", "S", "L", "ms", "U", "us", "N", "ns"}


def _infer_resample_rule(index: pd.DatetimeIndex) -> str | None:
    """
    Return a resample rule only when the native frequency is finer than hourly.

    WHY THIS EXISTS:
    - The old code always did resample('h').mean() on any DatetimeIndex.
    - For monthly data (freq='MS'), that upsamples 145 rows to ~104,000 hourly
      slots, fills them with NaN, then interpolates — producing a fake 104k-point
      series. This caused: wrong period detection (12→24), 52s ADF hang,
      211s ARIMA grid, and 20% sMAPE on a dataset that should give ~5%.
    - Fix: only resample when the data is genuinely sub-hourly (minutely, secondly,
      etc.) and needs to be coarsened. For hourly, daily, weekly, monthly data
      the index is already at the right resolution — leave it alone.
    """
    freq = pd.infer_freq(index)
    if freq is None:
        return None
    # Strip leading digits (e.g. '15T' → 'T', '30S' → 'S')
    base = freq.lstrip("0123456789")
    if base in _SUB_HOURLY:
        return "h"
    return None  # hourly / daily / weekly / monthly — preserve as-is


def process_csv(file_bytes: bytes, column: str | None = None) -> pd.Series:
    # ── Parse ─────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.StringIO(file_bytes.decode("utf-8")))
    except Exception:
        df = pd.read_csv(io.StringIO(file_bytes.decode("latin-1")))

    if df.shape[0] == 0:
        raise ValueError("Empty CSV file")

    # ── Datetime detection ────────────────────────────────────
    # Only test object/string columns — pd.to_datetime on numeric columns
    # interprets floats as Unix timestamps and falsely consumes value columns.
    for col in df.columns:
        if df[col].dtype not in (object, "string"):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0.8 * len(df):
            df[col] = parsed
            df = df.sort_values(by=col)
            df.set_index(col, inplace=True)
            break

    # ── Column selection ──────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric column found")

    if column and column in numeric_df.columns:
        ts = numeric_df[column].copy()
    else:
        ts = numeric_df.iloc[:, 0].copy()

    ts = pd.to_numeric(ts, errors="coerce")

    # ── Cleaning ──────────────────────────────────────────────
    ts = ts.replace([np.inf, -np.inf], np.nan)
    if ts.isnull().all():
        raise ValueError("Column contains only invalid values")

    # ── Frequency-aware resampling ────────────────────────────
    # Only resample when data is sub-hourly (minutely, secondly, etc.).
    # Monthly / daily / weekly / hourly data must NOT be resampled to 'h' —
    # that would upsample and inflate the series by orders of magnitude.
    if isinstance(ts.index, pd.DatetimeIndex):
        rule = _infer_resample_rule(ts.index)
        if rule is not None:
            ts = ts.resample(rule).mean()

    # ── Interpolate NaNs (gaps only, not structural zeros) ────
    ts = ts.interpolate(method="linear").bfill().ffill()

    # ── Validation ────────────────────────────────────────────
    # NOTE: zeros and negatives are intentionally kept.
    # The old code replaced them with NaN and interpolated, which silently
    # corrupted demand / financial series with legitimate zero values.
    # statistical.py handles log-transforms internally (detect_seasonality
    # uses log1p only when ts > 0; ARIMA/ETS work fine on non-positive data).
    if len(ts) < 10:
        raise ValueError(
            f"Dataset too small after processing: {len(ts)} rows, need ≥ 10"
        )
    if ts.std() == 0:
        raise ValueError("No variation in data — all values are identical")

    return ts