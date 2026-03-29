from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

from models.statistical import run_statistical_forecast
from models.supply_demand import run_supply_demand, get_nyc_slice, NYC_SCENARIOS
from models.finance import run_finance

from utils.parser import process_csv

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Existing routes ──────────────────────────────────────────

@app.post("/forecast/statistical")
async def statistical(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        ts = process_csv(contents)
        return run_statistical_forecast(ts)
    except Exception as e:
        return {"error": str(e)}


@app.post("/forecast/supply-demand")
async def supply_demand(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        ts = process_csv(contents)
        return run_supply_demand(ts)
    except Exception as e:
        return {"error": str(e)}


@app.post("/forecast/finance")
async def finance(file: UploadFile = File(None)):
    try:
        if file:
            contents = await file.read()
            ts = process_csv(contents)
            return run_finance(ts)
        else:
            return run_finance()
    except Exception as e:
        return {"error": str(e)}


# ── Demo routes ──────────────────────────────────────────────

@app.get(
    "/forecast/supply-demand/demo",
    summary="NYC Taxi Demo",
    description=(
        "Runs the supply-demand pipeline on a pre-defined NYC Taxi slice. "
        "No file upload required.\n\n"
        "**Scenarios**\n"
        "- `weekday_24h` — Tuesday 18 Nov 2014, 24-hour forecast\n"
        "- `weekend_24h` — Saturday 22 Nov 2014, 24-hour forecast\n"
        "- `mon_wed_72h` — Mon-Wed 17-19 Nov 2014, 72-hour forecast"
    ),
)
async def supply_demand_demo(
    scenario: Literal["weekday_24h", "weekend_24h", "mon_wed_72h"] = Query(
        default="mon_wed_72h",
        description="Pre-defined NYC Taxi time window to forecast.",
    ),
):
    try:
        end_date, test_hours = NYC_SCENARIOS[scenario]
        ts = get_nyc_slice(end_date)
        return run_supply_demand(ts, test_hours=test_hours, period=168)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/forecast/finance/demo",
    summary="BTC/USDT Finance Demo",
    description=(
        "Fetches live BTC/USDT 1H data from Binance and runs the full "
        "ETS -> GARCH -> Dual-Gated DL volatility correction pipeline. "
        "No file upload required.\n\n"
        "Returns per-hour GARCH baseline vol, DL-corrected vol, blend ratios, "
        "and coverage calibration metrics for both pre-DL and post-DL intervals."
    ),
)
async def finance_demo(
    test_hours: int = Query(
        default=5,
        ge=1,
        le=24,
        description="Number of hours to hold out as the test set (1-24).",
    ),
    n_candles: int = Query(
        default=6500,
        ge=500,
        le=8000,
        description="Number of 1H candles to fetch (~270 days at 6500).",
    ),
):
    try:
        return run_finance(ts=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))