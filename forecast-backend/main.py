from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from models.statistical import run_statistical_forecast
from models.supply_demand import run_supply_demand
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