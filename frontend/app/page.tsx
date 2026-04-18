"use client"

import { useState, useCallback } from "react"
import { CSVUpload } from "@/components/csv-upload"
import { Dashboard } from "@/components/dashboard"
import { parseCSV, type ParsedData } from "@/lib/csv-parser"

export default function Home() {
  const [data, setData] = useState<ParsedData | null>(null)
  const [fileName, setFileName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [rawCSV, setRawCSV] = useState<string>("")
  const [currentMode, setCurrentMode] = useState<string>("statistical")
  const [isLoading, setIsLoading] = useState(false)

  const syncWithBackend = useCallback(async (csvText: string, name: string, mode: string, col: string, baseParsed: ParsedData) => {
    try {
      const formData = new FormData()
      const file = new File([csvText], name, { type: "text/csv" })
      formData.append("file", file)
      formData.append("column", col)

      // ✅ FIX 1: correct port (8000 = FastAPI), no leading space in URL
      const response = await fetch(`http://localhost:8000/forecast/${mode}`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}: Failed to reach FastAPI`)
      }

      const pythonResult = await response.json()

      if (pythonResult.error || pythonResult.detail) {
        throw new Error(pythonResult.error || pythonResult.detail)
      }

      console.log(`✅ Python Logic Success (${mode}):`, pythonResult)

      let bestModel = "Unknown"
      let finalSmape = baseParsed.forecast.smape
      let backendPredictions: number[] | undefined = undefined
      let backendActuals: number[] | undefined = undefined
      let financeMetrics: any = undefined

      if (mode === "statistical") {
        bestModel = pythonResult.best_model || "Statistical Pipeline"
        finalSmape = typeof pythonResult.sMAPE === "number" ? pythonResult.sMAPE : finalSmape
        backendPredictions = pythonResult.predictions
        // ✅ FIX 2: use actuals from the backend so the split is always aligned
        backendActuals = pythonResult.actuals
      }
      else if (mode === "supply-demand") {
        bestModel = pythonResult.is_hybrid ? "ETS + Dual-Gated DL" : "ETS Baseline"
        finalSmape = typeof pythonResult.final_mape === "number" ? pythonResult.final_mape : finalSmape
        backendPredictions = pythonResult.final
      }
      else if (mode === "finance") {
        bestModel = "GARCH + Dual-Gated DL"
        backendPredictions = pythonResult.corrected_sigma_bps
        financeMetrics = {
          vol_mae_bps: pythonResult.metrics_post_dl?.vol_mae_bps,
          cov_2sig: pythonResult.metrics_post_dl?.cov_2sig,
        }
      }

      // ✅ FIX 3: for statistical mode, rebuild test array from backend actuals
      //    so predicted[i] is always compared against the correct actual[i]
      const rebuiltTest = (mode === "statistical" && backendActuals && backendPredictions)
        ? backendActuals.map((actual: number, index: number) => ({
            date: baseParsed.forecast.test[index]?.date ?? `T+${index + 1}`,
            actual,
            predicted: backendPredictions![index],
          }))
        : baseParsed.forecast.test.map((item, index) => ({
            ...item,
            predicted:
              backendPredictions && backendPredictions[index] !== undefined
                ? backendPredictions[index]
                : item.predicted,
          }))

      const finalParsed: ParsedData = {
        ...baseParsed,
        seasonal: pythonResult.is_seasonal ?? baseParsed.seasonal,
        seasonalStrength: pythonResult.seasonal_strength ?? baseParsed.seasonalStrength,
        forecast: {
          ...baseParsed.forecast,
          model: bestModel,
          smape: finalSmape,
          mape: finalSmape,
          differencing: pythonResult.differencing || 0,
          vol_mae_bps: financeMetrics?.vol_mae_bps,
          cov_2sig: financeMetrics?.cov_2sig,
          test: rebuiltTest,
        },
      }

      setData(finalParsed)
      setFileName(name)
    } catch (err) {
      console.log("❌ Backend not reachable or error occurred, using local fallback", err)
      setData(baseParsed)
      setFileName(name)
    }
  }, [])

  const handleFileLoaded = useCallback(async (csvText: string, name: string, mode: string) => {
    try {
      setError(null)
      setRawCSV(csvText)
      setCurrentMode(mode)
      setIsLoading(true)

      const parsed = parseCSV(csvText)
      await syncWithBackend(csvText, name, mode, parsed.targetCol, parsed)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to parse CSV")
    } finally {
      setIsLoading(false)
    }
  }, [syncWithBackend])

  const handleReset = useCallback(() => {
    setData(null)
    setFileName("")
    setError(null)
    setRawCSV("")
  }, [])

  const handleDemoLoaded = useCallback((result: any, fileName: string, mode: string) => {
    setFileName(fileName)
    setCurrentMode(mode)
    setError(null)

    if (mode === "supply-demand") {
      const testData = (result.timestamps as string[]).map((t, i) => ({
        date: t,
        actual: result.actual[i],
        predicted: result.final[i],
      }))
      const parsed: ParsedData = {
        headers: ["timestamp", "value"],
        rows: [],
        datetimeCol: "timestamp",
        targetCol: "value",
        numericCols: ["value"],
        timeSeries: (result.timestamps as string[]).map((t: string, i: number) => ({
          date: t, value: result.actual[i], timestamp: new Date(t).getTime(),
        })),
        stats: {
          count: result.actual.length, mean: 0, std: 0, min: 0, max: 0,
          median: 0, q1: 0, q3: 0, skewness: 0, trend: "stationary",
          volatility: 0, missingPercent: 0,
        },
        frequency: "Hourly",
        seasonal: true,
        seasonalStrength: 0.8,
        period: 168,
        stationarity: { adfStat: 0, pValue: 0.05, isStationary: true },
        decomposition: { trend: [], seasonal: [], residual: [] },
        forecast: {
          train: [], future: [],
          model: result.is_hybrid ? "ETS + Dual-Gated DL" : "ETS Baseline",
          smape: result.final_mape,
          mape: result.final_mape,
          differencing: 0,
          test: testData,
        },
      }
      setData(parsed)
    // Replace the finance branch in handleDemoLoaded with this:

} else if (mode === "finance") {
  // result now includes: cond_vol_train_bps, train_timestamps, series_stats
  const trainVols: number[]     = result.cond_vol_train_bps ?? []
  const trainTs: string[]       = result.train_timestamps   ?? []
  const testVols: number[]      = result.corrected_sigma_bps ?? []
  const garchVols: number[]     = result.garch_sigma_bps    ?? []
  const n                        = testVols.length

  // Full time series = in-sample GARCH vol + test corrected vol
  const fullTimeSeries = [
    ...trainTs.map((t: string, i: number) => ({
      date: t,
      value: trainVols[i],
      timestamp: new Date(t).getTime(),
    })),
    ...(result.timestamps as string[]).map((t: string, i: number) => ({
      date: t,
      value: testVols[i],
      timestamp: new Date(t).getTime(),
    })),
  ]

  // Test panel: garch baseline as "actual", DL-corrected as "predicted"
  const testData = Array.from({ length: n }, (_, i) => ({
    date: (result.timestamps as string[])[i],
    actual:    garchVols[i]  ?? 0,
    predicted: testVols[i]   ?? 0,
  }))

  // Stats from the full in-sample series (not hardcoded 0)
  const stats = result.series_stats ?? {
    count: trainVols.length + n,
    mean: 0, std: 0, min: 0, max: 0,
  }

  const parsed: ParsedData = {
    headers: ["timestamp", "sigma_bps"],
    rows: [],
    datetimeCol: "timestamp",
    targetCol: "sigma_bps",
    numericCols: ["sigma_bps"],
    timeSeries: fullTimeSeries,
    stats: {
      count:          stats.count,
      mean:           stats.mean,
      std:            stats.std,
      min:            stats.min,
      max:            stats.max,
      median:         0,
      q1:             0,
      q3:             0,
      skewness:       0,
      trend:          "stationary",
      volatility:     stats.std,
      missingPercent: 0,
    },
    frequency: "Hourly",
    seasonal: false,
    seasonalStrength: 0,
    period: null,
    stationarity: { adfStat: 0, pValue: 0.05, isStationary: true },
    decomposition: { trend: [], seasonal: [], residual: [] },
    forecast: {
      train: trainVols.map((v: number, i: number) => ({
        date: trainTs[i],
        value: v,           // forecast.train expects { date, value }
      })),
      future: [],
      model: "GARCH + Dual-Gated DL",
      smape: result.metrics_post_dl?.vol_mae_bps ?? 0,
      mape:  result.metrics_post_dl?.vol_mae_bps ?? 0,
      differencing: 0,
      vol_mae_bps: result.metrics_post_dl?.vol_mae_bps,
      cov_2sig:    result.metrics_post_dl?.cov_2sig,
      coverage_reliable: result.metrics_post_dl?.coverage_reliable ?? false,
      test: testData,
    },
  }
  setData(parsed)
}
  }, [])

  const handleColumnChange = useCallback(
    async (col: string) => {
      if (!rawCSV) return
      try {
        setIsLoading(true)
        const parsed = parseCSV(rawCSV, col)
        const numIdx = parsed.headers.indexOf(col)
        if (numIdx !== -1) {
          await syncWithBackend(rawCSV, fileName, currentMode, col, parsed)
        }
      } catch {
        // silently ignore
      } finally {
        setIsLoading(false)
      }
    },
    [rawCSV, fileName, currentMode, syncWithBackend]
  )

  // ✅ FIX 4: single loading block (duplicate removed)
  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background bg-grid">
        <div className="flex flex-col items-center gap-6 rounded-xl border border-primary/20 bg-card p-8 max-w-md">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="flex flex-col items-center gap-2 text-center">
            <h2 className="text-lg font-semibold text-foreground capitalize">
              Running {currentMode.replace("-", " ")} Model
            </h2>
            <p className="text-sm text-muted-foreground">
              {currentMode === "statistical"
                ? "Evaluating ARIMA, SARIMA, and ETS baselines to find the best fit..."
                : "Processing deep learning corrections in the backend. This might take a few seconds..."}
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background bg-grid">
        <div className="flex flex-col items-center gap-6 rounded-xl border border-destructive/30 bg-card p-8 max-w-md">
          <div className="h-12 w-12 rounded-full bg-destructive/10 flex items-center justify-center">
            <span className="text-destructive text-xl font-bold">!</span>
          </div>
          <div className="flex flex-col items-center gap-2 text-center">
            <h2 className="text-lg font-semibold text-foreground">Parse Error</h2>
            <p className="text-sm text-muted-foreground">{error}</p>
          </div>
          <button
            onClick={handleReset}
            className="rounded-md bg-secondary px-4 py-2 text-sm text-foreground hover:bg-secondary/80 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!data) {
    return <CSVUpload onFileLoaded={handleFileLoaded} onDemoLoaded={handleDemoLoaded} />
  }

  return (
    <Dashboard
      data={data}
      fileName={fileName}
      onReset={handleReset}
      onColumnChange={handleColumnChange}
    />
  )
}