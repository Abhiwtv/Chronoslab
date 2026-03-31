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
  const [isClient, setIsClient] = useState(false);

 const syncWithBackend = useCallback(async (csvText: string, name: string, mode: string, col: string, baseParsed: ParsedData) => {
    try {
      const formData = new FormData()
      const file = new File([csvText], name, { type: "text/csv" })
      formData.append("file", file)
      formData.append("column", col)

      const response = await fetch(`http://127.0.0.1:8000/forecast/${mode}`, {
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

      // 1. Dynamically map the correct keys based on the pipeline mode
      let bestModel = "Unknown"
      let finalSmape = baseParsed.forecast.smape
      let backendPredictions: number[] | undefined = undefined
      let financeMetrics: any = undefined

      if (mode === "statistical") {
        bestModel = pythonResult.best_model || "Statistical Pipeline"
        finalSmape = typeof pythonResult.sMAPE === 'number' ? pythonResult.sMAPE : finalSmape
        backendPredictions = pythonResult.predictions
      } 
      else if (mode === "supply-demand") {
        bestModel = pythonResult.is_hybrid ? "ETS + Dual-Gated DL" : "ETS Baseline"
        finalSmape = typeof pythonResult.final_mape === 'number' ? pythonResult.final_mape : finalSmape
        backendPredictions = pythonResult.final
      } 
      else if (mode === "finance") {
        bestModel = "GARCH + Dual-Gated DL"
        backendPredictions = pythonResult.corrected_sigma_bps
        financeMetrics = {
          vol_mae_bps: pythonResult.metrics_post_dl?.vol_mae_bps,
          cov_2sig: pythonResult.metrics_post_dl?.cov_2sig
        }
      }

      // 2. Overwrite the frontend state
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
          test: baseParsed.forecast.test.map((item, index) => ({
            ...item,
            predicted: backendPredictions && backendPredictions[index] !== undefined 
              ? backendPredictions[index] 
              : item.predicted
          }))
        }
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

      // 1. Run the local frontend parse first
      let parsed = parseCSV(csvText)
      
      // 2. Call the backend sync process
      await syncWithBackend(csvText, name, mode, parsed.targetCol, parsed)

    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to parse CSV")
    }
  }, [syncWithBackend])

  const handleReset = useCallback(() => {
    setData(null)
    setFileName("")
    setError(null)
    setRawCSV("")
  }, [])

  const handleColumnChange = useCallback(
    async (col: string) => {
      if (!rawCSV) return
      try {
        const parsed = parseCSV(rawCSV, col)
        const newHeaders = [...parsed.headers]
        const numIdx = newHeaders.indexOf(col)

        if (numIdx !== -1) {
          await syncWithBackend(rawCSV, fileName, currentMode, col, parsed)
        }
      } catch {
        // silently ignore
      }
    },
    [rawCSV, fileName, currentMode, syncWithBackend]
  )

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
    return <CSVUpload onFileLoaded={handleFileLoaded} />
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
