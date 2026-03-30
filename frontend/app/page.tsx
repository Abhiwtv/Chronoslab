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
      
      // Safety check in case FastAPI returned a dict with error via 200 somehow
      if (pythonResult.error || pythonResult.detail) {
        throw new Error(pythonResult.error || pythonResult.detail)
      }

      console.log("✅ Python Logic Success:", pythonResult.sMAPE)

      // OVERWRITE the local "dumb" forecast with the Python "smart" forecast
      const finalParsed: ParsedData = {
        ...baseParsed,
        seasonal: pythonResult.is_seasonal,
        seasonalStrength: pythonResult.seasonal_strength,
        forecast: {
          ...baseParsed.forecast,
          model: pythonResult.best_model || "Unknown",
          smape: typeof pythonResult.sMAPE === 'number' ? pythonResult.sMAPE : baseParsed.forecast.smape,
          mape: typeof pythonResult.MAPE === 'number' ? pythonResult.MAPE : baseParsed.forecast.mape,
          differencing: pythonResult.differencing || 0,
          test: baseParsed.forecast.test.map((item, index) => ({
            ...item,
            predicted: pythonResult.predictions && pythonResult.predictions[index] !== undefined 
              ? pythonResult.predictions[index] 
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
