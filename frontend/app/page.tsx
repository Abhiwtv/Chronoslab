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

  const [isClient, setIsClient] = useState(false);

 const handleFileLoaded = useCallback(async (csvText: string, name: string) => {
    try {
      setError(null)
      setRawCSV(csvText)

      // 1. Run the local frontend parse first (provides initial structures)
      let parsed = parseCSV(csvText)
      
      // 2. Call the Python Backend
      try {
        const formData = new FormData()
        const file = new File([csvText], name, { type: "text/csv" })
        formData.append("file", file)

        const response = await fetch("http://127.0.0.1:8000/forecast", {
          method: "POST",
          body: formData,
        })

        if (response.ok) {
          const pythonResult = await response.json()
          console.log("✅ Python Logic Success:", pythonResult.sMAPE)

          // 3. OVERWRITE the local "dumb" forecast with the Python "smart" forecast
          parsed = {
            ...parsed,
            seasonal: pythonResult.is_seasonal,
            seasonalStrength: pythonResult.seasonal_strength,
            forecast: {
              ...parsed.forecast,
              model: pythonResult.best_model,
              smape: pythonResult.sMAPE, // THIS is the 2.60%
              mape: pythonResult.MAPE,
              differencing: pythonResult.differencing,
              // Update the chart predictions to match the Python model
              test: parsed.forecast.test.map((item, index) => ({
                ...item,
                predicted: pythonResult.predictions[index] !== undefined 
                  ? pythonResult.predictions[index] 
                  : item.predicted
              }))
            }
          }
        }
      } catch (err) {
        console.log("❌ Backend not reachable, using local fallback", err)
      }

      // 4. Set the FINAL data (now containing the 2.60%)
      setData(parsed)
      setFileName(name)

    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to parse CSV")
    }
  }, [])

  const handleReset = useCallback(() => {
    setData(null)
    setFileName("")
    setError(null)
    setRawCSV("")
  }, [])

  const handleColumnChange = useCallback(
    (col: string) => {
      if (!rawCSV) return
      try {
        const lines = rawCSV.trim().split("\n")
        const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""))
        const colIdx = headers.indexOf(col)
        if (colIdx === -1) return

        const parsed = parseCSV(rawCSV)
        const newHeaders = [...parsed.headers]
        const numIdx = newHeaders.indexOf(col)

        if (numIdx !== -1) {
          const newData: ParsedData = {
            ...parsed,
            targetCol: col,
            timeSeries: parsed.rows
              .map((r) => {
                const dateStr = String(r[parsed.datetimeCol])
                const d = new Date(dateStr)
                const val = Number(r[col])
                if (isNaN(d.getTime()) || isNaN(val)) return null
                return {
                  date: d.toISOString().split("T")[0],
                  value: val,
                  timestamp: d.getTime(),
                }
              })
              .filter(Boolean) as ParsedData["timeSeries"],
          }

          newData.timeSeries.sort((a, b) => a.timestamp - b.timestamp)
          setData(newData)
        }

      } catch {
        // silently ignore
      }
    },
    [rawCSV]
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
