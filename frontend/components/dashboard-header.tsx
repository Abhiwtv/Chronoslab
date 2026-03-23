"use client"

import { Atom, Upload, ChevronDown, Signal } from "lucide-react"
import { useState, useEffect } from "react"
import type { ParsedData } from "@/lib/csv-parser"

interface DashboardHeaderProps {
  data: ParsedData
  fileName: string
  onReset: () => void
  onColumnChange: (col: string) => void
}

export function DashboardHeader({ data, fileName, onReset, onColumnChange }: DashboardHeaderProps) {
  const [showColumns, setShowColumns] = useState(false)
  const [clock, setClock] = useState("")

  useEffect(() => {
    const update = () => {
      const now = new Date()
      setClock(now.toLocaleTimeString("en-US", { hour12: false }))
    }
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/70 backdrop-blur-2xl">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-2.5">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="relative h-8 w-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
              <Atom className="h-4 w-4 text-primary" />
            </div>
            <span className="text-base font-bold text-foreground tracking-tight">
              Chronos<span className="text-primary">Lab</span>
            </span>
          </div>
          <div className="hidden h-5 w-px bg-border/50 md:block" />
          <div className="hidden items-center gap-3 md:flex">
            <div className="flex items-center gap-1.5">
              <Signal className="h-3 w-3 text-chart-2" />
              <span className="font-mono text-[10px] text-chart-2 uppercase tracking-wider">Live</span>
            </div>
            <span className="font-mono text-[10px] text-muted-foreground truncate max-w-[180px]">
              {fileName}
            </span>
            <div className="h-1 w-1 rounded-full bg-primary/30" />
            <span className="font-mono text-[10px] text-primary tabular-nums">
              {data.timeSeries.length.toLocaleString()} records
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Clock */}
          <span className="hidden font-mono text-[10px] text-muted-foreground tabular-nums lg:block">
            {clock}
          </span>

          <div className="hidden h-5 w-px bg-border/50 lg:block" />

          {/* Column selector */}
          {data.numericCols.length > 1 && (
            <div className="relative">
              <button
                onClick={() => setShowColumns(!showColumns)}
                className="flex items-center gap-2 rounded-lg border border-border/60 bg-secondary/50 px-3 py-1.5 font-mono text-[11px] text-muted-foreground hover:border-primary/30 hover:text-foreground transition-all"
              >
                <span className="text-primary/60 text-[9px] uppercase tracking-wider mr-1">Col:</span>
                {data.targetCol}
                <ChevronDown className={`h-3 w-3 transition-transform ${showColumns ? "rotate-180" : ""}`} />
              </button>
              {showColumns && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowColumns(false)} />
                  <div className="absolute right-0 top-full mt-1.5 z-50 rounded-lg border border-border/60 bg-card/95 backdrop-blur-xl py-1 shadow-2xl min-w-[180px]">
                    <div className="px-3 py-1.5 border-b border-border/40">
                      <span className="font-mono text-[9px] text-muted-foreground/60 uppercase tracking-widest">Numeric Columns</span>
                    </div>
                    {data.numericCols.map((col) => (
                      <button
                        key={col}
                        onClick={() => {
                          onColumnChange(col)
                          setShowColumns(false)
                        }}
                        className={`flex w-full items-center gap-2 px-3 py-2 text-left font-mono text-xs transition-colors hover:bg-secondary/50 ${
                          col === data.targetCol ? "text-primary" : "text-muted-foreground"
                        }`}
                      >
                        <div className={`h-1.5 w-1.5 rounded-full ${col === data.targetCol ? "bg-primary" : "bg-muted-foreground/20"}`} />
                        {col}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          <button
            onClick={onReset}
            className="flex items-center gap-2 rounded-lg border border-border/60 bg-secondary/50 px-3 py-1.5 text-[11px] text-muted-foreground hover:border-primary/30 hover:text-foreground transition-all"
          >
            <Upload className="h-3 w-3" />
            <span className="hidden sm:inline">New Dataset</span>
          </button>
        </div>
      </div>
    </header>
  )
}
