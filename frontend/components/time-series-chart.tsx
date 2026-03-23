"use client"

import { useMemo, useState } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush,
} from "recharts"
import type { ParsedData } from "@/lib/csv-parser"

interface TimeSeriesChartProps {
  data: ParsedData
}

export function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  const [showBrush, setShowBrush] = useState(false)

  const chartData = useMemo(() => {
    const step = Math.max(1, Math.floor(data.timeSeries.length / 600))
    return data.timeSeries.filter((_, i) => i % step === 0)
  }, [data.timeSeries])

  return (
    <div className="animate-float-up rounded-2xl border border-border/50 bg-card/80 p-6 backdrop-blur-sm card-hover" style={{ animationDelay: "200ms" }}>
      <div className="mb-6 flex items-center justify-between">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-primary animate-pulse-glow" />
            <h3 className="text-sm font-semibold text-foreground">Time Series</h3>
          </div>
          <p className="font-mono text-[10px] text-muted-foreground/60 uppercase tracking-wider pl-4">
            {data.targetCol} over {data.datetimeCol}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowBrush(!showBrush)}
            className={`rounded-md border px-2.5 py-1 font-mono text-[9px] uppercase tracking-wider transition-all ${
              showBrush
                ? "border-primary/30 bg-primary/10 text-primary"
                : "border-border/50 bg-secondary/30 text-muted-foreground hover:border-primary/20"
            }`}
          >
            Zoom
          </button>
          <div className="flex items-center gap-1.5 rounded-md bg-secondary/30 border border-border/30 px-2.5 py-1">
            <span className="font-mono text-[9px] text-muted-foreground tabular-nums">
              {data.timeSeries.length.toLocaleString()} pts
            </span>
          </div>
        </div>
      </div>
      <div className={showBrush ? "h-80" : "h-72"}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: showBrush ? 30 : 5 }}>
            <defs>
              <linearGradient id="areaGradientMain" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="oklch(0.75 0.18 195)" stopOpacity={0.25} />
                <stop offset="50%" stopColor="oklch(0.75 0.18 195)" stopOpacity={0.08} />
                <stop offset="100%" stopColor="oklch(0.75 0.18 195)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.22 0.015 260 / 0.5)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 9, fill: "oklch(0.5 0.02 260)", fontFamily: "var(--font-jetbrains)" }}
              tickLine={false}
              axisLine={{ stroke: "oklch(0.22 0.015 260 / 0.5)" }}
              interval="preserveStartEnd"
              tickFormatter={(v) => {
                const d = new Date(v)
                return `${d.getMonth() + 1}/${d.getFullYear().toString().slice(-2)}`
              }}
            />
            <YAxis
  
                domain={['dataMin', 'dataMax']} 
                tick={{ fontSize: 9, fill: "oklch(0.5 0.02 260)", fontFamily: "var(--font-jetbrains)" }}
                tickLine={false}
                axisLine={false}
                width={55}
                // This ensures the axis doesn't jump to 90k unless a data point actually hits 90k
                allowDataOverflow={false} 
                tickFormatter={(v) => {
                  if (Math.abs(v) >= 1000000) return `${(v / 1000).toFixed(1)}M`
                  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}K`
                  return Number(v).toFixed(1)
                }}
              />
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.12 0.008 260 / 0.95)",
                border: "1px solid oklch(0.3 0.015 260)",
                borderRadius: "10px",
                fontSize: "11px",
                fontFamily: "var(--font-jetbrains)",
                color: "oklch(0.95 0.005 260)",
                backdropFilter: "blur(12px)",
                boxShadow: "0 8px 32px oklch(0 0 0 / 0.4)",
              }}
              labelStyle={{ color: "oklch(0.5 0.02 260)", fontSize: "9px", marginBottom: "4px", letterSpacing: "0.05em" }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="oklch(0.75 0.18 195)"
              strokeWidth={1.5}
              fill="url(#areaGradientMain)"
              dot={false}
              activeDot={{ r: 4, fill: "oklch(0.75 0.18 195)", stroke: "oklch(0.14 0.008 260)", strokeWidth: 2 }}
            />
            {showBrush && (
              <Brush
                dataKey="date"
                height={20}
                stroke="oklch(0.75 0.18 195 / 0.3)"
                fill="oklch(0.12 0.008 260)"
                tickFormatter={(v) => {
                  const d = new Date(v)
                  return `${d.getMonth() + 1}/${d.getFullYear().toString().slice(-2)}`
                }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
