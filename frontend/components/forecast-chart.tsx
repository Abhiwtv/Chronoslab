"use client"

import { useMemo } from "react"
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"
import type { ParsedData } from "@/lib/csv-parser"

interface ForecastChartProps {
  data: ParsedData
}

export function ForecastChart({ data }: ForecastChartProps) {
  const chartData = useMemo(() => {
    const { train, test, future } = data.forecast

    const trainStep = Math.max(1, Math.floor(train.length / 200))
    const sampledTrain = train.filter((_, i) => i % trainStep === 0)

    const combined: Record<string, number | undefined | string>[] = [
      ...sampledTrain.map((t) => ({
        date: t.date,
        actual: t.value,
        predicted: undefined as number | undefined,
        lower: undefined as number | undefined,
        upper: undefined as number | undefined,
      })),
      ...test.map((t) => ({
        date: t.date,
        actual: t.actual,
        predicted: t.predicted,
        lower: undefined as number | undefined,
        upper: undefined as number | undefined,
      })),
      ...future.map((f) => ({
        date: f.date,
        actual: undefined as number | undefined,
        predicted: f.predicted,
        lower: f.lower,
        upper: f.upper,
      })),
    ]

    return {
      combined,
      splitDate: test[0]?.date,
      futureStart: future[0]?.date,
    }
  }, [data.forecast])

  return (
    <div
      className="animate-float-up rounded-2xl border border-border/50 bg-card/80 p-6 backdrop-blur-sm card-hover"
      style={{ animationDelay: "600ms" }}
    >
      <div className="mb-6 flex items-center justify-between flex-wrap gap-4">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-chart-4 animate-pulse-glow" />
            <h3 className="text-sm font-semibold text-foreground">Forecast Engine</h3>
          </div>
          <div className="flex items-center gap-2 pl-4">
            <span className="font-mono text-[10px] text-muted-foreground/60 uppercase tracking-wider">
              {data.forecast.model}
            </span>
            <span className="rounded-full border border-primary/20 bg-primary/5 px-2 py-0.5 font-mono text-[9px] text-primary">
              sMAPE {data.forecast.smape.toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className="h-px w-4 bg-primary" />
            <span className="font-mono text-[9px] text-muted-foreground/60">Actual</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-px w-4 border-t border-dashed" style={{ borderColor: "oklch(0.8 0.16 80)" }} />
            <span className="font-mono text-[9px] text-muted-foreground/60">Predicted</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2.5 w-4 rounded-sm" style={{ backgroundColor: "oklch(0.8 0.16 80 / 0.12)" }} />
            <span className="font-mono text-[9px] text-muted-foreground/60">95% CI</span>
          </div>
        </div>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData.combined} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <defs>
              <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="oklch(0.8 0.16 80)" stopOpacity={0.2} />
                <stop offset="100%" stopColor="oklch(0.8 0.16 80)" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.22 0.015 260)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "oklch(0.6 0.02 260)", fontFamily: "var(--font-jetbrains)" }}
              tickLine={false}
              axisLine={{ stroke: "oklch(0.22 0.015 260)" }}
              interval="preserveStartEnd"
              tickFormatter={(v) => {
                const d = new Date(v)
                return `${d.getMonth() + 1}/${d.getFullYear().toString().slice(-2)}`
              }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "oklch(0.6 0.02 260)", fontFamily: "var(--font-jetbrains)" }}
              tickLine={false}
              axisLine={false}
              width={60}
              tickFormatter={(v) => {
                if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`
                if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}K`
                return v.toFixed(1)
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.14 0.008 260)",
                border: "1px solid oklch(0.22 0.015 260)",
                borderRadius: "8px",
                fontSize: "11px",
                fontFamily: "var(--font-jetbrains)",
                color: "oklch(0.95 0.005 260)",
              }}
              labelStyle={{ color: "oklch(0.6 0.02 260)", fontSize: "10px", marginBottom: "4px" }}
            />
            {chartData.splitDate && (
              <ReferenceLine
                x={chartData.splitDate}
                stroke="oklch(0.75 0.18 195)"
                strokeDasharray="4 4"
                strokeOpacity={0.5}
              />
            )}
            {chartData.futureStart && (
              <ReferenceLine
                x={chartData.futureStart}
                stroke="oklch(0.65 0.2 300)"
                strokeDasharray="4 4"
                strokeOpacity={0.5}
              />
            )}
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="url(#ciGradient)"
              connectNulls={false}
            />
            <Area
              type="monotone"
              dataKey="lower"
              stroke="none"
              fill="oklch(0.1 0.005 260)"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="oklch(0.75 0.18 195)"
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="oklch(0.8 0.16 80)"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
