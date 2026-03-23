"use client"

import { useMemo } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"
import type { ParsedData } from "@/lib/csv-parser"

interface DistributionChartProps {
  data: ParsedData
}

export function DistributionChart({ data }: DistributionChartProps) {
  const { bins, mean } = useMemo(() => {
    const values = data.timeSeries.map((t) => t.value)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const binCount = Math.min(50, Math.max(10, Math.floor(Math.sqrt(values.length))))
    const binWidth = (max - min) / binCount || 1
    const mean = values.reduce((s, v) => s + v, 0) / values.length

    const bins = Array.from({ length: binCount }, (_, i) => ({
      range: `${(min + i * binWidth).toFixed(1)}`,
      count: 0,
      rangeEnd: min + (i + 1) * binWidth,
    }))

    for (const v of values) {
      const idx = Math.min(binCount - 1, Math.floor((v - min) / binWidth))
      bins[idx].count++
    }

    return { bins, mean }
  }, [data.timeSeries])

  return (
    <div className="animate-float-up rounded-xl border border-border bg-card p-6" style={{ animationDelay: "500ms" }}>
      <div className="mb-6 flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <h3 className="text-sm font-semibold text-foreground">Value Distribution</h3>
          <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">
            Histogram of {data.targetCol}
          </p>
        </div>
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.22 0.015 260)" vertical={false} />
            <XAxis
              dataKey="range"
              tick={{ fontSize: 9, fill: "oklch(0.6 0.02 260)", fontFamily: "var(--font-jetbrains)" }}
              tickLine={false}
              axisLine={{ stroke: "oklch(0.22 0.015 260)" }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 9, fill: "oklch(0.6 0.02 260)", fontFamily: "var(--font-jetbrains)" }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.14 0.008 260)",
                border: "1px solid oklch(0.22 0.015 260)",
                borderRadius: "6px",
                fontSize: "11px",
                fontFamily: "var(--font-jetbrains)",
                color: "oklch(0.95 0.005 260)",
              }}
            />
            <ReferenceLine
              x={bins.reduce((best, b) => (Math.abs(parseFloat(b.range) - mean) < Math.abs(parseFloat(best.range) - mean) ? b : best), bins[0]).range}
              stroke="oklch(0.75 0.18 195)"
              strokeDasharray="4 4"
              strokeOpacity={0.7}
              label={{ value: "Mean", position: "top", fill: "oklch(0.75 0.18 195)", fontSize: 10, fontFamily: "var(--font-jetbrains)" }}
            />
            <Bar
              dataKey="count"
              fill="oklch(0.75 0.18 195)"
              fillOpacity={0.6}
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
