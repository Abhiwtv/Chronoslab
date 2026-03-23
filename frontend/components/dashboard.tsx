"use client"

import type { ParsedData } from "@/lib/csv-parser"
import { DashboardHeader } from "./dashboard-header"
import { StatCard } from "./stat-card"
import { TimeSeriesChart } from "./time-series-chart"
import { DecompositionChart } from "./decomposition-chart"
import { ForecastChart } from "./forecast-chart"
import { DistributionChart } from "./distribution-chart"
import { AnalysisPanels } from "./analysis-panels"
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Clock,
  Database,
} from "lucide-react"

interface DashboardProps {
  data: ParsedData
  fileName: string
  onReset: () => void
  onColumnChange: (col: string) => void
}

export function Dashboard({ data, fileName, onReset, onColumnChange }: DashboardProps) {
  const trendIcon =
    data.stats.trend === "upward" ? (
      <TrendingUp className="h-4 w-4 text-chart-2" />
    ) : data.stats.trend === "downward" ? (
      <TrendingDown className="h-4 w-4 text-chart-5" />
    ) : (
      <Activity className="h-4 w-4 text-muted-foreground" />
    )

  return (
    <div className="min-h-screen bg-background bg-grid">
      <DashboardHeader
        data={data}
        fileName={fileName}
        onReset={onReset}
        onColumnChange={onColumnChange}
      />

      <main className="mx-auto max-w-[1600px] px-6 py-8">
        {/* Overview stat cards */}
        <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          <StatCard
            label="Data Points"
            value={data.stats.count}
            icon={<Database className="h-4 w-4" />}
            delay={0}
          />
          <StatCard
            label="Mean"
            value={data.stats.mean.toFixed(2)}
            sub={`Std: ${data.stats.std.toFixed(2)}`}
            icon={<BarChart3 className="h-4 w-4" />}
            delay={50}
          />
          <StatCard
            label="Trend"
            value={data.stats.trend}
            icon={trendIcon}
            accent={data.stats.trend !== "stationary"}
            delay={100}
          />
          <StatCard
            label="Frequency"
            value={data.frequency}
            icon={<Clock className="h-4 w-4" />}
            delay={150}
          />
          <StatCard
            label="sMAPE"
            value={`${data.forecast.smape.toFixed(1)}%`}
            sub={data.forecast.model}
            accent
            delay={200}
          />
          <StatCard
            label="Volatility"
            value={`${(data.stats.volatility * 100).toFixed(1)}%`}
            sub={data.stats.volatility > 0.1 ? "High" : "Low"}
            delay={250}
          />
        </div>

        {/* Main charts row */}
        <div className="mb-8 grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <TimeSeriesChart data={data} />
          </div>
          <DecompositionChart data={data} />
        </div>

        {/* Forecast */}
        <div className="mb-8">
          <ForecastChart data={data} />
        </div>

        {/* Distribution + Analysis */}
        <div className="mb-8">
          <DistributionChart data={data} />
        </div>

        {/* Detailed Analysis Panels */}
        <div className="mb-8">
          <div className="mb-4 flex items-center gap-3">
            <h2 className="text-base font-semibold text-foreground">Detailed Analysis</h2>
            <div className="h-px flex-1 bg-border" />
          </div>
          <AnalysisPanels data={data} />
        </div>

        {/* Raw data preview */}
        <div className="animate-float-up rounded-xl border border-border bg-card p-6" style={{ animationDelay: "900ms" }}>
          <div className="mb-4 flex items-center justify-between">
            <div className="flex flex-col gap-1">
              <h3 className="text-sm font-semibold text-foreground">Data Preview</h3>
              <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">
                First 20 rows
              </p>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border">
                  {data.headers.slice(0, 8).map((h) => (
                    <th
                      key={h}
                      className="px-3 py-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.rows.slice(0, 20).map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-border/30 transition-colors hover:bg-secondary/50"
                  >
                    {data.headers.slice(0, 8).map((h) => (
                      <td
                        key={h}
                        className="px-3 py-2 font-mono text-xs text-foreground/80 tabular-nums"
                      >
                        {String(row[h] ?? "").slice(0, 30)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 flex items-center justify-center py-8 border-t border-border">
          <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">
            ChronoForge Time Series Forecasting Engine
          </p>
        </footer>
      </main>
    </div>
  )
}
