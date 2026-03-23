"use client"

import type { ParsedData } from "@/lib/csv-parser"
import { StatusBadge } from "./stat-card"

interface AnalysisPanelsProps {
  data: ParsedData
}

function MetricRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className={`font-mono text-xs tabular-nums ${highlight ? "text-primary" : "text-foreground"}`}>
        {value}
      </span>
    </div>
  )
}

function ProgressBar({ value, max, color = "bg-primary" }: { value: number; max: number; color?: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
      <div className={`h-full rounded-full transition-all duration-1000 ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function AnalysisPanels({ data }: AnalysisPanelsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {/* Stationarity Panel */}
      <div className="animate-float-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "300ms" }}>
        <h4 className="mb-4 text-sm font-semibold text-foreground">Stationarity Test</h4>
        <div className="flex flex-col gap-1">
          <MetricRow label="ADF Statistic" value={data.stationarity.adfStat.toFixed(4)} />
<MetricRow label="ADF Statistic" value={data.stationarity.adfStat.toFixed(4)} />
<MetricRow
  label="p-value"
  value={data.stationarity.pValue.toFixed(4)}
  highlight={data.stationarity.pValue < 0.05}
/>
<MetricRow
  label="Differencing (d)"
  value={(data.forecast.differencing ?? 0).toString()}
  highlight={(data.forecast.differencing ?? 0) > 0}
/>

          <div className="mt-3">
            <StatusBadge
              label="Stationarity"
              active={data.stationarity.isStationary}
              activeLabel="Stationary"
              inactiveLabel="Non-stationary"
            />
          </div>
        </div>
      </div>

      {/* Seasonality Panel */}
      <div className="animate-float-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "400ms" }}>
        <h4 className="mb-4 text-sm font-semibold text-foreground">Seasonality Analysis</h4>
        <div className="flex flex-col gap-1">
          <MetricRow label="Detected" value={data.seasonal ? "Yes" : "No"} highlight={data.seasonal} />
          <MetricRow label="Strength" value={`${(data.seasonalStrength * 100).toFixed(1)}%`} />
          <MetricRow label="Period" value={data.period ? `${data.period}` : "N/A"} />
          <MetricRow label="Frequency" value={data.frequency} />
          <div className="mt-3">
            <div className="flex flex-col gap-1.5">
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Seasonal Strength</span>
              <ProgressBar value={data.seasonalStrength} max={1} />
            </div>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="animate-float-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "500ms" }}>
        <h4 className="mb-4 text-sm font-semibold text-foreground">Model Performance</h4>
        <div className="flex flex-col gap-1">
          <MetricRow label="Model" value={data.forecast.model} highlight />
          <MetricRow label="MAPE" value={`${(data.forecast.mape || data.forecast.smape).toFixed(2)}%`} highlight />
          <MetricRow 
      label="SARIMA Order" 
      value={data.forecast.seasonal_order ? `(${data.forecast.seasonal_order.join(',')})` : "N/A"} 
    />
          <MetricRow label="Train Size" value={data.forecast.train.length.toLocaleString()} />
          <MetricRow label="Test Size" value={data.forecast.test.length.toLocaleString()} />
          <MetricRow label="Forecast Steps" value={data.forecast.future.length.toLocaleString()} />
          <div className="mt-3">
            <div className="flex flex-col gap-1.5">
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Accuracy</span>
              <ProgressBar
                value={Math.max(0, 100 - data.forecast.smape)}
                max={100}
                color={data.forecast.smape < 10 ? "bg-chart-2" : data.forecast.smape < 25 ? "bg-chart-4" : "bg-chart-5"}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Distribution Panel */}
      <div className="animate-float-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "600ms" }}>
        <h4 className="mb-4 text-sm font-semibold text-foreground">Distribution</h4>
        <div className="flex flex-col gap-1">
          <MetricRow label="Mean" value={data.stats.mean.toFixed(4)} />
          <MetricRow label="Median" value={data.stats.median.toFixed(4)} />
          <MetricRow label="Std Dev" value={data.stats.std.toFixed(4)} />
          <MetricRow label="Skewness" value={data.stats.skewness.toFixed(4)} />
          <MetricRow label="Q1" value={data.stats.q1.toFixed(4)} />
          <MetricRow label="Q3" value={data.stats.q3.toFixed(4)} />
        </div>
      </div>

      {/* Range Panel */}
      <div className="animate-float-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "700ms" }}>
        <h4 className="mb-4 text-sm font-semibold text-foreground">Range & Volatility</h4>
        <div className="flex flex-col gap-1">
          <MetricRow label="Min" value={data.stats.min.toFixed(4)} />
          <MetricRow label="Max" value={data.stats.max.toFixed(4)} />
          <MetricRow label="Range" value={(data.stats.max - data.stats.min).toFixed(4)} />
          <MetricRow label="Volatility" value={`${(data.stats.volatility * 100).toFixed(2)}%`} />
          <MetricRow label="Trend" value={data.stats.trend} highlight={data.stats.trend !== "stationary"} />
        </div>
      </div>

      {/* Data Quality Panel */}
      <div className="animate-float-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "800ms" }}>
        <h4 className="mb-4 text-sm font-semibold text-foreground">Data Quality</h4>
        <div className="flex flex-col gap-1">
          <MetricRow label="Total Rows" value={data.rows.length.toLocaleString()} />
          <MetricRow label="Valid Points" value={data.timeSeries.length.toLocaleString()} />
          <MetricRow label="Missing" value={`${data.stats.missingPercent.toFixed(1)}%`} />
          <MetricRow label="Columns" value={data.headers.length.toString()} />
          <MetricRow label="Numeric Cols" value={data.numericCols.length.toString()} />
          <MetricRow label="Date Column" value={data.datetimeCol} highlight />
        </div>
      </div>
    </div>
  )
}
