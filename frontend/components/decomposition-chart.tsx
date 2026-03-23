"use client"

import { useMemo, useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import type { ParsedData } from "@/lib/csv-parser"

interface DecompositionChartProps {
  data: ParsedData
}

const layers = [
  { key: "trend", label: "Trend", color: "oklch(0.75 0.18 195)" },
  { key: "seasonal", label: "Seasonal", color: "oklch(0.7 0.15 145)" },
  { key: "residual", label: "Residual", color: "oklch(0.7 0.2 25)" },
] as const

function MiniChart({
  chartData,
  color,
  title,
  isActive,
  onClick,
}: {
  chartData: { date: string; value: number }[]
  color: string
  title: string
  isActive: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col gap-2 rounded-lg border p-3 transition-all ${
        isActive
          ? "border-border/60 bg-secondary/40"
          : "border-transparent bg-transparent hover:bg-secondary/20"
      }`}
    >
      <div className="flex items-center gap-2">
        <div className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground/70">
          {title}
        </span>
      </div>
      <div className="h-16">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </button>
  )
}

export function DecompositionChart({ data }: DecompositionChartProps) {
  const [activeLayer, setActiveLayer] = useState<"trend" | "seasonal" | "residual">("trend")

  const { datasets, activeData } = useMemo(() => {
    const step = Math.max(1, Math.floor(data.decomposition.trend.length / 300))
    const datasets = {
      trend: data.decomposition.trend.filter((_, i) => i % step === 0),
      seasonal: data.decomposition.seasonal.filter((_, i) => i % step === 0),
      residual: data.decomposition.residual.filter((_, i) => i % step === 0),
    }
    return { datasets, activeData: datasets[activeLayer] }
  }, [data.decomposition, activeLayer])

  const activeConfig = layers.find((l) => l.key === activeLayer)!

  return (
    <div
      className="animate-float-up rounded-2xl border border-border/50 bg-card/80 p-6 backdrop-blur-sm card-hover"
      style={{ animationDelay: "400ms" }}
    >
      <div className="mb-4 flex flex-col gap-1.5">
        <h3 className="text-sm font-semibold text-foreground">Decomposition</h3>
        <p className="font-mono text-[10px] text-muted-foreground/60 uppercase tracking-wider">
          Trend + Seasonal + Residual
        </p>
      </div>

      {/* Active chart expanded */}
      <div className="mb-4 h-40 rounded-lg border border-border/30 bg-background/50 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={activeData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.22 0.015 260 / 0.4)" />
            <XAxis dataKey="date" hide />
            <YAxis hide />
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.12 0.008 260 / 0.95)",
                border: "1px solid oklch(0.3 0.015 260)",
                borderRadius: "8px",
                fontSize: "10px",
                fontFamily: "var(--font-jetbrains)",
                color: "oklch(0.95 0.005 260)",
                backdropFilter: "blur(12px)",
              }}
              labelStyle={{ display: "none" }}
            />
            <Line type="monotone" dataKey="value" stroke={activeConfig.color} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Switcher mini-charts */}
      <div className="grid grid-cols-3 gap-2">
        {layers.map((layer) => (
          <MiniChart
            key={layer.key}
            chartData={datasets[layer.key]}
            color={layer.color}
            title={layer.label}
            isActive={activeLayer === layer.key}
            onClick={() => setActiveLayer(layer.key)}
          />
        ))}
      </div>
    </div>
  )
}
