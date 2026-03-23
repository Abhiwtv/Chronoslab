"use client"

import type { ReactNode } from "react"

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  icon?: ReactNode
  accent?: boolean
  delay?: number
}

export function StatCard({ label, value, sub, icon, accent, delay = 0 }: StatCardProps) {
  return (
    <div
      className="animate-float-up group flex flex-col gap-2.5 rounded-xl border border-border/50 bg-card/80 p-4 backdrop-blur-sm card-hover"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground/70">
          {label}
        </span>
        {icon && <span className="text-muted-foreground/50 group-hover:text-primary/50 transition-colors">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold tabular-nums tracking-tight ${accent ? "text-primary glow-text-cyan" : "text-foreground"}`}>
        {typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}
      </div>
      {sub && (
        <span className="font-mono text-[10px] text-muted-foreground/60">{sub}</span>
      )}
    </div>
  )
}

interface StatusBadgeProps {
  label: string
  active: boolean
  activeLabel?: string
  inactiveLabel?: string
}

export function StatusBadge({ label, active, activeLabel = "Yes", inactiveLabel = "No" }: StatusBadgeProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border/50 bg-secondary/30 p-3">
      <div className="relative">
        <div className={`h-2.5 w-2.5 rounded-full ${active ? "bg-primary" : "bg-muted-foreground/30"}`} />
        {active && <div className="absolute inset-0 h-2.5 w-2.5 rounded-full bg-primary animate-ping opacity-30" />}
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground/70">{label}</span>
        <span className={`text-sm font-semibold ${active ? "text-primary" : "text-muted-foreground"}`}>
          {active ? activeLabel : inactiveLabel}
        </span>
      </div>
    </div>
  )
}
