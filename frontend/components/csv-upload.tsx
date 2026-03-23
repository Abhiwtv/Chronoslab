"use client"

import { useCallback, useState, useEffect, useRef } from "react"
import { Upload, FileText, Atom, Database, BarChart3, Cpu } from "lucide-react"

interface CSVUploadProps {
  onFileLoaded: (csvText: string, fileName: string) => void
}

function FloatingParticle({ delay, x, duration }: { delay: number; x: number; duration: number }) {
  return (
    <div
      className="absolute w-px rounded-full bg-primary/30"
      style={{
        height: `${Math.random() * 60 + 20}px`,
        left: `${x}%`,
        bottom: "-10%",
        animation: `data-stream ${duration}s linear ${delay}s infinite`,
      }}
    />
  )
}

function PipelineStep({ icon, label, active, done }: { icon: React.ReactNode; label: string; active: boolean; done: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`flex h-10 w-10 items-center justify-center rounded-lg border transition-all duration-500 ${
          done
            ? "border-primary/50 bg-primary/10 text-primary glow-cyan"
            : active
              ? "border-primary/30 bg-primary/5 text-primary animate-breathe"
              : "border-border bg-secondary/50 text-muted-foreground"
        }`}
      >
        {icon}
      </div>
      <span className={`font-mono text-[9px] uppercase tracking-widest transition-colors duration-500 ${
        done || active ? "text-primary" : "text-muted-foreground/50"
      }`}>
        {label}
      </span>
    </div>
  )
}

export function CSVUpload({ onFileLoaded }: CSVUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [pipelineStep, setPipelineStep] = useState(0)
  const timerRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const handleFile = useCallback(
    (file: File) => {
      if (!file.name.endsWith(".csv")) return
      setIsProcessing(true)
      setPipelineStep(0)
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        const steps = [1, 2, 3, 4]
        let i = 0
        const advance = () => {
          if (i < steps.length) {
            setPipelineStep(steps[i])
            i++
            timerRef.current = setTimeout(advance, 600)
          } else {
            onFileLoaded(text, file.name)
            setIsProcessing(false)
          }
        }
        timerRef.current = setTimeout(advance, 400)
      }
      reader.readAsText(file)
    },
    [onFileLoaded]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const loadSampleData = useCallback(() => {
    setIsProcessing(true)
    setPipelineStep(0)
    const headers = "Date,Temperature,Humidity,Pressure\n"
    const rows: string[] = []
    const start = new Date("2021-01-01")
    let temp = 15, hum = 60, pres = 1013
    for (let i = 0; i < 730; i++) {
      const d = new Date(start.getTime() + i * 86400000)
      temp += (Math.random() - 0.48) * 2 + Math.sin((i / 365) * Math.PI * 2) * 0.5
      hum += (Math.random() - 0.5) * 3 + Math.cos((i / 365) * Math.PI * 2) * 0.3
      pres += (Math.random() - 0.5) * 2
      rows.push(`${d.toISOString().split("T")[0]},${temp.toFixed(1)},${Math.max(20, Math.min(95, hum)).toFixed(1)},${pres.toFixed(1)}`)
    }
    const text = headers + rows.join("\n")
    let step = 0
    const advance = () => {
      if (step < 5) {
        setPipelineStep(step)
        step++
        timerRef.current = setTimeout(advance, 500)
      } else {
        onFileLoaded(text, "weather_sample_2yr.csv")
        setIsProcessing(false)
      }
    }
    timerRef.current = setTimeout(advance, 300)
  }, [onFileLoaded])

  const triggerFileInput = useCallback(() => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = ".csv"
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (file) handleFile(file)
    }
    input.click()
  }, [handleFile])

  if (isProcessing) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background relative overflow-hidden">
        <div className="absolute inset-0 bg-dot-grid animate-grid-pulse" />
        {Array.from({ length: 30 }).map((_, i) => (
          <FloatingParticle key={i} delay={Math.random() * 5} x={Math.random() * 100} duration={Math.random() * 4 + 3} />
        ))}
        <div className="relative z-10 flex flex-col items-center gap-10">
          {/* Orbiting loader */}
          <div className="relative h-36 w-36">
            <div className="absolute inset-0 rounded-full border border-primary/10" />
            <div className="absolute inset-4 rounded-full border border-primary/15" />
            <div className="absolute inset-8 rounded-full border border-primary/20" />
            <div className="absolute inset-0 animate-spin rounded-full" style={{ animationDuration: "3s" }}>
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 h-2 w-2 rounded-full bg-primary glow-cyan-strong" />
            </div>
            <div className="absolute inset-4 animate-spin rounded-full" style={{ animationDuration: "2s", animationDirection: "reverse" }}>
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 h-1.5 w-1.5 rounded-full bg-chart-2" />
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <Atom className="h-8 w-8 text-primary animate-pulse" />
            </div>
          </div>

          {/* Pipeline visualization */}
          <div className="flex items-center gap-3">
            <PipelineStep icon={<Database className="h-4 w-4" />} label="Parse" active={pipelineStep === 1} done={pipelineStep > 1} />
            <div className={`h-px w-6 transition-colors duration-500 ${pipelineStep > 1 ? "bg-primary/50" : "bg-border"}`} />
            <PipelineStep icon={<BarChart3 className="h-4 w-4" />} label="Analyze" active={pipelineStep === 2} done={pipelineStep > 2} />
            <div className={`h-px w-6 transition-colors duration-500 ${pipelineStep > 2 ? "bg-primary/50" : "bg-border"}`} />
            <PipelineStep icon={<Cpu className="h-4 w-4" />} label="Decompose" active={pipelineStep === 3} done={pipelineStep > 3} />
            <div className={`h-px w-6 transition-colors duration-500 ${pipelineStep > 3 ? "bg-primary/50" : "bg-border"}`} />
            <PipelineStep icon={<Atom className="h-4 w-4" />} label="Forecast" active={pipelineStep === 4} done={pipelineStep > 4} />
          </div>

          <p className="font-mono text-xs tracking-[0.3em] text-primary/70 uppercase">
            Initializing temporal analysis engine
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-dot-grid opacity-50" />
      {Array.from({ length: 15 }).map((_, i) => (
        <FloatingParticle key={i} delay={Math.random() * 8} x={Math.random() * 100} duration={Math.random() * 5 + 4} />
      ))}

      {/* Radial gradient */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[600px] w-[600px] rounded-full bg-primary/[0.03] blur-3xl pointer-events-none" />

      <div className="relative z-10 flex flex-col items-center gap-14 px-6 w-full max-w-2xl">
        {/* Branding */}
        <div className="flex flex-col items-center gap-5 animate-float-up" style={{ animationDelay: "0ms" }}>
          <div className="relative">
            <div className="h-14 w-14 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center backdrop-blur-sm">
              <Atom className="h-7 w-7 text-primary" />
            </div>
            <div className="absolute -inset-3 rounded-2xl bg-primary/5 animate-breathe -z-10" />
            <div className="absolute -inset-6 rounded-3xl bg-primary/[0.02] animate-breathe -z-20" style={{ animationDelay: "1s" }} />
          </div>
          <div className="flex flex-col items-center gap-2">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              Chronos<span className="text-primary glow-text-cyan">Lab</span>
            </h1>
            <p className="font-mono text-xs tracking-[0.25em] text-muted-foreground uppercase">
              Temporal Intelligence Engine
            </p>
          </div>
        </div>

        {/* Drop Zone */}
        <div
          className="animate-float-up w-full"
          style={{ animationDelay: "100ms" }}
        >
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            className={`relative flex w-full cursor-pointer flex-col items-center gap-8 rounded-2xl border-2 border-dashed p-14 transition-all duration-300 noise-bg ${
              isDragging
                ? "border-primary bg-primary/[0.03] glow-cyan-strong scale-[1.01]"
                : "border-border/60 hover:border-primary/40 hover:bg-card/30"
            }`}
            onClick={triggerFileInput}
            role="button"
            tabIndex={0}
            aria-label="Upload CSV file"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                triggerFileInput()
              }
            }}
          >
            {isDragging && (
              <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
                <div className="absolute inset-x-0 h-px bg-primary/30 animate-scan-line" />
              </div>
            )}

            <div className="relative">
              <div className="flex h-20 w-20 items-center justify-center rounded-2xl border border-border/60 bg-secondary/50 backdrop-blur-sm">
                <Upload className="h-8 w-8 text-muted-foreground" />
              </div>
              {isDragging && <div className="absolute -inset-2 rounded-2xl border border-primary/20 animate-pulse" />}
            </div>

            <div className="flex flex-col items-center gap-2">
              <p className="text-lg font-medium text-foreground">
                Drop your dataset here
              </p>
              <p className="text-sm text-muted-foreground">
                or click to browse your files
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 rounded-md bg-secondary/80 px-3 py-1.5 border border-border/50">
                <FileText className="h-3 w-3 text-muted-foreground" />
                <span className="font-mono text-[10px] text-muted-foreground">.csv</span>
              </div>
              <div className="flex items-center gap-1.5 rounded-md bg-secondary/80 px-3 py-1.5 border border-border/50">
                <span className="font-mono text-[10px] text-muted-foreground">auto-detect columns</span>
              </div>
            </div>
          </div>
        </div>

        {/* OR divider */}
        <div className="flex items-center gap-4 w-full animate-float-up" style={{ animationDelay: "200ms" }}>
          <div className="h-px flex-1 bg-border/50" />
          <span className="text-[10px] text-muted-foreground font-mono uppercase tracking-[0.3em]">or</span>
          <div className="h-px flex-1 bg-border/50" />
        </div>

        {/* Sample data button */}
        <button
          onClick={loadSampleData}
          className="animate-float-up group flex items-center gap-4 rounded-xl border border-border/60 bg-card/50 px-8 py-4 transition-all hover:border-primary/30 hover:bg-card/80 hover:glow-cyan backdrop-blur-sm card-hover"
          style={{ animationDelay: "300ms" }}
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 border border-primary/20 group-hover:bg-primary/15 transition-colors">
            <Database className="h-4 w-4 text-primary" />
          </div>
          <div className="flex flex-col items-start gap-0.5">
            <span className="text-sm font-medium text-foreground">Load sample dataset</span>
            <span className="font-mono text-[10px] text-muted-foreground">730 days weather data</span>
          </div>
        </button>

        {/* Feature pills */}
        <div className="flex flex-wrap items-center justify-center gap-3 animate-float-up" style={{ animationDelay: "400ms" }}>
          {["Stationarity Testing", "Seasonal Decomposition", "Holt-Winters Forecast", "Distribution Analysis"].map(
            (feat) => (
              <span
                key={feat}
                className="rounded-full border border-border/40 bg-secondary/30 px-3 py-1 font-mono text-[9px] text-muted-foreground/70 uppercase tracking-wider"
              >
                {feat}
              </span>
            )
          )}
        </div>
      </div>
    </div>
  )
}
