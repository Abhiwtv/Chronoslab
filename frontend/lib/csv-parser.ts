export interface ParsedData {
  headers: string[]
  rows: Record<string, string | number>[]
  datetimeCol: string
  targetCol: string
  numericCols: string[]
  timeSeries: { date: string; value: number; timestamp: number }[]
  stats: {
    count: number
    mean: number
    std: number
    min: number
    max: number
    median: number
    q1: number
    q3: number
    skewness: number
    trend: "upward" | "downward" | "stationary"
    volatility: number
    missingPercent: number
  }
  frequency: string
  seasonal: boolean
  seasonalStrength: number
  period: number | null
  stationarity: { adfStat: number; pValue: number; isStationary: boolean }
  decomposition: {
    trend: { date: string; value: number }[]
    seasonal: { date: string; value: number }[]
    residual: { date: string; value: number }[]
  }
  forecast: {
    train: { date: string; value: number }[]
    test: { date: string; actual: number; predicted: number }[]
    future: { date: string; predicted: number; lower: number; upper: number }[]
    model: string
    smape: number
    mape?: number            
    differencing: number
    seasonal_order?: number[] 
    seasonal_differencing?: number
    // ADD THESE TWO LINES 👇
    vol_mae_bps?: number;
    cov_2sig?: number;
  }
}


function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ""
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    if (char === '"') {
      inQuotes = !inQuotes
    } else if (char === "," && !inQuotes) {
      result.push(current.trim())
      current = ""
    } else {
      current += char
    }
  }
  result.push(current.trim())
  return result
}

function isDateLike(val: string): boolean {
  if (!val || val.length < 6) return false
  const d = new Date(val)
  return !isNaN(d.getTime()) && d.getFullYear() > 1900 && d.getFullYear() < 2100
}

function inferFrequency(dates: Date[]): string {
  if (dates.length < 3) return "Unknown"
  const diffs: number[] = []
  for (let i = 1; i < Math.min(dates.length, 50); i++) {
    diffs.push(dates[i].getTime() - dates[i - 1].getTime())
  }
  const medianDiff = diffs.sort((a, b) => a - b)[Math.floor(diffs.length / 2)]
  const hours = medianDiff / (1000 * 60 * 60)
  if (hours < 0.1) return "Minutes"
  if (hours < 2) return "Hourly"
  if (hours < 36) return "Daily"
  if (hours < 200) return "Weekly"
  if (hours < 1000) return "Monthly"
  if (hours < 3000) return "Quarterly"
  return "Yearly"
}

function computeStats(values: number[]) {
  const n = values.length
  const sorted = [...values].sort((a, b) => a - b)
  const mean = values.reduce((s, v) => s + v, 0) / n
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / n
  const std = Math.sqrt(variance)
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)]
  const q1 = sorted[Math.floor(n * 0.25)]
  const q3 = sorted[Math.floor(n * 0.75)]
  const m3 = values.reduce((s, v) => s + ((v - mean) / (std || 1)) ** 3, 0) / n

  // trend
  const xMean = (n - 1) / 2
  let num = 0, den = 0
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (values[i] - mean)
    den += (i - xMean) ** 2
  }
  const slope = den ? num / den : 0
  const trendDir: "upward" | "downward" | "stationary" =
    Math.abs(slope) < std * 0.01 ? "stationary" : slope > 0 ? "upward" : "downward"

  // volatility
  const returns: number[] = []
  for (let i = 1; i < n; i++) {
    if (values[i - 1] !== 0) returns.push((values[i] - values[i - 1]) / Math.abs(values[i - 1]))
  }
  const retMean = returns.reduce((s, v) => s + v, 0) / (returns.length || 1)
  const volatility = Math.sqrt(returns.reduce((s, v) => s + (v - retMean) ** 2, 0) / (returns.length || 1))

  return {
    count: n,
    mean,
    std,
    min: sorted[0],
    max: sorted[n - 1],
    median,
    q1,
    q3,
    skewness: m3,
    trend: trendDir,
    volatility,
    missingPercent: 0,
  }
}

function simpleSeasonalDecompose(values: number[], period: number) {
  const n = values.length
  // Trend via moving average
  const trend: (number | null)[] = new Array(n).fill(null)
  const half = Math.floor(period / 2)
  for (let i = half; i < n - half; i++) {
    let sum = 0
    for (let j = i - half; j <= i + half; j++) sum += values[j]
    trend[i] = sum / period
  }
  // Fill edges
  for (let i = 0; i < half; i++) trend[i] = trend[half]
  for (let i = n - half; i < n; i++) trend[i] = trend[n - half - 1]

  // Seasonal
  const seasonal = new Array(n).fill(0)
  const detrended = values.map((v, i) => v - (trend[i] ?? v))
  const seasonalAvg = new Array(period).fill(0)
  const seasonalCount = new Array(period).fill(0)
  for (let i = 0; i < n; i++) {
    seasonalAvg[i % period] += detrended[i]
    seasonalCount[i % period]++
  }
  for (let i = 0; i < period; i++) {
    seasonalAvg[i] /= seasonalCount[i] || 1
  }
  for (let i = 0; i < n; i++) {
    seasonal[i] = seasonalAvg[i % period]
  }

  // Residual
  const residual = values.map((v, i) => v - (trend[i] ?? v) - seasonal[i])

  return { trend: trend as number[], seasonal, residual }
}

function detectSeasonality(values: number[]): { seasonal: boolean; strength: number; period: number | null } {
  const n = values.length
  if (n < 10) return { seasonal: false, strength: 0, period: null }

  // Simple ACF-based detection
  const mean = values.reduce((s, v) => s + v, 0) / n
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / n
  if (variance === 0) return { seasonal: false, strength: 0, period: null }

  const maxLag = Math.min(Math.floor(n / 2), 120)
  const acf: number[] = [1]
  for (let lag = 1; lag <= maxLag; lag++) {
    let sum = 0
    for (let i = 0; i < n - lag; i++) {
      sum += (values[i] - mean) * (values[i + lag] - mean)
    }
    acf.push(sum / (n * variance))
  }

  // Find peak
  let bestLag = 2, bestVal = -1
  for (let i = 2; i < acf.length; i++) {
    if (acf[i] > bestVal) { bestVal = acf[i]; bestLag = i }
  }

  const isSeasonal = bestVal > 0.25
  return { seasonal: isSeasonal, strength: Math.max(0, Math.min(1, bestVal)), period: isSeasonal ? bestLag : null }
}

function simpleADF(values: number[]): { adfStat: number; pValue: number; isStationary: boolean } {
  const n = values.length
  if (n < 10) return { adfStat: 0, pValue: 0.5, isStationary: false }
  const diffs = values.slice(1).map((v, i) => v - values[i])
  const lagged = values.slice(0, -1)
  const meanDiff = diffs.reduce((s, v) => s + v, 0) / diffs.length
  const meanLag = lagged.reduce((s, v) => s + v, 0) / lagged.length

  let num = 0, den = 0
  for (let i = 0; i < diffs.length; i++) {
    num += (lagged[i] - meanLag) * (diffs[i] - meanDiff)
    den += (lagged[i] - meanLag) ** 2
  }
  const rho = den ? num / den : 0
  const residuals = diffs.map((d, i) => d - meanDiff - rho * (lagged[i] - meanLag))
  const sigma = Math.sqrt(residuals.reduce((s, r) => s + r ** 2, 0) / (residuals.length - 1))
  const seRho = sigma / Math.sqrt(den || 1)
  const adfStat = seRho ? rho / seRho : 0

  // Approximate p-value
  let pValue = 0.5
  if (adfStat < -3.5) pValue = 0.01
  else if (adfStat < -2.9) pValue = 0.05
  else if (adfStat < -2.6) pValue = 0.1
  else pValue = 0.5 + Math.min(0.49, Math.max(0, (adfStat + 2.6) * 0.1))

  return { adfStat, pValue, isStationary: pValue < 0.05 }
}

function simpleForecast(
  values: number[],
  dates: string[],
  testSize: number = 0.2
) {
  const splitIdx = Math.floor(values.length * (1 - testSize))
  const trainVals = values.slice(0, splitIdx)
  const testVals = values.slice(splitIdx)
  const trainDates = dates.slice(0, splitIdx)
  const testDates = dates.slice(splitIdx)

  // Exponential smoothing (Holt's linear trend method)
  const alpha = 0.3
  const beta = 0.1

  let level = trainVals[0]
  let trend = trainVals.length > 1 ? trainVals[1] - trainVals[0] : 0

  const predictions: number[] = []
  for (let i = 0; i < testVals.length; i++) {
    const pred = level + (i + 1) * trend
    predictions.push(pred)
  }

  // Refine with simple walk-forward
  const refined: number[] = []
  let lvl = trainVals[trainVals.length - 1]
  let trd = (trainVals[trainVals.length - 1] - trainVals[Math.max(0, trainVals.length - 5)]) / 5

  for (let i = 0; i < testVals.length; i++) {
    const pred = lvl + trd
    refined.push(pred)
    lvl = alpha * testVals[i] + (1 - alpha) * (lvl + trd)
    trd = beta * (lvl - (i > 0 ? testVals[i - 1] : trainVals[trainVals.length - 1])) + (1 - beta) * trd
  }

  // sMAPE
  let smapeSum = 0
  for (let i = 0; i < testVals.length; i++) {
    const denom = Math.abs(testVals[i]) + Math.abs(refined[i])
    smapeSum += denom === 0 ? 0 : (2 * Math.abs(testVals[i] - refined[i])) / denom
  }
  const smape = (smapeSum / testVals.length) * 100

  // Future forecast
  const futureSteps = Math.max(5, Math.floor(testVals.length * 0.5))
  const lastVal = values[values.length - 1]
  const lastTrend = (values[values.length - 1] - values[Math.max(0, values.length - 10)]) / 10
  const residualStd = Math.sqrt(
    refined.reduce((s, v, i) => s + (v - testVals[i]) ** 2, 0) / refined.length
  )

  const lastDate = new Date(dates[dates.length - 1])
  const interval = dates.length > 1 ? new Date(dates[dates.length - 1]).getTime() - new Date(dates[dates.length - 2]).getTime() : 86400000

  const future: { date: string; predicted: number; lower: number; upper: number }[] = []
  for (let i = 1; i <= futureSteps; i++) {
    const predicted = lastVal + lastTrend * i
    const ci = residualStd * 1.96 * Math.sqrt(i)
    future.push({
      date: new Date(lastDate.getTime() + interval * i).toISOString().split("T")[0],
      predicted,
      lower: predicted - ci,
      upper: predicted + ci,
    })
  }

  return {
    train: trainDates.map((d, i) => ({ date: d, value: trainVals[i] })),
    test: testDates.map((d, i) => ({ date: d, actual: testVals[i], predicted: refined[i] })),
    future,
    model: "Holt-Winters ES",
    smape: smape, // 👈 MUST HAVE THIS
    mape: smape,
    differencing: 0,
    seasonal_order: null,
  }
}

export function parseCSV(csvText: string, targetColOverride?: string): ParsedData {
  const lines = csvText.trim().split("\n").filter(l => l.trim())
  if (lines.length < 3) throw new Error("CSV must have at least a header and 2 data rows")

  const headers = parseCSVLine(lines[0])
  const rows: Record<string, string | number>[] = []

  for (let i = 1; i < lines.length; i++) {
    const vals = parseCSVLine(lines[i])
    const row: Record<string, string | number> = {}
    headers.forEach((h, j) => {
      const val = vals[j] ?? ""
      const num = Number(val)
      row[h] = val !== "" && !isNaN(num) && !isDateLike(val) ? num : val
    })
    rows.push(row)
  }

  // Infer datetime column
  let datetimeCol = ""
  let bestRatio = 0
  for (const h of headers) {
    let dateCount = 0
    for (const r of rows.slice(0, 100)) {
      if (isDateLike(String(r[h]))) dateCount++
    }
    const ratio = dateCount / Math.min(rows.length, 100)
    if (ratio > bestRatio) { bestRatio = ratio; datetimeCol = h }
  }
  if (!datetimeCol) datetimeCol = headers[0]

  // Infer numeric columns
  const numericCols = headers.filter(h => {
    if (h === datetimeCol) return false
    let numCount = 0
    for (const r of rows.slice(0, 100)) {
      if (typeof r[h] === "number") numCount++
    }
    return numCount / Math.min(rows.length, 100) > 0.5
  })

  const targetCol = targetColOverride && numericCols.includes(targetColOverride) 
    ? targetColOverride 
    : (numericCols[0] || headers.find(h => h !== datetimeCol) || headers[1])

  // Build time series
  const timeSeries = rows
    .map(r => {
      const dateStr = String(r[datetimeCol])
      const d = new Date(dateStr)
      const val = Number(r[targetCol])
      if (isNaN(d.getTime()) || isNaN(val)) return null
      return { date: d.toISOString().split("T")[0], value: val, timestamp: d.getTime() }
    })
    .filter(Boolean) as { date: string; value: number; timestamp: number }[]

  timeSeries.sort((a, b) => a.timestamp - b.timestamp)

  const values = timeSeries.map(t => t.value)
  const dates = timeSeries.map(t => t.date)
  const dateObjects = timeSeries.map(t => new Date(t.timestamp))

  const stats = computeStats(values)
  stats.missingPercent = ((rows.length - timeSeries.length) / rows.length) * 100

  const frequency = inferFrequency(dateObjects)
  const { seasonal, strength: seasonalStrength, period } = detectSeasonality(values)
  const stationarity = simpleADF(values)

// Auto-differencing logic
let differencing = 0
let tempValues = [...values]
let testStationarity = stationarity

while (!testStationarity.isStationary && differencing < 2) {
  tempValues = tempValues.slice(1).map((v, i) => v - tempValues[i])
  testStationarity = simpleADF(tempValues)
  differencing++
}

const forecastBase = simpleForecast(values, dates)

  // 1. Explicitly build the forecast object to match your interface
  const forecast = {
    train: forecastBase.train,
    test: forecastBase.test,
    future: forecastBase.future,
    model: forecastBase.model,
    // Add both keys to ensure backward compatibility and new high-precision support
    smape: forecastBase.smape, 
    mape: forecastBase.smape, 
    differencing: differencing,
    seasonal_order: [] as number[],
    seasonal_differencing: 0
  }

  // 2. Decomposition (keep as is)
  const decomposePeriod = period || (frequency === "Monthly" ? 12 : frequency === "Daily" ? 7 : frequency === "Hourly" ? 24 : 4)
  const decomp = values.length > decomposePeriod * 2
    ? simpleSeasonalDecompose(values, decomposePeriod)
    : { trend: values, seasonal: values.map(() => 0), residual: values.map(() => 0) }

  const decomposition = {
    trend: dates.map((d, i) => ({ date: d, value: decomp.trend[i] })),
    seasonal: dates.map((d, i) => ({ date: d, value: decomp.seasonal[i] })),
    residual: dates.map((d, i) => ({ date: d, value: decomp.residual[i] })),
  }

  // 3. Final return
  return {
    headers,
    rows,
    datetimeCol,
    targetCol,
    numericCols,
    timeSeries,
    stats,
    frequency,
    seasonal,
    seasonalStrength,
    period,
    stationarity,
    decomposition,
    forecast // This now matches the interface perfectly
  }}