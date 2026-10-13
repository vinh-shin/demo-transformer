import type { Label } from './constants'

export interface Health {
  ready: boolean
  metrics_ready: boolean
  error: string | null
}

export interface ClassMetric {
  label: Label
  precision: number
  recall: number
  f1: number
  support: number
}

export interface Metrics {
  n: number
  accuracy: number
  macro_f1: number
  per_class: ClassMetric[]
  confusion_matrix: number[][]
}

export interface ModelInfo {
  config: {
    vocab_size: number
    d_model: number
    n_heads: number
    d_ff: number
    n_layers: number
    max_len: number
  }
  n_params: number
  labels: Label[]
  test_metrics: Metrics
}

export type Analysis =
  | { empty: true }
  | {
      empty: false
      prediction: Label
      probs: number[]
      cleaned: string // sau làm sạch
      segmented: string // sau tách từ (đầy đủ, chưa cắt MAX_LEN)
      n_tokens: number
      max_len: number
      tokens: string[] // đã cắt còn tối đa max_len
      is_unk: boolean[]
      attention: number[][][][] // layer × head × query × key
    }

export interface PredRow {
  prediction: Label | null
  probs: number[] | null
  true_label?: Label | null
  correct?: boolean | null
  text?: string
  data?: Record<string, unknown>
}

export interface BatchSummary {
  total: number
  empty: number
  counts: Record<Label, number>
  metrics: Metrics | null
}

export interface BatchResult {
  rows: PredRow[]
  summary: BatchSummary
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let msg = `Lỗi ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') msg = body.detail
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      /* phản hồi không phải JSON */
    }
    throw new Error(msg)
  }
  return res.json()
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  health: () => request<Health>('/api/health'),
  model: () => request<ModelInfo>('/api/model'),
  analyze: (text: string) => request<Analysis>('/api/analyze', json({ text })),
  samples: (n: number, onlyToxic: boolean) =>
    request<BatchResult>(`/api/samples?n=${n}&only_toxic=${onlyToxic}`),
}
