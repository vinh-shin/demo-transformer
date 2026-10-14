import type { Analysis } from '../api'
import { EMPTY_MSG, pct, type Label } from '../constants'
import { useAnalysis } from '../hooks'
import { AttentionView } from './AttentionView'
import { ProbBars } from './charts'
import { LabelBadge, Notice, Spinner } from './ui'

export function PredictionView({ data, trueLabel }: { data: Extract<Analysis, { empty: false }>; trueLabel?: Label | null }) {
  const conf = Math.max(...data.probs)
  return (
    <div className="space-y-4">
      <LabelBadge label={data.prediction} size="lg" />
      <p className="text-sm">
        Độ tin cậy <b>{pct(conf)}</b>
        {trueLabel && (
          <>
            {' · '}Nhãn thật <b>{trueLabel}</b> {trueLabel === data.prediction ? '✅ Đúng' : '❌ Sai'}
          </>
        )}
      </p>
      <ProbBars probs={data.probs} />
    </div>
  )
}

/** Kết quả dự đoán + attention xếp dọc, dùng cho tab ví dụ từ tập test. */
export function FullAnalysis({ text, trueLabel }: { text: string; trueLabel?: Label | null }) {
  const { data, error, loading } = useAnalysis(text)
  if (loading)
    return (
      <div className="card flex items-center gap-3 text-slate-500">
        <Spinner /> Đang phân tích...
      </div>
    )
  if (error) return <Notice tone="error">{error}</Notice>
  if (!data) return null
  if (data.empty) return <Notice tone="warn">{EMPTY_MSG}</Notice>
  return (
    <div className="space-y-4">
      <div className="card">
        <PredictionView data={data} trueLabel={trueLabel} />
      </div>
      <div className="card">
        <AttentionView tokens={data.tokens} isUnk={data.is_unk} attention={data.attention} />
      </div>
    </div>
  )
}
