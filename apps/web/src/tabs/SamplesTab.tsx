import { useCallback, useEffect, useState } from 'react'
import { api, type BatchResult } from '../api'
import { FullAnalysis } from '../components/AnalysisPanel'
import { LabelBadge, Notice, Spinner } from '../components/ui'
import { cssWidth, pct } from '../constants'

export function SamplesTab() {
  const [n, setN] = useState(10)
  const [onlyToxic, setOnlyToxic] = useState(false)
  const [result, setResult] = useState<BatchResult | null>(null)
  const [selected, setSelected] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSamples = useCallback(async (n: number, onlyToxic: boolean) => {
    try {
      setResult(await api.samples(n, onlyToxic))
      setSelected(0)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  const load = (n: number, onlyToxic: boolean) => {
    setLoading(true)
    setError(null)
    fetchSamples(n, onlyToxic)
  }

  // Lấy mẫu lần đầu khi mở tab.
  useEffect(() => {
    fetchSamples(10, false)
  }, [fetchSamples])

  const rows = result?.rows ?? []
  const nCorrect = rows.filter((r) => r.correct).length
  const current = rows[selected]

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-end gap-4">
        <label className="block w-28">
          <span className="label">Số câu</span>
          <input
            type="number"
            className="field"
            min={5}
            max={50}
            value={n}
            onChange={(e) => setN(Math.min(50, Math.max(5, Number(e.target.value) || 5)))}
          />
        </label>
        <label
          className="flex cursor-pointer items-center gap-2 pb-2 text-sm"
          title="CLEAN chiếm khoảng 83% tập test nên lấy ngẫu nhiên thường ra toàn câu sạch"
        >
          <input type="checkbox" className="size-4 accent-brand-600" checked={onlyToxic} onChange={(e) => setOnlyToxic(e.target.checked)} />
          Chỉ lấy câu độc hại (OFFENSIVE, HATE)
        </label>
        <button className="btn-primary ml-auto" onClick={() => load(n, onlyToxic)} disabled={loading}>
          {loading ? <Spinner className="size-4" /> : '🎲'} Lấy mẫu ngẫu nhiên
        </button>
      </div>

      {error && <Notice tone="error">{error}</Notice>}

      {result && (
        <div className="card overflow-hidden p-0">
          <p className="border-b border-slate-200 px-4 py-3 text-sm dark:border-slate-800">
            Đúng <b>
              {nCorrect}/{rows.length}
            </b>{' '}
            câu. Bấm vào một dòng để xem phân tích chi tiết.
          </p>
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-slate-50 text-left text-slate-500 dark:bg-slate-900">
                <tr>
                  <th className="px-4 py-2 font-medium">Bình luận</th>
                  <th className="px-2 py-2 font-medium">Nhãn thật</th>
                  <th className="px-2 py-2 font-medium">Dự đoán</th>
                  <th className="w-36 px-2 py-2 font-medium">Tin cậy</th>
                  <th className="px-4 py-2 text-center font-medium">Kết quả</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const conf = r.probs ? Math.max(...r.probs) : null
                  return (
                    <tr
                      key={i}
                      onClick={() => setSelected(i)}
                      className={`cursor-pointer border-t border-slate-100 dark:border-slate-800 ${
                        i === selected ? 'bg-brand-50 dark:bg-brand-700/15' : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                      }`}
                    >
                      <td className="px-4 py-2">{r.text}</td>
                      <td className="px-2 py-2">{r.true_label && <LabelBadge label={r.true_label} size="sm" />}</td>
                      <td className="px-2 py-2">{r.prediction ? <LabelBadge label={r.prediction} size="sm" /> : 'Trống'}</td>
                      <td className="px-2 py-2">
                        {conf !== null && (
                          <div className="flex items-center gap-2">
                            <div className="h-2 flex-1 rounded-full bg-slate-100 dark:bg-slate-800">
                              <div className="h-full rounded-full bg-brand-600" style={{ width: cssWidth(conf) }} />
                            </div>
                            <span className="w-12 text-right tabular-nums">{pct(conf)}</span>
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-2 text-center">{r.correct == null ? '' : r.correct ? '✅' : '❌'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {current?.text && (
        <>
          <p className="text-sm">
            <b>Câu đang phân tích:</b> {current.text}
          </p>
          <FullAnalysis text={current.text} trueLabel={current.true_label} />
        </>
      )}
    </div>
  )
}
