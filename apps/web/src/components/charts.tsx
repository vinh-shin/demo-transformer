import { BRAND_RGB, cssWidth, LABEL_COLOR, LABEL_VI, LABELS, num, pct } from '../constants'

/** Thanh xác suất ngang cho 3 lớp. */
export function ProbBars({ probs }: { probs: number[] }) {
  return (
    <div className="space-y-2">
      {LABELS.map((l, i) => (
        <div key={l} className="grid grid-cols-[9.5rem_1fr_3.5rem] items-center gap-3 text-sm">
          <span className="truncate">
            <b>{l}</b> <span className="text-slate-500">· {LABEL_VI[l]}</span>
          </span>
          <div className="h-3 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
            <div
              className="h-full rounded-full transition-[width] duration-500"
              style={{ width: cssWidth(probs[i]), background: LABEL_COLOR[l] }}
            />
          </div>
          <span className="text-right tabular-nums">{pct(probs[i])}</span>
        </div>
      ))}
    </div>
  )
}

/** Ma trận nhầm lẫn: hàng = thực tế, cột = dự đoán. */
export function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const max = Math.max(1, ...cm.flat())
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th className="px-2 text-left text-xs font-normal text-slate-500">Thực tế ↓ / Dự đoán →</th>
            {LABELS.map((l) => (
              <th key={l} className="px-2 pb-1 font-semibold">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={LABELS[i]}>
              <th className="px-2 text-right font-semibold">{LABELS[i]}</th>
              {row.map((v, j) => {
                const a = v / max
                return (
                  <td
                    key={j}
                    className={`h-14 min-w-20 rounded-md text-center font-semibold tabular-nums ${i === j ? 'ring-2 ring-brand-600/40' : ''}`}
                    style={{ background: `rgba(${BRAND_RGB},${0.06 + 0.84 * a})`, color: a > 0.5 ? 'white' : undefined }}
                  >
                    {num(v)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
