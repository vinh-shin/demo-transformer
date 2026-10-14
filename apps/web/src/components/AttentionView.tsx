import { useMemo, useState } from 'react'
import { BRAND_RGB, dec } from '../constants'
import { Select } from './ui'

const AVG = -1

function averageHeads(heads: number[][][]): number[][] {
  const T = heads[0].length
  return Array.from({ length: T }, (_, q) =>
    Array.from({ length: T }, (_, k) => heads.reduce((s, h) => s + h[q][k], 0) / heads.length),
  )
}

const display = (t: string) => t.replaceAll('_', ' ')

export function AttentionView({
  tokens,
  isUnk,
  attention,
}: {
  tokens: string[]
  isUnk: boolean[]
  attention: number[][][][]
}) {
  const nLayers = attention.length
  const nHeads = attention[0].length
  const [layer, setLayer] = useState(nLayers - 1)
  const [head, setHead] = useState(AVG)
  const [hover, setHover] = useState<[number, number] | null>(null)

  const matrix = useMemo(
    () => (head === AVG ? averageHeads(attention[layer]) : attention[layer][head]),
    [attention, layer, head],
  )
  // Mức attention trung bình mỗi token nhận được từ các token khác (trung bình theo cột).
  const received = useMemo(() => {
    const col = tokens.map((_, k) => matrix.reduce((s, row) => s + row[k], 0) / matrix.length)
    const max = Math.max(...col)
    return col.map((v) => v / max)
  }, [matrix, tokens])
  const max = useMemo(() => Math.max(...matrix.flat()), [matrix])
  const cell = tokens.length > 30 ? 'size-3.5' : tokens.length > 16 ? 'size-5' : 'size-8'

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h3 className="text-lg font-bold">🔎 Self-attention</h3>
        <div className="flex gap-3">
          <Select
            label="Layer"
            value={layer}
            onChange={setLayer}
            options={Array.from({ length: nLayers }, (_, i) => ({ value: i, label: `Layer ${i + 1}` }))}
          />
          <Select
            label="Head"
            value={head}
            onChange={setHead}
            options={[
              { value: AVG, label: 'Trung bình các head' },
              ...Array.from({ length: nHeads }, (_, i) => ({ value: i, label: `Head ${i + 1}` })),
            ]}
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 leading-loose">
        {tokens.map((t, i) => (
          <span
            key={i}
            title={isUnk[i] ? 'Không có trong vocab → <unk>' : `Attention nhận được: ${dec(received[i], 2)}`}
            className={`rounded-md px-2 py-0.5 ${isUnk[i] ? 'border-2 border-dashed border-slate-400' : 'border border-brand-600/20'}`}
            style={{ background: `rgba(${BRAND_RGB},${0.06 + 0.7 * received[i]})`, color: received[i] > 0.6 ? 'white' : undefined }}
          >
            {display(t)}
          </span>
        ))}
      </div>
      <p className="text-xs text-slate-500">
        Nền càng đậm thì token càng được các token khác chú ý nhiều. Viền đứt là từ không có trong vocab (mã hoá thành{' '}
        <code>&lt;unk&gt;</code>).
      </p>

      <div className="overflow-x-auto pb-2">
        <div className="inline-grid gap-px" style={{ gridTemplateColumns: `auto repeat(${tokens.length}, max-content)` }}>
          <div />
          {tokens.map((t, k) => (
            // Nhãn đặt absolute để không làm cột rộng ra theo độ dài chữ.
            <div key={k} className="relative h-24">
              <span
                className={`absolute bottom-1 left-1/2 origin-bottom-left -rotate-60 text-xs whitespace-nowrap ${hover?.[1] === k ? 'font-bold text-brand-600' : 'text-slate-500'}`}
              >
                {display(t)}
              </span>
            </div>
          ))}
          {matrix.map((row, q) => (
            <div key={q} className="contents">
              <div className={`pr-2 text-right text-xs whitespace-nowrap ${hover?.[0] === q ? 'font-bold text-brand-600' : 'text-slate-500'}`}>
                {display(tokens[q])}
              </div>
              {row.map((v, k) => (
                <div
                  key={k}
                  className={`${cell} rounded-[2px] ${hover?.[0] === q && hover?.[1] === k ? 'ring-2 ring-amber-400' : ''}`}
                  style={{ background: `rgba(${BRAND_RGB},${0.04 + 0.96 * (v / max)})` }}
                  onMouseEnter={() => setHover([q, k])}
                  onMouseLeave={() => setHover(null)}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      <p className="min-h-5 text-sm tabular-nums">
        {hover ? (
          <>
            Query <b>{display(tokens[hover[0]])}</b> chú ý vào Key <b>{display(tokens[hover[1]])}</b>: trọng số{' '}
            <b>{dec(matrix[hover[0]][hover[1]], 3)}</b>
          </>
        ) : (
          <span className="text-slate-500">
            Rê chuột lên ô để xem trọng số. Mỗi hàng là phân phối softmax của một query trên toàn bộ key (tổng hàng bằng
            1). Token <code>&lt;pad&gt;</code> đã bị mask nên không xuất hiện.
          </span>
        )}
      </p>
    </div>
  )
}
