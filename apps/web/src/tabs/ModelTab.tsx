import type { ModelInfo } from '../api'
import { ConfusionMatrix } from '../components/charts'
import { LabelBadge } from '../components/ui'
import { dec, LABEL_VI, LABELS, num } from '../constants'

export function ModelTab({ info }: { info: ModelInfo }) {
  const { config, n_params, test_metrics: m } = info
  const arch: [string, string][] = [
    ['Số encoder layer', String(config.n_layers)],
    ['Số attention head', String(config.n_heads)],
    ['d_model', String(config.d_model)],
    ['d_ff (feed-forward)', String(config.d_ff)],
    ['Độ dài tối đa', `${config.max_len} token`],
    ['Kích thước vocab', num(config.vocab_size)],
    ['Tổng tham số', num(n_params)],
  ]
  const steps = [
    'Làm sạch',
    'Tách từ (underthesea)',
    'Mã hoá token',
    'Embedding + Positional Encoding',
    `${config.n_layers} × Encoder (Multi-head Attention + FFN)`,
    'Mean pooling',
    'Linear + Softmax',
  ]

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="card space-y-5">
        <section>
          <h3 className="mb-2 font-bold">Kiến trúc</h3>
          <table className="w-full text-sm">
            <tbody>
              {arch.map(([k, v]) => (
                <tr key={k} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
                  <td className="py-1.5 text-slate-500">{k}</td>
                  <td className="py-1.5 text-right font-semibold tabular-nums">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section>
          <h3 className="mb-2 font-bold">Pipeline xử lý</h3>
          <ol className="flex flex-wrap items-center gap-1.5 text-sm">
            {steps.map((s, i) => (
              <li key={s} className="flex items-center gap-1.5">
                <span className="rounded-md border border-brand-600/30 bg-brand-600/10 px-2.5 py-1">{s}</span>
                {i < steps.length - 1 && <span className="text-slate-400">→</span>}
              </li>
            ))}
          </ol>
        </section>

        <section>
          <h3 className="mb-2 font-bold">Nhãn</h3>
          <ul className="space-y-1.5 text-sm">
            {LABELS.map((l) => (
              <li key={l} className="flex items-center gap-2">
                <LabelBadge label={l} size="sm" /> {LABEL_VI[l]}
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="card space-y-4">
        <h3 className="font-bold">Kết quả trên tập test ({num(m.n)} bình luận)</h3>
        <table className="w-full text-sm">
          <thead className="text-slate-500">
            <tr className="border-b border-slate-200 dark:border-slate-800">
              <th className="py-1.5 text-left font-medium">Lớp</th>
              <th className="py-1.5 text-right font-medium">Precision</th>
              <th className="py-1.5 text-right font-medium">Recall</th>
              <th className="py-1.5 text-right font-medium">F1</th>
              <th className="py-1.5 text-right font-medium">Support</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {m.per_class.map((c) => (
              <tr key={c.label} className="border-b border-slate-100 dark:border-slate-800">
                <td className="py-1.5">
                  <LabelBadge label={c.label} size="sm" />
                </td>
                <td className="py-1.5 text-right">{dec(c.precision, 4)}</td>
                <td className="py-1.5 text-right">{dec(c.recall, 4)}</td>
                <td className="py-1.5 text-right">{dec(c.f1, 4)}</td>
                <td className="py-1.5 text-right">{num(c.support)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <ConfusionMatrix cm={m.confusion_matrix} />
      </div>
    </div>
  )
}
