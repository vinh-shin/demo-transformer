import { useEffect, useState } from 'react'
import { api, type Health, type ModelInfo } from './api'
import { MetricCard, Notice, Spinner } from './components/ui'
import { dec, pct, TEAM } from './constants'
import { ChatTab } from './tabs/ChatTab'
import { ModelTab } from './tabs/ModelTab'
import { SamplesTab } from './tabs/SamplesTab'

const TABS = [
  { id: 'chat', label: '🗨️ Mô phỏng bình luận' },
  { id: 'samples', label: '🎲 Ví dụ từ tập test' },
  { id: 'model', label: '🧠 Mô hình' },
] as const
type TabId = (typeof TABS)[number]['id']

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.theme = dark ? 'dark' : 'light'
  }, [dark])
  return [dark, setDark] as const
}

/** Hỏi /api/health đến khi model sẵn sàng, rồi lấy thông tin model và kết quả tập test. */
function useBackend() {
  const [health, setHealth] = useState<Health | null>(null)
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let stop = false
    let timer: number
    const poll = async () => {
      try {
        const h = await api.health()
        if (stop) return
        setHealth(h)
        setError(null)
        if (h.error) return
        if (h.metrics_ready) {
          setInfo(await api.model())
          return
        }
      } catch {
        if (!stop) setError('Không kết nối được API. Hãy chạy `make start` (hoặc `make api`) ở thư mục gốc project.')
      }
      timer = window.setTimeout(poll, 1500)
    }
    poll()
    return () => {
      stop = true
      clearTimeout(timer)
    }
  }, [])
  return { health, info, error: error ?? health?.error ?? null }
}

function Sidebar() {
  return (
    <aside className="space-y-4">
      <div className="text-center">
        <img src="/hcmute_logo.png" alt="Logo HCM-UTE" className="mx-auto w-24" />
        <p className="mt-2 text-xs leading-snug">
          <b>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM</b>
          <br />
          Khoa Công nghệ Thông tin
        </p>
      </div>
      <div className="rounded-lg border-l-4 border-brand-600 bg-brand-600/8 p-3 text-sm">
        <div className="mb-1.5 text-base font-extrabold">👥 {TEAM.name}</div>
        {TEAM.members.map((m) => (
          <div key={m.id} className="flex justify-between py-0.5">
            <span>{m.name}</span>
            <span className="font-mono text-slate-500">{m.id}</span>
          </div>
        ))}
        <div className="mt-2 border-t border-brand-600/25 pt-2">
          <b>GVHD:</b> {TEAM.supervisor}
        </div>
      </div>
    </aside>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>('chat')
  const [dark, setDark] = useTheme()
  const { health, info, error } = useBackend()
  const ready = !!health?.ready && !error
  const m = info?.test_metrics

  return (
    <div className="mx-auto grid max-w-[1400px] gap-6 p-4 md:p-6 lg:grid-cols-[16rem_1fr]">
      <div className="lg:sticky lg:top-6 lg:self-start">
        <Sidebar />
      </div>

      <main className="min-w-0 space-y-5">
        <header className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-extrabold md:text-3xl">🛡️ Phát hiện bình luận độc hại tiếng Việt</h1>
            <p className="mt-1 text-slate-500">
              Ứng dụng kỹ thuật Transformer · Transformer Encoder tự cài đặt, huấn luyện từ đầu trên dataset ViHSD
            </p>
          </div>
          <div className="flex gap-2">
            <a className="btn-ghost" href="/docs" target="_blank" rel="noreferrer">
              📘 API docs
            </a>
            <button className="btn-ghost" onClick={() => setDark(!dark)} aria-label="Đổi giao diện sáng/tối">
              {dark ? '☀️ Sáng' : '🌙 Tối'}
            </button>
          </div>
        </header>

        {error && <Notice tone="error">{error}</Notice>}

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricCard label="Accuracy (test)" value={m && pct(m.accuracy, 2)} loading={!m} />
          <MetricCard label="Macro-F1 (test)" value={m && dec(m.macro_f1, 4)} loading={!m} />
          <MetricCard label="Layer × Head" value={info && `${info.config.n_layers} × ${info.config.n_heads}`} loading={!info} />
          <MetricCard label="Số tham số" value={info && `${dec(info.n_params / 1e6, 2)}M`} loading={!info} />
        </div>

        <nav className="flex gap-1 overflow-x-auto border-b border-slate-200 dark:border-slate-800">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px cursor-pointer border-b-2 px-4 py-2.5 text-sm font-medium whitespace-nowrap transition ${
                tab === t.id
                  ? 'border-brand-600 text-brand-600 dark:text-brand-500'
                  : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {!ready ? (
          !error && (
            <div className="card flex items-center gap-3 text-slate-500">
              <Spinner /> Đang nạp model...
            </div>
          )
        ) : (
          <>
            {/* Giữ các tab trong DOM để không mất kết quả khi chuyển tab. */}
            <div hidden={tab !== 'chat'}>
              <ChatTab />
            </div>
            <div hidden={tab !== 'samples'}>
              <SamplesTab />
            </div>
            <div hidden={tab !== 'model'}>
              {info ? (
                <ModelTab info={info} />
              ) : (
                <div className="card flex items-center gap-3 text-slate-500">
                  <Spinner /> Đang đánh giá model trên tập test (chỉ chạy lần đầu)...
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  )
}
