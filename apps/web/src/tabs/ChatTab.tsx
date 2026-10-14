import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { api } from '../api'
import { PredictionView } from '../components/AnalysisPanel'
import { AttentionView } from '../components/AttentionView'
import { Modal } from '../components/Modal'
import { LabelBadge, Notice, Spinner } from '../components/ui'
import { EMPTY_MSG, EXAMPLES, LABEL_COLOR, LABEL_VI, LABELS, num, type Label } from '../constants'
import { useAnalysis } from '../hooks'

interface ChatComment {
  id: string
  text: string
  time: number
  status: 'pending' | 'done' | 'empty' | 'error'
  prediction?: Label
  probs?: number[]
  error?: string
}

const STORE_KEY = 'vihsd-chat'
const HIDE_KEY = 'vihsd-chat-hide-toxic'
const MAX_CHARS = 2000
const QUICK = [...EXAMPLES, 'Đúng là đồ óc chó, nói mãi không hiểu', 'Mai đi cà phê không mọi người ơi']

function loadComments(): ChatComment[] {
  try {
    const saved: ChatComment[] = JSON.parse(localStorage.getItem(STORE_KEY) ?? '[]')
    // Bình luận đang chờ lúc đóng trang sẽ không bao giờ có kết quả, chuyển sang lỗi để thử lại.
    return saved.map((c) => (c.status === 'pending' ? { ...c, status: 'error', error: 'Bị gián đoạn khi tải lại trang.' } : c))
  } catch {
    return []
  }
}

const formatTime = (t: number) =>
  new Date(t).toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' })

const isToxic = (c: ChatComment) => c.prediction === 'OFFENSIVE' || c.prediction === 'HATE'

export function ChatTab() {
  const [comments, setComments] = useState<ChatComment[]>(loadComments)
  const [draft, setDraft] = useState('')
  const [hideToxic, setHideToxic] = useState(() => localStorage.getItem(HIDE_KEY) !== 'false')
  const [revealed, setRevealed] = useState<Set<string>>(new Set())
  const [openId, setOpenId] = useState<string | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Focus ô nhập nhưng không cuộn trang (autoFocus sẽ đẩy phần đầu trang ra khỏi màn hình).
  useEffect(() => inputRef.current?.focus({ preventScroll: true }), [])

  useEffect(() => localStorage.setItem(STORE_KEY, JSON.stringify(comments)), [comments])
  useEffect(() => localStorage.setItem(HIDE_KEY, String(hideToxic)), [hideToxic])

  // Cuộn xuống cuối khi có bình luận mới.
  const count = comments.length
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [count])

  useEffect(() => {
    if (!confirmClear) return
    const t = setTimeout(() => setConfirmClear(false), 3000)
    return () => clearTimeout(t)
  }, [confirmClear])

  const update = (id: string, patch: Partial<ChatComment>) =>
    setComments((cs) => cs.map((c) => (c.id === id ? { ...c, ...patch } : c)))

  const classify = async (id: string, text: string) => {
    update(id, { status: 'pending', error: undefined })
    try {
      const r = await api.analyze(text)
      update(id, r.empty ? { status: 'empty' } : { status: 'done', prediction: r.prediction, probs: r.probs })
    } catch (e) {
      update(id, { status: 'error', error: (e as Error).message })
    }
  }

  const send = (text = draft) => {
    const t = text.trim()
    if (!t) return
    const id = crypto.randomUUID()
    setComments((cs) => [...cs, { id, text: t, time: Date.now(), status: 'pending' }])
    if (text === draft) setDraft('')
    classify(id, t) // không chờ: người dùng có thể gửi tiếp ngay
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      send()
    }
  }

  const clear = () => {
    if (!confirmClear) return setConfirmClear(true)
    setComments([])
    setRevealed(new Set())
    setConfirmClear(false)
  }

  const counts = Object.fromEntries(LABELS.map((l) => [l, comments.filter((c) => c.prediction === l).length])) as Record<
    Label,
    number
  >
  const opened = comments.find((c) => c.id === openId)

  return (
    <div className="card flex flex-col p-0">
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <div className="mr-auto">
          <h3 className="font-bold">Mô phỏng khung bình luận</h3>
          <p className="text-xs text-slate-500">Mỗi bình luận được model gắn nhãn ngay khi gửi. Bấm vào bình luận để xem chi tiết.</p>
        </div>
        <div className="flex flex-wrap gap-2 text-sm">
          <span className="rounded-full bg-slate-100 px-2.5 py-0.5 dark:bg-slate-800">Tổng {num(comments.length)}</span>
          {LABELS.map((l) => (
            <span key={l} className="rounded-full px-2.5 py-0.5 text-white" style={{ background: LABEL_COLOR[l] }}>
              {l} {num(counts[l])}
            </span>
          ))}
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm select-none">
          <input
            type="checkbox"
            className="peer sr-only"
            checked={hideToxic}
            onChange={(e) => setHideToxic(e.target.checked)}
          />
          <span className="relative h-5 w-9 rounded-full bg-slate-300 transition peer-checked:bg-brand-600 after:absolute after:top-0.5 after:left-0.5 after:size-4 after:rounded-full after:bg-white after:transition peer-checked:after:translate-x-4 dark:bg-slate-700" />
          Ẩn bình luận độc hại
        </label>
        <button className="btn-ghost" onClick={clear} disabled={!comments.length}>
          {confirmClear ? 'Bấm lần nữa để xoá' : '🧹 Xoá lịch sử'}
        </button>
      </div>

      <div ref={listRef} className="h-[55vh] min-h-80 space-y-3 overflow-y-auto px-4 py-4">
        {comments.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-slate-500">
            <span className="text-4xl">💬</span>
            <p>Chưa có bình luận nào. Hãy nhập bình luận bên dưới hoặc chọn một câu thử nhanh.</p>
          </div>
        )}
        {comments.map((c) => {
          const hidden = hideToxic && isToxic(c) && !revealed.has(c.id)
          const clickable = c.status === 'done' || c.status === 'empty'
          return (
            <div key={c.id} className="flex gap-3">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-brand-600 font-bold text-white">B</div>
              <div className="min-w-0 flex-1">
                <div
                  role={clickable ? 'button' : undefined}
                  tabIndex={clickable ? 0 : undefined}
                  onClick={() => clickable && setOpenId(c.id)}
                  onKeyDown={(e) => clickable && e.key === 'Enter' && setOpenId(c.id)}
                  className={`inline-block max-w-full rounded-2xl rounded-tl-sm border-l-4 bg-slate-100 px-4 py-2 text-left dark:bg-slate-800 ${
                    clickable ? 'cursor-pointer transition hover:ring-2 hover:ring-brand-500/40' : ''
                  }`}
                  style={{ borderLeftColor: c.prediction ? LABEL_COLOR[c.prediction] : 'transparent' }}
                >
                  <div className="text-sm font-semibold">Bạn</div>
                  <p className={`break-words whitespace-pre-wrap ${hidden ? 'blur-sm select-none' : ''}`}>{c.text}</p>
                  {hidden && (
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                      ⚠️ Đã ẩn vì nội dung {LABEL_VI[c.prediction!].toLowerCase()}.{' '}
                      <button
                        className="cursor-pointer font-semibold text-brand-600 hover:underline dark:text-brand-500"
                        onClick={(e) => {
                          e.stopPropagation()
                          setRevealed((s) => new Set(s).add(c.id))
                        }}
                      >
                        Xem
                      </button>
                    </p>
                  )}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-2 pl-1 text-xs text-slate-500">
                  <span>{formatTime(c.time)}</span>
                  {c.status === 'pending' && (
                    <span className="flex items-center gap-1">
                      <Spinner className="size-3" /> Đang phân tích...
                    </span>
                  )}
                  {c.status === 'done' && c.prediction && (
                    <>
                      <LabelBadge label={c.prediction} size="sm" />
                      <span className="tabular-nums">{Math.round(Math.max(...c.probs!) * 100)}%</span>
                    </>
                  )}
                  {c.status === 'empty' && (
                    <span className="rounded-full bg-slate-400 px-2 py-0.5 font-bold text-white">TRỐNG</span>
                  )}
                  {c.status === 'error' && (
                    <span className="text-red-600">
                      Lỗi: {c.error}{' '}
                      <button className="cursor-pointer font-semibold underline" onClick={() => classify(c.id, c.text)}>
                        Thử lại
                      </button>
                    </span>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="space-y-2 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-slate-500">Thử nhanh:</span>
          {QUICK.map((q) => (
            <button key={q} className="btn-ghost px-2 py-0.5 text-xs" onClick={() => send(q)}>
              {q}
            </button>
          ))}
        </div>
        <div className="flex items-end gap-2">
          <textarea
            className="field min-h-11 resize-none"
            rows={Math.min(5, Math.max(1, draft.split('\n').length))}
            value={draft}
            maxLength={MAX_CHARS}
            placeholder="Viết bình luận... (Enter để gửi, Shift + Enter để xuống dòng)"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            ref={inputRef}
          />
          <button className="btn-primary h-11" onClick={() => send()} disabled={!draft.trim()}>
            Gửi
          </button>
        </div>
      </div>

      {opened && <CommentDetail comment={opened} onClose={() => setOpenId(null)} />}
    </div>
  )
}

function CommentDetail({ comment, onClose }: { comment: ChatComment; onClose: () => void }) {
  const { data, error, loading } = useAnalysis(comment.text)

  return (
    <Modal title="Chi tiết bình luận" onClose={onClose}>
      <section>
        <div className="mb-1 text-xs text-slate-500">Bình luận gốc · {formatTime(comment.time)}</div>
        <p className="rounded-lg bg-slate-100 px-4 py-3 break-words whitespace-pre-wrap dark:bg-slate-800">{comment.text}</p>
      </section>

      {loading && (
        <div className="flex items-center gap-3 text-slate-500">
          <Spinner /> Đang phân tích...
        </div>
      )}
      {error && <Notice tone="error">{error}</Notice>}
      {data?.empty && <Notice tone="warn">{EMPTY_MSG}</Notice>}

      {data && !data.empty && (
        <>
          <section className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
            <PredictionView data={data} />
          </section>

          <section className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
            <h3 className="mb-2 font-bold">Tiền xử lý</h3>
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-slate-500">Sau làm sạch</dt>
              <dd className="break-words">{data.cleaned}</dd>
              <dt className="text-slate-500">Sau tách từ</dt>
              <dd className="font-mono break-words">{data.segmented}</dd>
              <dt className="text-slate-500">Số token</dt>
              <dd>
                {num(data.n_tokens)}
                {data.n_tokens > data.max_len && (
                  <span className="text-amber-600"> (vượt MAX_LEN, model chỉ đọc {data.max_len} token đầu)</span>
                )}
              </dd>
              <dt className="text-slate-500">Token &lt;unk&gt;</dt>
              <dd>
                {data.is_unk.filter(Boolean).length}/{data.tokens.length}
                <span className="text-slate-500"> (từ không có trong vocab)</span>
              </dd>
            </dl>
          </section>

          <section className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
            <AttentionView tokens={data.tokens} isUnk={data.is_unk} attention={data.attention} />
          </section>
        </>
      )}
    </Modal>
  )
}
