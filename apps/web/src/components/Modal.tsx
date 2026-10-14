import { useEffect, type ReactNode } from 'react'

/** Pop-up giữa màn hình; đóng bằng Esc, nút ✕ hoặc bấm ra ngoài. */
export function Modal({ title, onClose, children }: { title: ReactNode; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden' // khoá cuộn trang phía sau
    window.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = overflow
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-950/60 p-4 backdrop-blur-sm md:p-10"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-4xl rounded-2xl bg-white shadow-2xl dark:bg-slate-900">
        <div className="sticky top-0 z-10 flex items-center justify-between gap-4 rounded-t-2xl border-b border-slate-200 bg-white/95 px-5 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-900/95">
          <h2 className="text-lg font-bold">{title}</h2>
          <button className="btn-ghost size-9 px-0" onClick={onClose} aria-label="Đóng">
            ✕
          </button>
        </div>
        <div className="space-y-5 p-5">{children}</div>
      </div>
    </div>
  )
}
