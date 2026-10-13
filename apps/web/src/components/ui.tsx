import type { ReactNode } from 'react'
import { LABEL_COLOR, LABEL_VI, type Label } from '../constants'

export function Spinner({ className = 'size-5' }: { className?: string }) {
  return (
    <span
      className={`inline-block animate-spin rounded-full border-2 border-current border-r-transparent ${className}`}
      aria-label="Đang tải"
    />
  )
}

export function Notice({ tone = 'info', children }: { tone?: 'info' | 'warn' | 'error'; children: ReactNode }) {
  const styles = {
    info: 'border-brand-200 bg-brand-50 text-brand-700 dark:border-brand-700/40 dark:bg-brand-700/10 dark:text-brand-100',
    warn: 'border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700/50 dark:bg-amber-900/20 dark:text-amber-200',
    error: 'border-red-300 bg-red-50 text-red-800 dark:border-red-800/60 dark:bg-red-900/20 dark:text-red-200',
  }[tone]
  return <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>{children}</div>
}

export function LabelBadge({ label, size = 'md' }: { label: Label; size?: 'sm' | 'md' | 'lg' }) {
  const sizes = { sm: 'px-2 py-0.5 text-xs', md: 'px-3 py-1 text-sm', lg: 'px-5 py-1.5 text-lg' }[size]
  return (
    <span
      className={`inline-block rounded-full font-bold whitespace-nowrap text-white ${sizes}`}
      style={{ background: LABEL_COLOR[label] }}
    >
      {label}
      {size === 'lg' && ` · ${LABEL_VI[label]}`}
    </span>
  )
}

export function MetricCard({ label, value, loading }: { label: string; value: ReactNode; loading?: boolean }) {
  return (
    <div className="card">
      <div className="text-sm text-slate-500 dark:text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-bold tabular-nums">
        {loading ? <span className="inline-block h-7 w-24 animate-pulse rounded bg-slate-200 dark:bg-slate-800" /> : value}
      </div>
    </div>
  )
}

export function Select<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <select
        className="field"
        value={String(value)}
        onChange={(e) => onChange(options.find((o) => String(o.value) === e.target.value)!.value)}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}
