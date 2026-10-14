import { useEffect, useState } from 'react'
import { api, type Analysis } from './api'

/** Gọi /api/analyze cho `text`; kết quả gắn với câu đã hỏi nên không hiện nhầm kết quả cũ. */
export function useAnalysis(text: string | null) {
  const [state, setState] = useState<{ text: string; data?: Analysis; error?: string } | null>(null)

  useEffect(() => {
    if (!text) return
    let cancelled = false
    api
      .analyze(text)
      .then((data) => !cancelled && setState({ text, data }))
      .catch((e: Error) => !cancelled && setState({ text, error: e.message }))
    return () => {
      cancelled = true
    }
  }, [text])

  const current = text && state?.text === text ? state : null
  return { data: current?.data, error: current?.error, loading: !!text && !current }
}
