import { useState, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export function usePing() {
  const [ready, setReady]     = useState(false)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const start   = Date.now()
    const timer   = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 500)
    let cancelled = false

    async function poll() {
      while (!cancelled) {
        try {
          const res = await fetch(`${API}/ping`, { signal: AbortSignal.timeout(5000) })
          if (res.ok) { setReady(true); break }
        } catch (_) { /* server still cold */ }
        await new Promise(r => setTimeout(r, 2000))
      }
    }

    poll()
    return () => { cancelled = true; clearInterval(timer) }
  }, [])

  return { ready, elapsed }
}

export { API }
