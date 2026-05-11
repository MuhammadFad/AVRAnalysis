import { useState, useRef, useCallback } from 'react'
import { API } from '../hooks/usePing'

function DropZone({ label, file, onFile }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) onFile(f)
  }, [onFile])

  const preview = file ? URL.createObjectURL(file) : null

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`relative flex flex-col items-center justify-center rounded-xl border-2 transition-all cursor-pointer select-none overflow-hidden
        ${dragging ? 'border-accent bg-accent/10' : file ? 'border-green/40 bg-green/5' : 'border-border hover:border-muted bg-panel'}`}
      style={{ minHeight: 220 }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={e => { const f = e.target.files[0]; if (f) onFile(f) }}
      />
      {preview ? (
        <>
          <img src={preview} alt={label} className="absolute inset-0 w-full h-full object-cover opacity-60" />
          <div className="relative z-10 bg-ink/80 rounded-lg px-4 py-2 text-center">
            <p className="font-mono text-green text-xs font-medium">✓ {file.name}</p>
            <p className="font-mono text-dim text-xs mt-0.5">click to replace</p>
          </div>
        </>
      ) : (
        <div className="text-center p-8">
          <div className="w-12 h-12 rounded-xl border border-border flex items-center justify-center mx-auto mb-4">
            <svg className="w-5 h-5 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="font-display text-text font-medium">{label}</p>
          <p className="font-mono text-dim text-xs mt-1">drag & drop or click</p>
        </div>
      )}
    </div>
  )
}

export default function UploadScreen({ onResult }) {
  const [baseline,  setBaseline]  = useState(null)
  const [optimized, setOptimized] = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState('')

  async function handleAnalyze() {
    if (!baseline || !optimized) return
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('baseline',  baseline)
      form.append('optimized', optimized)

      const res = await fetch(`${API}/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: 'Server error' }))
        throw new Error(detail.detail || `HTTP ${res.status}`)
      }
      onResult(await res.json())
    } catch (e) {
      setError(e.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-ink animate-fade-in">
      <div className="absolute inset-0 opacity-[0.03]"
           style={{ backgroundImage: 'linear-gradient(#5b6ef5 1px,transparent 1px),linear-gradient(90deg,#5b6ef5 1px,transparent 1px)', backgroundSize: '40px 40px' }} />

      <div className="relative w-full max-w-2xl">
        {/* Header */}
        <div className="mb-10 text-center">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg border border-accent/40 flex items-center justify-center">
              <span className="font-mono text-accent text-xs font-semibold">AVR</span>
            </div>
            <span className="font-mono text-dim text-sm tracking-widest uppercase">Visual Regression Analyzer</span>
          </div>
          <h1 className="font-display text-3xl font-semibold text-bright mb-2">
            Compare two images.
          </h1>
          <p className="font-mono text-dim text-sm max-w-sm mx-auto leading-relaxed">
            SSIM · Canny edges · Bhattacharyya color — three passes, one diagnosis.
          </p>
        </div>

        {/* Drop zones */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <DropZone label="Baseline  (reference)" file={baseline}  onFile={setBaseline} />
          <DropZone label="Optimized  (test)"     file={optimized} onFile={setOptimized} />
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-red/10 border border-red/30 font-mono text-red text-sm">
            {error}
          </div>
        )}

        {/* Analyze button */}
        <button
          onClick={handleAnalyze}
          disabled={!baseline || !optimized || loading}
          className={`w-full py-3.5 rounded-xl font-display font-semibold text-sm tracking-wide transition-all
            ${baseline && optimized && !loading
              ? 'bg-accent text-white hover:bg-accent/90 shadow-lg shadow-accent/20'
              : 'bg-panel text-muted border border-border cursor-not-allowed'}`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analyzing…
            </span>
          ) : 'Run Analysis'}
        </button>
      </div>
    </div>
  )
}
