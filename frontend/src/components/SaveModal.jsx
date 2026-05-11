import { useState } from 'react'
import { createPortal } from 'react-dom'

function b64toBlob(b64, mime = 'image/png') {
  const binary = atob(b64)
  const arr    = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i)
  return new Blob([arr], { type: mime })
}

export default function SaveModal({ result, onClose }) {
  const [name, setName]         = useState(`avr-report-${Date.now()}`)
  const [incSSIM,  setSSIM]     = useState(false)
  const [incEdge,  setEdge]     = useState(false)
  const [incColor, setColor]    = useState(false)
  const [incHeat,  setHeat]     = useState(true)
  const [saving,   setSaving]   = useState(false)

  async function handleSave() {
    setSaving(true)
    try {
      const JSZip = (await import('jszip')).default
      const zip   = new JSZip()

      // Always include JSON
      const { heatmap: _h, source_images: _s, passes, regions, ...reportRest } = result
      const cleanReport = {
        ...reportRest,
        passes: {
          ssim:  { score: passes.ssim.score,  verdict: passes.ssim.verdict },
          edge:  { score: passes.edge.score,  verdict: passes.edge.verdict },
          color: { score: passes.color.score, verdict: passes.color.verdict,
                   per_channel_distance: passes.color.per_channel_distance },
        },
        regions: regions.map(({ crops: _c, ...r }) => r),
      }
      zip.file('regression_report.json', JSON.stringify(cleanReport, null, 2))

      // Heatmap
      if (incHeat && result.heatmap)
        zip.file('heatmap_composite.png', b64toBlob(result.heatmap))

      // SSIM intermediates
      if (incSSIM) {
        const ints = passes.ssim.intermediates || {}
        for (const [k, v] of Object.entries(ints))
          if (v) zip.file(`ssim/${k}.png`, b64toBlob(v))
      }

      // Edge intermediates
      if (incEdge) {
        const ints = passes.edge.intermediates || {}
        for (const [k, v] of Object.entries(ints))
          if (v) zip.file(`edge/${k}.png`, b64toBlob(v))
      }

      // Color intermediates
      if (incColor) {
        const ints = passes.color.intermediates || {}
        for (const [k, v] of Object.entries(ints))
          if (v) zip.file(`color/${k}.png`, b64toBlob(v))
      }

      const blob = await zip.generateAsync({ type: 'blob' })
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `${name || 'avr-report'}.zip`
      a.click()
      URL.revokeObjectURL(url)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
         onClick={onClose}>
      <div className="bg-panel border border-border rounded-2xl w-full max-w-sm animate-slide-up"
           onClick={e => e.stopPropagation()}>

        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <span className="font-display font-semibold text-text text-sm">Save report</span>
          <button onClick={onClose} className="font-mono text-dim hover:text-text text-sm">✕</button>
        </div>

        <div className="p-5 space-y-4">
          {/* Filename */}
          <div>
            <label className="font-mono text-dim text-xs block mb-1.5">Filename</label>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              className="w-full bg-ink border border-border rounded-lg px-3 py-2 font-mono text-xs text-text focus:outline-none focus:border-accent"
            />
          </div>

          {/* Checkboxes */}
          <div className="space-y-2">
            <p className="font-mono text-dim text-xs uppercase tracking-wide">Include</p>
            {[
              { label: 'regression_report.json',     checked: true,     disabled: true,  set: null },
              { label: 'Heatmap composite',           checked: incHeat,  disabled: false, set: setHeat },
              { label: 'SSIM intermediates',          checked: incSSIM,  disabled: false, set: setSSIM },
              { label: 'Edge intermediates',          checked: incEdge,  disabled: false, set: setEdge },
              { label: 'Color intermediates',         checked: incColor, disabled: false, set: setColor },
            ].map(({ label, checked, disabled, set }) => (
              <label key={label} className={`flex items-center gap-3 cursor-pointer ${disabled ? 'opacity-50' : ''}`}>
                <input type="checkbox" checked={checked} disabled={disabled}
                       onChange={set ? e => set(e.target.checked) : undefined}
                       className="accent-accent" />
                <span className="font-mono text-xs text-text">{label}</span>
              </label>
            ))}
          </div>

          <button onClick={handleSave} disabled={saving}
                  className="w-full py-2.5 rounded-lg font-display font-semibold text-sm bg-accent text-white hover:bg-accent/90 disabled:opacity-50 transition-colors">
            {saving ? 'Zipping…' : 'Download .zip'}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
