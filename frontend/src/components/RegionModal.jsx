import { useEffect } from 'react'
import { createPortal } from 'react-dom'

const CONFIDENCE_STYLES = {
  clean:      'bg-green/10 text-green border-green/30',
  low:        'bg-blue-500/10 text-blue-400 border-blue-500/30',
  medium:     'bg-amber/10 text-amber border-amber/30',
  high:       'bg-orange-500/10 text-orange-400 border-orange-500/30',
  very_high:  'bg-red/10 text-red border-red/30',
  definitive: 'bg-red/20 text-red border-red/50',
}

const PASS_DESCRIPTIONS = {
  ssim:  'Perceptual similarity loss — the region looks visibly different to the human eye.',
  edge:  'Structural edges lost — outlines, geometry, or fine detail missing in this area.',
  color: 'Color distribution shifted — hue, saturation, or brightness changed significantly.',
}

export default function RegionModal({ region, index, onClose }) {
  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  const [bx, by, bw, bh] = region.bbox
  const conf = region.confidence

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
         onClick={onClose}>
      <div className="bg-panel border border-border rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-auto animate-slide-up"
           onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <span className="font-mono text-dim text-xs">Region {index + 1}</span>
            <span className={`font-mono text-xs font-semibold border px-2 py-0.5 rounded ${CONFIDENCE_STYLES[conf] || CONFIDENCE_STYLES.low}`}>
              {conf.replace('_', ' ')}
            </span>
          </div>
          <button onClick={onClose} className="font-mono text-dim hover:text-text text-sm">✕</button>
        </div>

        <div className="p-6 grid grid-cols-2 gap-6">
          {/* Left: crops */}
          <div className="space-y-3">
            <div>
              <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">Baseline crop</p>
              {region.crops?.baseline ? (
                <img src={`data:image/png;base64,${region.crops.baseline}`}
                     alt="baseline crop" className="w-full rounded-lg border border-border object-contain bg-ink" />
              ) : <div className="h-32 rounded-lg border border-border bg-ink flex items-center justify-center text-muted text-xs font-mono">no crop</div>}
            </div>
            <div>
              <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">Optimized crop</p>
              {region.crops?.optimized ? (
                <img src={`data:image/png;base64,${region.crops.optimized}`}
                     alt="optimized crop" className="w-full rounded-lg border border-border object-contain bg-ink" />
              ) : <div className="h-32 rounded-lg border border-border bg-ink flex items-center justify-center text-muted text-xs font-mono">no crop</div>}
            </div>
          </div>

          {/* Right: analysis */}
          <div className="space-y-5">
            {/* Hypothesis */}
            <div>
              <p className="font-mono text-dim text-xs uppercase tracking-wide mb-2">Cause hypothesis</p>
              <p className="font-display text-text text-sm leading-relaxed">
                {region.cause_hypothesis}
              </p>
            </div>

            {/* Failed passes */}
            <div>
              <p className="font-mono text-dim text-xs uppercase tracking-wide mb-2">Failed passes</p>
              <div className="space-y-2">
                {region.passes_failed.map(pass => (
                  <div key={pass} className="rounded-lg border border-border bg-ink/50 px-3 py-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-xs font-semibold text-bright uppercase">{pass}</span>
                      <span className="font-mono text-xs text-red">
                        {pass === 'ssim'  && region.local_scores.ssim  != null && `${region.local_scores.ssim.toFixed(3)}`}
                        {pass === 'edge'  && region.local_scores.edge  != null && `${(region.local_scores.edge * 100).toFixed(1)}%`}
                        {pass === 'color' && region.local_scores.color != null && `d=${region.local_scores.color.toFixed(3)}`}
                      </span>
                    </div>
                    <p className="font-mono text-dim text-xs leading-relaxed">
                      {PASS_DESCRIPTIONS[pass]}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Coordinates */}
            <div>
              <p className="font-mono text-dim text-xs uppercase tracking-wide mb-2">Bounding box</p>
              <div className="font-mono text-xs text-dim space-y-0.5">
                <p>origin: ({bx}, {by}) px</p>
                <p>size:   {bw} × {bh} px</p>
                <p>area:   {(bw * bh).toLocaleString()} px²</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  )
}
