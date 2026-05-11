import { useState } from 'react'
import ZoomOverlay from '../ZoomOverlay'

const SCORE_COLOR = s => s >= 0.9 ? 'text-green' : s >= 0.75 ? 'text-amber' : 'text-red'

function IntermediateImg({ b64, label, onClick }) {
  if (!b64) return null
  return (
    <div>
      <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">{label}</p>
      <img src={`data:image/png;base64,${b64}`} alt={label}
           className="w-full rounded-lg border border-border cursor-zoom-in"
           onClick={onClick} />
    </div>
  )
}

export default function SSIMTab({ result }) {
  const [zoom, setZoom] = useState(null)
  const ssim = result.passes.ssim
  const ints = ssim.intermediates || {}
  const src  = result.source_images

  const components = [
    { key: 'luminance',  label: 'Luminance component (L)',  desc: 'Local brightness similarity' },
    { key: 'contrast',   label: 'Contrast component (C)',   desc: 'Local texture variation' },
    { key: 'structure',  label: 'Structure component (S)',  desc: 'Edge-pattern correlation' },
    { key: 'ssim_map',   label: 'SSIM map',                 desc: 'Combined per-pixel score' },
  ]

  return (
    <div className="flex h-full">
      {/* Left: images */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="grid grid-cols-2 gap-4">
          {components.map(({ key, label }) => (
            <IntermediateImg key={key} b64={ints[key]} label={label}
                             onClick={() => setZoom(key)} />
          ))}
        </div>
      </div>

      {/* Right panel */}
      <aside className="w-80 border-l border-border p-5 overflow-auto space-y-5">
        {/* Score */}
        <div>
          <p className="font-mono text-dim text-xs mb-1">SSIM Score</p>
          <p className={`font-mono text-4xl font-semibold ${SCORE_COLOR(ssim.score)}`}>
            {ssim.score.toFixed(4)}
          </p>
          <div className={`inline-flex mt-2 font-mono text-xs border px-2 py-1 rounded
            ${ssim.verdict === 'PASS' ? 'bg-green/10 text-green border-green/30' : 'bg-red/10 text-red border-red/30'}`}>
            {ssim.verdict}  ·  threshold ≥ 0.90
          </div>
        </div>

        {/* Component breakdown */}
        <div>
          <p className="font-mono text-dim text-xs uppercase tracking-wide mb-3">Component interpretation</p>
          <div className="space-y-2">
            {components.slice(0,3).map(({ key, label, desc }) => (
              <div key={key} className="rounded-lg border border-border bg-ink/50 px-3 py-2">
                <p className="font-mono text-xs text-bright">{label}</p>
                <p className="font-mono text-dim text-xs">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Gradient bar */}
        <div>
          <p className="font-mono text-dim text-xs uppercase tracking-wide mb-2">Map legend</p>
          <div className="h-4 rounded" style={{ background: 'linear-gradient(to right, #000, #ff3c00, #ffff00, #ffffff)' }} />
          <div className="flex justify-between font-mono text-dim text-xs mt-1">
            <span>dissimilar</span><span>similar</span>
          </div>
          <p className="font-mono text-dim text-xs mt-2 leading-relaxed">
            Black = very different. White = near-identical. Hot colors indicate degraded regions.
          </p>
        </div>

        {/* Explanation */}
        <div className="rounded-xl border border-border bg-ink/30 p-4">
          <p className="font-mono text-dim text-xs uppercase tracking-wide mb-2">What is SSIM?</p>
          <p className="font-mono text-dim text-xs leading-relaxed">
            Structural Similarity measures perceptual image quality across three dimensions simultaneously —
            local brightness (L), texture variation (C), and edge structure (S). A score of 1.0 is identical;
            0.90 is the typical quality threshold for rendered content.
          </p>
        </div>
      </aside>

      {zoom && (
        <ZoomOverlay
          b64={ints[zoom]}
          label={components.find(c => c.key === zoom)?.label || zoom}
          b64Baseline={src.baseline}
          b64Optimized={src.optimized}
          onClose={() => setZoom(null)}
        />
      )}
    </div>
  )
}
