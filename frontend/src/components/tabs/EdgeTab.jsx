import { useState } from 'react'
import ZoomOverlay from '../ZoomOverlay'

const SCORE_COLOR = s => s >= 0.9 ? 'text-green' : s >= 0.75 ? 'text-amber' : 'text-red'

export default function EdgeTab({ result }) {
  const [zoom, setZoom] = useState(null)
  const edge = result.passes.edge
  const ints = edge.intermediates || {}
  const src  = result.source_images

  const images = [
    { key: 'baseline_edges',  label: 'Baseline edge map' },
    { key: 'optimized_edges', label: 'Optimized edge map' },
    { key: 'edge_diff',       label: 'Edge diff  (green = survived · red = lost)' },
  ]

  return (
    <div className="flex h-full">
      {/* Left: images */}
      <div className="flex-1 p-6 overflow-auto">
        {/* Baseline + Optimized side by side */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          {images.slice(0, 2).map(({ key, label }) => ints[key] && (
            <div key={key}>
              <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">{label}</p>
              <img src={`data:image/png;base64,${ints[key]}`} alt={label}
                   className="w-full rounded-lg border border-border cursor-zoom-in"
                   onClick={() => setZoom(key)} />
            </div>
          ))}
        </div>
        {/* Diff map — full width below */}
        {ints['edge_diff'] && (
          <div>
            <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">
              Edge diff&nbsp;&nbsp;<span className="text-green normal-case">■ survived</span>
              &nbsp;&nbsp;<span className="text-red normal-case">■ lost</span>
            </p>
            <img src={`data:image/png;base64,${ints['edge_diff']}`} alt="edge diff"
                 className="w-full rounded-lg border border-border cursor-zoom-in"
                 onClick={() => setZoom('edge_diff')} />
          </div>
        )}
      </div>

      {/* Right panel */}
      <aside className="w-80 border-l border-border p-5 overflow-auto space-y-5">
        <div>
          <p className="font-mono text-dim text-xs mb-1">Edge match score</p>
          <p className={`font-mono text-4xl font-semibold ${SCORE_COLOR(edge.score)}`}>
            {(edge.score * 100).toFixed(1)}%
          </p>
          <div className={`inline-flex mt-2 font-mono text-xs border px-2 py-1 rounded
            ${edge.verdict === 'PASS' ? 'bg-green/10 text-green border-green/30' : 'bg-red/10 text-red border-red/30'}`}>
            {edge.verdict}  ·  threshold ≥ 85%
          </div>
        </div>

        <div className="rounded-xl border border-border bg-ink/50 px-4 py-3">
          <p className="font-mono text-dim text-xs mb-1">Baseline edges retained</p>
          <p className="font-mono text-bright text-lg font-semibold">
            {(edge.score * 100).toFixed(1)}% of baseline edges preserved
          </p>
        </div>

        <div className="rounded-xl border border-border bg-ink/30 p-4 space-y-3">
          <p className="font-mono text-dim text-xs uppercase tracking-wide">What does edge loss mean?</p>
          <p className="font-mono text-dim text-xs leading-relaxed">
            Each pixel in the baseline edge map represents a structural boundary —
            an object outline, material edge, or fine geometric detail. When those pixels are absent
            in the optimized image, that structural information has been lost.
          </p>
          <p className="font-mono text-dim text-xs leading-relaxed">
            ⚠ <span className="text-amber">Lighting caveat:</span> a change in lighting can shift
            edge positions without any geometric damage. This is why edge-only failures carry
            <span className="text-amber"> low confidence</span> — they need SSIM agreement to be meaningful.
          </p>
        </div>
      </aside>

      {zoom && (
        <ZoomOverlay
          b64={ints[zoom]}
          label={images.find(i => i.key === zoom)?.label || zoom}
          b64Baseline={src.baseline}
          b64Optimized={src.optimized}
          onClose={() => setZoom(null)}
        />
      )}
    </div>
  )
}
