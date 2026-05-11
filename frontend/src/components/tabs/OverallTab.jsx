import { useState } from 'react'
import { createPortal } from 'react-dom'
import ZoomOverlay from '../ZoomOverlay'
import RegionModal from '../RegionModal'

const CONFIDENCE_ORDER = ['definitive','very_high','high','medium','low','clean']

const CONFIDENCE_STYLES = {
  clean:      'bg-green/10 text-green border-green/30',
  low:        'bg-blue-500/10 text-blue-400 border-blue-500/30',
  medium:     'bg-amber/10 text-amber border-amber/30',
  high:       'bg-orange-500/10 text-orange-400 border-orange-500/30',
  very_high:  'bg-red/10 text-red border-red/30',
  definitive: 'bg-red/20 text-red border-red/50',
}

const TRUTH_TABLE = [
  ['✗','✓','✓','Texture/surface detail lost. Geometry and lighting intact.','high'],
  ['✓','✗','✓','Likely lighting change shifting edges. Low geometry concern.','low'],
  ['✓','✓','✗','Global lighting/tone shift. No structural damage.','high'],
  ['✗','✗','✓','Compression blocking artifact. DCT grid + perceptual blurring.','very_high'],
  ['✗','✓','✗','Material/shader broke. Surface wrong, geometry fine.','high'],
  ['✓','✗','✗','Lighting rig changed. Edge shift is shadow-driven.','medium'],
  ['✗','✗','✗','Total visual failure. All three dimensions degraded.','definitive'],
  ['✓','✓','✓','No degradation detected.','clean'],
]

const SCORE_COLOR = s => s >= 0.9 ? 'text-green' : s >= 0.75 ? 'text-amber' : 'text-red'
const VERDICT_STYLE = v => v === 'PASS' ? 'bg-green/10 text-green border-green/30' : 'bg-red/10 text-red border-red/30'

function ScoreCard({ label, score, verdict, threshold, unit = '' }) {
  return (
    <div className="rounded-xl border border-border bg-ink/50 p-4">
      <p className="font-mono text-dim text-xs mb-1">{label}</p>
      <p className={`font-mono text-2xl font-semibold ${SCORE_COLOR(score)}`}>
        {unit === '%' ? `${(score * 100).toFixed(1)}%` : score.toFixed(4)}
      </p>
      <div className="flex items-center gap-2 mt-1">
        <span className={`font-mono text-xs border px-1.5 py-0.5 rounded ${VERDICT_STYLE(verdict)}`}>{verdict}</span>
        <span className="font-mono text-dim text-xs">thr {threshold}</span>
      </div>
    </div>
  )
}

function RegionCard({ region, index, onClick }) {
  const conf = region.confidence
  return (
    <button onClick={onClick}
            className="w-full text-left rounded-lg border border-border hover:border-muted bg-ink/30 hover:bg-ink/60 px-4 py-3 transition-all">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-dim text-xs">#{index + 1}</span>
        <span className={`font-mono text-xs border px-1.5 py-0.5 rounded ${CONFIDENCE_STYLES[conf] || ''}`}>
          {conf.replace('_',' ')}
        </span>
        <span className="font-mono text-muted text-xs ml-auto">
          {region.bbox[2]}×{region.bbox[3]}px
        </span>
      </div>
      <p className="font-mono text-xs text-text leading-snug line-clamp-2">{region.cause_hypothesis}</p>
    </button>
  )
}

export default function OverallTab({ result }) {
  const [zoom,        setZoom]        = useState(null)
  const [regionModal, setRegionModal] = useState(null)
  const [subTab,      setSubTab]      = useState('summary')
  const [truthOpen,   setTruthOpen]   = useState(false)

  const { passes, regions, heatmap, source_images, overall_verdict } = result
  const sorted = [...regions].sort((a, b) =>
    CONFIDENCE_ORDER.indexOf(a.confidence) - CONFIDENCE_ORDER.indexOf(b.confidence)
  )

  const LEGEND_ENTRIES = [
    { color: 'bg-yellow-400', label: 'SSIM only — perceptual diff' },
    { color: 'bg-orange-500', label: 'SSIM + Edge — structural loss' },
    { color: 'bg-blue-500',   label: 'SSIM + Color — color shift' },
    { color: 'bg-red',        label: 'All three — full degradation' },
  ]

  return (
    <div className="flex h-full">
      {/* Left: heatmap */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="mb-3 flex items-center justify-between">
          <p className="font-mono text-dim text-xs uppercase tracking-wide">Heatmap composite</p>
          <button onClick={() => setTruthOpen(true)}
                  className="font-mono text-dim text-xs hover:text-text border border-border rounded px-2 py-1">
            ⓘ truth table
          </button>
        </div>

        {heatmap && (
          <img
            src={`data:image/png;base64,${heatmap}`}
            alt="heatmap"
            className="w-full rounded-xl border border-border cursor-zoom-in"
            onClick={() => setZoom('heatmap')}
          />
        )}

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-3">
          {LEGEND_ENTRIES.map(({ color, label }) => (
            <div key={label} className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-sm ${color}`} />
              <span className="font-mono text-dim text-xs">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel */}
      <aside className="w-80 border-l border-border flex flex-col">
        {/* Sub-tab bar */}
        <div className="flex border-b border-border">
          {['summary','regions'].map(t => (
            <button key={t} onClick={() => setSubTab(t)}
                    className={`flex-1 py-3 font-mono text-xs capitalize transition-colors
                      ${subTab === t ? 'text-bright border-b-2 border-accent' : 'text-dim hover:text-text'}`}>
              {t}
              {t === 'regions' && regions.length > 0 && (
                <span className="ml-1.5 font-mono text-[10px] bg-muted/30 text-dim rounded px-1">{regions.length}</span>
              )}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-auto p-4">
          {subTab === 'summary' && (
            <div className="space-y-3">
              {/* Verdict */}
              <div className="rounded-xl border border-border bg-ink/50 p-4 text-center">
                <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">Overall verdict</p>
                <p className={`font-display text-4xl font-bold ${overall_verdict === 'PASS' ? 'text-green' : 'text-red'}`}>
                  {overall_verdict}
                </p>
              </div>

              <ScoreCard label="SSIM" score={passes.ssim.score}
                         verdict={passes.ssim.verdict} threshold="≥ 0.90" />
              <ScoreCard label="Edge match" score={passes.edge.score}
                         verdict={passes.edge.verdict} threshold="≥ 0.85" unit="%" />
              <ScoreCard label="Color distance" score={passes.color.score}
                         verdict={passes.color.verdict} threshold="≤ 0.30" />

              <div className="rounded-xl border border-border bg-ink/50 p-4">
                <p className="font-mono text-dim text-xs mb-2">Regions flagged</p>
                <p className="font-mono text-2xl font-semibold text-bright">{regions.length}</p>
                {Object.entries(
                  regions.reduce((acc, r) => {
                    acc[r.cause_tag] = (acc[r.cause_tag] || 0) + 1; return acc
                  }, {})
                ).map(([tag, count]) => (
                  <p key={tag} className="font-mono text-dim text-xs mt-1">{count}× {tag}</p>
                ))}
              </div>
            </div>
          )}

          {subTab === 'regions' && (
            <div className="space-y-2">
              {sorted.length === 0
                ? <p className="font-mono text-dim text-xs text-center py-8">no regions flagged ✓</p>
                : sorted.map((r, i) => (
                    <RegionCard key={i} region={r} index={i}
                                onClick={() => setRegionModal({ region: r, index: i })} />
                  ))
              }
            </div>
          )}
        </div>
      </aside>

      {/* Zoom overlay */}
      {zoom === 'heatmap' && (
        <ZoomOverlay
          b64={heatmap}
          label="Heatmap"
          b64Baseline={source_images.baseline}
          b64Optimized={source_images.optimized}
          onClose={() => setZoom(null)}
        />
      )}

      {/* Region modal */}
      {regionModal && (
        <RegionModal
          region={regionModal.region}
          index={regionModal.index}
          onClose={() => setRegionModal(null)}
        />
      )}

      {/* Truth table modal */}
      {truthOpen && createPortal(
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
             onClick={() => setTruthOpen(false)}>
          <div className="bg-panel border border-border rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-auto animate-slide-up p-6"
               onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display font-semibold text-text">Combinator Truth Table</h3>
              <button onClick={() => setTruthOpen(false)} className="font-mono text-dim text-sm">✕</button>
            </div>
            <table className="w-full font-mono text-xs">
              <thead>
                <tr className="text-dim border-b border-border">
                  <th className="text-left py-2 pr-3">SSIM</th>
                  <th className="text-left py-2 pr-3">Edge</th>
                  <th className="text-left py-2 pr-3">Color</th>
                  <th className="text-left py-2 pr-6">Hypothesis</th>
                  <th className="text-left py-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {TRUTH_TABLE.map(([s,e,c,hyp,conf], i) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className={`py-2 pr-3 ${s==='✗'?'text-red':'text-green'}`}>{s}</td>
                    <td className={`py-2 pr-3 ${e==='✗'?'text-red':'text-green'}`}>{e}</td>
                    <td className={`py-2 pr-3 ${c==='✗'?'text-red':'text-green'}`}>{c}</td>
                    <td className="py-2 pr-6 text-text leading-relaxed">{hyp}</td>
                    <td className={`py-2`}>
                      <span className={`border px-1.5 py-0.5 rounded text-[10px] ${CONFIDENCE_STYLES[conf]||''}`}>
                        {conf.replace('_',' ')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}
