import { useState } from 'react'
import SSIMTab    from './tabs/SSIMTab'
import EdgeTab    from './tabs/EdgeTab'
import ColorTab   from './tabs/ColorTab'
import OverallTab from './tabs/OverallTab'
import SaveModal  from './SaveModal'

const TABS = ['overall', 'ssim', 'edge', 'color']

const VERDICT_STYLES = {
  PASS: 'text-green border-green/30 bg-green/10',
  FAIL: 'text-red border-red/30 bg-red/10',
}

export default function ResultsScreen({ result, onReset }) {
  const [tab,       setTab]       = useState('overall')
  const [saveOpen,  setSaveOpen]  = useState(false)

  const verdict = result.overall_verdict

  return (
    <div className="min-h-screen bg-ink flex flex-col animate-fade-in">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg border border-accent/40 flex items-center justify-center">
            <span className="font-mono text-accent text-[10px] font-semibold">AVR</span>
          </div>
          <span className="font-mono text-dim text-xs tracking-widest uppercase">Visual Regression Analyzer</span>
        </div>

        <div className="flex items-center gap-3">
          {result.resolution_mismatch && (
            <span className="font-mono text-amber text-xs border border-amber/30 bg-amber/10 px-2 py-1 rounded">
              ⚠ resolution mismatch — images aligned automatically
            </span>
          )}
          <span className={`font-mono text-xs font-semibold border px-3 py-1 rounded ${VERDICT_STYLES[verdict]}`}>
            {verdict}
          </span>
          <button onClick={() => setSaveOpen(true)}
                  className="font-mono text-xs text-dim hover:text-text border border-border hover:border-muted px-3 py-1 rounded transition-colors">
            Save ↓
          </button>
          <button onClick={onReset}
                  className="font-mono text-xs text-dim hover:text-text border border-border hover:border-muted px-3 py-1 rounded transition-colors">
            ← New
          </button>
        </div>
      </header>

      {/* Tab bar */}
      <nav className="flex gap-1 px-6 pt-3 pb-0 border-b border-border shrink-0">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`font-mono text-xs px-4 py-2 rounded-t transition-colors capitalize tracking-wide
              ${tab === t
                ? 'text-bright bg-panel border-t border-l border-r border-border'
                : 'text-dim hover:text-text'}`}
          >
            {t === 'ssim' ? 'SSIM' : t === 'overall' ? 'Overall' : t.charAt(0).toUpperCase() + t.slice(1)}
            {/* Pass/Fail dot */}
            {t !== 'overall' && (
              <span className={`ml-2 inline-block w-1.5 h-1.5 rounded-full align-middle
                ${result.passes[t]?.verdict === 'PASS' ? 'bg-green' : 'bg-red'}`} />
            )}
          </button>
        ))}
      </nav>

      {/* Tab content */}
      <div className="flex-1 overflow-auto">
        {tab === 'overall' && <OverallTab result={result} />}
        {tab === 'ssim'    && <SSIMTab    result={result} />}
        {tab === 'edge'    && <EdgeTab    result={result} />}
        {tab === 'color'   && <ColorTab   result={result} />}
      </div>

      {saveOpen && <SaveModal result={result} onClose={() => setSaveOpen(false)} />}
    </div>
  )
}
