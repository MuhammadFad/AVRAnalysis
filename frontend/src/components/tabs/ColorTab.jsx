import { useState, useRef, useEffect } from 'react'
import ZoomOverlay from '../ZoomOverlay'

const SCORE_COLOR = s => s <= 0.1 ? 'text-green' : s <= 0.3 ? 'text-amber' : 'text-red'

/**
 * Renders a base64 PNG into an offscreen canvas, then draws individual
 * H, S, V channels side-by-side onto a visible canvas.
 * Pure frontend — no backend round-trip needed.
 */
function HSVChannelStrip({ b64, label }) {
  const canvasRef = useRef()

  useEffect(() => {
    if (!b64 || !canvasRef.current) return
    const img = new Image()
    img.onload = () => {
      const W = img.naturalWidth
      const H = img.naturalHeight

      // Decode into offscreen canvas to get pixel data
      const off = document.createElement('canvas')
      off.width  = W
      off.height = H
      const octx = off.getContext('2d')
      octx.drawImage(img, 0, 0)
      const { data } = octx.getImageData(0, 0, W, H)

      // Build three channel images side by side
      const out = canvasRef.current
      out.width  = W * 3
      out.height = H
      const ctx  = out.getContext('2d')
      const outData = ctx.createImageData(W * 3, H)
      const od = outData.data

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const src = (y * W + x) * 4
          const r = data[src] / 255
          const g = data[src + 1] / 255
          const b = data[src + 2] / 255

          // RGB → HSV
          const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min
          let h = 0
          if (d > 0) {
            if (max === r)      h = ((g - b) / d + 6) % 6
            else if (max === g) h = (b - r) / d + 2
            else                h = (r - g) / d + 4
            h /= 6
          }
          const s = max > 0 ? d / max : 0
          const v = max

          // Hue channel: render as hue-colored swatch at full S & V
          const hDeg = h * 6
          const hI = Math.floor(hDeg)
          const hF = hDeg - hI
          const p = 0, q = 1 - hF, t = hF
          let hr = 0, hg = 0, hb = 0
          switch (hI % 6) {
            case 0: hr=1;  hg=t;  hb=p;  break
            case 1: hr=q;  hg=1;  hb=p;  break
            case 2: hr=p;  hg=1;  hb=t;  break
            case 3: hr=p;  hg=q;  hb=1;  break
            case 4: hr=t;  hg=p;  hb=1;  break
            case 5: hr=1;  hg=p;  hb=q;  break
          }

          // Column 0: Hue (false-color)
          const c0 = (y * W * 3 + 0 * W + x) * 4
          od[c0]   = hr * 255; od[c0+1] = hg * 255; od[c0+2] = hb * 255; od[c0+3] = 255

          // Column 1: Saturation (grayscale)
          const c1 = (y * W * 3 + 1 * W + x) * 4
          const sv = Math.round(s * 255)
          od[c1] = sv; od[c1+1] = sv; od[c1+2] = sv; od[c1+3] = 255

          // Column 2: Value (grayscale)
          const c2 = (y * W * 3 + 2 * W + x) * 4
          const vv = Math.round(v * 255)
          od[c2] = vv; od[c2+1] = vv; od[c2+2] = vv; od[c2+3] = 255
        }
      }
      ctx.putImageData(outData, 0, 0)
    }
    img.src = `data:image/png;base64,${b64}`
  }, [b64])

  return (
    <div>
      <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">
        {label} — <span className="text-text">H</span>
        <span className="text-muted"> / </span><span className="text-text">S</span>
        <span className="text-muted"> / </span><span className="text-text">V</span>
      </p>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border" style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}

export default function ColorTab({ result }) {
  const [zoom,       setZoom]      = useState(null)
  const [overlayOn,  setOverlayOn] = useState(false)
  const [channelMode, setMode]     = useState('hsv')  // 'hsv' | 'rgb'

  const color = result.passes.color
  const ints  = color.intermediates || {}
  const src   = result.source_images
  const pcd   = color.per_channel_distance || {}

  const CHANNEL_LABELS = { H: 'Hue', S: 'Saturation', V: 'Value (brightness)' }

  return (
    <div className="flex h-full">
      {/* Left */}
      <div className="flex-1 p-6 overflow-auto space-y-6">

        {/* Mode toggle */}
        <div className="flex items-center gap-2">
          <p className="font-mono text-dim text-xs uppercase tracking-wide mr-2">Channel view</p>
          {['hsv', 'rgb'].map(m => (
            <button key={m} onClick={() => setMode(m)}
                    className={`font-mono text-xs border px-3 py-1 rounded uppercase transition-colors
                      ${channelMode === m ? 'border-accent text-accent bg-accent/10' : 'border-border text-dim hover:text-text'}`}>
              {m}
            </button>
          ))}
        </div>

        {channelMode === 'hsv' ? (
          /* HSV rows — computed client-side from source images */
          <>
            <HSVChannelStrip b64={src.baseline}  label="Baseline" />
            <HSVChannelStrip b64={src.optimized} label="Optimized" />
          </>
        ) : (
          /* RGB rows — just the original images split into R G B channels via canvas */
          <>
            <RGBChannelStrip b64={src.baseline}  label="Baseline" />
            <RGBChannelStrip b64={src.optimized} label="Optimized" />
          </>
        )}

        {/* HSV overlay toggle — overlays the two HSV strips at 50% */}
        <div>
          <div className="flex items-center gap-3 mb-3">
            <p className="font-mono text-dim text-xs uppercase tracking-wide">
              {channelMode.toUpperCase()} channel overlay
            </p>
            <button onClick={() => setOverlayOn(v => !v)}
                    className={`font-mono text-xs border px-2 py-1 rounded transition-colors
                      ${overlayOn ? 'border-accent text-accent' : 'border-border text-dim hover:text-text'}`}>
              {overlayOn ? 'hide' : 'show'} overlay
            </button>
          </div>
          {overlayOn && (
            <ChannelOverlay b64Baseline={src.baseline} b64Optimized={src.optimized} mode={channelMode} />
          )}
        </div>

        {/* Histogram overlay */}
        {ints.overlay_hist && (
          <div>
            <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">HSV histogram overlay</p>
            <img src={`data:image/png;base64,${ints.overlay_hist}`} alt="histogram overlay"
                 className="w-full rounded-lg border border-border cursor-zoom-in"
                 onClick={() => setZoom('overlay_hist')} />
          </div>
        )}
      </div>

      {/* Right panel */}
      <aside className="w-80 border-l border-border p-5 overflow-auto space-y-5">
        <div>
          <p className="font-mono text-dim text-xs mb-1">Mean Bhattacharyya distance</p>
          <p className={`font-mono text-4xl font-semibold ${SCORE_COLOR(color.score)}`}>
            {color.score.toFixed(4)}
          </p>
          <div className={`inline-flex mt-2 font-mono text-xs border px-2 py-1 rounded
            ${color.verdict === 'PASS' ? 'bg-green/10 text-green border-green/30' : 'bg-red/10 text-red border-red/30'}`}>
            {color.verdict}  ·  threshold ≤ 0.30
          </div>
        </div>

        <div>
          <p className="font-mono text-dim text-xs uppercase tracking-wide mb-3">Per-channel distances</p>
          <div className="space-y-2">
            {Object.entries(pcd).map(([ch, d]) => (
              <div key={ch} className="rounded-lg border border-border bg-ink/50 px-3 py-2 flex items-center justify-between">
                <div>
                  <span className="font-mono text-xs text-bright">{ch}</span>
                  <span className="font-mono text-dim text-xs ml-2">— {CHANNEL_LABELS[ch]}</span>
                </div>
                <span className={`font-mono text-sm font-semibold ${SCORE_COLOR(d)}`}>{d.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-ink/30 p-4 space-y-3">
          <p className="font-mono text-dim text-xs uppercase tracking-wide">What is Bhattacharyya distance?</p>
          <p className="font-mono text-dim text-xs leading-relaxed">
            Measures the overlap between two color probability distributions.
            0.0 = identical. Higher = more divergence.
            Computed in HSV space, cleanly separating color (H, S) from brightness (V).
          </p>
          <p className="font-mono text-dim text-xs leading-relaxed">
            ⚠ <span className="text-amber">Blind spot:</span> this is a global metric.
            A small localized color shift can be masked by the rest of the image.
            Check the heatmap for spatially-localized color regions.
          </p>
        </div>
      </aside>

      {zoom && (
        <ZoomOverlay
          b64={ints[zoom]}
          label={zoom}
          b64Baseline={src.baseline}
          b64Optimized={src.optimized}
          onClose={() => setZoom(null)}
        />
      )}
    </div>
  )
}

// ---- RGB channel strip ----
function RGBChannelStrip({ b64, label }) {
  const canvasRef = useRef()
  useEffect(() => {
    if (!b64 || !canvasRef.current) return
    const img = new Image()
    img.onload = () => {
      const W = img.naturalWidth, H = img.naturalHeight
      const off = document.createElement('canvas')
      off.width = W; off.height = H
      const octx = off.getContext('2d')
      octx.drawImage(img, 0, 0)
      const { data } = octx.getImageData(0, 0, W, H)

      const out = canvasRef.current
      out.width = W * 3; out.height = H
      const ctx = out.getContext('2d')
      const od  = ctx.createImageData(W * 3, H)
      const d2  = od.data

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const s = (y * W + x) * 4
          // R channel → red tinted
          const c0 = (y * W * 3 + 0 * W + x) * 4
          d2[c0] = data[s]; d2[c0+1] = 0; d2[c0+2] = 0; d2[c0+3] = 255
          // G channel → green tinted
          const c1 = (y * W * 3 + 1 * W + x) * 4
          d2[c1] = 0; d2[c1+1] = data[s+1]; d2[c1+2] = 0; d2[c1+3] = 255
          // B channel → blue tinted
          const c2 = (y * W * 3 + 2 * W + x) * 4
          d2[c2] = 0; d2[c2+1] = 0; d2[c2+2] = data[s+2]; d2[c2+3] = 255
        }
      }
      ctx.putImageData(od, 0, 0)
    }
    img.src = `data:image/png;base64,${b64}`
  }, [b64])

  return (
    <div>
      <p className="font-mono text-dim text-xs mb-2 uppercase tracking-wide">
        {label} — <span className="text-red">R</span>
        <span className="text-muted"> / </span><span className="text-green">G</span>
        <span className="text-muted"> / </span><span className="text-blue-400">B</span>
      </p>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border" style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}

// ---- Overlay: two HSV/RGB strips blended 50/50 via canvas ----
function ChannelOverlay({ b64Baseline, b64Optimized, mode }) {
  const canvasRef = useRef()

  useEffect(() => {
    if (!b64Baseline || !b64Optimized || !canvasRef.current) return

    const loadImg = src => new Promise(res => {
      const img = new Image()
      img.onload = () => res(img)
      img.src = `data:image/png;base64,${src}`
    })

    Promise.all([loadImg(b64Baseline), loadImg(b64Optimized)]).then(([imgB, imgO]) => {
      const W = imgB.naturalWidth, H = imgB.naturalHeight
      const decode = img => {
        const off = document.createElement('canvas')
        off.width = W; off.height = H
        const ctx = off.getContext('2d')
        ctx.drawImage(img, 0, 0)
        return ctx.getImageData(0, 0, W, H).data
      }
      const dB = decode(imgB)
      const dO = decode(imgO)

      const toChannels = (data) => {
        const out = new Uint8ClampedArray(W * 3 * H * 4)
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const s = (y * W + x) * 4
            const r = data[s]/255, g = data[s+1]/255, b = data[s+2]/255

            let ch0r, ch0g, ch0b, ch1v, ch2v
            if (mode === 'hsv') {
              const max = Math.max(r,g,b), min = Math.min(r,g,b), d = max - min
              let h = 0
              if (d > 0) {
                if (max === r) h = ((g-b)/d+6)%6
                else if (max === g) h = (b-r)/d+2
                else h = (r-g)/d+4
                h /= 6
              }
              const sv = max > 0 ? d/max : 0
              const hDeg = h*6, hI = Math.floor(hDeg), hF = hDeg - hI
              const p=0, q=1-hF, t=hF
              let hr=0,hg2=0,hb2=0
              switch (hI%6) {
                case 0:hr=1;hg2=t;hb2=p;break; case 1:hr=q;hg2=1;hb2=p;break
                case 2:hr=p;hg2=1;hb2=t;break; case 3:hr=p;hg2=q;hb2=1;break
                case 4:hr=t;hg2=p;hb2=1;break; case 5:hr=1;hg2=p;hb2=q;break
              }
              ch0r=hr*255; ch0g=hg2*255; ch0b=hb2*255
              ch1v=sv*255; ch2v=max*255
            } else {
              ch0r=r*255; ch0g=0; ch0b=0
              ch1v=g*255; ch2v=b*255
            }

            const c0 = (y * W*3 + 0*W + x)*4
            out[c0]=ch0r; out[c0+1]=ch0g; out[c0+2]=ch0b; out[c0+3]=255
            const c1 = (y * W*3 + 1*W + x)*4
            out[c1]=ch1v; out[c1+1]=mode==='hsv'?ch1v:0; out[c1+2]=0; out[c1+3]=255
            const c2 = (y * W*3 + 2*W + x)*4
            out[c2]=0; out[c2+1]=mode==='hsv'?ch2v:0; out[c2+2]=mode==='hsv'?ch2v:ch2v; out[c2+3]=255
          }
        }
        return out
      }

      const chB = toChannels(dB)
      const chO = toChannels(dO)

      const canvas = canvasRef.current
      canvas.width = W*3; canvas.height = H
      const ctx = canvas.getContext('2d')
      const od  = ctx.createImageData(W*3, H)

      for (let i = 0; i < od.data.length; i += 4) {
        od.data[i]   = Math.round((chB[i]   + chO[i])   / 2)
        od.data[i+1] = Math.round((chB[i+1] + chO[i+1]) / 2)
        od.data[i+2] = Math.round((chB[i+2] + chO[i+2]) / 2)
        od.data[i+3] = 255
      }
      ctx.putImageData(od, 0, 0)
    })
  }, [b64Baseline, b64Optimized, mode])

  return (
    <div>
      <p className="font-mono text-dim text-xs mb-2 leading-relaxed">
        Baseline + Optimized blended 50/50 — aligned peaks = matching color, divergence = shift
      </p>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border" style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}

