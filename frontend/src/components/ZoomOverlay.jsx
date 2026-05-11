import { useRef, useState, useCallback, useEffect } from 'react'
import { createPortal } from 'react-dom'

/**
 * Draw a 2x-zoomed crop centred at (cx, cy) from a base64 image onto a canvas.
 * Returns void — used imperatively.
 */
function drawCrop(canvas, b64, naturalW, naturalH, cx, cy, cropPx = 120) {
  if (!canvas || !b64) return
  const ctx = canvas.getContext('2d')
  const img = new Image()
  img.onload = () => {
    const half  = cropPx / 2
    const sx    = Math.max(0, cx - half)
    const sy    = Math.max(0, cy - half)
    const sw    = Math.min(naturalW - sx, cropPx)
    const sh    = Math.min(naturalH - sy, cropPx)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height)
  }
  img.src = `data:image/png;base64,${b64}`
}

export default function ZoomOverlay({ b64, label, b64Baseline, b64Optimized, onClose }) {
  const [hoverCard, setHoverCard] = useState(null)   // { x, y, imgX, imgY } screen coords
  const imgRef      = useRef()
  const baseRef     = useRef()
  const curRef      = useRef()
  const optRef      = useRef()

  // Natural dimensions of the main image — needed for coordinate mapping
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 })

  const handleMouseMove = useCallback(e => {
    const rect = imgRef.current?.getBoundingClientRect()
    if (!rect) return

    const relX = e.clientX - rect.left
    const relY = e.clientY - rect.top

    // Normalize to [0,1] then map to natural image coords
    const nx = (relX / rect.width)  * naturalSize.w
    const ny = (relY / rect.height) * naturalSize.h

    setHoverCard({ screenX: e.clientX, screenY: e.clientY, imgX: nx, imgY: ny })

    // Draw crops
    drawCrop(baseRef.current, b64Baseline, naturalSize.w, naturalSize.h, nx, ny)
    drawCrop(curRef.current,  b64,         naturalSize.w, naturalSize.h, nx, ny)
    drawCrop(optRef.current,  b64Optimized,naturalSize.w, naturalSize.h, nx, ny)
  }, [naturalSize, b64, b64Baseline, b64Optimized])

  const handleMouseLeave = () => setHoverCard(null)

  useEffect(() => {
    const handler = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  // Card position: clamp to viewport using clientX/Y (viewport coords)
  const cardW = 364
  const cardH = 148
  let cardLeft = hoverCard ? hoverCard.screenX + 20 : 0
  let cardTop  = hoverCard ? hoverCard.screenY + 20 : 0
  if (hoverCard && cardLeft + cardW > window.innerWidth  - 8) cardLeft = hoverCard.screenX - cardW - 20
  if (hoverCard && cardTop  + cardH > window.innerHeight - 8) cardTop  = hoverCard.screenY - cardH - 20

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90 backdrop-blur-sm"
         onClick={onClose}>
      <div className="relative" onClick={e => e.stopPropagation()}>
        <img
          ref={imgRef}
          src={`data:image/png;base64,${b64}`}
          alt={label}
          className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg block"
          onLoad={e => setNaturalSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ cursor: 'crosshair' }}
        />
        <div className="absolute top-3 left-3 font-mono text-xs text-dim bg-ink/80 px-2 py-1 rounded">
          {label}
        </div>
        <button onClick={onClose}
                className="absolute top-3 right-3 font-mono text-xs text-dim hover:text-text bg-ink/80 px-2 py-1 rounded">
          ✕ ESC
        </button>
      </div>

      {/* Hover card — rendered at fixed viewport coords, never clipped by scroll */}
      {hoverCard && (
        <div
          className="fixed pointer-events-none z-[10000] rounded-xl border border-border bg-panel shadow-2xl p-3 flex gap-2"
          style={{ left: cardLeft, top: cardTop, width: cardW }}
        >
          {[
            { ref: baseRef, label: 'Baseline' },
            { ref: curRef,  label: label },
            { ref: optRef,  label: 'Optimized' },
          ].map(({ ref, label: l }) => (
            <div key={l} className="flex-1 text-center">
              <canvas ref={ref} width={100} height={100}
                      className="w-full aspect-square rounded bg-ink border border-border" />
              <p className="font-mono text-dim text-[10px] mt-1 truncate">{l}</p>
            </div>
          ))}
        </div>
      )}
    </div>,
    document.body
  )
}
