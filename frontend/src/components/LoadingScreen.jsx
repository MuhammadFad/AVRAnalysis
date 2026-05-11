export default function LoadingScreen({ elapsed }) {
  const dots = '.'.repeat((Math.floor(elapsed / 0.5) % 4))

  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center bg-ink gap-6">
      {/* Animated grid background */}
      <div className="absolute inset-0 opacity-[0.04]"
           style={{ backgroundImage: 'linear-gradient(#5b6ef5 1px,transparent 1px),linear-gradient(90deg,#5b6ef5 1px,transparent 1px)', backgroundSize: '40px 40px' }} />

      {/* Pulsing logo mark */}
      <div className="relative">
        <div className="w-16 h-16 rounded-2xl border border-accent/40 flex items-center justify-center">
          <span className="font-mono text-accent text-xl font-semibold">AVR</span>
        </div>
        <div className="absolute inset-0 rounded-2xl border border-accent/20 scale-125 animate-ping" />
      </div>

      <div className="text-center">
        <p className="font-mono text-dim text-sm tracking-widest uppercase">
          turning the lights on{dots}&nbsp;
        </p>
        {elapsed > 5 && (
          <p className="font-mono text-muted text-xs mt-2 animate-fade-in">
            render free tier cold start — usually ~20s
          </p>
        )}
        {elapsed > 0 && (
          <p className="font-mono text-muted/60 text-xs mt-1">
            {elapsed}s elapsed
          </p>
        )}
      </div>
    </div>
  )
}
