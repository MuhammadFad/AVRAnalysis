import { useState } from 'react'
import { usePing }      from './hooks/usePing'
import LoadingScreen    from './components/LoadingScreen'
import UploadScreen     from './components/UploadScreen'
import ResultsScreen    from './components/ResultsScreen'

export default function App() {
  const { ready, elapsed } = usePing()
  const [result, setResult] = useState(null)

  if (!ready) return <LoadingScreen elapsed={elapsed} />
  if (!result) return <UploadScreen onResult={setResult} />
  return <ResultsScreen result={result} onReset={() => setResult(null)} />
}
