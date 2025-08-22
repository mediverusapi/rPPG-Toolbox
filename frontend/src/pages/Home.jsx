import React, { useState } from 'react'
import useUpload from '../hooks/useUpload'
import UploadForm from '../components/UploadForm'
import VideoPreview from '../components/VideoPreview'
import MetricsCard from '../components/MetricsCard'
import useLive from '../hooks/useLive'
import LivePanel from '../components/LivePanel'

export default function Home() {
  const { upload, progress, loading, error, result } = useUpload()
  const live = useLive()
  const [mode, setMode] = useState('upload') // 'upload' | 'live'

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button onClick={()=>setMode('upload')} className={`px-3 py-1.5 rounded border ${mode==='upload'?'bg-emerald-50 border-emerald-600 text-emerald-700':'bg-white'}`}>Upload</button>
        <button onClick={()=>setMode('live')} className={`px-3 py-1.5 rounded border ${mode==='live'?'bg-emerald-50 border-emerald-600 text-emerald-700':'bg-white'}`}>Live</button>
      </div>

      {mode === 'upload' && (
        <>
          <UploadForm onSubmit={upload} loading={loading} progress={progress} />
          {error && <div className="text-red-600 text-sm">{String(error.message || error)}</div>}
          {result && (
            <div className="space-y-4">
              <VideoPreview
                previewPath={result?.preprocessing?.enhanced_preview}
                waveformPath={result?.preprocessing?.waveform_preview}
                spectrumPath={result?.preprocessing?.spectrum_preview}
                peaksPath={result?.preprocessing?.peaks_preview}
              />
              <MetricsCard data={result} />
            </div>
          )}
        </>
      )}

      {mode === 'live' && (
        <LivePanel
          connected={live.connected}
          connect={live.connect}
          disconnect={live.disconnect}
          sendFrame={live.sendFrame}
          metrics={live.metrics}
          waveform={live.waveform}
        />
      )}
    </div>
  )
}

