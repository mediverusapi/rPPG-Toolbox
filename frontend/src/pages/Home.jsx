import React from 'react'
import useUpload from '../hooks/useUpload'
import UploadForm from '../components/UploadForm'
import VideoPreview from '../components/VideoPreview'
import MetricsCard from '../components/MetricsCard'

export default function Home() {
  const { upload, progress, loading, error, result } = useUpload()

  return (
    <div className="space-y-4">
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
    </div>
  )
}

