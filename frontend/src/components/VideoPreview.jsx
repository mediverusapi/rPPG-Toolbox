import React from 'react'
import { previewUrl } from '../api/client'

export default function VideoPreview({ previewPath, waveformPath, spectrumPath, peaksPath }) {
  const vUrl = previewUrl(previewPath)
  const wUrl = previewUrl(waveformPath)
  const sUrl = previewUrl(spectrumPath)
  const pUrl = previewUrl(peaksPath)
  if (!vUrl && !wUrl && !sUrl && !pUrl) return null

  return (
    <div className="grid md:grid-cols-2 gap-4">
      {vUrl && (
        <div>
          <div className="text-sm font-medium mb-1">Enhanced video</div>
          <video src={vUrl} controls className="w-full rounded border" />
          <a href={vUrl} download className="text-emerald-700 text-sm inline-block mt-1">Download</a>
        </div>
      )}
      {wUrl && (
        <div>
          <div className="text-sm font-medium mb-1">Waveform</div>
          <img src={wUrl} className="w-full rounded border bg-white" />
        </div>
      )}
      {sUrl && (
        <div>
          <div className="text-sm font-medium mb-1">Spectrum</div>
          <img src={sUrl} className="w-full rounded border bg-white" />
        </div>
      )}
      {pUrl && (
        <div>
          <div className="text-sm font-medium mb-1">Peaks</div>
          <img src={pUrl} className="w-full rounded border bg-white" />
        </div>
      )}
    </div>
  )
}

