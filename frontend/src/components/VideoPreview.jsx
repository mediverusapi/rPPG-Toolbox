import React, { useState } from 'react'
import { previewUrl } from '../api/client'

function ZoomModal({ src, onClose }) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" onClick={onClose}>
      <div className="max-w-[95vw] max-h-[95vh] overflow-auto bg-white rounded p-2" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium">BVP with Peaks (click & drag to pan)</span>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
        </div>
        <img src={src} className="block" style={{minWidth: '1600px', cursor: 'grab'}} />
      </div>
    </div>
  )
}

export default function VideoPreview({ previewPath, waveformPath, spectrumPath, peaksPath }) {
  const [zoomSrc, setZoomSrc] = useState(null)
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
          <div className="text-sm font-medium mb-1">BVP with Detected Peaks (click to zoom)</div>
          <div className="w-full h-80 rounded border bg-white overflow-auto cursor-pointer" onClick={() => setZoomSrc(pUrl)}>
            <img src={pUrl} className="min-w-full h-full object-contain" style={{minWidth: '1200px'}} />
          </div>
        </div>
      )}
      {zoomSrc && <ZoomModal src={zoomSrc} onClose={() => setZoomSrc(null)} />}
    </div>
  )
}

