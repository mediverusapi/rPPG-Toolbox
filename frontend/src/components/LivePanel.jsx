import React, { useEffect, useRef, useState } from 'react'

export default function LivePanel({ connected, connect, disconnect, sendFrame, metrics, waveform }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    let stream
    let raf
    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
        if (videoRef.current) videoRef.current.srcObject = stream
        setRunning(true)
        loop()
      } catch (e) {
        setRunning(false)
      }
    }
    function loop() {
      if (!canvasRef.current || !videoRef.current) return
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      const w = 320
      const h = Math.round((video.videoHeight || 240) * (w / (video.videoWidth || 320)))
      canvas.width = w
      canvas.height = h
      ctx.drawImage(video, 0, 0, w, h)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
      sendFrame(dataUrl)
      raf = requestAnimationFrame(loop)
    }
    start()
    return () => {
      setRunning(false)
      if (raf) cancelAnimationFrame(raf)
      if (stream) stream.getTracks().forEach(t => t.stop())
    }
  }, [sendFrame])

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        {!connected ? (
          <button onClick={connect} className="bg-emerald-600 text-white px-3 py-1.5 rounded">Connect</button>
        ) : (
          <button onClick={disconnect} className="bg-gray-700 text-white px-3 py-1.5 rounded">Disconnect</button>
        )}
        <span className={`text-sm ${connected ? 'text-emerald-700' : 'text-gray-500'}`}>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>
      <div>
        <div className="text-sm font-medium mb-2">Camera</div>
        <video ref={videoRef} autoPlay playsInline muted className="w-full rounded border bg-black max-w-xl" />
        <canvas ref={canvasRef} className="hidden" />
      </div>

      <div>
        <div className="text-sm font-medium mb-2">Live Waveform (last ~20s)</div>
        <div className="h-56 rounded border bg-white p-2">
          <svg viewBox="0 0 500 200" className="w-full h-full">
            <polyline fill="none" stroke="#10b981" strokeWidth="2"
              points={waveform.map((y, i) => `${i},${100 - (y * 70)}`).join(' ')} />
          </svg>
        </div>
        {metrics && (
          <div className="mt-2 text-sm grid grid-cols-2 md:grid-cols-4 gap-2">
            <div>FFT HR: <b>{metrics.heart_rate?.fft_bpm}</b> bpm</div>
            <div>Peak HR: <b>{metrics.heart_rate?.peak_bpm}</b> bpm</div>
            <div>SNR: <b>{metrics.signal_quality?.snr_db}</b> dB</div>
            <div>Method: <b>{metrics.signal_quality?.method}</b></div>
          </div>
        )}
      </div>
    </div>
  )
}

