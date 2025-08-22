import { useCallback, useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8002'

export default function useLive() {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [waveform, setWaveform] = useState([])
  const [error, setError] = useState(null)

  const connect = useCallback(() => {
    setError(null)
    try {
      const ws = new WebSocket(API_BASE.replace(/^http/, 'ws') + '/ws/live')
      ws.onopen = () => setConnected(true)
      ws.onclose = () => { setConnected(false); wsRef.current = null }
      ws.onerror = (e) => setError(e)
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data)
          if (msg.type === 'metrics') {
            setMetrics({
              heart_rate: msg.heart_rate,
              heart_rate_variability: msg.heart_rate_variability,
              signal_quality: msg.signal_quality,
            })
            if (Array.isArray(msg.waveform)) setWaveform(msg.waveform)
          }
        } catch {}
      }
      wsRef.current = ws
    } catch (e) {
      setError(e)
    }
  }, [])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.send(JSON.stringify({ type: 'close' })) } catch {}
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  useEffect(() => () => { if (wsRef.current) wsRef.current.close() }, [])

  const sendFrame = useCallback((dataUrl) => {
    if (!wsRef.current || wsRef.current.readyState !== 1) return
    wsRef.current.send(JSON.stringify({ type: 'frame', data: dataUrl, ts: Date.now() }))
  }, [])

  return { connected, connect, disconnect, sendFrame, metrics, waveform, error }
}

