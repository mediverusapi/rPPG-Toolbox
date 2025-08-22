import { useCallback, useState } from 'react'
import { uploadVideo } from '../api/client'

export default function useUpload() {
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const upload = useCallback(async (file, extras = {}) => {
    setError(null)
    setLoading(true)
    setProgress(0)
    setResult(null)
    try {
      const res = await uploadVideo(file, extras, setProgress)
      setResult(res)
      return res
    } catch (e) {
      setError(e)
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  return { upload, progress, loading, error, result }
}

