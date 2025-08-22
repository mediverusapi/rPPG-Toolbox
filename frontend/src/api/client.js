const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8002'

export async function uploadVideo(file, { age, bmi } = {}, onProgress) {
  const form = new FormData()
  form.append('video', file)
  if (age != null) form.append('age', age)
  if (bmi != null) form.append('bmi', bmi)

  const xhr = new XMLHttpRequest()
  const url = `${API_BASE}/predict`

  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', url)
    xhr.responseType = 'json'
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response)
      else reject(new Error(xhr.response?.detail || `Upload failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Network error'))
    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }
    xhr.send(form)
  })

  return promise
}

export function previewUrl(path) {
  if (!path) return null
  // path example: /preview/xyz.mp4 or /waveform/abc.png
  return `${API_BASE}${path}`
}

