import React, { useRef, useState } from 'react'

export default function UploadForm({ onSubmit, loading, progress }) {
  const fileRef = useRef(null)
  const [age, setAge] = useState('')
  const [bmi, setBmi] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const file = fileRef.current?.files?.[0]
    if (!file) return
    onSubmit(file, { age: age || undefined, bmi: bmi || undefined })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="flex items-center gap-3">
        <input type="file" accept="video/*" ref={fileRef} disabled={loading} className="block" />
        <input type="number" inputMode="decimal" placeholder="Age (optional)" value={age} onChange={e=>setAge(e.target.value)} className="border rounded px-2 py-1 w-36" />
        <input type="number" inputMode="decimal" placeholder="BMI (optional)" value={bmi} onChange={e=>setBmi(e.target.value)} className="border rounded px-2 py-1 w-36" />
        <button disabled={loading} className="bg-emerald-600 text-white px-3 py-1.5 rounded disabled:opacity-50">{loading ? 'Uploading…' : 'Analyze'}</button>
      </div>
      {loading ? (
        <div className="h-2 bg-gray-200 rounded overflow-hidden">
          <div className="h-full bg-emerald-500" style={{ width: `${progress}%` }} />
        </div>
      ) : null}
    </form>
  )
}

