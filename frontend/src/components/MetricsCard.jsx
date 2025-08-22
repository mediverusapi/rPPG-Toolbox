import React from 'react'

function Row({ label, value, unit }) {
  return (
    <div className="flex justify-between text-sm py-0.5">
      <div className="text-gray-600">{label}</div>
      <div className="font-medium">{value}{unit ? ` ${unit}` : ''}</div>
    </div>
  )
}

export default function MetricsCard({ data }) {
  if (!data) return null
  const hr = data.heart_rate || {}
  const hrv = data.heart_rate_variability || {}
  const stress = data.stress_and_relaxation || {}
  const qual = data.signal_quality || {}
  const pm = data.per_method || {}

  return (
    <div className="bg-white border rounded p-4 space-y-3">
      <div className="text-base font-semibold">Metrics</div>
      <div className="space-y-4">
        <div>
          <div className="font-medium mb-1">Heart Rate</div>
          <Row label="FFT" value={hr.fft_bpm} unit="bpm" />
          <Row label="Peak" value={hr.peak_bpm} unit="bpm" />
        </div>
        <div>
          <div className="font-medium mb-1">Quality</div>
          <Row label="SNR" value={qual.snr_db} unit="dB" />
          <Row label="Quality Score" value={qual.quality_score} />
          <Row label="Selected Method" value={qual.method} />
        </div>
        <div>
          <div className="font-medium mb-1">HRV</div>
          <Row label="RMSSD" value={hrv.rmssd_ms} unit="ms" />
          <Row label="SDNN" value={hrv.sdnn_ms} unit="ms" />
          <Row label="Mean RR" value={hrv.mean_rr_ms} unit="ms" />
          <Row label="pNN50" value={hrv.pnn50_percent} unit="%" />
        </div>
        <div>
          <div className="font-medium mb-1">Stress</div>
          <Row label="Stress Index" value={stress.stress_index} />
          <Row label="Parasympathetic" value={stress.parasympathetic_tone} />
          <Row label="Level" value={stress.stress_level} />
          <Row label="Relaxation" value={stress.relaxation_state} />
        </div>
      </div>

      {Object.keys(pm).length > 0 && (
        <div>
          <div className="font-medium mb-2">Per-method comparison</div>
          <div className="grid md:grid-cols-3 gap-3">
            {Object.entries(pm).map(([name, v]) => (
              <div key={name} className="border rounded p-3 text-sm bg-gray-50">
                <div className="font-semibold mb-1">{name}</div>
                <Row label="SNR" value={v.snr_db} unit="dB" />
                <Row label="FFT HR" value={v.fft_bpm} unit="bpm" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

