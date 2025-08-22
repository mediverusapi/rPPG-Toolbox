export function round(value, digits = 2) {
  if (value == null || isNaN(value)) return ''
  const p = Math.pow(10, digits)
  return Math.round(Number(value) * p) / p
}

