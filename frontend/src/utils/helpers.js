export const fmt = {
  number: (n) => (n ?? 0).toLocaleString(),
  pct:    (n) => `${(n ?? 0).toFixed(1)}%`,
  f1:     (n) => (n ?? 0).toFixed(4),
}

export const CASE_TYPE_COLORS = {
  CIVIL:    '#3b82f6',
  CRIMINAL: '#ef4444',
  CONTRACT: '#10b981',
  PROPERTY: '#f59e0b',
  TORTS:    '#8b5cf6',
}

export const VERDICT_COLORS = {
  AFFIRMED: '#10b981',
  REVERSED: '#ef4444',
  DENIED:   '#f59e0b',
  GRANTED:  '#3b82f6',
  OTHER:    '#6b7280',
}

export function verdictBadge(v) {
  const map = {
    AFFIRMED: 'badge-green',
    REVERSED: 'badge-red',
    DENIED:   'badge-yellow',
    GRANTED:  'badge-blue',
  }
  return map[v] || 'badge-gray'
}

export function caseTypeBadge(t) {
  const map = {
    CIVIL:    'badge-blue',
    CRIMINAL: 'badge-red',
    CONTRACT: 'badge-green',
    PROPERTY: 'badge-yellow',
    TORTS:    'badge-purple',
  }
  return map[t] || 'badge-gray'
}

export function truncate(str, n = 120) {
  if (!str) return '—'
  return str.length > n ? str.slice(0, n) + '…' : str
}
