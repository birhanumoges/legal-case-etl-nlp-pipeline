import React from 'react'
import { verdictBadge, caseTypeBadge } from '../utils/helpers'

export function VerdictBadge({ verdict }) {
  if (!verdict) return <span className="badge badge-gray">—</span>
  return <span className={verdictBadge(verdict)}>{verdict}</span>
}

export function CaseTypeBadge({ type }) {
  if (!type) return <span className="badge badge-gray">—</span>
  return <span className={caseTypeBadge(type)}>{type}</span>
}

export function SubTypeBadge({ subType }) {
  if (!subType) return <span className="badge badge-gray">—</span>
  // format: "CIVIL__Appeal" → "Appeal"
  const label = subType.includes(':') ? subType.split(':').pop().trim()
              : subType.includes('__') ? subType.split('__').pop()
              : subType
  return <span className="badge badge-purple">{label}</span>
}
