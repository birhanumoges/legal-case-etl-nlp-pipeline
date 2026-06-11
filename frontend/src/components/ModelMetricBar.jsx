import React from 'react'

export default function ModelMetricBar({ label, value, max = 1, color = 'blue' }) {
  const pct = Math.min((value / max) * 100, 100)
  const colors = {
    blue:   'bg-blue-500',
    green:  'bg-green-500',
    yellow: 'bg-yellow-500',
    red:    'bg-red-500',
  }
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-600 mb-0.5">
        <span>{label}</span>
        <span className="font-medium">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all ${colors[color]}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
