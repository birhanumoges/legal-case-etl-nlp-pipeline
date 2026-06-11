import React from 'react'

export default function Loading({ text = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-3">
      <div className="w-8 h-8 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
      <p className="text-sm text-gray-500">{text}</p>
    </div>
  )
}

export function Skeleton({ className = '' }) {
  return <div className={`bg-gray-200 animate-pulse rounded ${className}`} />
}
