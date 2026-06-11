import React from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'

export default function Pagination({ page, pages, onPage }) {
  if (pages <= 1) return null
  const prev = page > 1
  const next = page < pages

  const pageNums = []
  for (let i = Math.max(1, page - 2); i <= Math.min(pages, page + 2); i++) {
    pageNums.push(i)
  }

  return (
    <div className="flex items-center gap-1 justify-center mt-4">
      <button
        className="p-1.5 rounded border border-gray-200 text-gray-600
                   hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
        disabled={!prev}
        onClick={() => onPage(page - 1)}
      >
        <ChevronLeft size={16} />
      </button>

      {pageNums[0] > 1 && (
        <>
          <button className="px-3 py-1 rounded border border-gray-200 text-sm hover:bg-gray-50"
                  onClick={() => onPage(1)}>1</button>
          {pageNums[0] > 2 && <span className="text-gray-400 text-sm">…</span>}
        </>
      )}

      {pageNums.map(n => (
        <button
          key={n}
          onClick={() => onPage(n)}
          className={`px-3 py-1 rounded border text-sm transition-colors
            ${n === page
              ? 'bg-primary-600 text-white border-primary-600'
              : 'border-gray-200 text-gray-700 hover:bg-gray-50'}`}
        >
          {n}
        </button>
      ))}

      {pageNums[pageNums.length - 1] < pages && (
        <>
          {pageNums[pageNums.length - 1] < pages - 1 && (
            <span className="text-gray-400 text-sm">…</span>
          )}
          <button className="px-3 py-1 rounded border border-gray-200 text-sm hover:bg-gray-50"
                  onClick={() => onPage(pages)}>{pages}</button>
        </>
      )}

      <button
        className="p-1.5 rounded border border-gray-200 text-gray-600
                   hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
        disabled={!next}
        onClick={() => onPage(page + 1)}
      >
        <ChevronRight size={16} />
      </button>
    </div>
  )
}
