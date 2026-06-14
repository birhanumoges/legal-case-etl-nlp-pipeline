import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Filter, X } from 'lucide-react'
import { useQuery } from 'react-query'
import { casesAPI } from '../services/api'
import { VerdictBadge, CaseTypeBadge } from '../components/Badge'
import Pagination from '../components/Pagination'
import Loading from '../components/Loading'
import { truncate, fmt } from '../utils/helpers'

const CASE_TYPES = ['', 'CIVIL', 'CRIMINAL', 'CONTRACT', 'PROPERTY', 'TORTS']
const VERDICTS   = ['', 'AFFIRMED', 'REVERSED', 'DENIED', 'GRANTED']

export default function Cases() {
  const navigate = useNavigate()
  const [params, setParams] = useState({ page: 1, size: 20 })
  const [filters, setFilters] = useState({ case_type: '', verdict: '', court: '', year_from: '', year_to: '' })
  const [showFilters, setShowFilters] = useState(false)
  const [q, setQ] = useState('')

  const queryParams = {
    ...params,
    ...(filters.case_type && { case_type: filters.case_type }),
    ...(filters.verdict    && { verdict:   filters.verdict }),
    ...(filters.court      && { court:     filters.court }),
    ...(filters.year_from  && { year_from: Number(filters.year_from) }),
    ...(filters.year_to    && { year_to:   Number(filters.year_to)   }),
  }

  const { data, isLoading, isFetching } = useQuery(
    ['cases', queryParams],
    () => casesAPI.list(queryParams).then(r => r.data),
    { keepPreviousData: true }
  )

  const handleSearch = (e) => {
    e.preventDefault()
    // text search uses POST /cases/search; update params to trigger re-fetch
    setParams(p => ({ ...p, page: 1 }))
  }

  const clearFilters = () => {
    setFilters({ case_type: '', verdict: '', court: '', year_from: '', year_to: '' })
    setParams({ page: 1, size: 20 })
  }

  const hasFilters = Object.values(filters).some(Boolean)

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h1>Cases</h1>
        <span className="text-sm text-gray-500">
          {data ? fmt.number(data.total) + ' total' : ''}
        </span>
      </div>

      {/* Search + filter bar */}
      <div className="card p-4 space-y-3">
        <form onSubmit={handleSearch} className="flex gap-2">
          <div className="relative flex-1">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              className="input pl-9"
              placeholder="Search case text…"
              value={q}
              onChange={e => setQ(e.target.value)}
            />
          </div>
          <button type="submit" className="btn-primary px-4">Search</button>
          <button
            type="button"
            className={`btn-secondary px-3 ${showFilters ? 'bg-primary-50 border-primary-300' : ''}`}
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter size={16} />
          </button>
          {hasFilters && (
            <button type="button" className="btn-secondary px-3 text-red-500"
                    onClick={clearFilters}>
              <X size={16} />
            </button>
          )}
        </form>

        {showFilters && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 pt-1">
            <div>
              <label className="label text-xs">Case Type</label>
              <select className="input text-sm" value={filters.case_type}
                      onChange={e => setFilters({ ...filters, case_type: e.target.value })}>
                {CASE_TYPES.map(t => <option key={t} value={t}>{t || 'All'}</option>)}
              </select>
            </div>
            <div>
              <label className="label text-xs">Verdict</label>
              <select className="input text-sm" value={filters.verdict}
                      onChange={e => setFilters({ ...filters, verdict: e.target.value })}>
                {VERDICTS.map(v => <option key={v} value={v}>{v || 'All'}</option>)}
              </select>
            </div>
            <div>
              <label className="label text-xs">Court</label>
              <input className="input text-sm" placeholder="e.g. Superior"
                     value={filters.court}
                     onChange={e => setFilters({ ...filters, court: e.target.value })} />
            </div>
            <div>
              <label className="label text-xs">Year From</label>
              <input className="input text-sm" type="number" placeholder="1800"
                     value={filters.year_from}
                     onChange={e => setFilters({ ...filters, year_from: e.target.value })} />
            </div>
            <div>
              <label className="label text-xs">Year To</label>
              <input className="input text-sm" type="number" placeholder="2020"
                     value={filters.year_to}
                     onChange={e => setFilters({ ...filters, year_to: e.target.value })} />
            </div>
          </div>
        )}
      </div>

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        {(isLoading || isFetching) && (
          <div className="h-1 bg-primary-200">
            <div className="h-1 bg-primary-500 animate-pulse" style={{ width: '60%' }} />
          </div>
        )}
        {isLoading ? <Loading /> : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-100">
                  {['Case ID', 'Case Name', 'Year', 'Court', 'Type', 'Verdict', 'Citations'].map(h => (
                    <th key={h} className="table-header">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {(data?.items || []).map(c => (
                  <tr
                    key={c.case_id}
                    className="hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => navigate(`/cases/${c.case_id}`)}
                  >
                    <td className="table-cell font-mono text-xs text-gray-500">{c.case_id}</td>
                    <td className="table-cell max-w-xs">
                      <span className="text-sm">{truncate(c.case_name, 50)}</span>
                    </td>
                    <td className="table-cell text-gray-500">{c.year || '—'}</td>
                    <td className="table-cell text-xs text-gray-500 max-w-[150px] truncate">
                      {truncate(c.court, 30)}
                    </td>
                    <td className="table-cell"><CaseTypeBadge type={c.case_type_mapped} /></td>
                    <td className="table-cell"><VerdictBadge verdict={c.verdict_mapped} /></td>
                    <td className="table-cell text-gray-500">{c.num_citations ?? 0}</td>
                  </tr>
                ))}
                {(data?.items || []).length === 0 && (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-gray-400 text-sm">
                      No cases found matching your criteria.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Pagination
        page={params.page}
        pages={data?.pages || 1}
        onPage={p => setParams(prev => ({ ...prev, page: p }))}
      />
    </div>
  )
}
