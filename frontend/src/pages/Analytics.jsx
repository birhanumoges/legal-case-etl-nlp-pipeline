import React, { useState } from 'react'
import { useQuery } from 'react-query'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'
import { BarChart2, TrendingUp, Calendar, BookOpen } from 'lucide-react'
import { analyticsAPI } from '../services/api'
import Loading from '../components/Loading'
import { fmt, CASE_TYPE_COLORS, VERDICT_COLORS } from '../utils/helpers'

const TABS = [
  { id: 'overview',  label: 'Overview',       icon: BarChart2   },
  { id: 'trends',    label: 'Yearly Trends',   icon: TrendingUp  },
  { id: 'forecast',  label: 'Forecast',        icon: Calendar    },
  { id: 'subtype',   label: 'Sub-Types',       icon: BookOpen    },
]

function SectionTitle({ title, sub }) {
  return (
    <div className="mb-4">
      <h3>{title}</h3>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  )
}

export default function Analytics() {
  const [tab, setTab] = useState('overview')

  const { data: caseTypeDist }  = useQuery('caseTypeDist',  () => analyticsAPI.caseTypeDist().then(r => r.data))
  const { data: verdictDist }   = useQuery('verdictDist',   () => analyticsAPI.verdictDist().then(r => r.data))
  const { data: subTypeDist }   = useQuery('subTypeDist',   () => analyticsAPI.subTypeDist().then(r => r.data))
  const { data: yearly }        = useQuery('yearly',        () => analyticsAPI.yearly().then(r => r.data))
  const { data: forecastData }  = useQuery('forecast',      () => analyticsAPI.forecast().then(r => r.data))

  const yearlyItems = yearly?.items || []
  const forecast    = forecastData?.forecast || []

  // combine history + forecast for a single chart
  const combinedTimeline = [
    ...yearlyItems.slice(-20).map(y => ({ year: y.year, cases: y.n_cases, type: 'historical' })),
    ...forecast.map(f => ({ year: f.year, forecast: Math.round(f.forecast), type: 'forecast' })),
  ]

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2">
        <BarChart2 size={20} className="text-primary-500" />
        <h1>Analytics</h1>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm
                        font-medium transition-colors
                        ${tab === id
                          ? 'bg-white text-primary-700 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'}`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Case Type bar */}
          <div className="card">
            <SectionTitle title="Case Type Distribution"
                          sub="Proportion of each top-level case category" />
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={caseTypeDist || []}
                        layout="vertical"
                        margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="label" width={80} tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(v, n, p) => [
                    `${fmt.number(v)} (${p.payload.percent?.toFixed(1)}%)`, 'Cases',
                  ]}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {(caseTypeDist || []).map((d, i) => (
                    <Cell key={i} fill={CASE_TYPE_COLORS[d.label] || '#6b7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Verdict bar */}
          <div className="card">
            <SectionTitle title="Verdict Distribution"
                          sub="Canonical verdict class proportions" />
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={verdictDist || []}
                        margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(v, n, p) => [
                    `${fmt.number(v)} (${p.payload.percent?.toFixed(1)}%)`, 'Cases',
                  ]}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {(verdictDist || []).map((d, i) => (
                    <Cell key={i} fill={VERDICT_COLORS[d.label] || '#6b7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── YEARLY TRENDS TAB ── */}
      {tab === 'trends' && (
        <div className="space-y-6">
          {yearlyItems.length === 0 ? (
            <div className="card text-center py-12 text-gray-400">
              No yearly data available. Run main.py first.
            </div>
          ) : (
            <>
              {/* Volume line */}
              <div className="card">
                <SectionTitle title="Annual Case Volume"
                              sub="Number of cases per year (metadata-recovered records)" />
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={yearlyItems}
                             margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="year"  tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip formatter={v => [fmt.number(v), 'Cases']} />
                    <Line type="monotone" dataKey="n_cases" stroke="#3b82f6"
                          strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Citations line */}
              <div className="card">
                <SectionTitle title="Average Citations Per Case"
                              sub="Year-on-year citation density trend" />
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={yearlyItems}
                             margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip formatter={v => [v?.toFixed(2), 'Avg Citations']} />
                    <Line type="monotone" dataKey="avg_citations" stroke="#10b981"
                          strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Table */}
              <div className="card p-0 overflow-hidden">
                <div className="px-5 py-3 border-b border-gray-100">
                  <h3 className="text-sm">Yearly Statistics Table</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-100">
                        {['Year', 'Cases', 'Avg Citations', 'Total Citations', 'Top Type', 'Top Verdict'].map(h => (
                          <th key={h} className="table-header">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                      {[...yearlyItems].reverse().map((row, i) => (
                        <tr key={i} className={i % 2 === 0 ? '' : 'bg-gray-50/50'}>
                          <td className="table-cell font-medium">{row.year}</td>
                          <td className="table-cell">{fmt.number(row.n_cases)}</td>
                          <td className="table-cell">{row.avg_citations?.toFixed(2)}</td>
                          <td className="table-cell">{fmt.number(row.total_citations)}</td>
                          <td className="table-cell">
                            <span className="badge badge-blue text-xs">
                              {row.top_case_type || '—'}
                            </span>
                          </td>
                          <td className="table-cell">
                            <span className="badge badge-green text-xs">
                              {row.top_verdict || '—'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── FORECAST TAB ── */}
      {tab === 'forecast' && (
        <div className="space-y-6">
          {/* Model info */}
          {forecastData && (
            <div className="card flex gap-6">
              <div>
                <p className="text-xs text-gray-500 uppercase">Method</p>
                <p className="font-semibold">{forecastData.method}</p>
              </div>
              {forecastData.order && (
                <div>
                  <p className="text-xs text-gray-500 uppercase">Order (p,d,q)</p>
                  <p className="font-semibold">({forecastData.order.join(',')})</p>
                </div>
              )}
              {forecastData.aic && (
                <div>
                  <p className="text-xs text-gray-500 uppercase">AIC</p>
                  <p className="font-semibold">{forecastData.aic?.toFixed(2)}</p>
                </div>
              )}
            </div>
          )}

          {/* Combined chart */}
          <div className="card">
            <SectionTitle title="Case Volume Forecast (2020–2024)"
                          sub="Blue bars = historical; orange bars = ARIMA forecast" />
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={combinedTimeline}
                        margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="cases"    name="Historical" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="forecast" name="Forecast"   fill="#f97316" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Forecast table */}
          {forecast.length > 0 && (
            <div className="card">
              <h3 className="mb-3">Forecast Values</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-100">
                      <th className="table-header">Year</th>
                      <th className="table-header">Forecast Cases</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {forecast.map((f, i) => (
                      <tr key={i}>
                        <td className="table-cell font-medium">{f.year}</td>
                        <td className="table-cell">{Math.round(f.forecast).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── SUBTYPE TAB ── */}
      {tab === 'subtype' && (
        <div className="card">
          <SectionTitle title="Sub-Type Distribution (Top 21)"
                        sub="Fine-grained hierarchical case sub-categories" />
          {!subTypeDist ? <Loading /> : (
            <ResponsiveContainer width="100%" height={560}>
              <BarChart
                data={(subTypeDist || []).map(d => ({
                  ...d,
                  label: d.label.includes(':')  ? d.label.split(':').pop().trim()
                       : d.label.includes('__') ? d.label.split('__').pop()
                       : d.label,
                  fullLabel: d.label,
                }))}
                layout="vertical"
                margin={{ left: 12, right: 40, top: 0, bottom: 0 }}
              >
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(v, n, p) => [
                    `${fmt.number(v)} (${p.payload.percent?.toFixed(1)}%)`,
                    p.payload.fullLabel,
                  ]}
                />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
    </div>
  )
}
