import React from 'react'
import { useQuery } from 'react-query'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { FileText, CheckCircle, HelpCircle, Quote, TrendingUp } from 'lucide-react'
import { analyticsAPI } from '../services/api'
import StatCard from '../components/StatCard'
import Loading from '../components/Loading'
import { fmt, CASE_TYPE_COLORS, VERDICT_COLORS } from '../utils/helpers'

export default function Dashboard() {
  const { data: statsRes, isLoading } = useQuery('stats', () =>
    analyticsAPI.stats().then(r => r.data))
  const { data: forecastRes } = useQuery('forecast', () =>
    analyticsAPI.forecast().then(r => r.data))

  if (isLoading) return <Loading text="Loading dashboard…" />

  const stats = statsRes || {}
  const forecast = forecastRes?.forecast || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Legal NLP Pipeline — corpus overview and key metrics
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={FileText}     label="Total Cases"       value={fmt.number(stats.total_cases)}    color="blue"   />
        <StatCard icon={CheckCircle}  label="Clean Labelled"    value={fmt.number(stats.clean_cases)}    color="green"  />
        <StatCard icon={HelpCircle}   label="Unknown / Unlabel" value={fmt.number(stats.unknown_cases)}  color="yellow" />
        <StatCard icon={Quote}        label="Avg Citations"      value={stats.avg_citations?.toFixed(2)}  color="purple" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Case Type */}
        <div className="card">
          <h3 className="mb-4">Case Type Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={stats.case_type_dist || []} layout="vertical"
                      margin={{ left: 10, right: 20, top: 0, bottom: 0 }}>
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" width={75} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.number(v), 'Cases']} />
              <Bar dataKey="count" radius={[0,4,4,0]}>
                {(stats.case_type_dist || []).map((d, i) => (
                  <Cell key={i} fill={CASE_TYPE_COLORS[d.label] || '#6b7280'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Verdict */}
        <div className="card">
          <h3 className="mb-4">Verdict Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={stats.verdict_dist || []}
                dataKey="count"
                nameKey="label"
                cx="50%" cy="50%"
                outerRadius={80}
                label={({ label, percent }) => `${label} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {(stats.verdict_dist || []).map((d, i) => (
                  <Cell key={i} fill={VERDICT_COLORS[d.label] || '#6b7280'} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.number(v), 'Cases']} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Forecast */}
      {forecast.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp size={18} className="text-primary-500" />
            <h3>ARIMA(2,2,1) — 5-Year Case Volume Forecast</h3>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={forecast} margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
              <XAxis dataKey="year" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [Math.round(v), 'Forecast Cases']} />
              <Bar dataKey="forecast" fill="#3b82f6" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top courts */}
      {(stats.top_courts || []).length > 0 && (
        <div className="card">
          <h3 className="mb-4">Top Courts</h3>
          <div className="space-y-2">
            {stats.top_courts.slice(0, 6).map((c, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-4 text-right">{i + 1}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between text-xs mb-0.5">
                    <span className="truncate text-gray-700">{c.label}</span>
                    <span className="text-gray-400 ml-2 flex-shrink-0">{fmt.number(c.count)}</span>
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full bg-primary-400"
                      style={{ width: `${c.percent}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
