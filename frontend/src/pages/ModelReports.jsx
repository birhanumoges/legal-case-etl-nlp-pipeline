import React, { useState } from 'react'
import { useQuery } from 'react-query'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, Legend,
} from 'recharts'
import { Search, Trophy, Clock } from 'lucide-react'
import { analyticsAPI } from '../services/api'
import Loading from '../components/Loading'
import ModelMetricBar from '../components/ModelMetricBar'
import { fmt } from '../utils/helpers'

const TARGET_LABELS = {
  Case_Type_Mapped: 'Case Type',
  Sub_Type_Mapped:  'Sub-Type',
  Verdict_Mapped:   'Verdict',
}

const MODEL_COLORS = {
  'Logistic Regression': '#3b82f6',
  'Linear SVM':          '#10b981',
  'XGBoost':             '#f59e0b',
  'Hierarchical SVM':    '#8b5cf6',
}

function ReportCard({ report }) {
  const label = TARGET_LABELS[report.target] || report.target

  const radarData = [
    { metric: 'Accuracy',  value: report.test_accuracy    * 100 },
    { metric: 'Macro-F1',  value: report.test_macro_f1    * 100 },
    { metric: 'Precision', value: report.test_precision   * 100 },
    { metric: 'Recall',    value: report.test_recall      * 100 },
    { metric: 'Wtd-F1',    value: report.test_weighted_f1 * 100 },
  ]

  const compData = (report.all_model_comparison || []).map(m => ({
    model:      m.model,
    'Val F1':   +(m.val_macro_f1  * 100).toFixed(1),
    'Val Acc':  +(m.val_accuracy  * 100).toFixed(1),
    'Time (s)': m.train_time_s,
  }))

  return (
    <div className="card space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="flex items-center gap-2">
            {label}
            <span className="badge badge-yellow text-xs flex items-center gap-1">
              <Trophy size={10} /> {report.best_model}
            </span>
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">Test set results</p>
        </div>
      </div>

      {/* Metric bars */}
      <div className="space-y-2.5">
        <ModelMetricBar label="Accuracy"           value={report.test_accuracy}    color="blue"  />
        <ModelMetricBar label="Macro F1"           value={report.test_macro_f1}    color="green" />
        <ModelMetricBar label="Weighted F1"        value={report.test_weighted_f1} color="green" />
        <ModelMetricBar label="Macro Precision"    value={report.test_precision}   color="yellow"/>
        <ModelMetricBar label="Macro Recall"       value={report.test_recall}      color="red"   />
      </div>

      {/* Radar + comparison charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Radar */}
        <div>
          <p className="text-xs text-gray-500 mb-2">Best Model Radar</p>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={radarData} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
              <Radar name={report.best_model} dataKey="value"
                     stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Model comparison bars */}
        <div>
          <p className="text-xs text-gray-500 mb-2">Validation Macro-F1 Comparison</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={compData} margin={{ left: 0, right: 8, top: 5, bottom: 0 }}>
              <XAxis dataKey="model" tick={{ fontSize: 9 }}
                     tickFormatter={v => v.replace('Logistic Regression', 'LR')
                                         .replace('Linear SVM', 'SVM')
                                         .replace('Hierarchical SVM', 'H-SVM')} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 9 }} unit="%" />
              <Tooltip formatter={v => [`${v}%`, 'Val F1']} />
              <Bar dataKey="Val F1" radius={[4, 4, 0, 0]}>
                {compData.map((d, i) => (
                  <Cell key={i}
                        fill={d.model === report.best_model ? '#f59e0b' : '#94a3b8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model comparison table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-100">
              {['Model', 'Val Macro-F1', 'Val Accuracy', 'Train Time'].map(h => (
                <th key={h} className="table-header text-xs py-2">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {(report.all_model_comparison || []).map((m, i) => (
              <tr key={i}
                  className={m.model === report.best_model ? 'bg-yellow-50' : ''}>
                <td className="table-cell text-xs font-medium flex items-center gap-1">
                  {m.model === report.best_model && <Trophy size={11} className="text-yellow-500" />}
                  {m.model}
                </td>
                <td className="table-cell text-xs">{(m.val_macro_f1 * 100).toFixed(2)}%</td>
                <td className="table-cell text-xs">{(m.val_accuracy  * 100).toFixed(2)}%</td>
                <td className="table-cell text-xs flex items-center gap-1">
                  <Clock size={10} className="text-gray-400" />
                  {m.train_time_s >= 3600
                    ? `${(m.train_time_s / 3600).toFixed(1)} h`
                    : m.train_time_s >= 60
                      ? `${(m.train_time_s / 60).toFixed(1)} min`
                      : `${m.train_time_s.toFixed(0)} s`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Classification report */}
      {report.classification_report && (
        <details className="group">
          <summary className="cursor-pointer text-xs text-gray-500 hover:text-gray-700
                              flex items-center gap-1">
            <span className="group-open:rotate-90 inline-block transition-transform">▶</span>
            Full Classification Report
          </summary>
          <pre className="mt-2 bg-gray-50 rounded-lg p-3 text-xs font-mono
                          text-gray-700 overflow-x-auto whitespace-pre">
            {report.classification_report}
          </pre>
        </details>
      )}
    </div>
  )
}

export default function ModelReports() {
  const { data: reports, isLoading } = useQuery(
    'models',
    () => analyticsAPI.models().then(r => r.data)
  )

  // Summary comparison across all targets
  const summaryData = (reports || []).map(r => ({
    target:     TARGET_LABELS[r.target] || r.target,
    'Accuracy': +(r.test_accuracy    * 100).toFixed(1),
    'Macro F1': +(r.test_macro_f1    * 100).toFixed(1),
    'Precision':+(r.test_precision   * 100).toFixed(1),
    'Recall':   +(r.test_recall      * 100).toFixed(1),
    model:      r.best_model,
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Search size={20} className="text-primary-500" />
        <h1>Model Reports</h1>
      </div>

      {isLoading ? <Loading text="Loading model reports…" /> : (
        <>
          {/* Cross-target summary */}
          {summaryData.length > 0 && (
            <div className="card">
              <h3 className="mb-4">Cross-Target Performance Summary — XGBoost Best Model</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={summaryData}
                          margin={{ left: 8, right: 24, top: 5, bottom: 0 }}>
                  <XAxis dataKey="target" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                  <Tooltip formatter={v => [`${v}%`]} />
                  <Legend />
                  <Bar dataKey="Accuracy"  fill="#3b82f6" radius={[4,4,0,0]} />
                  <Bar dataKey="Macro F1"  fill="#10b981" radius={[4,4,0,0]} />
                  <Bar dataKey="Precision" fill="#f59e0b" radius={[4,4,0,0]} />
                  <Bar dataKey="Recall"    fill="#ef4444" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>

              {/* Summary table */}
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-100">
                      {['Target', 'Best Model', 'Accuracy', 'Macro-F1', 'Weighted-F1', 'Precision', 'Recall'].map(h => (
                        <th key={h} className="table-header">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {(reports || []).map((r, i) => (
                      <tr key={i}>
                        <td className="table-cell font-medium">
                          {TARGET_LABELS[r.target] || r.target}
                        </td>
                        <td className="table-cell">
                          <span className="badge badge-yellow">{r.best_model}</span>
                        </td>
                        <td className="table-cell">{(r.test_accuracy    * 100).toFixed(2)}%</td>
                        <td className="table-cell font-medium text-primary-700">
                          {(r.test_macro_f1    * 100).toFixed(2)}%
                        </td>
                        <td className="table-cell">{(r.test_weighted_f1 * 100).toFixed(2)}%</td>
                        <td className="table-cell">{(r.test_precision   * 100).toFixed(2)}%</td>
                        <td className="table-cell">{(r.test_recall      * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* No reports yet */}
          {(reports || []).length === 0 && (
            <div className="card text-center py-16 text-gray-400">
              <Search size={32} className="mx-auto mb-3 opacity-30" />
              <p>No model reports found.</p>
              <p className="text-xs mt-1">Run <code className="bg-gray-100 px-1 rounded">python main.py</code> to generate reports.</p>
            </div>
          )}

          {/* Per-target report cards */}
          <div className="space-y-6">
            {(reports || []).map((r, i) => (
              <ReportCard key={i} report={r} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
