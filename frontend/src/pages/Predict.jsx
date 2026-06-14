import React, { useState } from 'react'
import { Brain, PlusCircle, Trash2, Download } from 'lucide-react'
import toast from 'react-hot-toast'
import { predictAPI } from '../services/api'
import { VerdictBadge, CaseTypeBadge, SubTypeBadge } from '../components/Badge'

function ResultCard({ result, title = 'Prediction Result' }) {
  return (
    <div className="card border-l-4 border-primary-500 space-y-3">
      <h3 className="text-primary-700">{title}</h3>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-xs text-gray-500 mb-1">Case Type</p>
          <CaseTypeBadge type={result.case_type} />
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Sub-Type</p>
          <SubTypeBadge subType={result.sub_type} />
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Verdict</p>
          <VerdictBadge verdict={result.verdict} />
        </div>
      </div>
      {result.confidence && (
        <p className="text-xs text-gray-400">
          Confidence: {(result.confidence * 100).toFixed(1)}%
        </p>
      )}
    </div>
  )
}

export default function Predict() {
  const [mode, setMode] = useState('single') // 'single' | 'batch'

  // Single
  const [form,     setForm]     = useState({ case_text: '', court: '', num_citations: 0 })
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)

  // Batch
  const [items,       setItems]       = useState([{ id: '1', case_text: '', court: '', num_citations: 0 }])
  const [batchResult, setBatchResult] = useState(null)
  const [batchLoading, setBatchLoading] = useState(false)

  // ── Single predict ──────────────────────────────────────────────
  const handleSingle = async (e) => {
    e.preventDefault()
    if (!form.case_text.trim()) return toast.error('Please enter case text')
    setLoading(true)
    try {
      const { data } = await predictAPI.single(form)
      setResult(data)
    } catch {
      toast.error('Prediction failed. Check the API is running.')
    } finally {
      setLoading(false)
    }
  }

  // ── Batch predict ───────────────────────────────────────────────
  const handleBatch = async () => {
    const valid = items.filter(i => i.case_text.trim())
    if (!valid.length) return toast.error('Add at least one case text')
    setBatchLoading(true)
    try {
      const { data } = await predictAPI.batch(valid)
      setBatchResult(data)
    } catch {
      toast.error('Batch prediction failed.')
    } finally {
      setBatchLoading(false)
    }
  }

  const downloadCSV = () => {
    if (!batchResult) return
    const rows = [
      ['ID', 'Case Type', 'Sub-Type', 'Verdict', 'Error'],
      ...batchResult.results.map(r =>
        [r.id, r.case_type, r.sub_type, r.verdict, r.error || ''])
    ]
    const csv  = rows.map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a'); a.href = url
    a.download = 'batch_predictions.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <Brain size={22} className="text-primary-500" />
        <h1>Case Predictor</h1>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2">
        <button
          className={`px-4 py-1.5 rounded-lg text-sm font-medium border transition-colors
            ${mode === 'single' ? 'bg-primary-600 text-white border-primary-600'
                               : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'}`}
          onClick={() => setMode('single')}
        >Single</button>
        <button
          className={`px-4 py-1.5 rounded-lg text-sm font-medium border transition-colors
            ${mode === 'batch'  ? 'bg-primary-600 text-white border-primary-600'
                               : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'}`}
          onClick={() => setMode('batch')}
        >Batch (up to 50)</button>
      </div>

      {/* ── SINGLE MODE ── */}
      {mode === 'single' && (
        <form onSubmit={handleSingle} className="card space-y-4">
          <div>
            <label className="label">Case Text *</label>
            <textarea
              className="input h-40 resize-none"
              placeholder="Paste the full case opinion text here…"
              value={form.case_text}
              onChange={e => setForm({ ...form, case_text: e.target.value })}
              required
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Court (optional)</label>
              <input className="input" placeholder="e.g. Superior Court"
                     value={form.court}
                     onChange={e => setForm({ ...form, court: e.target.value })} />
            </div>
            <div>
              <label className="label">Citation Count</label>
              <input className="input" type="number" min={0} value={form.num_citations}
                     onChange={e => setForm({ ...form, num_citations: Number(e.target.value) })} />
            </div>
          </div>
          <button className="btn-primary w-full py-2.5" disabled={loading}>
            {loading ? 'Predicting…' : 'Predict'}
          </button>
        </form>
      )}

      {result && mode === 'single' && (
        <ResultCard result={result} title="Prediction Result" />
      )}

      {/* ── BATCH MODE ── */}
      {mode === 'batch' && (
        <div className="card space-y-4">
          <div className="space-y-3">
            {items.map((item, i) => (
              <div key={i} className="p-3 border border-gray-200 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-600">Case #{i + 1}</span>
                  {items.length > 1 && (
                    <button onClick={() => setItems(items.filter((_, j) => j !== i))}
                            className="text-red-400 hover:text-red-600">
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
                <textarea
                  className="input h-24 resize-none text-xs"
                  placeholder="Case text…"
                  value={item.case_text}
                  onChange={e => {
                    const next = [...items]
                    next[i] = { ...item, case_text: e.target.value }
                    setItems(next)
                  }}
                />
                <div className="grid grid-cols-2 gap-2">
                  <input className="input text-xs" placeholder="Court (optional)"
                         value={item.court}
                         onChange={e => {
                           const next = [...items]
                           next[i] = { ...item, court: e.target.value }
                           setItems(next)
                         }} />
                  <input className="input text-xs" type="number" min={0}
                         placeholder="Citations"
                         value={item.num_citations}
                         onChange={e => {
                           const next = [...items]
                           next[i] = { ...item, num_citations: Number(e.target.value) }
                           setItems(next)
                         }} />
                </div>
              </div>
            ))}
          </div>

          <div className="flex gap-2">
            <button
              className="btn-secondary flex items-center gap-1.5 text-sm"
              onClick={() => setItems([...items, {
                id: String(items.length + 1), case_text: '', court: '', num_citations: 0
              }])}
              disabled={items.length >= 50}
            >
              <PlusCircle size={15} /> Add Case
            </button>
            <button className="btn-primary flex-1 py-2" onClick={handleBatch}
                    disabled={batchLoading}>
              {batchLoading ? 'Running…' : `Predict ${items.length} Case(s)`}
            </button>
          </div>
        </div>
      )}

      {batchResult && mode === 'batch' && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3>Batch Results ({batchResult.total} cases)</h3>
            <button className="btn-secondary text-xs flex items-center gap-1"
                    onClick={downloadCSV}>
              <Download size={13} /> CSV
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100">
                  {['ID', 'Case Type', 'Sub-Type', 'Verdict', 'Error'].map(h => (
                    <th key={h} className="table-header">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {batchResult.results.map((r, i) => (
                  <tr key={i}>
                    <td className="table-cell font-mono text-xs">{r.id}</td>
                    <td className="table-cell"><CaseTypeBadge type={r.case_type} /></td>
                    <td className="table-cell"><SubTypeBadge subType={r.sub_type} /></td>
                    <td className="table-cell"><VerdictBadge verdict={r.verdict} /></td>
                    <td className="table-cell text-red-500 text-xs">{r.error || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
