import React from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from 'react-query'
import { ArrowLeft, Link as LinkIcon } from 'lucide-react'
import { casesAPI } from '../services/api'
import { VerdictBadge, CaseTypeBadge, SubTypeBadge } from '../components/Badge'
import Loading from '../components/Loading'
import { fmt, truncate } from '../utils/helpers'

function Field({ label, value }) {
  return (
    <div>
      <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</dt>
      <dd className="mt-0.5 text-sm text-gray-900">{value || '—'}</dd>
    </div>
  )
}

export default function CaseDetail() {
  const { id } = useParams()
  const navigate = useNavigate()

  const { data: cas, isLoading } = useQuery(
    ['case', id],
    () => casesAPI.get(id).then(r => r.data),
    { enabled: !!id }
  )

  const { data: similar } = useQuery(
    ['similar', id],
    () => casesAPI.similar(id, 5).then(r => r.data),
    { enabled: !!id }
  )

  if (isLoading) return <Loading />

  if (!cas) return (
    <div className="text-center py-20 text-gray-400">Case not found.</div>
  )

  return (
    <div className="space-y-5 max-w-4xl">
      {/* Back */}
      <button onClick={() => navigate(-1)}
              className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-900">
        <ArrowLeft size={15} /> Back to cases
      </button>

      {/* Header */}
      <div className="card space-y-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-lg">{cas.case_name || cas.case_id}</h1>
            <p className="text-xs text-gray-400 mt-0.5 font-mono">{cas.case_id}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <VerdictBadge verdict={cas.verdict_mapped} />
            <CaseTypeBadge type={cas.case_type_mapped} />
          </div>
        </div>

        {/* Meta grid */}
        <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 pt-2
                       border-t border-gray-100">
          <Field label="Year"          value={cas.year} />
          <Field label="Court"         value={cas.court} />
          <Field label="Source"        value={cas.source_folder} />
          <Field label="Citations"     value={cas.num_citations} />
          <Field label="Raw Type"      value={cas.case_type} />
          <Field label="Canonical Type" value={cas.case_type_mapped} />
          <Field label="Sub-Type"      value={cas.sub_type_mapped} />
          <Field label="Raw Verdict"   value={cas.verdict} />
          <Field label="Text Length"   value={cas.text_length ? fmt.number(cas.text_length) + ' chars' : null} />
          <Field label="Word Count"    value={cas.word_count  ? fmt.number(cas.word_count)  + ' words' : null} />
        </dl>

        {/* Sub-type badge row */}
        {cas.sub_type_mapped && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Sub-type:</span>
            <SubTypeBadge subType={cas.sub_type_mapped} />
          </div>
        )}
      </div>

      {/* Citations */}
      {cas.legal_citations && (
        <div className="card">
          <h3 className="mb-2 flex items-center gap-1.5">
            <LinkIcon size={15} className="text-gray-400" /> Legal Citations
          </h3>
          <p className="text-sm text-gray-600 leading-relaxed">
            {cas.legal_citations}
          </p>
        </div>
      )}

      {/* Opinion text */}
      {cas.case_text && (
        <div className="card">
          <h3 className="mb-3">Opinion Text</h3>
          <div className="prose prose-sm max-w-none bg-gray-50 rounded-lg p-4
                          text-gray-700 leading-relaxed whitespace-pre-wrap text-sm
                          max-h-96 overflow-y-auto">
            {cas.case_text}
          </div>
        </div>
      )}

      {/* Similar cases */}
      {similar?.results?.length > 0 && (
        <div className="card">
          <h3 className="mb-3">Semantically Similar Cases</h3>
          <div className="space-y-2">
            {similar.results.map((s, i) => (
              <div key={i}
                   className="flex items-start gap-3 p-3 rounded-lg bg-gray-50
                              hover:bg-gray-100 cursor-pointer transition-colors"
                   onClick={() => navigate(`/cases/${s.id.split('_chunk_')[0]}`)}>
                <span className="text-xs text-primary-500 font-medium flex-shrink-0 mt-0.5">
                  {(s.score * 100).toFixed(0)}%
                </span>
                <div className="min-w-0">
                  <p className="text-xs font-mono text-gray-500">{s.id}</p>
                  <p className="text-sm text-gray-700 mt-0.5">{truncate(s.text, 100)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
