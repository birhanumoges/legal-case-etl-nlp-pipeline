import React, { useState, useRef } from 'react'
import {
  Key, Copy, CheckCheck, Play, ChevronDown, ChevronRight,
  Lock, Unlock, Terminal, Eye, EyeOff, Zap, RefreshCw,
  Code, Globe, Shield,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import api, { authAPI } from '../services/api'

// ── Colour map for HTTP methods ───────────────────────────────────
const METHOD_STYLE = {
  GET:    'bg-blue-100   text-blue-700   border-blue-300',
  POST:   'bg-green-100  text-green-700  border-green-300',
  PUT:    'bg-yellow-100 text-yellow-700 border-yellow-300',
  DELETE: 'bg-red-100    text-red-700    border-red-300',
}

// ── All API endpoints definition ──────────────────────────────────
const ENDPOINTS = [
  {
    group: '🔐 Authentication',
    color: 'purple',
    items: [
      {
        id:     'login',
        method: 'POST',
        path:   '/api/v1/auth/login',
        summary:'Get JWT access token — no auth required',
        auth:   false,
        body: { username: 'admin', password: 'admin123' },
        fields: [
          { key: 'username', label: 'Username', type: 'text',     placeholder: 'admin',    required: true },
          { key: 'password', label: 'Password', type: 'password', placeholder: 'admin123', required: true },
        ],
      },
    ],
  },
  {
    group: '❤️ Health',
    color: 'green',
    items: [
      {
        id:     'health',
        method: 'GET',
        path:   '/api/v1/health',
        summary:'Check API server status — no auth required',
        auth:   false,
        body:   null,
        fields: [],
      },
    ],
  },
  {
    group: '🤖 Prediction',
    color: 'blue',
    items: [
      {
        id:     'predict_single',
        method: 'POST',
        path:   '/api/v1/predict',
        summary:'Predict case type, sub-type, and verdict for a single case',
        auth:   true,
        body: {
          case_text: 'The plaintiff filed an action for breach of contract. The court finds for the plaintiff. Judgment affirmed.',
          court: 'Connecticut Superior Court',
          num_citations: 3,
        },
        fields: [
          { key: 'case_text',     label: 'Case Text',      type: 'textarea', placeholder: 'Paste legal opinion text here…', required: true },
          { key: 'court',         label: 'Court',          type: 'text',     placeholder: 'e.g. Superior Court' },
          { key: 'num_citations', label: 'Citation Count', type: 'number',   placeholder: '0' },
        ],
      },
      {
        id:     'predict_batch',
        method: 'POST',
        path:   '/api/v1/predict/batch',
        summary:'Predict up to 50 cases at once (requires auth)',
        auth:   true,
        body: {
          items: [
            { id: '1', case_text: 'Defendant was indicted for larceny. Judgment reversed.', court: 'Superior Court', num_citations: 1 },
            { id: '2', case_text: 'The demurrer is sustained. Appeal dismissed.', court: 'Supreme Court', num_citations: 2 },
          ],
        },
        fields: [],
        rawJson: true,
      },
    ],
  },
  {
    group: '💬 RAG Search',
    color: 'indigo',
    items: [
      {
        id:     'rag_query',
        method: 'POST',
        path:   '/api/v1/rag/query',
        summary:'Ask a question and get an answer grounded in retrieved case text',
        auth:   true,
        body: { question: 'What are the common outcomes in contract bond cases?', top_k: 5 },
        fields: [
          { key: 'question', label: 'Question', type: 'text',   placeholder: 'Ask about the legal corpus…', required: true },
          { key: 'top_k',    label: 'Sources',  type: 'number', placeholder: '5' },
        ],
      },
    ],
  },
  {
    group: '📂 Cases',
    color: 'cyan',
    items: [
      {
        id:     'cases_list',
        method: 'GET',
        path:   '/api/v1/cases',
        summary:'List cases with pagination and optional filters',
        auth:   false,
        body:   null,
        fields: [
          { key: 'page',      label: 'Page',      type: 'number', placeholder: '1' },
          { key: 'size',      label: 'Page Size', type: 'number', placeholder: '10' },
          { key: 'case_type', label: 'Case Type', type: 'select',
            options: ['', 'CIVIL', 'CRIMINAL', 'CONTRACT', 'PROPERTY', 'TORTS'] },
          { key: 'verdict',   label: 'Verdict',   type: 'select',
            options: ['', 'AFFIRMED', 'REVERSED', 'DENIED', 'GRANTED'] },
        ],
        isQuery: true,
      },
      {
        id:     'case_detail',
        method: 'GET',
        path:   '/api/v1/cases/{case_id}',
        summary:'Get full detail of a single case by its ID',
        auth:   false,
        body:   null,
        fields: [
          { key: 'case_id', label: 'Case ID', type: 'text', placeholder: 'e.g. 1234', required: true, isPathParam: true },
        ],
      },
    ],
  },
  {
    group: '📊 Analytics',
    color: 'orange',
    items: [
      {
        id:     'stats',
        method: 'GET',
        path:   '/api/v1/analytics/stats',
        summary:'High-level corpus statistics (total cases, distributions, year range)',
        auth:   false,
        body:   null,
        fields: [],
      },
      {
        id:     'yearly',
        method: 'GET',
        path:   '/api/v1/analytics/yearly',
        summary:'Per-year case volume and citation statistics',
        auth:   false,
        body:   null,
        fields: [],
      },
      {
        id:     'forecast',
        method: 'GET',
        path:   '/api/v1/analytics/forecast',
        summary:'ARIMA(2,2,1) five-year case volume forecast',
        auth:   false,
        body:   null,
        fields: [],
      },
      {
        id:     'models',
        method: 'GET',
        path:   '/api/v1/analytics/models',
        summary:'All three model performance reports (Case Type, Sub-Type, Verdict)',
        auth:   false,
        body:   null,
        fields: [],
      },
    ],
  },
]

// ── Token box ─────────────────────────────────────────────────────
function TokenBox({ token, onClear }) {
  const [show, setShow] = useState(false)
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(token)
    setCopied(true)
    toast.success('Token copied!')
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-xl border border-green-300 bg-green-50 p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm font-semibold text-green-800">
            ✓ Authenticated — Token Active
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShow(!show)}
            className="text-green-600 hover:text-green-800 p-1 rounded"
            title={show ? 'Hide token' : 'Show token'}
          >
            {show ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
          <button
            onClick={copy}
            className="flex items-center gap-1 text-xs bg-green-600 hover:bg-green-700
                       text-white px-2.5 py-1 rounded-lg transition-colors"
          >
            {copied ? <CheckCheck size={13} /> : <Copy size={13} />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <button
            onClick={onClear}
            className="text-xs text-red-500 hover:text-red-700 px-2 py-1 rounded"
          >
            Clear
          </button>
        </div>
      </div>
      <div className="font-mono text-xs bg-white border border-green-200 rounded-lg
                      px-3 py-2 break-all text-gray-600 max-h-16 overflow-y-auto">
        {show ? token : `${token.slice(0, 30)}${'•'.repeat(20)}…`}
      </div>
      <p className="text-xs text-green-600">
        This token is automatically used for all authenticated requests below.
      </p>
    </div>
  )
}

// ── Login widget (inline inside explorer) ────────────────────────
function InlineLogin({ onToken }) {
  const [creds, setCreds]   = useState({ username: 'admin', password: 'admin123' })
  const [loading, setLoading] = useState(false)

  const login = async () => {
    setLoading(true)
    try {
      const { data } = await authAPI.login(creds.username, creds.password)
      onToken(data.access_token)
      toast.success('Logged in! Token is ready.')
    } catch {
      toast.error('Login failed — check username/password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-xl border border-yellow-300 bg-yellow-50 p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Lock size={16} className="text-yellow-600" />
        <span className="font-semibold text-yellow-800 text-sm">
          Get your access token to unlock authenticated endpoints
        </span>
      </div>

      {/* Credential cards */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { user: 'admin',   pass: 'admin123',   role: 'Admin',   color: 'purple' },
          { user: 'analyst', pass: 'analyst123', role: 'Analyst', color: 'blue'   },
        ].map(({ user, pass, role, color }) => (
          <button
            key={user}
            onClick={() => { setCreds({ username: user, password: pass }); }}
            className={`text-left p-3 rounded-lg border-2 transition-all
              ${creds.username === user
                ? `border-${color}-400 bg-${color}-50`
                : 'border-gray-200 bg-white hover:border-gray-300'}`}
          >
            <p className="font-medium text-sm text-gray-800">{role}</p>
            <p className="text-xs text-gray-500 mt-0.5">
              {user} / {pass}
            </p>
          </button>
        ))}
      </div>

      <div className="flex gap-3">
        <div className="flex-1">
          <label className="block text-xs font-medium text-gray-600 mb-1">Username</label>
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-yellow-400"
            value={creds.username}
            onChange={e => setCreds({ ...creds, username: e.target.value })}
          />
        </div>
        <div className="flex-1">
          <label className="block text-xs font-medium text-gray-600 mb-1">Password</label>
          <input
            type="password"
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-yellow-400"
            value={creds.password}
            onChange={e => setCreds({ ...creds, password: e.target.value })}
          />
        </div>
      </div>

      <button
        onClick={login}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 bg-yellow-500
                   hover:bg-yellow-600 text-white font-semibold py-2.5 rounded-lg
                   transition-colors disabled:opacity-50"
      >
        <Key size={16} />
        {loading ? 'Logging in…' : 'Get Token & Unlock All Endpoints'}
      </button>
    </div>
  )
}

// ── Response viewer ───────────────────────────────────────────────
function ResponseViewer({ response, error, loading }) {
  const [copied, setCopied] = useState(false)

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-6 justify-center">
        <RefreshCw size={16} className="animate-spin text-gray-400" />
        <span className="text-sm text-gray-400">Sending request…</span>
      </div>
    )
  }

  if (!response && !error) return null

  const isError   = !!error
  const statusOk  = response?.status >= 200 && response?.status < 300
  const bodyStr   = JSON.stringify(response?.data || error, null, 2)

  const copy = () => {
    navigator.clipboard.writeText(bodyStr)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold px-2 py-0.5 rounded
            ${isError || !statusOk
              ? 'bg-red-100 text-red-700'
              : 'bg-green-100 text-green-700'}`}>
            {isError ? 'ERROR' : response?.status}
          </span>
          <span className="text-xs text-gray-500">
            {isError ? 'Request failed' : 'Response'}
          </span>
        </div>
        <button onClick={copy}
                className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700">
          {copied ? <CheckCheck size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="bg-gray-900 text-green-300 rounded-xl p-4 text-xs font-mono
                      overflow-x-auto max-h-72 whitespace-pre-wrap break-all">
        {bodyStr}
      </pre>
    </div>
  )
}

// ── Single endpoint card ──────────────────────────────────────────
function EndpointCard({ ep, globalToken }) {
  const [open,     setOpen]     = useState(false)
  const [fields,   setFields]   = useState(() => {
    // pre-fill from ep.body
    const init = {}
    ep.fields.forEach(f => {
      if (ep.body && ep.body[f.key] !== undefined) {
        init[f.key] = ep.body[f.key]
      } else {
        init[f.key] = f.type === 'number' ? 0 : ''
      }
    })
    return init
  })
  const [rawJson,  setRawJson]  = useState(
    ep.rawJson ? JSON.stringify(ep.body, null, 2) : ''
  )
  const [response, setResponse] = useState(null)
  const [error,    setError]    = useState(null)
  const [loading,  setLoading]  = useState(false)

  const needsAuth = ep.auth
  const isLocked  = needsAuth && !globalToken

  const run = async () => {
    setLoading(true)
    setResponse(null)
    setError(null)

    try {
      // Build headers
      const headers = { 'Content-Type': 'application/json' }
      if (globalToken) headers['Authorization'] = `Bearer ${globalToken}`

      let path = ep.path
      let body = null
      let params = {}

      // Substitute path params
      ep.fields.filter(f => f.isPathParam).forEach(f => {
        path = path.replace(`{${f.key}}`, encodeURIComponent(fields[f.key] || ''))
      })

      // Query params for GET
      if (ep.isQuery) {
        ep.fields.filter(f => !f.isPathParam && fields[f.key]).forEach(f => {
          params[f.key] = fields[f.key]
        })
      }

      // Body
      if (ep.method !== 'GET') {
        if (ep.rawJson) {
          body = JSON.parse(rawJson)
        } else {
          body = {}
          ep.fields.filter(f => !f.isPathParam).forEach(f => {
            if (fields[f.key] !== '' && fields[f.key] !== undefined) {
              body[f.key] = f.type === 'number' ? Number(fields[f.key]) : fields[f.key]
            }
          })
        }
      }

      const res = await api.request({
        method:  ep.method.toLowerCase(),
        url:     path,
        data:    body,
        params,
        headers,
      })
      setResponse(res)
    } catch (err) {
      setError(err.response?.data || { message: err.message })
    } finally {
      setLoading(false)
    }
  }

  const curlCmd = () => {
    let path = ep.path
    ep.fields.filter(f => f.isPathParam).forEach(f => {
      path = path.replace(`{${f.key}}`, fields[f.key] || '{id}')
    })
    const url = `http://localhost:8000${path}`
    let cmd = `curl -X ${ep.method} "${url}"`
    if (globalToken) cmd += ` \\\n  -H "Authorization: Bearer ${globalToken.slice(0,20)}..."`
    if (ep.method !== 'GET') {
      cmd += ` \\\n  -H "Content-Type: application/json"`
      const b = ep.rawJson ? rawJson : JSON.stringify(
        Object.fromEntries(ep.fields.filter(f => !f.isPathParam).map(f => [f.key, fields[f.key]]))
      )
      cmd += ` \\\n  -d '${b}'`
    }
    return cmd
  }

  const [showCurl, setShowCurl] = useState(false)
  const [curlCopied, setCurlCopied] = useState(false)

  return (
    <div className={`border rounded-xl overflow-hidden transition-all
                     ${isLocked ? 'border-gray-200 opacity-75' : 'border-gray-200 hover:border-gray-300'}`}>
      {/* Header row */}
      <button
        className="w-full flex items-center gap-3 px-5 py-3.5 bg-white hover:bg-gray-50
                   transition-colors text-left"
        onClick={() => !isLocked && setOpen(!open)}
      >
        <span className={`text-xs font-bold px-2.5 py-1 rounded border font-mono flex-shrink-0
                          ${METHOD_STYLE[ep.method]}`}>
          {ep.method}
        </span>
        <span className="font-mono text-sm text-gray-700 flex-1">{ep.path}</span>
        {needsAuth && (
          <span title="Requires authentication">
            {globalToken
              ? <Unlock size={13} className="text-green-500 flex-shrink-0" />
              : <Lock   size={13} className="text-gray-400 flex-shrink-0" />}
          </span>
        )}
        <span className="text-xs text-gray-400 hidden md:block flex-shrink-0 max-w-xs truncate">
          {ep.summary}
        </span>
        {!isLocked && (
          open
            ? <ChevronDown  size={16} className="text-gray-400 flex-shrink-0" />
            : <ChevronRight size={16} className="text-gray-400 flex-shrink-0" />
        )}
        {isLocked && (
          <span className="text-xs text-gray-400 flex-shrink-0">🔒 Login first</span>
        )}
      </button>

      {/* Expanded body */}
      {open && !isLocked && (
        <div className="border-t border-gray-100 bg-gray-50 px-5 py-4 space-y-4">
          <p className="text-sm text-gray-500">{ep.summary}</p>

          {/* Fields */}
          {ep.fields.length > 0 && !ep.rawJson && (
            <div className={`grid gap-3 ${ep.fields.length > 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
              {ep.fields.map(f => (
                <div key={f.key}>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    {f.label}
                    {f.required && <span className="text-red-400 ml-0.5">*</span>}
                    {f.isPathParam && <span className="text-blue-400 ml-1 text-xs">(path)</span>}
                  </label>
                  {f.type === 'textarea' ? (
                    <textarea
                      rows={4}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm
                                 focus:outline-none focus:ring-2 focus:ring-primary-400 bg-white resize-none"
                      placeholder={f.placeholder}
                      value={fields[f.key] || ''}
                      onChange={e => setFields({ ...fields, [f.key]: e.target.value })}
                    />
                  ) : f.type === 'select' ? (
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm
                                 focus:outline-none focus:ring-2 focus:ring-primary-400 bg-white"
                      value={fields[f.key] || ''}
                      onChange={e => setFields({ ...fields, [f.key]: e.target.value })}
                    >
                      {f.options.map(o => <option key={o} value={o}>{o || 'All'}</option>)}
                    </select>
                  ) : (
                    <input
                      type={f.type}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm
                                 focus:outline-none focus:ring-2 focus:ring-primary-400 bg-white"
                      placeholder={f.placeholder}
                      value={fields[f.key] !== undefined ? fields[f.key] : ''}
                      onChange={e => setFields({ ...fields, [f.key]: e.target.value })}
                    />
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Raw JSON editor */}
          {ep.rawJson && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Request Body (JSON)
              </label>
              <textarea
                rows={8}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs
                           font-mono focus:outline-none focus:ring-2 focus:ring-primary-400
                           bg-white resize-y"
                value={rawJson}
                onChange={e => setRawJson(e.target.value)}
              />
            </div>
          )}

          {/* Action row */}
          <div className="flex items-center gap-3">
            <button
              onClick={run}
              disabled={loading}
              className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700
                         text-white font-semibold px-5 py-2 rounded-lg transition-colors
                         disabled:opacity-50 text-sm"
            >
              <Play size={14} />
              {loading ? 'Running…' : 'Send Request'}
            </button>
            <button
              onClick={() => setShowCurl(!showCurl)}
              className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700
                         border border-gray-300 px-3 py-2 rounded-lg bg-white"
            >
              <Terminal size={14} /> cURL
            </button>
          </div>

          {/* cURL snippet */}
          {showCurl && (
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500 flex items-center gap-1">
                  <Code size={12} /> cURL command
                </span>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(curlCmd())
                    setCurlCopied(true)
                    setTimeout(() => setCurlCopied(false), 2000)
                  }}
                  className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
                >
                  {curlCopied ? <CheckCheck size={12} /> : <Copy size={12} />}
                  {curlCopied ? 'Copied' : 'Copy'}
                </button>
              </div>
              <pre className="bg-gray-900 text-yellow-300 rounded-xl p-3 text-xs
                              font-mono overflow-x-auto whitespace-pre-wrap">
                {curlCmd()}
              </pre>
            </div>
          )}

          {/* Response */}
          <ResponseViewer response={response} error={error} loading={loading} />
        </div>
      )}
    </div>
  )
}

// ── Main APIExplorer page ─────────────────────────────────────────
export default function APIExplorer() {
  const { token: contextToken } = useAuth()
  const [token,       setToken]       = useState(contextToken || '')
  const [openGroups,  setOpenGroups]  = useState(
    () => Object.fromEntries(ENDPOINTS.map(g => [g.group, true]))
  )

  const handleToken = (t) => {
    setToken(t)
    // also persist to axios header so other pages benefit
    api.defaults.headers.common['Authorization'] = `Bearer ${t}`
    localStorage.setItem('token', t)
  }

  const toggleGroup = (g) =>
    setOpenGroups(prev => ({ ...prev, [g]: !prev[g] }))

  return (
    <div className="max-w-4xl space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="flex items-center gap-2">
            <Globe size={22} className="text-primary-500" />
            API Explorer
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Interactive REST API playground — try every endpoint directly from the browser
          </p>
        </div>
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-primary-600
                     border border-gray-300 px-3 py-1.5 rounded-lg bg-white transition-colors"
        >
          <Zap size={12} /> Swagger UI ↗
        </a>
      </div>

      {/* Token section */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2">
          <Shield size={16} className="text-primary-500" />
          <h3>Authentication</h3>
        </div>

        {token ? (
          <TokenBox token={token} onClear={() => {
            setToken('')
            delete api.defaults.headers.common['Authorization']
            localStorage.removeItem('token')
          }} />
        ) : (
          <InlineLogin onToken={handleToken} />
        )}
      </div>

      {/* Endpoint groups */}
      {ENDPOINTS.map(group => (
        <div key={group.group} className="space-y-2">
          {/* Group header */}
          <button
            className="flex items-center gap-2 w-full text-left py-1"
            onClick={() => toggleGroup(group.group)}
          >
            {openGroups[group.group]
              ? <ChevronDown  size={16} className="text-gray-400" />
              : <ChevronRight size={16} className="text-gray-400" />}
            <span className="font-semibold text-gray-800">{group.group}</span>
            <span className="text-xs text-gray-400">
              {group.items.length} endpoint{group.items.length !== 1 ? 's' : ''}
            </span>
          </button>

          {/* Endpoint cards */}
          {openGroups[group.group] && (
            <div className="space-y-2 pl-5">
              {group.items.map(ep => (
                <EndpointCard
                  key={ep.id}
                  ep={ep}
                  globalToken={token}
                />
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Footer note */}
      <div className="text-xs text-gray-400 text-center pb-4">
        API runs on <code className="bg-gray-100 px-1 rounded">http://localhost:8000</code> ·
        Full Swagger docs at{' '}
        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
           className="text-primary-500 hover:underline">
          /docs
        </a>
      </div>
    </div>
  )
}
