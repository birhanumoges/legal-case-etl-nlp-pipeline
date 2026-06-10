import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach stored token on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Redirect to login on 401
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api

// ── Auth ──────────────────────────────────────────────────────────
export const authAPI = {
  login: (username, password) =>
    api.post('/auth/login', { username, password }),
}

// ── Health ────────────────────────────────────────────────────────
export const healthAPI = {
  check: () => api.get('/health'),
}

// ── Cases ─────────────────────────────────────────────────────────
export const casesAPI = {
  list: (params) => api.get('/cases', { params }),
  search: (body)  => api.post('/cases/search', body),
  get: (id)       => api.get(`/cases/${id}`),
  similar: (id, top_k = 5) => api.get(`/cases/${id}/similar`, { params: { top_k } }),
}

// ── Prediction ────────────────────────────────────────────────────
export const predictAPI = {
  single: (payload)      => api.post('/predict', payload),
  batch:  (items)        => api.post('/predict/batch', { items }),
  rag:    (question, top_k = 5) => api.post('/rag/query', { question, top_k }),
}

// ── Analytics ─────────────────────────────────────────────────────
export const analyticsAPI = {
  stats:           ()         => api.get('/analytics/stats'),
  yearly:          ()         => api.get('/analytics/yearly'),
  forecast:        ()         => api.get('/analytics/forecast'),
  caseTypeDist:    ()         => api.get('/analytics/distribution/case-type'),
  verdictDist:     ()         => api.get('/analytics/distribution/verdict'),
  subTypeDist:     ()         => api.get('/analytics/distribution/sub-type'),
  models:          ()         => api.get('/analytics/models'),
  modelReport:     (target)   => api.get(`/analytics/models/${target}`),
}
