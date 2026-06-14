import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Scale } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'

export default function Login() {
  const { login, loading } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ username: '', password: '' })

  const handleSubmit = async (e) => {
    e.preventDefault()
    const ok = await login(form.username, form.password)
    if (ok) {
      toast.success('Welcome back!')
      navigate('/')
    } else {
      toast.error('Invalid username or password')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br
                    from-legal-dark to-legal-mid p-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl
                          bg-white/10 mb-4">
            <Scale size={32} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Legal NLP Platform</h1>
          <p className="text-white/60 text-sm mt-1">Case Analysis & Prediction</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-xl p-8 space-y-5">
          <h2 className="text-lg font-semibold text-gray-800">Sign In</h2>

          <div>
            <label className="label">Username</label>
            <input
              className="input"
              placeholder="admin"
              value={form.username}
              onChange={e => setForm({ ...form, username: e.target.value })}
              required
              autoFocus
            />
          </div>

          <div>
            <label className="label">Password</label>
            <input
              className="input"
              type="password"
              placeholder="••••••••"
              value={form.password}
              onChange={e => setForm({ ...form, password: e.target.value })}
              required
            />
          </div>

          <button className="btn-primary w-full py-2.5 mt-2" disabled={loading}>
            {loading ? 'Signing in…' : 'Sign In'}
          </button>

          <p className="text-xs text-gray-400 text-center">
            Demo credentials: admin / admin123
          </p>
        </form>
      </div>
    </div>
  )
}
