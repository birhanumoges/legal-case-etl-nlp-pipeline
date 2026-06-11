import React, { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, Search, Brain, MessageSquare,
  BarChart2, FileText, LogOut, Menu, X, Scale, Globe,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const NAV = [
  { to: '/',           icon: LayoutDashboard, label: 'Dashboard'    },
  { to: '/cases',      icon: FileText,        label: 'Cases'        },
  { to: '/predict',    icon: Brain,           label: 'Predict'      },
  { to: '/rag',        icon: MessageSquare,   label: 'RAG Search'   },
  { to: '/analytics',  icon: BarChart2,       label: 'Analytics'    },
  { to: '/models',     icon: Search,          label: 'Model Reports'},
  { to: '/api',        icon: Globe,           label: 'API Explorer' },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate  = useNavigate()
  const [open, setOpen] = useState(true)

  const handleLogout = () => { logout(); navigate('/login') }

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <aside className={`${open ? 'w-56' : 'w-16'} flex-shrink-0 bg-legal-dark
                         text-white flex flex-col transition-all duration-200`}>
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-white/10">
          <Scale size={22} className="text-primary-400 flex-shrink-0" />
          {open && <span className="font-bold text-sm leading-tight">Legal NLP<br/><span className="font-normal text-xs text-white/60">Case Analysis</span></span>}
        </div>

        {/* Nav links */}
        <nav className="flex-1 overflow-y-auto py-4 space-y-1 px-2">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors
                 ${isActive
                   ? 'bg-primary-600 text-white'
                   : 'text-white/70 hover:bg-white/10 hover:text-white'}`}
            >
              <Icon size={18} className="flex-shrink-0" />
              {open && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Bottom */}
        <div className="border-t border-white/10 p-3 space-y-1">
          {open && user && (
            <p className="text-xs text-white/50 px-2 pb-1 truncate">@{user}</p>
          )}
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-3 py-2 rounded-lg
                       text-sm text-white/70 hover:bg-white/10 hover:text-white transition-colors"
          >
            <LogOut size={18} className="flex-shrink-0" />
            {open && <span>Logout</span>}
          </button>
          <button
            onClick={() => setOpen(!open)}
            className="flex items-center gap-3 w-full px-3 py-2 rounded-lg
                       text-sm text-white/70 hover:bg-white/10 hover:text-white transition-colors"
          >
            {open ? <X size={18} /> : <Menu size={18} />}
            {open && <span className="text-xs">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
