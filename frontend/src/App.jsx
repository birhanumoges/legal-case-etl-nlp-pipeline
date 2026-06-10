import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'

import Layout       from './components/Layout'
import Login        from './pages/Login'
import Dashboard    from './pages/Dashboard'
import Cases        from './pages/Cases'
import CaseDetail   from './pages/CaseDetail'
import Predict      from './pages/Predict'
import RAG          from './pages/RAG'
import Analytics    from './pages/Analytics'
import ModelReports from './pages/ModelReports'
import APIExplorer  from './pages/APIExplorer'

function PrivateRoute({ children }) {
  const { token } = useAuth()
  return token ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />

      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route index              element={<Dashboard />}    />
        <Route path="cases"       element={<Cases />}        />
        <Route path="cases/:id"   element={<CaseDetail />}   />
        <Route path="predict"     element={<Predict />}      />
        <Route path="rag"         element={<RAG />}          />
        <Route path="analytics"   element={<Analytics />}    />
        <Route path="models"      element={<ModelReports />} />
        <Route path="api"         element={<APIExplorer />}  />
        <Route path="*"           element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
