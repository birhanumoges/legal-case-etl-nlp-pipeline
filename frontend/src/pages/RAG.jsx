import React, { useState, useRef, useEffect } from 'react'
import { MessageSquare, Send, Bot, User, Trash2, BookOpen } from 'lucide-react'
import toast from 'react-hot-toast'
import { predictAPI } from '../services/api'
import { truncate } from '../utils/helpers'

const SAMPLE_QUESTIONS = [
  'What are the common outcomes in contract bond cases?',
  'How are larceny cases typically decided?',
  'What verdict is most common in civil appeal cases?',
  'Describe the typical outcome of negligence tort cases.',
  'How do courts handle demurrer in civil procedure?',
]

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
                       ${isUser ? 'bg-primary-100 text-primary-600' : 'bg-gray-100 text-gray-600'}`}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed
                       ${isUser
                         ? 'bg-primary-600 text-white rounded-tr-sm'
                         : 'bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm'}`}>
        {msg.content}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-200/40">
            <p className="text-xs opacity-70 flex items-center gap-1 mb-1">
              <BookOpen size={11} /> Sources ({msg.sources.length})
            </p>
            <div className="flex flex-wrap gap-1">
              {msg.sources.map((s, i) => (
                <span key={i}
                      className="text-xs bg-white/20 rounded px-1.5 py-0.5 font-mono">
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function RAG() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m the Legal NLP assistant. Ask me anything about the legal case corpus — case outcomes, trends, legal patterns, or specific case types.',
    },
  ])
  const [input,   setInput]   = useState('')
  const [topK,    setTopK]    = useState(5)
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (question) => {
    const q = question || input.trim()
    if (!q) return
    setInput('')

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)

    try {
      const { data } = await predictAPI.rag(q, topK)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer || 'No answer generated.',
        sources: data.sources || [],
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ The RAG service is currently unavailable. Please ensure the backend is running and the vectorstore has been built by running main.py.',
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-7rem)] max-w-3xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <MessageSquare size={20} className="text-primary-500" />
          <h1>RAG Legal Search</h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-gray-500">Sources</label>
            <select
              className="text-xs border border-gray-200 rounded px-2 py-1 bg-white"
              value={topK}
              onChange={e => setTopK(Number(e.target.value))}
            >
              {[3, 5, 8, 10].map(k => <option key={k} value={k}>{k}</option>)}
            </select>
          </div>
          <button
            className="text-xs text-gray-400 hover:text-red-500 flex items-center gap-1"
            onClick={() => setMessages([{
              role: 'assistant',
              content: 'Conversation cleared. How can I help you?',
            }])}
          >
            <Trash2 size={13} /> Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 pb-2">
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
              <Bot size={16} className="text-gray-600" />
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm
                            px-4 py-3 shadow-sm">
              <div className="flex gap-1 items-center h-5">
                {[0, 1, 2].map(i => (
                  <div key={i}
                       className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
                       style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Sample questions */}
      <div className="py-2 flex gap-2 overflow-x-auto flex-nowrap">
        {SAMPLE_QUESTIONS.map((q, i) => (
          <button
            key={i}
            onClick={() => send(q)}
            disabled={loading}
            className="flex-shrink-0 text-xs bg-white border border-gray-200 rounded-full
                       px-3 py-1.5 text-gray-600 hover:bg-primary-50 hover:border-primary-300
                       hover:text-primary-700 transition-colors disabled:opacity-50"
          >
            {truncate(q, 45)}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="flex gap-2 pt-2 border-t border-gray-200">
        <textarea
          className="input flex-1 resize-none h-12 py-3 text-sm"
          placeholder="Ask a question about the legal case corpus… (Enter to send)"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          rows={1}
        />
        <button
          className="btn-primary px-4 flex items-center gap-1.5 self-end h-10"
          onClick={() => send()}
          disabled={loading || !input.trim()}
        >
          <Send size={15} />
        </button>
      </div>
    </div>
  )
}
