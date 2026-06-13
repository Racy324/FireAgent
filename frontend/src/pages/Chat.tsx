import { useState, useRef, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send,
  Trash2,
  BookOpen,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronUp,
  Plus,
  MessageSquare,
  X,
} from 'lucide-react'
import { useChatStore } from '../stores/chatStore'
import MarkdownRenderer from '../components/MarkdownRenderer'
import type { ChatMessage, CitationItem, SessionItem, SSEStageEvent } from '../api/types'
import { formatElapsed } from '../utils/format'

const STAGE_LABELS: Record<string, string> = {
  intent: '意图识别',
  rewrite: '查询改写',
  retrieve: '检索召回',
  fusion: '混合融合',
  rerank: '重排序',
  sufficiency: '充分性检查',
  web_search: '联网搜索',
  context: '上下文构建',
}

export default function Chat() {
  const [searchParams] = useSearchParams()
  const {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
    isLoadingSessions,
    loadSessions,
    selectSession,
    newSession,
    deleteSession,
    sendMessage,
    clearMessages,
  } = useChatStore()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const hasSentInitialQuery = useRef(false)

  useEffect(() => {
    void loadSessions()
  }, [loadSessions])

  // 从 URL 参数自动发送问题
  useEffect(() => {
    const q = searchParams.get('q')
    if (q && !hasSentInitialQuery.current) {
      hasSentInitialQuery.current = true
      sendMessage(q)
    }
  }, [searchParams, sendMessage])

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const q = input.trim()
    if (!q || isStreaming) return
    sendMessage(q)
    setInput('')
    // 重置 textarea 高度
    if (inputRef.current) inputRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // 自动调整 textarea 高度
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] gap-4">
      <SessionSidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        isLoading={isLoadingSessions}
        onNew={newSession}
        onSelect={(sessionId) => void selectSession(sessionId)}
        onDelete={(sessionId) => void deleteSession(sessionId)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* 顶栏 */}
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold fire-text">💬 智能问答</h1>
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/5"
            >
              <Trash2 size={14} /> 新建空会话
            </button>
          )}
        </div>

        {/* 消息区域 */}
        <div className="flex-1 overflow-y-auto space-y-4 pb-4 pr-1">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="text-5xl mb-4">🔥</div>
              <h2 className="text-xl font-bold text-gray-300 mb-2">你好，我是 FireAgent</h2>
              <p className="text-sm text-gray-500 max-w-md">
                火灾领域知识问答助手，基于 69 篇火灾论文提供有据可查的回答。
                输入你的问题开始对话。
              </p>
            </div>
          )}

          <AnimatePresence>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[85%] ${msg.role === 'user' ? '' : 'w-full'}`}>
                  {msg.role === 'user' ? (
                    <div className="bg-fire-500/20 text-fire-100 px-4 py-3 rounded-2xl rounded-tr-md text-sm leading-relaxed">
                      {msg.content}
                    </div>
                  ) : (
                    <AssistantBubble message={msg} />
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* 输入区 */}
        <div className="glass-card p-3 flex items-end gap-2 mt-2">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="输入你的问题... (Shift+Enter 换行)"
            className="flex-1 bg-transparent outline-none text-sm text-gray-200 placeholder:text-gray-500 resize-none py-2 px-2 max-h-[120px]"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="p-2.5 bg-gradient-to-r from-fire-500 to-fire-600 text-white rounded-xl hover:from-fire-400 hover:to-fire-500 transition-all disabled:opacity-30 disabled:cursor-not-allowed active:scale-95 flex-shrink-0"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  )
}

function SessionSidebar({
  sessions,
  currentSessionId,
  isLoading,
  onNew,
  onSelect,
  onDelete,
}: {
  sessions: SessionItem[]
  currentSessionId?: string
  isLoading: boolean
  onNew: () => void
  onSelect: (sessionId: string) => void
  onDelete: (sessionId: string) => void
}) {
  return (
    <aside className="glass-card hidden w-72 shrink-0 flex-col p-3 md:flex">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-200">
          <MessageSquare size={16} className="text-fire-400" />
          会话历史
        </div>
        <button
          onClick={onNew}
          className="rounded-lg p-1.5 text-gray-400 transition-colors hover:bg-white/5 hover:text-fire-300"
          title="新建会话"
        >
          <Plus size={16} />
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {isLoading && <div className="px-2 py-3 text-xs text-gray-500">加载会话中...</div>}
        {!isLoading && sessions.length === 0 && (
          <div className="px-2 py-3 text-xs leading-relaxed text-gray-500">
            暂无历史会话，发送第一条消息后会自动创建。
          </div>
        )}
        {sessions.map((session) => {
          const active = session.session_id === currentSessionId
          return (
            <div
              key={session.session_id}
              className={`group flex w-full items-start gap-2 rounded-lg transition-colors ${
                active ? 'bg-fire-500/15 text-fire-100' : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'
              }`}
            >
              <button
                onClick={() => onSelect(session.session_id)}
                className="flex min-w-0 flex-1 items-start gap-2 px-2 py-2 text-left"
              >
                <MessageSquare size={14} className="mt-0.5 shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-medium">{session.title || '新会话'}</span>
                  <span className="mt-0.5 block text-[10px] text-gray-600">
                    {formatSessionTime(session.updated_at)}
                  </span>
                </span>
              </button>
              <button
                onClick={(event) => {
                  event.stopPropagation()
                  onDelete(session.session_id)
                }}
                className="mr-1 mt-1.5 rounded p-1 text-gray-600 opacity-0 transition-opacity hover:bg-red-500/10 hover:text-red-300 group-hover:opacity-100"
                title="删除会话"
              >
                <X size={12} />
              </button>
            </div>
          )
        })}
      </div>
    </aside>
  )
}

function formatSessionTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** AI 回答气泡 */
function AssistantBubble({ message }: { message: ChatMessage }) {
  const [showCitations, setShowCitations] = useState(false)
  const [showStages, setShowStages] = useState(false)

  return (
    <div className="glass-card p-4 space-y-3">
      {/* 阶段进度 */}
      {message.stages && message.stages.length > 0 && (
        <div>
          <button
            onClick={() => setShowStages(!showStages)}
            className="flex items-center gap-1.5 text-[10px] text-gray-500 hover:text-gray-400 transition-colors"
          >
            {showStages ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            检索过程 ({message.stages.length} 个阶段)
          </button>
          <AnimatePresence>
            {showStages && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {message.stages.map((stage, i) => (
                    <StageTag key={i} stage={stage} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* 回答内容 */}
      {message.content ? (
        <div className="text-sm">
          <MarkdownRenderer content={message.content} />
          {message.isStreaming && <span className="cursor-blink" />}
        </div>
      ) : message.isStreaming ? (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <div className="sse-dot" />
          <span>正在思考...</span>
        </div>
      ) : null}

      {/* 底部信息栏 */}
      {!message.isStreaming && message.content && (
        <div className="flex items-center gap-3 pt-2 border-t border-white/5">
          {message.elapsed !== undefined && (
            <span className="flex items-center gap-1 text-[10px] text-gray-500">
              <Clock size={10} /> {formatElapsed(message.elapsed)}
            </span>
          )}
          {message.intent && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-fire-500/10 text-fire-400">
              {message.intent === 'rag' ? '知识问答' : message.intent === 'paper' ? '论文分析' : message.intent}
            </span>
          )}
          {message.evidence_sufficient !== undefined && (
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${message.evidence_sufficient ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'}`}>
              {message.evidence_sufficient ? '证据充足' : '证据不足'}
            </span>
          )}
        </div>
      )}

      {/* 安全提醒 */}
      {message.safety_notice && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <AlertTriangle size={14} className="text-red-400 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-red-300">{message.safety_notice}</p>
        </div>
      )}

      {/* 引用来源 */}
      {(() => {
        const hasUsed = message.used_citations && message.used_citations.length > 0
        const hasLegacy = message.citations && message.citations.length > 0
        if (!hasUsed && !hasLegacy) return null
        const citeCount = hasUsed ? message.used_citations!.length : message.citations!.length
        return (
          <div>
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="flex items-center gap-1.5 text-[10px] text-gray-500 hover:text-gray-400 transition-colors"
            >
              <BookOpen size={12} />
              {showCitations ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              引用来源 ({citeCount})
            </button>
            <AnimatePresence>
              {showCitations && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-2 space-y-1.5">
                    {hasUsed
                      ? message.used_citations!.map((cite) => (
                          <CitationCard key={cite.citation_id} citation={cite} />
                        ))
                      : message.citations!.map((cite, i) => (
                          <div key={i} className="text-[11px] text-gray-400 pl-3 border-l border-fire-500/20 py-0.5">
                            {cite}
                          </div>
                        ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })()}
    </div>
  )
}

/** 阶段标签 */
function StageTag({ stage }: { stage: SSEStageEvent }) {
  const label = STAGE_LABELS[stage.stage] || stage.stage
  const detail = stage.stage === 'retrieve'
    ? `dense:${stage.dense_count} sparse:${stage.sparse_count}`
    : stage.stage === 'rerank'
    ? `${stage.count} 条`
    : stage.stage === 'sufficiency'
    ? stage.sufficient ? '充足' : '不足'
    : ''

  return (
    <span className="text-[10px] px-2 py-0.5 rounded-full bg-bg-tertiary text-gray-400 border border-white/5">
      {label}{detail ? ` · ${detail}` : ''}
    </span>
  )
}

/** 结构化引用卡片 */
function CitationCard({ citation }: { citation: CitationItem }) {
  const isWeb = citation.source_type.startsWith('web')
  const pageText = citation.page_start
    ? citation.page_end && citation.page_end !== citation.page_start
      ? `第${citation.page_start}-${citation.page_end}页`
      : `第${citation.page_start}页`
    : ''

  return (
    <div className="text-[11px] pl-3 border-l border-fire-500/20 py-1 space-y-0.5">
      <div className="flex items-center gap-1.5">
        <span className="font-mono text-fire-400 font-semibold">{citation.marker}</span>
        <span className="text-gray-300">{citation.title || '题名未知'}</span>
      </div>
      {!isWeb && (
        <div className="text-gray-500">
          {citation.authors.length > 0 ? citation.authors.join('、') : '作者未知'}
          {citation.year ? `，${citation.year}` : ''}
          {citation.section_title ? `，${citation.section_title}` : ''}
          {pageText ? `，${pageText}` : ''}
        </div>
      )}
      {isWeb && citation.url && (
        <a
          href={citation.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-fire-400 hover:text-fire-300 underline underline-offset-2 truncate block max-w-full"
        >
          {citation.url}
        </a>
      )}
    </div>
  )
}
