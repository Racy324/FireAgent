import { useState, useRef, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Trash2, BookOpen, AlertTriangle, Clock, ChevronDown, ChevronUp } from 'lucide-react'
import { useChatStore } from '../stores/chatStore'
import MarkdownRenderer from '../components/MarkdownRenderer'
import type { ChatMessage, SSEStageEvent } from '../api/types'
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
  const { messages, isStreaming, sendMessage, clearMessages } = useChatStore()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const hasSentInitialQuery = useRef(false)

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
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* 顶栏 */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold fire-text">💬 智能问答</h1>
        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/5"
          >
            <Trash2 size={14} /> 清空对话
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
  )
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
      {message.citations && message.citations.length > 0 && (
        <div>
          <button
            onClick={() => setShowCitations(!showCitations)}
            className="flex items-center gap-1.5 text-[10px] text-gray-500 hover:text-gray-400 transition-colors"
          >
            <BookOpen size={12} />
            {showCitations ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            引用来源 ({message.citations.length})
          </button>
          <AnimatePresence>
            {showCitations && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1">
                  {message.citations.map((cite, i) => (
                    <div key={i} className="text-[11px] text-gray-400 pl-3 border-l border-fire-500/20 py-0.5">
                      {cite}
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
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
