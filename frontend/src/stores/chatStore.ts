import { create } from 'zustand'
import type { ChatMessage, CitationItem, MessageItem, SessionItem, SSEStageEvent } from '../api/types'
import {
  deleteSession as deleteSessionApi,
  listSessionMessages,
  listSessions,
  sendStreamChat,
} from '../api/chat'

interface ChatState {
  sessions: SessionItem[]
  currentSessionId?: string
  messages: ChatMessage[]
  isStreaming: boolean
  isLoadingSessions: boolean
  currentStages: SSEStageEvent[]
  loadSessions: () => Promise<void>
  selectSession: (sessionId: string) => Promise<void>
  newSession: () => void
  deleteSession: (sessionId: string) => Promise<void>
  sendMessage: (query: string) => void
  clearMessages: () => void
}

let abortController: AbortController | null = null

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: localStorage.getItem('fireagent.currentSessionId') || undefined,
  messages: [],
  isStreaming: false,
  isLoadingSessions: false,
  currentStages: [],

  loadSessions: async () => {
    set({ isLoadingSessions: true })
    try {
      const sessions = await listSessions()
      set({ sessions, isLoadingSessions: false })
      const current = get().currentSessionId
      if (current && sessions.some((item) => item.session_id === current) && get().messages.length === 0) {
        await get().selectSession(current)
      }
    } catch {
      set({ isLoadingSessions: false })
    }
  },

  selectSession: async (sessionId: string) => {
    if (abortController) abortController.abort()
    const payload = await listSessionMessages(sessionId)
    const messages = payload.messages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map(messageItemToChatMessage)
    localStorage.setItem('fireagent.currentSessionId', sessionId)
    set({
      currentSessionId: sessionId,
      messages,
      isStreaming: false,
      currentStages: [],
    })
  },

  newSession: () => {
    if (abortController) abortController.abort()
    localStorage.removeItem('fireagent.currentSessionId')
    set({
      currentSessionId: undefined,
      messages: [],
      isStreaming: false,
      currentStages: [],
    })
  },

  deleteSession: async (sessionId: string) => {
    if (abortController) abortController.abort()
    await deleteSessionApi(sessionId)
    const currentSessionId = get().currentSessionId
    if (currentSessionId === sessionId) {
      localStorage.removeItem('fireagent.currentSessionId')
      set({ currentSessionId: undefined, messages: [], currentStages: [], isStreaming: false })
    }
    await get().loadSessions()
  },

  sendMessage: (query: string) => {
    if (get().isStreaming) return

    const sessionId = get().currentSessionId
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      session_id: sessionId,
      role: 'user',
      content: query,
    }

    const assistantMsg: ChatMessage = {
      id: `assistant-${Date.now()}`,
      session_id: sessionId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      stages: [],
    }

    set((s) => ({
      messages: [...s.messages, userMsg, assistantMsg],
      isStreaming: true,
      currentStages: [],
    }))

    // 取消之前的请求
    if (abortController) abortController.abort()

    abortController = sendStreamChat(query, sessionId, {
      onStage: (event) => {
        set((s) => ({
          currentStages: [...s.currentStages, event],
        }))
        set((s) => {
          const msgs = [...s.messages]
          const last = msgs[msgs.length - 1]
          if (last && last.role === 'assistant') {
            last.stages = [...(last.stages || []), event]
          }
          return { messages: msgs }
        })
      },
      onToken: (token) => {
        set((s) => {
          const msgs = [...s.messages]
          const last = msgs[msgs.length - 1]
          if (last && last.role === 'assistant') {
            last.content += token
          }
          return { messages: msgs }
        })
      },
      onDone: (data) => {
        set((s) => {
          const msgs = [...s.messages]
          const last = msgs[msgs.length - 1]
          if (last && last.role === 'assistant') {
            last.id = data.message_id || last.id
            last.session_id = data.session_id || s.currentSessionId
            last.isStreaming = false
            last.citations = data.citations
            last.used_citations = data.used_citations || []
            last.used_citation_markers = data.used_citation_markers || []
            last.invalid_citation_markers = data.invalid_citation_markers || []
            last.intent = data.intent
            last.evidence_sufficient = data.evidence_sufficient
            last.safety_notice = data.safety_notice
            last.elapsed = data.elapsed
          }
          const nextSessionId = data.session_id || s.currentSessionId
          if (nextSessionId) {
            localStorage.setItem('fireagent.currentSessionId', nextSessionId)
          }
          return { messages: msgs, currentSessionId: nextSessionId, isStreaming: false }
        })
        void get().loadSessions()
      },
      onError: (error) => {
        set((s) => {
          const msgs = [...s.messages]
          const last = msgs[msgs.length - 1]
          if (last && last.role === 'assistant') {
            last.content = `⚠️ 请求失败：${error}`
            last.isStreaming = false
          }
          return { messages: msgs, isStreaming: false }
        })
      },
    })
  },

  clearMessages: () => {
    if (abortController) abortController.abort()
    localStorage.removeItem('fireagent.currentSessionId')
    set({ currentSessionId: undefined, messages: [], isStreaming: false, currentStages: [] })
  },
}))

function messageItemToChatMessage(message: MessageItem): ChatMessage {
  return {
    id: message.message_id,
    session_id: message.session_id,
    role: message.role === 'assistant' ? 'assistant' : 'user',
    content: message.content,
    citations: message.citations || [],
    used_citations: Array.isArray(message.metadata?.used_citations)
      ? (message.metadata.used_citations as CitationItem[])
      : [],
    used_citation_markers: Array.isArray(message.metadata?.used_citation_markers)
      ? (message.metadata.used_citation_markers as string[])
      : [],
    invalid_citation_markers: Array.isArray(message.metadata?.invalid_citation_markers)
      ? (message.metadata.invalid_citation_markers as string[])
      : [],
    intent: message.intent,
    evidence_sufficient: Boolean(message.metadata?.evidence_sufficient),
  }
}
