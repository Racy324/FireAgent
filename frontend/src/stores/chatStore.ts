import { create } from 'zustand'
import type { ChatMessage, SSEStageEvent } from '../api/types'
import { sendStreamChat } from '../api/chat'

interface ChatState {
  messages: ChatMessage[]
  isStreaming: boolean
  currentStages: SSEStageEvent[]
  sendMessage: (query: string) => void
  clearMessages: () => void
}

let abortController: AbortController | null = null

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  currentStages: [],

  sendMessage: (query: string) => {
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
    }

    const assistantMsg: ChatMessage = {
      id: `assistant-${Date.now()}`,
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

    abortController = sendStreamChat(query, {
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
            last.isStreaming = false
            last.citations = data.citations
            last.intent = data.intent
            last.evidence_sufficient = data.evidence_sufficient
            last.safety_notice = data.safety_notice
            last.elapsed = data.elapsed
          }
          return { messages: msgs, isStreaming: false }
        })
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
    set({ messages: [], isStreaming: false, currentStages: [] })
  },
}))
