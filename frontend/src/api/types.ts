// FireAgent API 类型定义

export interface ChatRequest {
  query: string
  session_id?: string
  include_context?: boolean
  include_debug?: boolean
}

export interface ChatResponse {
  answer: string
  session_id: string
  message_id: string
  intent: string
  evidence_sufficient: boolean
  citations: string[]
  context?: string
  errors: string[]
  debug: Record<string, unknown>
}

export interface PaperItem {
  doc_id: string
  title: string
  authors: string[]
  year: number | null
  abstract: string
  keywords: string[]
  chunk_count: number
}

export interface PaperChunk {
  chunk_id: string
  text: string
  section_title: string
  page_start: number | null
  page_end: number | null
  chunk_type: string
}

export interface PaperDetail {
  doc_id: string
  title: string
  authors: string[]
  year: number | null
  abstract: string
  keywords: string[]
  chunk_count: number
  chunks: PaperChunk[]
}

export interface PapersStats {
  total_papers: number
  total_chunks: number
  year_distribution: Record<string, number>
  top_keywords: Record<string, number>
}

export interface PapersListResponse {
  total: number
  page: number
  page_size: number
  papers: PaperItem[]
}

export interface SSEStageEvent {
  stage: string
  [key: string]: unknown
}

export interface SSETokenEvent {
  token: string
}

export interface SSEDoneEvent {
  citations: string[]
  session_id?: string
  message_id?: string
  intent: string
  evidence_sufficient: boolean
  safety_notice?: string
  elapsed?: number
}

export interface SessionItem {
  session_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface SessionsListResponse {
  sessions: SessionItem[]
}

export interface MessageItem {
  message_id: string
  session_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  created_at: string
  intent?: string
  citations?: string[]
  metadata?: Record<string, unknown>
}

export interface SessionMessagesResponse {
  session_id: string
  messages: MessageItem[]
}

export interface ChatMessage {
  id: string
  session_id?: string
  role: 'user' | 'assistant'
  content: string
  citations?: string[]
  intent?: string
  evidence_sufficient?: boolean
  safety_notice?: string
  elapsed?: number
  stages?: SSEStageEvent[]
  isStreaming?: boolean
}
