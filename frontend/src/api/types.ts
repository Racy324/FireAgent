// FireAgent API 类型定义

export interface ChatRequest {
  query: string
  session_id?: string
  include_context?: boolean
  include_debug?: boolean
}

export interface CitationItem {
  citation_id: string
  marker: string
  source_type: string
  title: string
  authors: string[]
  year: number | null
  doc_id: string
  chunk_id: string
  parent_id: string
  section_title: string
  section_path: string[]
  page_start: number | null
  page_end: number | null
  url: string
  score: number
  text_preview: string
  metadata: Record<string, unknown>
}

export interface ChatResponse {
  answer: string
  session_id: string
  message_id: string
  intent: string
  evidence_sufficient: boolean
  citations: string[]
  used_citations: CitationItem[]
  used_citation_markers: string[]
  invalid_citation_markers: string[]
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
  used_citations?: CitationItem[]
  used_citation_markers?: string[]
  invalid_citation_markers?: string[]
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
  used_citations?: CitationItem[]
  used_citation_markers?: string[]
  invalid_citation_markers?: string[]
  intent?: string
  evidence_sufficient?: boolean
  safety_notice?: string
  elapsed?: number
  stages?: SSEStageEvent[]
  isStreaming?: boolean
}

// ── Ingest 类型 ──

export interface UploadIngestResult {
  status: 'indexed' | 'skipped' | 'error'
  doc_id: string
  filename: string
  content_hash: string
  chunks: number
  message: string
  old_chunks_deleted: number
}

// ── Detection 类型 ──

export interface DetectionModelInfo {
  model_id: string
  display_name: string
  type: string
  ready: boolean
  labels: string[]
  default_conf_threshold: number
}

export interface VideoJobInfo {
  job_id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  progress: number
  input_filename: string
  output_url: string | null
  error: string | null
  metrics: {
    frames_total?: number
    frames_done?: number
    avg_latency_ms?: number
  }
}
