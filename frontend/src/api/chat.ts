import { API_BASE_URL } from './client'
import type { SSEStageEvent } from './types'

export interface StreamCallbacks {
  onStage?: (event: SSEStageEvent) => void
  onToken?: (token: string) => void
  onDone?: (data: {
    citations: string[]
    intent: string
    evidence_sufficient: boolean
    safety_notice?: string
    elapsed?: number
  }) => void
  onError?: (error: string) => void
}

/**
 * 发送流式问答请求，通过 SSE 逐 token 接收回答。
 */
export function sendStreamChat(query: string, callbacks: StreamCallbacks): AbortController {
  const controller = new AbortController()
  const streamUrl = `${API_BASE_URL}/chat/stream`

  fetch(streamUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('无法读取响应流')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            const dataStr = line.slice(6)
            try {
              const data = JSON.parse(dataStr)
              switch (currentEvent) {
                case 'stage':
                  callbacks.onStage?.(data)
                  break
                case 'token':
                  callbacks.onToken?.(data.token)
                  break
                case 'done':
                  callbacks.onDone?.(data)
                  break
                case 'error':
                  callbacks.onError?.(data.error)
                  break
              }
            } catch {
              // 忽略解析错误
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        callbacks.onError?.(err.message)
      }
    })

  return controller
}
