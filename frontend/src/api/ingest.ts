import api from './client'
import type { UploadIngestResult } from './types'

export async function uploadPdf(file: File, maxPages?: number): Promise<UploadIngestResult> {
  const formData = new FormData()
  formData.append('file', file)
  if (maxPages) formData.append('max_pages', String(maxPages))

  const response = await api.post<UploadIngestResult>('/ingest/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
  return response.data
}
