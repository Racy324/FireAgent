import api from './client'
import type { PapersListResponse, PaperDetail, PapersStats } from './types'

export async function fetchPapers(params: {
  page?: number
  page_size?: number
  keyword?: string
  year?: number
  sort?: string
}): Promise<PapersListResponse> {
  const { data } = await api.get('/papers', { params })
  return data
}

export async function fetchPaperDetail(docId: string): Promise<PaperDetail> {
  const { data } = await api.get(`/papers/${docId}`)
  return data
}

export async function fetchPapersStats(): Promise<PapersStats> {
  const { data } = await api.get('/papers/stats')
  return data
}
