import { API_BASE_URL } from './client'
import api from './client'
import type { DetectionModelInfo, VideoJobInfo } from './types'

export async function listDetectionModels(): Promise<DetectionModelInfo[]> {
  const response = await api.get<{ models: DetectionModelInfo[] }>('/detect/models')
  return Array.isArray(response.data?.models) ? response.data.models : []
}

export async function createVideoJob(
  file: File,
  modelId: string,
  confThreshold: number,
  iouThreshold: number,
): Promise<{ job_id: string; status: string }> {
  const formData = new FormData()
  formData.append('video', file)
  formData.append('model_id', modelId)
  formData.append('conf_threshold', String(confThreshold))
  formData.append('iou_threshold', String(iouThreshold))

  const response = await api.post<{ job_id: string; status: string }>(
    '/detect/videos',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000,
    },
  )
  return response.data
}

export async function getVideoJob(jobId: string): Promise<VideoJobInfo> {
  const response = await api.get<VideoJobInfo>(`/detect/videos/${jobId}`)
  return response.data
}

export function getVideoOutputUrl(jobId: string): string {
  return `${API_BASE_URL}/detect/videos/${jobId}/output`
}
