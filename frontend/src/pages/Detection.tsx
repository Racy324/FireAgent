import { useEffect, useState, useRef, useCallback } from 'react'
import { Upload, Play, Download, Loader2, AlertCircle, CheckCircle2, Video } from 'lucide-react'
import {
  listDetectionModels,
  createVideoJob,
  getVideoJob,
  getVideoOutputUrl,
} from '../api/detection'
import type { DetectionModelInfo, VideoJobInfo } from '../api/types'

export default function Detection() {
  // 模型
  const [models, setModels] = useState<DetectionModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [confThreshold, setConfThreshold] = useState(0.35)
  const [iouThreshold, setIouThreshold] = useState(0.7)
  const [loadingModels, setLoadingModels] = useState(true)

  // 上传视频
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobInfo, setJobInfo] = useState<VideoJobInfo | null>(null)
  const [error, setError] = useState('')

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 加载模型列表
  useEffect(() => {
    listDetectionModels()
      .then((m) => {
        setModels(m)
        if (m.length > 0) setSelectedModel(m[0].model_id)
      })
      .catch(() => setError('无法加载模型列表'))
      .finally(() => setLoadingModels(false))
  }, [])

  // 轮询任务状态
  const startPolling = useCallback((id: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const info = await getVideoJob(id)
        setJobInfo(info)
        if (info.status === 'succeeded' || info.status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          if (info.status === 'failed') setError(info.error || '检测失败')
        }
      } catch {
        // 忽略单次轮询失败
      }
    }, 1500)
  }, [])

  // 清理
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const handleUpload = async () => {
    if (!file || !selectedModel) return
    setError('')
    setUploading(true)
    setJobInfo(null)
    setJobId(null)

    try {
      const result = await createVideoJob(file, selectedModel, confThreshold, iouThreshold)
      setJobId(result.job_id)
      startPolling(result.job_id)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '上传失败'
      setError(msg)
    } finally {
      setUploading(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setJobId(null)
      setJobInfo(null)
      setError('')
    }
  }

  const isProcessing = jobInfo?.status === 'queued' || jobInfo?.status === 'running'
  const isDone = jobInfo?.status === 'succeeded'
  const isFailed = jobInfo?.status === 'failed'

  const selectedModelInfo = models.find((m) => m.model_id === selectedModel)

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold fire-text">🎯 火灾视觉检测</h1>

      {/* 工具栏 */}
      <div className="glass-card p-4 space-y-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* 模型选择 */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">模型</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loadingModels || isProcessing}
              className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-fire-500/50"
            >
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.display_name}{m.ready ? '' : ' (首次使用将自动加载)'}
                </option>
              ))}
              {models.length === 0 && !loadingModels && <option>暂无模型</option>}
            </select>
          </div>

          {/* 置信度 */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">置信度</label>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={confThreshold}
              onChange={(e) => setConfThreshold(Number(e.target.value))}
              disabled={isProcessing}
              className="w-24 accent-fire-500"
            />
            <span className="text-sm text-gray-300 w-10">{confThreshold.toFixed(2)}</span>
          </div>

          {/* IoU */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">IoU</label>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={iouThreshold}
              onChange={(e) => setIouThreshold(Number(e.target.value))}
              disabled={isProcessing}
              className="w-24 accent-fire-500"
            />
            <span className="text-sm text-gray-300 w-10">{iouThreshold.toFixed(2)}</span>
          </div>
        </div>

        {selectedModelInfo && (
          <div className="flex items-center gap-3 text-xs text-gray-500">
            <span>类别：{selectedModelInfo.labels.join(', ')}</span>
            <span>状态：{selectedModelInfo.ready ? '✅ 已加载' : '⏳ 未加载（首次使用自动加载）'}</span>
          </div>
        )}
      </div>

      {/* 上传区域 */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-4 flex-wrap">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-gray-300 hover:bg-white/10 transition-colors"
          >
            <Upload size={16} />
            {file ? file.name : '选择视频文件'}
          </button>

          {file && (
            <span className="text-xs text-gray-500">
              {(file.size / 1024 / 1024).toFixed(1)} MB
            </span>
          )}

          <button
            onClick={handleUpload}
            disabled={!file || !selectedModel || uploading || isProcessing}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-fire-500/20 border border-fire-500/30 text-sm text-fire-300 hover:bg-fire-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {uploading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {uploading ? '上传中...' : '开始检测'}
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-400">
            <AlertCircle size={16} />
            {error}
          </div>
        )}
      </div>

      {/* 进度 */}
      {jobInfo && (
        <div className="glass-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isProcessing && <Loader2 size={18} className="animate-spin text-fire-400" />}
              {isDone && <CheckCircle2 size={18} className="text-green-400" />}
              {isFailed && <AlertCircle size={18} className="text-red-400" />}
              <span className="text-sm text-gray-300">
                {jobInfo.status === 'queued' && '排队中...'}
                {jobInfo.status === 'running' && '检测中...'}
                {jobInfo.status === 'succeeded' && '检测完成'}
                {jobInfo.status === 'failed' && '检测失败'}
              </span>
            </div>
            <span className="text-xs text-gray-500">{jobInfo.input_filename}</span>
          </div>

          {/* 进度条 */}
          {(isProcessing || isDone) && (
            <div>
              <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span>
                  {jobInfo.metrics?.frames_done ?? 0} / {jobInfo.metrics?.frames_total ?? '?'} 帧
                </span>
                <span>{(jobInfo.progress * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full bg-fire-500/60 transition-all duration-500"
                  style={{ width: `${(jobInfo.progress * 100).toFixed(1)}%` }}
                />
              </div>
              {jobInfo.metrics?.avg_latency_ms != null && (
                <div className="mt-1 text-xs text-gray-600">
                  平均推理：{jobInfo.metrics.avg_latency_ms.toFixed(1)} ms / 帧
                </div>
              )}
            </div>
          )}

          {/* 错误 */}
          {isFailed && jobInfo.error && (
            <div className="text-sm text-red-400 bg-red-500/10 rounded-lg p-3">
              {jobInfo.error}
            </div>
          )}
        </div>
      )}

      {/* 结果播放 */}
      {isDone && jobId && (
        <div className="glass-card p-6 space-y-4">
          <h2 className="text-lg font-semibold text-gray-200 flex items-center gap-2">
            <Video size={20} className="text-fire-400" />
            检测结果
          </h2>

          <div className="w-full max-h-[600px] bg-black/50 rounded-lg border border-white/10 flex items-center justify-center overflow-hidden">
            <video
              controls
              className="max-h-[600px] w-auto max-w-full object-contain"
              src={getVideoOutputUrl(jobId)}
            />
          </div>

          <a
            href={getVideoOutputUrl(jobId)}
            download
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-gray-300 hover:bg-white/10 transition-colors"
          >
            <Download size={16} />
            下载结果视频
          </a>
        </div>
      )}
    </div>
  )
}
