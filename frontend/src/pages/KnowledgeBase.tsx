import { useEffect, useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, FileText, ChevronLeft, ChevronRight, X, Layers, Upload, CheckCircle2, AlertCircle, SkipForward } from 'lucide-react'
import { fetchPapers, fetchPaperDetail } from '../api/papers'
import { uploadPdf } from '../api/ingest'
import type { PaperItem, PaperDetail, UploadIngestResult } from '../api/types'
import { truncate } from '../utils/format'

export default function KnowledgeBase() {
  const [papers, setPapers] = useState<PaperItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [keyword, setKeyword] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedPaper, setSelectedPaper] = useState<PaperDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // 上传状态
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<UploadIngestResult | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadPapers = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetchPapers({ page, page_size: 12, keyword, sort: 'year_desc' })
      setPapers(res.papers)
      setTotal(res.total)
    } catch { /* ignore */ }
    setLoading(false)
  }, [page, keyword])

  useEffect(() => { loadPapers() }, [loadPapers])

  const handleSearch = () => {
    setKeyword(searchInput)
    setPage(1)
  }

  const openDetail = async (docId: string) => {
    setDetailLoading(true)
    try {
      const detail = await fetchPaperDetail(docId)
      setSelectedPaper(detail)
    } catch { /* ignore */ }
    setDetailLoading(false)
  }

  const totalPages = Math.ceil(total / 12)

  // 上传处理
  const handleFileSelect = (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) return
    setUploadFile(file)
    setUploadResult(null)
  }

  const handleUpload = async () => {
    if (!uploadFile || uploading) return
    setUploading(true)
    setUploadResult(null)
    try {
      const result = await uploadPdf(uploadFile)
      setUploadResult(result)
      if (result.status === 'indexed') {
        setUploadFile(null)
        loadPapers()
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '上传失败'
      setUploadResult({ status: 'error', doc_id: '', filename: uploadFile.name, content_hash: '', chunks: 0, message: msg, old_chunks_deleted: 0 })
    }
    setUploading(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const clearUpload = () => {
    setUploadFile(null)
    setUploadResult(null)
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold fire-text">📚 知识库</h1>

      {/* PDF 上传区域 */}
      <div
        className={`glass-card p-4 transition-colors ${dragOver ? 'border-fire-400 bg-fire-500/5' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
            dragOver ? 'border-fire-400 bg-fire-500/10' : 'border-white/10 hover:border-fire-500/30 hover:bg-white/[0.02]'
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFileSelect(file)
              e.target.value = ''
            }}
          />
          <Upload className="mx-auto mb-2 text-gray-500" size={24} />
          <p className="text-sm text-gray-400">
            拖拽 PDF 到此处，或<span className="text-fire-400 underline ml-1">点击选择文件</span>
          </p>
          <p className="text-xs text-gray-600 mt-1">支持 .pdf，增量更新已有知识库</p>
        </div>

        {/* 已选择文件 + 上传按钮 */}
        {uploadFile && (
          <div className="flex items-center gap-3 mt-3 px-2">
            <FileText size={16} className="text-fire-400 flex-shrink-0" />
            <span className="text-sm text-gray-300 truncate flex-1">{uploadFile.name}</span>
            <span className="text-xs text-gray-500">{(uploadFile.size / 1024 / 1024).toFixed(1)} MB</span>
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="px-4 py-1.5 bg-fire-500 text-white rounded-lg text-sm hover:bg-fire-600 disabled:opacity-50 transition-colors"
            >
              {uploading ? '索引中...' : '上传并索引'}
            </button>
            <button onClick={clearUpload} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
              <X size={16} className="text-gray-500" />
            </button>
          </div>
        )}

        {/* 上传结果 */}
        {uploadResult && (
          <div className={`flex items-center gap-2 mt-3 px-2 py-2 rounded-lg text-sm ${
            uploadResult.status === 'indexed' ? 'bg-green-500/10 text-green-400' :
            uploadResult.status === 'skipped' ? 'bg-blue-500/10 text-blue-400' :
            'bg-red-500/10 text-red-400'
          }`}>
            {uploadResult.status === 'indexed' && <CheckCircle2 size={16} />}
            {uploadResult.status === 'skipped' && <SkipForward size={16} />}
            {uploadResult.status === 'error' && <AlertCircle size={16} />}
            <span>{uploadResult.message}</span>
          </div>
        )}
      </div>

      {/* 搜索栏 */}
      <div className="glass-card p-3 flex items-center gap-2">
        <Search className="ml-2 text-gray-500" size={18} />
        <input
          type="text"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="搜索论文标题、作者、关键词..."
          className="flex-1 bg-transparent outline-none py-2 px-1 text-sm text-gray-200 placeholder:text-gray-500"
        />
        <button
          onClick={handleSearch}
          className="px-4 py-2 bg-fire-500/20 text-fire-400 rounded-lg text-sm hover:bg-fire-500/30 transition-colors"
        >
          搜索
        </button>
      </div>

      {/* 论文列表 */}
      {loading ? (
        <div className="text-center text-gray-500 py-20">加载中...</div>
      ) : papers.length === 0 ? (
        <div className="text-center text-gray-500 py-20">未找到匹配的论文</div>
      ) : (
        <div className="grid gap-3">
          {papers.map((paper, i) => (
            <motion.div
              key={paper.doc_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
              className="glass-card p-4 cursor-pointer group"
              onClick={() => openDetail(paper.doc_id)}
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-fire-500/10 flex items-center justify-center text-fire-400 flex-shrink-0 mt-0.5">
                  <FileText size={18} />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-gray-200 group-hover:text-fire-300 transition-colors truncate">
                    {paper.title}
                  </h3>
                  <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                    <span>{paper.authors.join(', ') || '作者未知'}</span>
                    {paper.year && <span>· {paper.year}</span>}
                    <span>· {paper.chunk_count} 个知识片段</span>
                  </div>
                  {paper.abstract && (
                    <p className="text-xs text-gray-400 mt-2 leading-relaxed">
                      {truncate(paper.abstract, 150)}
                    </p>
                  )}
                  {paper.keywords.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {paper.keywords.slice(0, 5).map((kw) => (
                        <span key={kw} className="text-[10px] px-2 py-0.5 rounded-full bg-fire-500/10 text-fire-400">
                          {kw}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 text-sm text-gray-400">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            className="p-2 rounded-lg hover:bg-white/5 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft size={18} />
          </button>
          <span>{page} / {totalPages}</span>
          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
            className="p-2 rounded-lg hover:bg-white/5 disabled:opacity-30 transition-colors"
          >
            <ChevronRight size={18} />
          </button>
          <span className="text-gray-500">共 {total} 篇</span>
        </div>
      )}

      {/* 论文详情侧滑面板 */}
      <AnimatePresence>
        {selectedPaper && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40"
              onClick={() => setSelectedPaper(null)}
            />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 bottom-0 w-full max-w-xl bg-bg-secondary border-l border-white/5 z-50 overflow-y-auto"
            >
              <div className="p-6 space-y-5">
                <div className="flex items-start justify-between">
                  <h2 className="text-lg font-bold text-gray-100 pr-4">{selectedPaper.title}</h2>
                  <button onClick={() => setSelectedPaper(null)} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
                    <X size={20} className="text-gray-400" />
                  </button>
                </div>

                <div className="flex flex-wrap gap-2 text-xs text-gray-400">
                  <span>👤 {selectedPaper.authors.join(', ') || '作者未知'}</span>
                  {selectedPaper.year && <span>📅 {selectedPaper.year}</span>}
                  <span>📦 {selectedPaper.chunk_count} 个知识片段</span>
                </div>

                {selectedPaper.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {selectedPaper.keywords.map((kw) => (
                      <span key={kw} className="text-xs px-2 py-1 rounded-full bg-fire-500/10 text-fire-400">{kw}</span>
                    ))}
                  </div>
                )}

                {selectedPaper.abstract && (
                  <div className="glass-card p-4">
                    <h4 className="text-xs font-medium text-gray-400 mb-2">📝 摘要</h4>
                    <p className="text-sm text-gray-300 leading-relaxed">{selectedPaper.abstract}</p>
                  </div>
                )}

                <div>
                  <h4 className="text-xs font-medium text-gray-400 mb-3 flex items-center gap-1">
                    <Layers size={14} /> 知识片段 ({selectedPaper.chunks.length})
                  </h4>
                  <div className="space-y-2">
                    {selectedPaper.chunks.map((chunk) => (
                      <div key={chunk.chunk_id} className="glass-card p-3">
                        <div className="flex items-center gap-2 text-[10px] text-gray-500 mb-1">
                          {chunk.section_title && <span className="text-fire-400">{chunk.section_title}</span>}
                          {chunk.page_start && <span>第{chunk.page_start}{chunk.page_end && chunk.page_end !== chunk.page_start ? `-${chunk.page_end}` : ''}页</span>}
                          {chunk.chunk_type && <span className="px-1.5 py-0.5 rounded bg-white/5">{chunk.chunk_type}</span>}
                        </div>
                        <p className="text-xs text-gray-300 leading-relaxed">{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
