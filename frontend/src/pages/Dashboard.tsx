import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FileText, Layers, MessageCircle, Search } from 'lucide-react'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import 'echarts-wordcloud'
import StatCard from '../components/StatCard'
import { fetchPapersStats, fetchPapers } from '../api/papers'
import { formatYearDistribution, formatKeywordCloud, extractKeywordsFromTitles } from '../utils/format'
import type { PapersStats, PaperItem } from '../api/types'

const HOT_QUESTIONS = [
  '隧道火灾烟气对人员疏散有什么影响',
  '森林火灾蔓延预测模型有哪些',
  '高层建筑火灾疏散优化策略',
  '基于深度学习的火灾检测方法',
  '火灾场景下人员疏散仿真',
]

export default function Dashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState<PapersStats | null>(null)
  const [papers, setPapers] = useState<PaperItem[]>([])
  const [question, setQuestion] = useState('')

  useEffect(() => {
    fetchPapersStats().then(setStats).catch(() => {})
    fetchPapers({ page: 1, page_size: 100 }).then((r) => setPapers(r.papers)).catch(() => {})
  }, [])

  const handleSubmit = (q?: string) => {
    const query = q || question
    if (query.trim()) navigate(`/chat?q=${encodeURIComponent(query.trim())}`)
  }

  const yearData = stats ? formatYearDistribution(stats.year_distribution) : []
  const keywordData = stats && Object.keys(stats.top_keywords).length > 0
    ? formatKeywordCloud(stats.top_keywords)
    : papers.length > 0
    ? extractKeywordsFromTitles(papers)
    : []

  return (
    <div className="space-y-8">
      {/* 标题区 */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-6"
      >
        <h1 className="text-4xl lg:text-5xl font-black fire-text mb-3">FireAgent</h1>
        <p className="text-gray-400 text-lg">火灾领域智能知识问答系统</p>
      </motion.div>

      {/* 搜索框 */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        className="max-w-2xl mx-auto"
      >
        <div className="glass-card p-2 flex items-center gap-2 focus-within:border-fire-500/40 transition-colors">
          <Search className="ml-3 text-gray-500" size={20} />
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="输入你的火灾领域问题..."
            className="flex-1 bg-transparent outline-none py-3 px-2 text-gray-200 placeholder:text-gray-500"
          />
          <button
            onClick={() => handleSubmit()}
            className="px-6 py-2.5 bg-gradient-to-r from-fire-500 to-fire-600 text-white rounded-xl font-medium hover:from-fire-400 hover:to-fire-500 transition-all active:scale-95"
          >
            提问
          </button>
        </div>
        {/* 热门问题 */}
        <div className="flex flex-wrap gap-2 mt-3 justify-center">
          {HOT_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => handleSubmit(q)}
              className="text-xs px-3 py-1.5 rounded-full bg-bg-tertiary text-gray-400 hover:text-fire-400 hover:bg-fire-500/10 transition-all border border-white/5 hover:border-fire-500/20"
            >
              {q}
            </button>
          ))}
        </div>
      </motion.div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard icon={<FileText size={24} />} label="论文数量" value={stats?.total_papers ?? '...'} delay={0.3} />
        <StatCard icon={<Layers size={24} />} label="知识片段" value={stats?.total_chunks ?? '...'} delay={0.4} />
        <StatCard icon={<MessageCircle size={24} />} label="研究方向" value={Object.keys(stats?.top_keywords ?? {}).length || '...'} delay={0.5} />
      </div>

      {/* 图表区 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 年份分布 */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }} className="glass-card p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">📊 论文年份分布</h3>
          {yearData.length > 0 ? (
            <ReactECharts
              style={{ height: 250 }}
              option={{
                grid: { top: 10, right: 20, bottom: 30, left: 40 },
                xAxis: { type: 'category', data: yearData.map((d) => d.year), axisLabel: { color: '#6B6B80', fontSize: 11 }, axisLine: { lineStyle: { color: '#333' } } },
                yAxis: { type: 'value', axisLabel: { color: '#6B6B80', fontSize: 11 }, splitLine: { lineStyle: { color: '#222' } } },
                series: [{
                  type: 'bar',
                  data: yearData.map((d) => d.count),
                  itemStyle: {
                    borderRadius: [4, 4, 0, 0],
                    color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#FB923C' }, { offset: 1, color: '#C2410C' }] },
                  },
                  barWidth: '60%',
                }],
                tooltip: { trigger: 'axis', backgroundColor: '#1A1A24', borderColor: '#333', textStyle: { color: '#F5F5F5' } },
              }}
            />
          ) : (
            <div className="h-[250px] flex items-center justify-center text-gray-500 text-sm">加载中...</div>
          )}
        </motion.div>

        {/* 关键词词云 */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }} className="glass-card p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">🏷️ 热门研究关键词</h3>
          {keywordData.length > 0 ? (
            <ReactECharts
              style={{ height: 250 }}
              option={{
                series: [{
                  type: 'wordCloud',
                  shape: 'circle',
                  left: 'center',
                  top: 'center',
                  width: '90%',
                  height: '90%',
                  sizeRange: [14, 40],
                  rotationRange: [-30, 30],
                  gridSize: 8,
                  textStyle: {
                    fontFamily: '"Noto Sans SC", sans-serif',
                    fontWeight: 500,
                    color: () => {
                      const colors = ['#FB923C', '#F97316', '#EA580C', '#FDBA74', '#C2410C', '#FDE68A']
                      return colors[Math.floor(Math.random() * colors.length)]
                    },
                  },
                  data: keywordData,
                }],
              }}
            />
          ) : (
            <div className="h-[250px] flex items-center justify-center text-gray-500 text-sm">加载中...</div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
