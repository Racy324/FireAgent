import { motion } from 'framer-motion'

const TECH_STACK = [
  { category: '前端', items: ['React 18', 'TypeScript', 'Tailwind CSS', 'ECharts', 'Framer Motion'] },
  { category: '后端', items: ['FastAPI', 'LangGraph', 'Pydantic v2', 'Qdrant', 'Tavily'] },
  { category: 'AI 模型', items: ['bge-m3 (Embedding)', 'bge-reranker-v2-m3', 'qwen3.5-plus (LLM)'] },
  { category: '数据处理', items: ['pdfplumber', 'MinerU', '语义切块', 'BM25 稀疏编码'] },
]

const WORKFLOW_STEPS = [
  '意图路由',
  '查询改写',
  'Dense + Sparse 并行检索',
  'Weighted RRF 融合',
  'Cross-Encoder 重排',
  '证据充分性检查',
  '联网搜索兜底 (Tavily)',
  '上下文构建',
  'LLM 回答生成',
  '幻觉检查',
]

export default function About() {
  return (
    <div className="space-y-8 max-w-3xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold fire-text mb-2">关于 FireAgent</h1>
        <p className="text-gray-400 leading-relaxed">
          FireAgent 是一个面向火灾领域论文知识库的专家问答系统。以本地火灾论文 PDF 为主要知识来源，
          离线解析论文内容，进行结构感知切块和混合检索；当本地证据不足，或问题涉及最新政策、标准、法规、
          事故等时效信息时，使用 Tavily 联网搜索作为临时证据兜底。
        </p>
      </motion.div>

      {/* 技术架构 */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <h2 className="text-lg font-bold text-gray-200 mb-3">🛠 技术架构</h2>
        <div className="grid grid-cols-2 gap-3">
          {TECH_STACK.map((group) => (
            <div key={group.category} className="glass-card p-4">
              <h3 className="text-sm font-medium text-fire-400 mb-2">{group.category}</h3>
              <div className="flex flex-wrap gap-1.5">
                {group.items.map((item) => (
                  <span key={item} className="text-xs px-2 py-1 rounded-full bg-bg-tertiary text-gray-400 border border-white/5">
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* 工作流 */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <h2 className="text-lg font-bold text-gray-200 mb-3">⚙️ RAG 工作流</h2>
        <div className="glass-card p-5">
          <div className="space-y-2">
            {WORKFLOW_STEPS.map((step, i) => (
              <div key={step} className="flex items-center gap-3">
                <div className="w-7 h-7 rounded-full bg-fire-500/15 flex items-center justify-center text-xs font-bold text-fire-400 flex-shrink-0">
                  {i + 1}
                </div>
                <span className="text-sm text-gray-300">{step}</span>
                {i < WORKFLOW_STEPS.length - 1 && (
                  <div className="flex-1 border-b border-dashed border-white/5 ml-2" />
                )}
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* 数据说明 */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <h2 className="text-lg font-bold text-gray-200 mb-3">📋 数据说明</h2>
        <div className="glass-card p-4 text-sm text-gray-400 space-y-2">
          <p>• 知识库基于本地火灾领域学术论文 PDF，涵盖隧道火灾、森林防火、建筑消防、火灾检测、人员疏散、风险评估等方向。</p>
          <p>• 回答严格基于检索到的论文证据，区分本地论文证据和联网资料证据。</p>
          <p>• 涉及应急、法规、标准、政策和事故等高风险内容时，请核对官方发布源。</p>
          <p>• <strong className="text-red-400">现场火灾请优先拨打 119 并撤离。</strong></p>
        </div>
      </motion.div>
    </div>
  )
}
