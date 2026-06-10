import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, AlertTriangle, Flame, TreePine, Building2, Scan, Users, BarChart3 } from 'lucide-react'

interface ScienceCard {
  icon: React.ReactNode
  title: string
  subtitle: string
  description: string
  keyData: { label: string; value: string }[]
  query: string
  gradient: string
}

const CARDS: ScienceCard[] = [
  {
    icon: <Flame size={24} />,
    title: '隧道火灾安全',
    subtitle: '烟气控制与通风排烟',
    description: '隧道火灾烟气是最主要的致死因素。烟气中含大量 CO 和有毒气体，能见度急剧下降严重阻碍人员疏散。有效的通风排烟措施是保障人员安全的关键。',
    keyData: [
      { label: 'CO 浓度', value: '>500ppm 致死' },
      { label: '能见度', value: '<10m 危险' },
      { label: '温度', value: '>60℃ 灼伤' },
      { label: '黄金逃生', value: '<5 分钟' },
    ],
    query: '隧道火灾烟气控制方法有哪些',
    gradient: 'from-red-500/20 to-orange-500/20',
  },
  {
    icon: <TreePine size={24} />,
    title: '森林火灾防控',
    subtitle: '蔓延预测与风险评估',
    description: '森林火灾蔓延受地形、气象、可燃物类型多重因素影响。基于深度学习的预测模型可辅助火情预判，遥感技术实现全天候监测。',
    keyData: [
      { label: '蔓延速度', value: '受风速影响' },
      { label: '监测手段', value: '遥感+无人机' },
      { label: '预测模型', value: 'DL+元胞自动机' },
    ],
    query: '森林火灾蔓延预测模型',
    gradient: 'from-green-500/20 to-emerald-500/20',
  },
  {
    icon: <Building2 size={24} />,
    title: '建筑消防安全',
    subtitle: '高层疏散与风险评估',
    description: '高层建筑火灾疏散面临垂直距离长、人员密集、烟气蔓延快等挑战。动态风险评估和疏散路径优化可显著提升安全性。',
    keyData: [
      { label: '疏散难点', value: '垂直距离' },
      { label: '外墙材料', value: '保温材料易燃' },
      { label: '评估方法', value: '动态风险模型' },
    ],
    query: '高层建筑火灾疏散优化策略',
    gradient: 'from-blue-500/20 to-cyan-500/20',
  },
  {
    icon: <Scan size={24} />,
    title: '火灾智能检测',
    subtitle: 'YOLO 与深度学习',
    description: '基于改进 YOLO、Transformer 等深度学习模型的火灾检测方法，在复杂场景下实现高精度火焰和烟雾识别，支持视频监控和红外图像。',
    keyData: [
      { label: '主流模型', value: 'YOLOv8/v5' },
      { label: '检测类型', value: '火焰+烟雾' },
      { label: '应用场景', value: '视频/红外/遥感' },
    ],
    query: '基于深度学习的火灾检测方法',
    gradient: 'from-purple-500/20 to-violet-500/20',
  },
  {
    icon: <Users size={24} />,
    title: '人员疏散仿真',
    subtitle: '路径规划与行为建模',
    description: '人员疏散仿真通过元胞自动机、社会力模型等方法模拟人群行为，结合 VR 技术进行应急演练，优化疏散路线和出口设计。',
    keyData: [
      { label: '仿真模型', value: '元胞自动机' },
      { label: '影响因素', value: '能见度/恐慌' },
      { label: '技术手段', value: 'VR+LLM' },
    ],
    query: '火灾场景下人员疏散仿真方法',
    gradient: 'from-amber-500/20 to-yellow-500/20',
  },
  {
    icon: <BarChart3 size={24} />,
    title: '火灾风险评估',
    subtitle: '量化模型与决策支持',
    description: '火灾风险评估通过量化模型评估火灾发生概率和后果严重程度，为消防规划和应急管理提供数据支撑。',
    keyData: [
      { label: '评估维度', value: '概率+后果' },
      { label: '数据来源', value: '历史+传感器' },
      { label: '输出', value: '风险等级地图' },
    ],
    query: '火灾风险评估量化模型',
    gradient: 'from-teal-500/20 to-cyan-500/20',
  },
]

const EMERGENCY_TIPS = [
  { title: '🔴 发生火灾怎么办？', query: '发生火灾如何正确逃生自救' },
  { title: '🟡 灭火器使用方法', query: '灭火器的正确使用方法是什么' },
  { title: '🟢 高层逃生指南', query: '高层建筑发生火灾如何逃生' },
  { title: '🔵 隧道火灾自救', query: '隧道内发生火灾如何自救' },
]

export default function PopularScience() {
  const navigate = useNavigate()

  const handleAsk = (query: string) => {
    navigate(`/chat?q=${encodeURIComponent(query)}`)
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold fire-text">🔥 火灾知识科普</h1>
        <p className="text-sm text-gray-400 mt-1">基于 69 篇火灾领域论文，科普关键安全知识</p>
      </div>

      {/* 知识卡片网格 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {CARDS.map((card, i) => (
          <motion.div
            key={card.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="glass-card p-5 space-y-4 group cursor-pointer"
            onClick={() => handleAsk(card.query)}
          >
            <div className="flex items-start gap-3">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${card.gradient} flex items-center justify-center text-fire-300 flex-shrink-0`}>
                {card.icon}
              </div>
              <div>
                <h3 className="font-bold text-gray-100 group-hover:text-fire-300 transition-colors">{card.title}</h3>
                <p className="text-xs text-gray-500">{card.subtitle}</p>
              </div>
            </div>

            <p className="text-sm text-gray-400 leading-relaxed">{card.description}</p>

            {/* 关键数据 */}
            <div className="grid grid-cols-2 gap-2">
              {card.keyData.map((d) => (
                <div key={d.label} className="bg-bg-primary/50 rounded-lg px-3 py-2">
                  <div className="text-[10px] text-gray-500">{d.label}</div>
                  <div className="text-sm font-medium text-fire-300">{d.value}</div>
                </div>
              ))}
            </div>

            <div className="flex items-center justify-end text-xs text-fire-400 group-hover:text-fire-300 transition-colors">
              深入了解 <ArrowRight size={14} className="ml-1" />
            </div>
          </motion.div>
        ))}
      </div>

      {/* 应急知识速查 */}
      <div>
        <h2 className="text-lg font-bold text-gray-200 mb-3 flex items-center gap-2">
          <AlertTriangle size={18} className="text-yellow-400" /> 应急知识速查
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {EMERGENCY_TIPS.map((tip) => (
            <button
              key={tip.title}
              onClick={() => handleAsk(tip.query)}
              className="glass-card p-4 text-left hover:border-fire-500/30 transition-all group"
            >
              <span className="text-sm font-medium text-gray-300 group-hover:text-fire-300 transition-colors">
                {tip.title}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
