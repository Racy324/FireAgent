import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface Props {
  icon: ReactNode
  label: string
  value: string | number
  delay?: number
}

export default function StatCard({ icon, label, value, delay = 0 }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="glass-card p-5 flex items-center gap-4"
    >
      <div className="w-12 h-12 rounded-xl bg-fire-500/10 flex items-center justify-center text-fire-400">
        {icon}
      </div>
      <div>
        <div className="text-2xl font-bold fire-text">{value}</div>
        <div className="text-sm text-gray-400">{label}</div>
      </div>
    </motion.div>
  )
}
