import { NavLink } from 'react-router-dom'
import { Home, BookOpen, Flame, MessageCircle, Info } from 'lucide-react'

const navItems = [
  { to: '/', label: '首页', icon: Home },
  { to: '/knowledge', label: '知识库', icon: BookOpen },
  { to: '/science', label: '科普', icon: Flame },
  { to: '/chat', label: '问答', icon: MessageCircle },
  { to: '/about', label: '关于', icon: Info },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 bottom-0 w-20 lg:w-56 bg-bg-secondary/80 backdrop-blur-xl border-r border-white/5 z-30 flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-white/5">
        <span className="text-2xl">🔥</span>
        <span className="hidden lg:block ml-3 text-lg font-bold fire-text">FireAgent</span>
      </div>

      {/* 导航 */}
      <nav className="flex-1 py-4 flex flex-col gap-1 px-2 lg:px-3">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-3 rounded-xl text-sm transition-all ${
                isActive
                  ? 'bg-fire-500/15 text-fire-400 font-medium'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
              }`
            }
          >
            <Icon size={20} strokeWidth={1.8} />
            <span className="hidden lg:block">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* 底部 */}
      <div className="px-3 py-4 border-t border-white/5">
        <div className="hidden lg:block text-xs text-gray-500 text-center">
          FireAgent v0.1.0
        </div>
      </div>
    </aside>
  )
}
