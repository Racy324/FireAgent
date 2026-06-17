import { useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { Home, BookOpen, Flame, MessageCircle, Info, Plus, X, ScanSearch } from 'lucide-react'
import { useChatStore } from '../stores/chatStore'

const navItems = [
  { to: '/', label: '首页', icon: Home },
  { to: '/knowledge', label: '知识库', icon: BookOpen },
  { to: '/science', label: '科普', icon: Flame },
  { to: '/chat', label: '问答', icon: MessageCircle },
  { to: '/detect', label: '检测', icon: ScanSearch },
  { to: '/about', label: '关于', icon: Info },
]

export default function Sidebar() {
  const navigate = useNavigate()
  const {
    sessions,
    currentSessionId,
    isLoadingSessions,
    loadSessions,
    selectSession,
    newSession,
    deleteSession,
  } = useChatStore()

  useEffect(() => {
    void loadSessions()
  }, [loadSessions])

  const handleNewChat = () => {
    newSession()
    navigate('/chat')
  }

  const handleSelectSession = (sessionId: string) => {
    void selectSession(sessionId).then(() => navigate('/chat'))
  }

  const handleDeleteSession = (sessionId: string) => {
    void deleteSession(sessionId)
  }

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-20 lg:w-56 bg-bg-secondary/80 backdrop-blur-xl border-r border-white/5 z-30 flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-white/5">
        <span className="text-2xl">🔥</span>
        <span className="hidden lg:block ml-3 text-lg font-bold fire-text">FireAgent</span>
      </div>

      {/* 导航 */}
      <nav className="py-4 flex flex-col gap-1 px-2 lg:px-3">
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

      {/* 最近会话 */}
      <section className="hidden min-h-0 flex-1 flex-col px-3 pb-4 lg:flex">
        <div className="mb-2 flex items-center justify-between px-1">
          <h2 className="text-xs font-semibold text-gray-500">最近</h2>
          <button
            onClick={handleNewChat}
            className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-white/5 hover:text-fire-300"
            title="新建会话"
          >
            <Plus size={14} />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
          {isLoadingSessions && (
            <div className="px-2 py-2 text-xs text-gray-600">加载中...</div>
          )}
          {!isLoadingSessions && sessions.length === 0 && (
            <div className="px-2 py-2 text-xs leading-relaxed text-gray-600">
              暂无历史会话
            </div>
          )}
          {sessions.map((session) => {
            const active = session.session_id === currentSessionId
            return (
              <div
                key={session.session_id}
                className={`group flex items-center rounded-lg text-sm transition-colors ${
                  active
                    ? 'bg-fire-500/15 text-fire-100'
                    : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'
                }`}
              >
                <button
                  onClick={() => handleSelectSession(session.session_id)}
                  className="min-w-0 flex-1 truncate px-2 py-2 text-left"
                  title={session.title || '新会话'}
                >
                  {session.title || '新会话'}
                </button>
                <button
                  onClick={(event) => {
                    event.stopPropagation()
                    handleDeleteSession(session.session_id)
                  }}
                  className="mr-1 rounded p-1 text-gray-600 opacity-0 transition-opacity hover:bg-red-500/10 hover:text-red-300 group-hover:opacity-100"
                  title="删除会话"
                >
                  <X size={12} />
                </button>
              </div>
            )
          })}
        </div>
      </section>

      {/* 底部 */}
      <div className="px-3 py-4 border-t border-white/5">
        <div className="hidden lg:block text-xs text-gray-500 text-center">
          FireAgent v0.1.0
        </div>
      </div>
    </aside>
  )
}
