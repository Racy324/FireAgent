import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import FireParticles from './FireParticles'

export default function Layout() {
  return (
    <div className="min-h-screen bg-bg-primary text-gray-100">
      <FireParticles />
      <Sidebar />
      <main className="ml-20 lg:ml-56 min-h-screen relative z-10">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
