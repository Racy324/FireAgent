import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import KnowledgeBase from './pages/KnowledgeBase'
import Chat from './pages/Chat'
import PopularScience from './pages/PopularScience'
import Detection from './pages/Detection'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/knowledge" element={<KnowledgeBase />} />
        <Route path="/science" element={<PopularScience />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/detect" element={<Detection />} />
      </Route>
    </Routes>
  )
}
