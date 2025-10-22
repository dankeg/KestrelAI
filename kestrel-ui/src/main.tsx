import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './index.css'
import KestrelAIApp from './KestrelAIApp'
import ChatInterface from './ChatInterface'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<KestrelAIApp />} />
        <Route path="/test" element={<ChatInterface />} />
      </Routes>
    </Router>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)