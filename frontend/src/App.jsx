import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [alerts, setAlerts] = useState([])
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    // Connect to WebSocket for real-time alerts
    const wsUrl = window.location.protocol === 'https:' ? `wss://${window.location.host}/ws/alerts` : `ws://${window.location.host}/ws/alerts`;
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      setIsConnected(true)
    }
    
    ws.onmessage = (event) => {
      const newAlert = JSON.parse(event.data)
      setAlerts(prev => [newAlert, ...prev].slice(0, 50)) // Keep last 50 alerts
    }
    
    ws.onclose = () => {
      setIsConnected(false)
    }
    
    return () => {
      ws.close()
    }
  }, [])

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>IBVAP Command Center</h1>
        <div className={`status ${isConnected ? 'online' : 'offline'}`}>
          {isConnected ? 'System Online' : 'Connecting...'}
        </div>
      </header>
      
      <main className="main-content">
        <section className="video-section">
          <h2>Live Surveillance Feed</h2>
          <div className="video-wrapper">
            <img 
              src="/api/video_feed" 
              alt="Live RTSP Feed" 
              onError={(e) => { e.target.onerror = null; e.target.src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='480'><rect width='100%' height='100%' fill='%23333'/><text x='50%' y='50%' font-size='20' fill='white' text-anchor='middle'>Stream Offline</text></svg>" }}
            />
          </div>
        </section>
        
        <aside className="alerts-sidebar">
          <h2>Real-Time Alerts</h2>
          <div className="alerts-list">
            {alerts.length === 0 ? (
              <div className="no-alerts">No alerts yet...</div>
            ) : (
              alerts.map((alert, idx) => (
                <div key={idx} className={`alert-card ${alert.type}`}>
                  <div className="alert-time">
                    {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                  </div>
                  <div className="alert-message">{alert.message}</div>
                </div>
              ))
            )}
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
