import { useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import MetricsBar from './components/MetricsBar'
import UploadZone from './components/UploadZone'
import ResultPanel from './components/ResultPanel'
import History from './components/History'

const API_URL = 'http://localhost:8000'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [stats, setStats] = useState({ total: 0, diseases: 0, healthy: 0 })
  const [deviceInfo, setDeviceInfo] = useState('Loading...')

  // Fetch device info on mount
  useState(() => {
    fetch(`${API_URL}/api/health`)
      .then(r => r.json())
      .then(data => setDeviceInfo(data.device === 'cuda:0' ? 'NVIDIA GPU' : 'CPU'))
      .catch(() => setDeviceInfo('Offline'))
  }, [])

  const handleImageSelect = useCallback((file) => {
    setSelectedImage(file)
    setResult(null)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target.result)
    reader.readAsDataURL(file)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedImage) return
    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      setResult(data)

      // Update stats
      setStats(prev => ({
        total: prev.total + 1,
        diseases: prev.diseases + (data.class_name !== 'Healthy' && data.class_name !== 'Not_Arecanut' ? 1 : 0),
        healthy: prev.healthy + (data.class_name === 'Healthy' ? 1 : 0),
      }))

      // Add to history
      setHistory(prev => [{
        filename: data.filename,
        result: data.display_name,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString(),
        class_name: data.class_name,
      }, ...prev].slice(0, 15))

    } catch (error) {
      console.error('Prediction failed:', error)
      setResult({ error: 'Failed to connect to AI engine. Make sure the backend is running.' })
    } finally {
      setLoading(false)
    }
  }, [selectedImage])

  return (
    <div className="app-layout">
      <Sidebar stats={stats} />
      <main className="main-content">
        <Header />
        <MetricsBar stats={stats} device={deviceInfo} />

        <div className="workspace">
          <div>
            <UploadZone
              onImageSelect={handleImageSelect}
              imagePreview={imagePreview}
              selectedImage={selectedImage}
              onAnalyze={handleAnalyze}
              loading={loading}
            />
          </div>
          <div>
            <ResultPanel result={result} loading={loading} />
          </div>
        </div>

        {history.length > 0 && <History entries={history} />}
      </main>
    </div>
  )
}

export default App
