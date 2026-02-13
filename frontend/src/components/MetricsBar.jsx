export default function MetricsBar({ stats, device }) {
    return (
        <div className="metrics-bar">
            <div className="metric-card">
                <div className="metric-value accent">99.84%</div>
                <div className="metric-label">Model Accuracy</div>
            </div>
            <div className="metric-card">
                <div className="metric-value">7</div>
                <div className="metric-label">Disease Classes</div>
            </div>
            <div className="metric-card">
                <div className="metric-value small">{device}</div>
                <div className="metric-label">Compute Engine</div>
            </div>
            <div className="metric-card">
                <div className="metric-value">{stats.total}</div>
                <div className="metric-label">Total Scans</div>
            </div>
        </div>
    )
}
