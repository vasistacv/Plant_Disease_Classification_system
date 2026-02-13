export default function Sidebar({ stats }) {
    const conditions = [
        { name: 'Healthy', color: '#059669' },
        { name: 'Mahali Koleroga', color: '#dc2626' },
        { name: 'Yellow Leaf Disease', color: '#d97706' },
        { name: 'Stem Cracking', color: '#d97706' },
        { name: 'Stem Bleeding', color: '#dc2626' },
        { name: 'Bud Borer', color: '#dc2626' },
        { name: 'Non-Arecanut', color: '#94a3b8' },
    ]

    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <div className="sidebar-brand-icon">🌿</div>
                <h2>KrishiSethu AI</h2>
                <span>Plant Intelligence</span>
            </div>

            <div className="sidebar-section">
                <h3>Performance</h3>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Accuracy</span>
                    <span className="sidebar-spec-value">99.84%</span>
                </div>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Classes</span>
                    <span className="sidebar-spec-value">7</span>
                </div>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Test Samples</span>
                    <span className="sidebar-spec-value">3,090</span>
                </div>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Correct</span>
                    <span className="sidebar-spec-value">3,085</span>
                </div>
            </div>

            <div className="sidebar-section">
                <h3>Session</h3>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Scans</span>
                    <span className="sidebar-spec-value">{stats.total}</span>
                </div>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Diseases</span>
                    <span className="sidebar-spec-value" style={{ color: '#dc2626' }}>{stats.diseases}</span>
                </div>
                <div className="sidebar-spec">
                    <span className="sidebar-spec-label">Healthy</span>
                    <span className="sidebar-spec-value">{stats.healthy}</span>
                </div>
            </div>

            <div className="sidebar-section">
                <h3>Detectable Conditions</h3>
                {conditions.map(c => (
                    <div className="sidebar-condition" key={c.name}>
                        <div className="sidebar-condition-dot" style={{ background: c.color }} />
                        <span className="sidebar-condition-name">{c.name}</span>
                    </div>
                ))}
            </div>

            <div className="sidebar-footer">
                <p>&copy; 2026 KrishiSethu AI<br />Deep Learning Powered</p>
            </div>
        </aside>
    )
}
