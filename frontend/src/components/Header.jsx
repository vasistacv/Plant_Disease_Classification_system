export default function Header() {
    return (
        <header className="header">
            <div className="header-left">
                <h1><span className="accent-text">Arecanut</span> Disease Intelligence</h1>
                <p>AI-powered diagnostic system for real-time plant health monitoring and disease classification</p>
            </div>
            <div className="header-status">
                <div className="status-dot" />
                <span className="status-text">System Online</span>
            </div>
        </header>
    )
}
