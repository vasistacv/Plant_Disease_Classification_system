export default function History({ entries }) {
    const getBadgeClass = (className) => {
        if (className === 'Healthy') return 'healthy'
        if (className === 'Not_Arecanut') return 'non-arecanut'
        return 'disease'
    }

    return (
        <div className="history-panel">
            <h4>🕐 Recent Scan History</h4>
            {entries.map((entry, i) => (
                <div className="history-row" key={i}>
                    <div>
                        <span className="history-name">{entry.filename}</span>
                        <span className="history-time">{entry.time}</span>
                    </div>
                    <div className="history-right">
                        <span className="history-confidence">{entry.confidence}%</span>
                        <span className={`history-badge ${getBadgeClass(entry.class_name)}`}>
                            {entry.result}
                        </span>
                    </div>
                </div>
            ))}
        </div>
    )
}
