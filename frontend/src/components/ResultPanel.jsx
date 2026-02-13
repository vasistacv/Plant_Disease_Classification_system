export default function ResultPanel({ result, loading }) {
    if (loading) {
        return (
            <div className="loading-overlay">
                <div className="spinner" />
                <div className="loading-text">Analyzing Patterns...</div>
                <div className="loading-subtext">Running neural network inference</div>
            </div>
        )
    }

    if (!result) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">🧬</div>
                <div className="empty-state-text">Upload an image to start analysis</div>
                <div className="empty-state-subtext">AI-powered disease classification in milliseconds</div>
            </div>
        )
    }

    if (result.error) {
        return (
            <div className="result-panel neutral">
                <div className="result-header">
                    <div>
                        <div className="result-label">Error</div>
                        <div className="result-disease-name" style={{ color: '#dc2626', fontSize: '1.2rem' }}>
                            Connection Failed
                        </div>
                    </div>
                </div>
                <p className="result-description">{result.error}</p>
            </div>
        )
    }

    let panelClass = 'neutral'
    if (result.class_name === 'Healthy') panelClass = 'healthy'
    else if (result.class_name === 'Not_Arecanut') panelClass = 'neutral'
    else if (result.severity === 'Critical') panelClass = 'critical'
    else panelClass = 'warning'

    // If low confidence, override to warning styling
    if (result.low_confidence && result.class_name !== 'Not_Arecanut') panelClass = 'warning'

    let confColor = '#059669'
    if (result.confidence < 80) confColor = '#dc2626'
    else if (result.confidence < 95) confColor = '#d97706'

    const severityStyles = {
        'None': { bg: '#ecfdf5', text: '#059669', border: '#059669' },
        'Medium': { bg: '#fffbeb', text: '#d97706', border: '#d97706' },
        'High': { bg: '#fef2f2', text: '#dc2626', border: '#dc2626' },
        'Critical': { bg: '#fef2f2', text: '#dc2626', border: '#dc2626' },
        'N/A': { bg: '#f1f5f9', text: '#64748b', border: '#94a3b8' },
    }
    const sev = severityStyles[result.severity] || severityStyles['N/A']

    const sortedProbs = Object.entries(result.all_probabilities || {}).sort(([, a], [, b]) => b - a)

    return (
        <div>
            {/* Low Confidence Warning Banner */}
            {result.low_confidence && (
                <div style={{
                    background: '#fffbeb',
                    border: '1px solid #fcd34d',
                    borderRadius: '12px',
                    padding: '14px 18px',
                    marginBottom: '14px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '10px',
                    animation: 'fadeInUp 0.3s ease-out',
                }}>
                    <span style={{ fontSize: '1.3rem', flexShrink: 0 }}>⚠️</span>
                    <div>
                        <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#92400e', marginBottom: '3px' }}>
                            Low Confidence Warning
                        </div>
                        <div style={{ fontSize: '0.8rem', color: '#a16207', lineHeight: 1.5 }}>
                            {result.confidence_warning || 'The model is not confident about this prediction. This image may not be an Arecanut plant.'}
                        </div>
                    </div>
                </div>
            )}

            <div className={`result-panel ${panelClass}`}>
                <div className="result-header">
                    <div>
                        <div className={`result-id-tag ${result.is_arecanut ? 'arecanut' : 'non-arecanut'}`}>
                            {result.is_arecanut ? '✓ Arecanut Identified' : '✗ Non-Arecanut Plant'}
                        </div>
                        <div className="result-label">Diagnosis Result</div>
                        <div className="result-disease-name">{result.display_name}</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                        <span className="severity-badge" style={{ background: sev.bg, color: sev.text, border: `1px solid ${sev.border}20` }}>
                            {result.severity} Severity
                        </span>
                        <div className="inference-time">{result.inference_ms}ms inference</div>
                    </div>
                </div>

                <div className="confidence-section">
                    <span className="confidence-label">Confidence</span>
                    <div className="confidence-bar-bg">
                        <div className="confidence-bar-fill" style={{ width: `${result.confidence}%`, background: `linear-gradient(90deg, ${confColor}, ${confColor}cc)` }} />
                    </div>
                    <div className="confidence-number" style={{ color: confColor }}>{result.confidence}%</div>
                </div>

                <p className="result-description">{result.description}</p>
            </div>

            {result.class_name !== 'Not_Arecanut' && (
                <div className="treatment-card">
                    <h4>💊 Recommended Treatment</h4>
                    <p>{result.treatment}</p>
                </div>
            )}

            {result.prevention && result.prevention.length > 0 && (
                <div className="prevention-card">
                    <h4>🛡 Prevention Guidelines</h4>
                    {result.prevention.map((tip, i) => (
                        <div className="prevention-item" key={i}>
                            <div className="prevention-dot" />
                            <div className="prevention-text">{tip}</div>
                        </div>
                    ))}
                </div>
            )}

            {sortedProbs.length > 0 && (
                <div className="probability-card">
                    <h4>📊 Confidence Distribution</h4>
                    {sortedProbs.map(([className, prob]) => {
                        const isActive = className === result.class_name
                        return (
                            <div className="prob-row" key={className}>
                                <div className={`prob-label ${isActive ? 'active' : 'inactive'}`}>{className.replace(/_/g, ' ')}</div>
                                <div className="prob-bar-bg">
                                    <div className={`prob-bar-fill ${isActive ? 'active' : 'inactive'}`} style={{ width: `${prob}%` }} />
                                </div>
                                <div className={`prob-value ${isActive ? 'active' : 'inactive'}`}>{prob}%</div>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
