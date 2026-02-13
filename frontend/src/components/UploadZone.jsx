import { useRef, useState, useCallback } from 'react'

export default function UploadZone({ onImageSelect, imagePreview, selectedImage, onAnalyze, loading }) {
    const fileInputRef = useRef(null)
    const [dragging, setDragging] = useState(false)

    const handleDragOver = useCallback((e) => { e.preventDefault(); setDragging(true) }, [])
    const handleDragLeave = useCallback(() => setDragging(false), [])
    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setDragging(false)
        const file = e.dataTransfer.files[0]
        if (file && file.type.startsWith('image/')) onImageSelect(file)
    }, [onImageSelect])
    const handleFileChange = useCallback((e) => {
        const file = e.target.files[0]
        if (file) onImageSelect(file)
    }, [onImageSelect])

    if (!imagePreview) {
        return (
            <>
                <div
                    className={`upload-zone ${dragging ? 'dragging' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <div className="upload-icon">📸</div>
                    <div className="upload-text">Upload Arecanut Image</div>
                    <div className="upload-subtext">Drag & drop or click to browse · JPG, PNG supported</div>
                    <button className="upload-browse" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}>
                        Browse Files
                    </button>
                </div>
                <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
            </>
        )
    }

    const fileSizeKB = selectedImage ? (selectedImage.size / 1024).toFixed(1) : '—'
    const fileFormat = selectedImage?.name?.split('.').pop()?.toUpperCase() || 'JPG'
    const shortName = selectedImage?.name?.length > 20 ? selectedImage.name.substring(0, 20) + '...' : selectedImage?.name

    return (
        <div>
            <div className="image-preview-container">
                <img src={imagePreview} alt="Uploaded" className="image-preview" />
                <div className="image-meta">
                    <div className="image-meta-item">
                        <div className="image-meta-label">File</div>
                        <div className="image-meta-value" style={{ fontSize: '0.75rem' }}>{shortName}</div>
                    </div>
                    <div className="image-meta-item">
                        <div className="image-meta-label">Size</div>
                        <div className="image-meta-value">{fileSizeKB} KB</div>
                    </div>
                    <div className="image-meta-item">
                        <div className="image-meta-label">Format</div>
                        <div className="image-meta-value">{fileFormat}</div>
                    </div>
                </div>
            </div>

            <button className="analyze-btn" onClick={onAnalyze} disabled={loading}>
                {loading ? '⏳  Analyzing...' : '🔬  Analyze Image'}
            </button>

            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
            <button
                className="upload-browse"
                style={{ width: '100%', marginTop: '8px', background: '#f7f8fa', color: '#475569', border: '1px solid #e5e8ef', boxShadow: 'none' }}
                onClick={() => fileInputRef.current?.click()}
            >
                Change Image
            </button>
        </div>
    )
}
