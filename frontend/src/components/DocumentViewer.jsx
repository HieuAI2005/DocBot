import React, { useState, useMemo } from 'react';
import { Document, Page } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, FileText } from 'lucide-react';
import '../pdfConfig'; // Initialize PDF.js worker
import './DocumentViewer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

const DocumentViewer = ({ document }) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);
    const [scale, setScale] = useState(1.0);
    const [loading, setLoading] = useState(true);

    const onDocumentLoadSuccess = ({ numPages }) => {
        setNumPages(numPages);
        setPageNumber(1);
        setLoading(false);
    };

    const onDocumentLoadError = (error) => {
        console.error('Error loading PDF:', error);
        setLoading(false);
    };

    const goToPrevPage = () => {
        setPageNumber(prev => Math.max(1, prev - 1));
    };

    const goToNextPage = () => {
        setPageNumber(prev => Math.min(numPages, prev + 1));
    };

    const zoomIn = () => {
        setScale(prev => Math.min(3.0, prev + 0.2));
    };

    const zoomOut = () => {
        setScale(prev => Math.max(0.5, prev - 0.2));
    };

    if (!document) {
        return (
            <div className="document-viewer-empty">
                <FileText size={64} />
                <h3>Chưa chọn tài liệu</h3>
                <p>Chọn một tài liệu từ sidebar để xem</p>
            </div>
        );
    }

    // Memoize file config to prevent unnecessary reloads
    const fileConfig = useMemo(() => {
        const filename = document.file_path.split('/').pop();
        const pdfUrl = `http://localhost:8000/uploads/${filename}`;

        console.log('PDF Config:', { filename, pdfUrl });

        return {
            url: pdfUrl,
            httpHeaders: {
                'Accept': 'application/pdf'
            },
            withCredentials: false
        };
    }, [document.file_path]);

    return (
        <div className="document-viewer">
            <div className="document-viewer-header">
                <div className="document-controls">
                    <button onClick={zoomOut} className="control-btn" title="Zoom out">
                        <ZoomOut size={18} />
                    </button>
                    <span className="zoom-level">{Math.round(scale * 100)}%</span>
                    <button onClick={zoomIn} className="control-btn" title="Zoom in">
                        <ZoomIn size={18} />
                    </button>
                </div>
            </div>

            <div className="pdf-container">
                <Document
                    file={fileConfig}
                    onLoadSuccess={onDocumentLoadSuccess}
                    onLoadError={onDocumentLoadError}
                    loading={
                        <div className="pdf-loading">
                            <div className="spinner"></div>
                            <p>Đang tải PDF...</p>
                        </div>
                    }
                    error={
                        <div className="pdf-error">
                            <p>❌ Không thể tải PDF</p>
                            <small style={{ marginTop: '8px', display: 'block' }}>
                                {fileConfig.url.split('/').pop()}
                            </small>
                            <small style={{ color: 'var(--text-tertiary)', marginTop: '4px', display: 'block' }}>
                                Kiểm tra console để xem lỗi chi tiết
                            </small>
                        </div>
                    }
                >
                    <Page
                        pageNumber={pageNumber}
                        scale={scale}
                        renderTextLayer={false}
                        renderAnnotationLayer={false}
                    />
                </Document>
            </div>

            {numPages && (
                <div className="document-viewer-footer">
                    <button
                        onClick={goToPrevPage}
                        disabled={pageNumber <= 1}
                        className="nav-btn"
                    >
                        <ChevronLeft size={18} />
                        Trang trước
                    </button>
                    <span className="page-info">
                        Trang {pageNumber} / {numPages}
                    </span>
                    <button
                        onClick={goToNextPage}
                        disabled={pageNumber >= numPages}
                        className="nav-btn"
                    >
                        Trang sau
                        <ChevronRight size={18} />
                    </button>
                </div>
            )}
        </div>
    );
};

export default DocumentViewer;
