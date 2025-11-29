import React, { useState, useEffect } from 'react';
import { Menu, FileText, Trash2, RefreshCw } from 'lucide-react';
import { documentsAPI } from '../services/api';
import DocumentUpload from './DocumentUpload';
import './Sidebar.css';

const Sidebar = ({ onDocumentSelect, selectedDocument, onGlobalChat }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadDocuments();
    }, []);

    const loadDocuments = async () => {
        setLoading(true);
        try {
            const response = await documentsAPI.list();
            setDocuments(response.documents || []);
        } catch (error) {
            console.error('Error loading documents:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (docId, e) => {
        e.stopPropagation();
        if (confirm('Bạn có chắc muốn xóa tài liệu này?')) {
            try {
                await documentsAPI.delete(docId);
                await loadDocuments();
            } catch (error) {
                console.error('Error deleting document:', error);
                alert('Không thể xóa tài liệu');
            }
        }
    };

    const getStatusBadge = (status) => {
        const badges = {
            completed: { text: 'Hoàn thành', color: 'var(--success)' },
            processing: { text: 'Đang xử lý', color: 'var(--warning)' },
            pending: { text: 'Chờ xử lý', color: 'var(--info)' },
            failed: { text: 'Lỗi', color: 'var(--error)' },
        };

        const badge = badges[status] || badges.pending;
        return (
            <span className="status-badge" style={{ color: badge.color }}>
                {badge.text}
            </span>
        );
    };

    return (
        <>
            <button className="sidebar-toggle" onClick={() => setIsOpen(!isOpen)}>
                <Menu size={24} />
            </button>

            <div className={`sidebar ${isOpen ? 'open' : ''} `}>
                <div className="sidebar-header">
                    <h2
                        onClick={onGlobalChat}
                        style={{ cursor: 'pointer' }}
                        title="Click để chat với tất cả documents"
                    >
                        DocBot
                    </h2>
                    <DocumentUpload onUploadComplete={loadDocuments} />
                </div>

                <div className="sidebar-body">
                    <div className="documents-header">
                        <h3>Tài liệu</h3>
                        <button className="refresh-btn" onClick={loadDocuments} disabled={loading}>
                            <RefreshCw size={16} className={loading ? 'icon-spin' : ''} />
                        </button>
                    </div>

                    {loading ? (
                        <div className="loading-state">
                            <RefreshCw size={24} className="icon-spin" />
                            <p>Đang tải...</p>
                        </div>
                    ) : documents.length === 0 ? (
                        <div className="empty-documents">
                            <FileText size={48} />
                            <p>Chưa có tài liệu nào</p>
                            <p className="subtext">Upload file PDF để bắt đầu</p>
                        </div>
                    ) : (
                        <div className="documents-list">
                            {documents.map((doc) => (
                                <div
                                    key={doc.id}
                                    className={`document-item ${selectedDocument?.id === doc.id ? 'selected' : ''}`}
                                    onClick={() => onDocumentSelect?.(doc)}
                                >
                                    <div className="document-icon">
                                        <FileText size={20} />
                                    </div>
                                    <div className="document-info">
                                        <div className="document-name">{doc.filename}</div>
                                        <div className="document-meta">
                                            {getStatusBadge(doc.processing_status)}
                                            {doc.num_chunks && (
                                                <span className="chunk-count">{doc.num_chunks} chunks</span>
                                            )}
                                        </div>
                                    </div>
                                    <button
                                        className="delete-btn"
                                        onClick={(e) => handleDelete(doc.id, e)}
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {isOpen && (
                <div className="sidebar-overlay" onClick={() => setIsOpen(false)} />
            )}
        </>
    );
};

export default Sidebar;
