import React, { useState, useRef } from 'react';
import { Upload, X, FileText, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { documentsAPI } from '../services/api';
import './DocumentUpload.css';

const DocumentUpload = ({ onUploadComplete }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState(null); // 'success', 'error', null
    const [uploadMessage, setUploadMessage] = useState('');
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        const files = Array.from(e.dataTransfer.files);
        const pdfFile = files.find(file => file.type === 'application/pdf');

        if (pdfFile) {
            handleUpload(pdfFile);
        } else {
            setUploadStatus('error');
            setUploadMessage('Vui lòng chọn file PDF');
            setTimeout(() => setUploadStatus(null), 3000);
        }
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleUpload(file);
        }
    };

    const handleUpload = async (file) => {
        setUploading(true);
        setUploadProgress(0);
        setUploadStatus(null);

        try {
            const response = await documentsAPI.upload(file, (progress) => {
                setUploadProgress(progress);
            });

            setUploadStatus('success');
            setUploadMessage(`Đã upload thành công: ${file.name}`);

            if (onUploadComplete) {
                onUploadComplete(response);
            }

            setTimeout(() => {
                setIsOpen(false);
                setUploadStatus(null);
                setUploading(false);
            }, 2000);
        } catch (error) {
            console.error('Upload error:', error);
            setUploadStatus('error');
            setUploadMessage(error.response?.data?.detail || 'Lỗi khi upload file');
            setUploading(false);
        }
    };

    return (
        <>
            <button className="upload-trigger-btn" onClick={() => setIsOpen(true)}>
                <Upload size={20} />
                <span>Upload PDF</span>
            </button>

            {isOpen && (
                <div className="upload-modal-overlay" onClick={() => !uploading && setIsOpen(false)}>
                    <div className="upload-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="upload-modal-header">
                            <h3>Upload Tài Liệu</h3>
                            {!uploading && (
                                <button className="close-btn" onClick={() => setIsOpen(false)}>
                                    <X size={20} />
                                </button>
                            )}
                        </div>

                        <div className="upload-modal-body">
                            {!uploading ? (
                                <div
                                    className={`drop-zone ${isDragging ? 'dragging' : ''}`}
                                    onDragOver={handleDragOver}
                                    onDragLeave={handleDragLeave}
                                    onDrop={handleDrop}
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <FileText size={48} className="drop-zone-icon" />
                                    <p className="drop-zone-text">
                                        Kéo và thả file PDF vào đây
                                    </p>
                                    <p className="drop-zone-subtext">hoặc click để chọn file</p>
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept=".pdf,application/pdf"
                                        onChange={handleFileSelect}
                                        style={{ display: 'none' }}
                                    />
                                </div>
                            ) : (
                                <div className="upload-progress-container">
                                    {uploadStatus === null && (
                                        <>
                                            <Loader2 size={48} className="icon-spin progress-icon" />
                                            <p className="progress-text">Đang upload và xử lý...</p>
                                            <div className="progress-bar">
                                                <div
                                                    className="progress-fill"
                                                    style={{ width: `${uploadProgress}%` }}
                                                />
                                            </div>
                                            <p className="progress-percent">{uploadProgress}%</p>
                                        </>
                                    )}

                                    {uploadStatus === 'success' && (
                                        <>
                                            <CheckCircle size={48} className="success-icon" />
                                            <p className="success-text">{uploadMessage}</p>
                                        </>
                                    )}

                                    {uploadStatus === 'error' && (
                                        <>
                                            <XCircle size={48} className="error-icon" />
                                            <p className="error-text">{uploadMessage}</p>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default DocumentUpload;
