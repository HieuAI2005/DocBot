import { pdfjs } from 'react-pdf';

// Configure PDF.js worker - use CDN with exact version to avoid mismatch
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js`;

export default pdfjs;
