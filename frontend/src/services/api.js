import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Documents API
export const documentsAPI = {
    upload: async (file, onProgress) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post('/api/documents/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
                if (onProgress && progressEvent.total) {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    onProgress(percentCompleted);
                }
            },
        });

        return response.data;
    },

    list: async () => {
        const response = await api.get('/api/documents');
        return response.data;
    },

    get: async (id) => {
        const response = await api.get(`/api/documents/${id}`);
        return response.data;
    },

    delete: async (id) => {
        const response = await api.delete(`/api/documents/${id}`);
        return response.data;
    },
};

// Chat API with SSE streaming
export const chatAPI = {
    streamChat: async (message, conversationId, documentId, onChunk, onError) => {
        try {
            const requestBody = {
                message,
                conversation_id: conversationId,
                use_adaptive_rag: true,
            };

            // Add document_id if provided
            if (documentId) {
                requestBody.document_id = documentId;
            }

            const response = await fetch(`${API_BASE_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();

                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            onChunk(data);
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Chat streaming error:', error);
            onError(error);
        }
    },
};

// Conversations API
export const conversationsAPI = {
    list: async () => {
        const response = await api.get('/api/conversations');
        return response.data;
    },

    get: async (id) => {
        const response = await api.get(`/api/conversations/${id}`);
        return response.data;
    },
};

// Feedback API
export const feedbackAPI = {
    submit: async (messageId, isHelpful, rating, comment) => {
        const response = await api.post('/api/feedback', {
            message_id: messageId,
            is_helpful: isHelpful,
            rating,
            comment,
        });
        return response.data;
    },
};

export default api;
