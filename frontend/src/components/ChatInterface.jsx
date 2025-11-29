import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import MessageInput from './MessageInput';
import { chatAPI, feedbackAPI } from '../services/api';
import { Loader2, AlertCircle } from 'lucide-react';
import './ChatInterface.css';

const ChatInterface = ({ conversationId, selectedDocument }) => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentConversationId, setCurrentConversationId] = useState(conversationId);
    const messagesEndRef = useRef(null);
    const streamingMessageRef = useRef('');

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (messageText) => {
        setError(null);
        setIsLoading(true);

        // Add user message
        const userMessage = {
            role: 'user',
            content: messageText,
            id: Date.now(),
        };
        setMessages(prev => [...prev, userMessage]);

        // Add placeholder for assistant message
        const assistantMessageId = Date.now() + 1;
        setMessages(prev => [...prev, {
            role: 'assistant',
            content: '',
            reasoning: '',
            id: assistantMessageId,
            isStreaming: true,
        }]);

        streamingMessageRef.current = '';
        let reasoningText = '';
        let sources = null;
        let finalMessageId = null;

        try {
            await chatAPI.streamChat(
                messageText,
                currentConversationId,
                selectedDocument?.id,  // Pass document ID if available
                (chunk) => {
                    if (chunk.type === 'sources') {
                        sources = chunk.sources;
                        setMessages(prev => prev.map(msg =>
                            msg.id === assistantMessageId
                                ? { ...msg, sources }
                                : msg
                        ));
                    } else if (chunk.type === 'reasoning') {
                        // Handle reasoning chunk
                        reasoningText += chunk.content;
                        setMessages(prev => prev.map(msg =>
                            msg.id === assistantMessageId
                                ? { ...msg, reasoning: reasoningText }
                                : msg
                        ));
                    } else if (chunk.type === 'token') {
                        streamingMessageRef.current += chunk.content;
                        setMessages(prev => prev.map(msg =>
                            msg.id === assistantMessageId
                                ? { ...msg, content: streamingMessageRef.current }
                                : msg
                        ));
                    } else if (chunk.type === 'done') {
                        finalMessageId = chunk.message_id;
                        if (chunk.conversation_id && !currentConversationId) {
                            setCurrentConversationId(chunk.conversation_id);
                        }
                        setMessages(prev => prev.map(msg =>
                            msg.id === assistantMessageId
                                ? {
                                    ...msg,
                                    id: finalMessageId || msg.id,
                                    isStreaming: false,
                                }
                                : msg
                        ));
                        setIsLoading(false);
                    } else if (chunk.type === 'error') {
                        setError(chunk.error || 'An error occurred');
                        setIsLoading(false);
                    }
                },
                (err) => {
                    console.error('Streaming error:', err);
                    setError(err.message || 'Failed to connect to server');
                    setIsLoading(false);
                }
            );
        } catch (err) {
            console.error('Send message error:', err);
            setError(err.message || 'Failed to send message');
            setIsLoading(false);
        }
    };

    const handleFeedback = async (messageId, isHelpful) => {
        try {
            await feedbackAPI.submit(messageId, isHelpful);
            console.log('Feedback submitted:', messageId, isHelpful);
        } catch (err) {
            console.error('Error submitting feedback:', err);
        }
    };

    return (
        <div className="chat-interface">
            <div className="chat-messages">
                {messages.length === 0 && !isLoading ? (
                    <div className="empty-state">
                        <h2>Chào mừng đến với DocBot!</h2>
                        {selectedDocument ? (
                            <p>Hỏi bất kỳ điều gì về tài liệu <strong>{selectedDocument.filename}</strong></p>
                        ) : (
                            <p>Upload tài liệu PDF hoặc hỏi về tất cả tài liệu đã upload.</p>
                        )}
                    </div>
                ) : null}

                {messages.map((message) => (
                    <Message
                        key={message.id}
                        message={message}
                        onFeedback={handleFeedback}
                        isLoading={message.isStreaming && isLoading}
                    />
                ))}

                {error && (
                    <div className="error-message">
                        <AlertCircle size={20} />
                        <span>{error}</span>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <MessageInput onSend={handleSendMessage} disabled={isLoading} />
        </div>
    );
};

export default ChatInterface;
