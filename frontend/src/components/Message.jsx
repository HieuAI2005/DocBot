import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { User, Bot, ThumbsUp, ThumbsDown, FileText, ChevronDown, Lightbulb } from 'lucide-react';
import './Message.css';

const Message = ({ message, onFeedback, isLoading }) => {
    const isUser = message.role === 'user';
    const [isReasoningExpanded, setIsReasoningExpanded] = useState(true);

    const handleFeedback = (isHelpful) => {
        if (onFeedback && message.id) {
            onFeedback(message.id, isHelpful);
        }
    };

    return (
        <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
            <div className="message-content-wrapper">
                <div className="message-avatar">
                    {isUser ? <User size={20} /> : <Bot size={20} />}
                </div>

                <div className="message-content">
                    {/* Reasoning Section - Only for assistant messages */}
                    {!isUser && message.reasoning && (
                        <div className="message-reasoning">
                            <button
                                className="reasoning-toggle"
                                onClick={() => setIsReasoningExpanded(!isReasoningExpanded)}
                            >
                                <Lightbulb size={16} />
                                <span>Quá trình suy luận</span>
                                <ChevronDown
                                    size={16}
                                    className={`chevron ${isReasoningExpanded ? 'expanded' : ''}`}
                                />
                            </button>

                            {isReasoningExpanded && (
                                <div className="reasoning-content">
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        components={{
                                            code({ node, inline, className, children, ...props }) {
                                                const match = /language-(\w+)/.exec(className || '');
                                                return !inline && match ? (
                                                    <SyntaxHighlighter
                                                        style={vscDarkPlus}
                                                        language={match[1]}
                                                        PreTag="div"
                                                        {...props}
                                                    >
                                                        {String(children).replace(/\n$/, '')}
                                                    </SyntaxHighlighter>
                                                ) : (
                                                    <code className={className} {...props}>
                                                        {children}
                                                    </code>
                                                );
                                            },
                                        }}
                                    >
                                        {message.reasoning}
                                    </ReactMarkdown>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Message Text */}
                    <div className="message-text">
                        {isUser ? (
                            <p>{message.content}</p>
                        ) : isLoading && (!message.content || message.content.length === 0) ? (
                            <div className="typing-indicator">
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                            </div>
                        ) : (
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                    code({ node, inline, className, children, ...props }) {
                                        const match = /language-(\w+)/.exec(className || '');
                                        return !inline && match ? (
                                            <SyntaxHighlighter
                                                style={vscDarkPlus}
                                                language={match[1]}
                                                PreTag="div"
                                                {...props}
                                            >
                                                {String(children).replace(/\n$/, '')}
                                            </SyntaxHighlighter>
                                        ) : (
                                            <code className={className} {...props}>
                                                {children}
                                            </code>
                                        );
                                    },
                                }}
                            >
                                {message.content}
                            </ReactMarkdown>
                        )}
                    </div>

                    {/* Sources */}
                    {!isUser && message.sources && message.sources.length > 0 && (
                        <div className="message-sources">
                            <div className="sources-header">
                                <FileText size={14} />
                                <span>Nguồn tài liệu:</span>
                            </div>
                            <div className="sources-list">
                                {message.sources.slice(0, 3).map((source, idx) => (
                                    <div key={idx} className="source-item">
                                        <span className="source-badge">{idx + 1}</span>
                                        <span className="source-text">
                                            {source.doc_id} - Chunk {source.chunk_id}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Feedback buttons */}
                    {!isUser && message.id && (
                        <div className="message-actions">
                            <button
                                className="feedback-btn"
                                onClick={() => handleFeedback(true)}
                                title="Helpful"
                            >
                                <ThumbsUp size={16} />
                            </button>
                            <button
                                className="feedback-btn"
                                onClick={() => handleFeedback(false)}
                                title="Not helpful"
                            >
                                <ThumbsDown size={16} />
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Message;
