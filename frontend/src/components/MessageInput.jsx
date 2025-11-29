import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import './MessageInput.css';

const MessageInput = ({ onSend, disabled }) => {
    const [message, setMessage] = useState('');
    const [isComposing, setIsComposing] = useState(false);
    const textareaRef = useRef(null);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [message]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (message.trim() && !disabled) {
            onSend(message.trim());
            setMessage('');
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <form className="message-input-container" onSubmit={handleSubmit}>
            <div className="message-input-wrapper">
                <textarea
                    ref={textareaRef}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onCompositionStart={() => setIsComposing(true)}
                    onCompositionEnd={() => setIsComposing(false)}
                    placeholder="Hỏi về tài liệu của bạn..."
                    disabled={disabled}
                    rows={1}
                    className="message-input"
                />
                <button
                    type="submit"
                    disabled={disabled || !message.trim()}
                    className="send-button"
                >
                    {disabled ? (
                        <Loader2 className="icon-spin" size={20} />
                    ) : (
                        <Send size={20} />
                    )}
                </button>
            </div>
        </form>
    );
};

export default MessageInput;
