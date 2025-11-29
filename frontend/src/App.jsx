import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import DocumentViewer from './components/DocumentViewer';
import './App.css';

function App() {
    const [selectedDocument, setSelectedDocument] = useState(null);
    const [conversationId, setConversationId] = useState(null);

    const handleDocumentSelect = (doc) => {
        setSelectedDocument(doc);
        console.log('Selected document:', doc);
    };

    const handleGlobalChat = () => {
        setSelectedDocument(null);
        console.log('Switched to global chat mode');
    };

    return (
        <div className="app">
            <Sidebar
                onDocumentSelect={handleDocumentSelect}
                selectedDocument={selectedDocument}
                onGlobalChat={handleGlobalChat}
            />

            {selectedDocument ? (
                <>
                    <div className="document-section">
                        <DocumentViewer document={selectedDocument} />
                    </div>
                    <div className="chat-section">
                        <ChatInterface
                            conversationId={conversationId}
                            selectedDocument={selectedDocument}
                        />
                    </div>
                </>
            ) : (
                <main className="main-content">
                    <ChatInterface
                        conversationId={conversationId}
                        selectedDocument={null}
                    />
                </main>
            )}
        </div>
    );
}

export default App;
