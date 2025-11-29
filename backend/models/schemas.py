from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============ Document Schemas ============
class DocumentBase(BaseModel):
    filename: str
    
    
class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: int
    file_path: str
    upload_time: datetime
    processing_status: str
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None
    doc_metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ============ Message Schemas ============
class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    conversation_id: Optional[int] = None
    sources: Optional[List[Dict[str, Any]]] = None


class MessageResponse(MessageBase):
    id: int
    conversation_id: int
    created_at: datetime
    sources: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        from_attributes = True


# ============ Conversation Schemas ============
class ConversationBase(BaseModel):
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    pass


class ConversationResponse(ConversationBase):
    id: int
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []
    
    class Config:
        from_attributes = True


class ConversationList(BaseModel):
    conversations: List[ConversationResponse]
    total: int


# ============ Chat Schemas ============
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    document_id: Optional[int] = None  # Filter RAG to specific document
    use_adaptive_rag: bool = True
    max_chunks: Optional[int] = None


class ChatStreamChunk(BaseModel):
    type: str  # "token", "sources", "done", "error"
    content: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    message_id: Optional[int] = None
    error: Optional[str] = None


# ============ Feedback Schemas ============
class FeedbackCreate(BaseModel):
    message_id: int
    rating: Optional[int] = Field(None, ge=1, le=5)
    is_helpful: Optional[bool] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    rating: Optional[int] = None
    is_helpful: Optional[int] = None
    comment: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
