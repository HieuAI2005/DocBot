import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.models.db_manager import get_db
from backend.models.database import Conversation, Message
from backend.models.schemas import (
    ChatRequest, ConversationResponse, ConversationList,
    MessageResponse, FeedbackCreate, FeedbackResponse
)
from backend.services.chat_service import get_chat_service
from backend.models.database import Feedback

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat endpoint with streaming response using Server-Sent Events (SSE)
    """
    chat_service = get_chat_service()
    
    # Get or create conversation
    conversation_id = request.conversation_id
    if conversation_id is None:
        # Create new conversation
        conv = Conversation(title=request.message[:50])
        db.add(conv)
        await db.flush()
        await db.refresh(conv)
        conversation_id = conv.id
    else:
        # Verify conversation exists
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(404, "Conversation not found")
    
    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.message
    )
    db.add(user_msg)
    await db.flush()
    
    # Get conversation history
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    history_msgs = list(result.scalars().all())
    
    # Format history for chat service
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_msgs[:-1]  # Exclude the just-added user message
    ]
    
    # Stream response
    async def event_generator():
        assistant_content = ""
        sources = None
        message_id = None
        
        try:
            async for chunk in chat_service.generate_response(
                query=request.message,
                conversation_history=conversation_history,
                use_adaptive_rag=request.use_adaptive_rag,
                max_chunks=request.max_chunks,
                document_id=request.document_id
            ):
                chunk_type = chunk.get("type")
                
                if chunk_type == "sources":
                    sources = chunk.get("sources", [])
                    # Send sources to client
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                elif chunk_type == "token":
                    content = chunk.get("content", "")
                    assistant_content += content
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                elif chunk_type == "done":
                    # Save assistant message
                    assistant_msg = Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_content,
                        sources=sources
                    )
                    db.add(assistant_msg)
                    await db.flush()
                    await db.refresh(assistant_msg)
                    message_id = assistant_msg.id
                    
                    # Send done with message_id
                    yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'conversation_id': conversation_id})}\n\n"
                
                elif chunk_type == "error":
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            await db.commit()
        
        except Exception as e:
            logger.error(f"Error in chat streaming: {e}", exc_info=True)
            await db.rollback()
            error_chunk = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/conversations", response_model=ConversationList)
async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """List all conversations"""
    result = await db.execute(
        select(Conversation)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    conversations = list(result.scalars().all())
    
    # Count total
    count_result = await db.execute(select(Conversation))
    total = len(list(count_result.scalars().all()))
    
    return ConversationList(conversations=conversations, total=total)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get conversation with all messages"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    
    if not conv:
        raise HTTPException(404, "Conversation not found")
    
    # Load messages
    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    conv.messages = list(msg_result.scalars().all())
    
    return conv


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback for a message"""
    
    # Verify message exists
    result = await db.execute(
        select(Message).where(Message.id == feedback.message_id)
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(404, "Message not found")
    
    # Create feedback
    fb = Feedback(
        message_id=feedback.message_id,
        rating=feedback.rating,
        is_helpful=1 if feedback.is_helpful else 0 if feedback.is_helpful is False else None,
        comment=feedback.comment
    )
    db.add(fb)
    await db.flush()
    await db.refresh(fb)
    
    # Update adaptive RAG with feedback
    if message.sources and feedback.is_helpful is not None:
        from backend.services.adaptive_rag import get_adaptive_rag
        adaptive_rag = get_adaptive_rag()
        
        chunk_ids = [source.get("id") for source in message.sources if source.get("id") is not None]
        if chunk_ids:
            adaptive_rag.add_feedback(chunk_ids, feedback.is_helpful)
    
    await db.commit()
    return fb
