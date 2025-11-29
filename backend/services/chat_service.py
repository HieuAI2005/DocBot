import sys
from pathlib import Path
import torch
from typing import List, Dict, Optional, AsyncGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.process.config import LLM, LLM_4BIT, SEED
MAX_RESPONSE_TOKENS = 256  # Reduced from 512 to save memory
MAX_CONTEXT_CHARS = 2500
from backend.config import STREAM_CHUNK_SIZE
from backend.services.adaptive_rag import get_adaptive_rag

logger = logging.getLogger(__name__)
torch.manual_seed(SEED)


class ChatService:
    """Service for handling chat interactions with streaming"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.adaptive_rag = get_adaptive_rag()
        # Don't load LLM at startup - lazy load when needed
    
    def _load_llm(self):
        """Load LLM model lazily"""
        try:
            logger.info(f"Loading LLM: {LLM}")
            
            # Use CPU to avoid OOM on small GPUs
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM,
                device_map="cpu",  # Use CPU instead of GPU
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(LLM)
            
            logger.info("LLM loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}", exc_info=True)
            raise
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("LLM loaded successfully")
    
    async def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_adaptive_rag: bool = True,
        max_chunks: Optional[int] = None,
        document_id: Optional[int] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Generate streaming response
        Yields chunks with type: 'sources', 'token', 'done', 'error'
        """
        try:
            # Lazy load LLM if not already loaded
            if self.model is None or self.tokenizer is None:
                logger.info("Lazy loading LLM on first chat request...")
                self._load_llm()
            
            # Retrieve relevant chunks
            sources = []
            if use_adaptive_rag:
                history_texts = []
                if conversation_history:
                    history_texts = [msg["content"] for msg in conversation_history[-3:]]
                
                sources = self.adaptive_rag.retrieve(
                    query,
                    conversation_history=history_texts,
                    top_k=max_chunks,
                    document_id=str(document_id) if document_id else None
                )
                
                # Yield sources first
                yield {
                    "type": "sources",
                    "sources": sources
                }
            
            # Build prompt
            prompt = self._build_prompt(query, sources, conversation_history)
            
            # Generate with streaming
            async for chunk in self._stream_generate(prompt):
                yield chunk
            
            # Done
            yield {"type": "done"}
        
        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def _build_prompt(
        self,
        query: str,
        sources: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build prompt with RAG context and conversation history"""
        
        # System prompt with clear instructions
        system = """Bạn là trợ lý AI thông minh hỗ trợ người dùng trả lời câu hỏi dựa trên tài liệu được cung cấp.

QUAN TRỌNG: 
- Bạn PHẢI sử dụng thông tin từ CONTEXT bên dưới để trả lời
- Nếu có CONTEXT, hãy dựa vào đó để trả lời câu hỏi
- KHÔNG nói rằng "chưa có tài liệu" nếu CONTEXT đã được cung cấp
- Trả lời dựa trên bằng chứng từ CONTEXT
- TRẢ LỜI BẰNG TIẾNG VIỆT - Dù câu hỏi bằng tiếng gì, luôn trả lời bằng tiếng Việt

Hãy tuân theo cấu trúc trả lời sau:

1. Trước tiên, hãy suy nghĩ và phân tích (bắt đầu bằng <reasoning>):
   - Xác định câu hỏi đang hỏi gì
   - Tìm kiếm thông tin liên quan trong CONTEXT
   - Phân tích và tổng hợp thông tin
   - Đánh giá độ tin cậy của thông tin
Kết thúc phần suy luận bằng </reasoning>

2. Sau đó, đưa ra câu trả lời cuối cùng (bắt đầu bằng <answer>):
   - Trả lời rõ ràng, chính xác BẰNG TIẾNG VIỆT
   - Dựa trên bằng chứng từ CONTEXT
   - Trích dẫn nguồn nếu cần
Kết thúc câu trả lời bằng </answer>
"""
        
        # Build context from sources
        context = ""
        if sources:
            context = "\n\nCONTEXT (Thông tin từ tài liệu):\n"
            context += "=" * 50 + "\n"
            for i, src in enumerate(sources[:6], 1):  # Limit to top 6
                context += f"\n[Đoạn {i}]:\n{src['text']}\n"
            context += "=" * 50 + "\n"
        
        # Build conversation history
        history = ""
        if conversation_history:
            history = "\n\nLịch sử hội thoại:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                history += f"{role}: {msg['content']}\n"
        
        # Build final prompt
        prompt = f"""{system}
{context}
{history}

Câu hỏi hiện tại: {query}

Hãy trả lời theo cấu trúc đã định (reasoning và answer):"""
        
        # Build messages for chat model
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n{history}\n\nCâu hỏi: {query}\n\nHãy trả lời theo cấu trúc (reasoning và answer):"}
        ]
        
        # Format for text generation
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return text
    
    async def _stream_generate(self, prompt: str) -> AsyncGenerator[Dict, None]:
        """Stream generate tokens, separating reasoning from answer"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": MAX_RESPONSE_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "streamer": streamer,
        }
        
        # Start generation in thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Stream tokens and parse reasoning vs answer
        full_text = ""
        in_reasoning = False
        in_answer = False
        current_section = ""
        
        for text in streamer:
            full_text += text
            current_section += text
            
            # Check for reasoning tags
            if "<reasoning>" in current_section and not in_reasoning:
                in_reasoning = True
                # Extract text before tag as potential answer start
                parts = current_section.split("<reasoning>")
                current_section = parts[1] if len(parts) > 1 else ""
                
            elif "</reasoning>" in current_section and in_reasoning:
                in_reasoning = False
                # Extract reasoning content without closing tag
                parts = current_section.split("</reasoning>")
                reasoning_content = parts[0].strip()
                if reasoning_content:
                    yield {
                        "type": "reasoning",
                        "content": reasoning_content
                    }
                current_section = parts[1] if len(parts) > 1 else ""
                
            elif "<answer>" in current_section and not in_answer:
                in_answer = True
                parts = current_section.split("<answer>")
                current_section = parts[1] if len(parts) > 1 else ""
                
            elif "</answer>" in current_section and in_answer:
                in_answer = False
                # Extract answer content without closing tag
                parts = current_section.split("</answer>")
                answer_content = parts[0].strip()
                if answer_content:
                    yield {
                        "type": "token",
                        "content": answer_content
                    }
                current_section = ""  # Clear section after answer
                
            # Stream content while in sections (but not the tags themselves)
            elif in_reasoning:
                # Check if we have accumulated enough for a chunk
                if len(current_section) > 20 and "</reasoning>" not in current_section:
                    # Don't yield yet, wait for complete reasoning
                    pass
                    
            elif in_answer:
                # Stream answer tokens as they come
                if text and not any(tag in text for tag in ["<answer>", "</answer>"]):
                    yield {
                        "type": "token",
                        "content": text
                    }
            else:
                # Before any tags, assume it's answer (fallback)
                yield {
                    "type": "token",
                    "content": text
                }
                current_section = ""
        
        thread.join()


# Singleton instance
_chat_service_instance: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create ChatService singleton"""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
