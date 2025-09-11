"""Main chatbot engine that orchestrates conversations."""

import uuid
import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from datetime import datetime

from ..config.manager import Config
from ..core.models import Message, Conversation, ChatRequest, ChatResponse, StreamingChatResponse
from ..core.manager import ProviderManager
from ..memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)


class ChatEngine:
    """Main chatbot engine that handles conversations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.provider_manager = ProviderManager(config)
        self.memory = ConversationMemory(config)
        self.system_message = self._get_system_message()
    
    def _get_system_message(self) -> str:
        """Get the default system message."""
        return """You are SynergyX, an advanced AI assistant with access to powerful analysis tools.
        
You can help with:
- Text analysis (summarization, sentiment, keywords)
- Code analysis (metrics, complexity, linting)
- Data analysis (CSV/JSON processing, statistics)
- Web content fetching and analysis
- Document retrieval and question answering

When appropriate, you can call tools to provide more detailed analysis. 
Always be helpful, accurate, and explain your reasoning."""
    
    async def chat(
        self,
        request: ChatRequest,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> ChatResponse:
        """Process a chat request and return a response."""
        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        conversation = await self.memory.get_conversation(conversation_id)
        
        if not conversation:
            # Create new conversation with system message
            conversation = Conversation(
                id=conversation_id,
                messages=[Message(role="system", content=self.system_message)]
            )
        
        # Add user message
        user_message = Message(role="user", content=request.message)
        conversation.add_message(user_message)
        
        # Get active provider
        provider = await self.provider_manager.get_active_provider()
        
        # Generate response
        response = await provider.chat_completion(
            messages=conversation.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=tools
        )
        
        # Update response with conversation ID
        response.conversation_id = conversation_id
        
        # Add assistant message to conversation
        assistant_message = Message(
            role="assistant",
            content=response.message,
            tool_calls=response.tool_calls
        )
        conversation.add_message(assistant_message)
        
        # Trim conversation if needed
        max_messages = self.config.get("memory.max_messages", 100)
        conversation.trim_to_limit(max_messages)
        
        # Save conversation
        await self.memory.save_conversation(conversation)
        
        return response
    
    async def stream_chat(
        self,
        request: ChatRequest,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[StreamingChatResponse]:
        """Process a streaming chat request."""
        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        conversation = await self.memory.get_conversation(conversation_id)
        
        if not conversation:
            conversation = Conversation(
                id=conversation_id,
                messages=[Message(role="system", content=self.system_message)]
            )
        
        # Add user message
        user_message = Message(role="user", content=request.message)
        conversation.add_message(user_message)
        
        # Get active provider
        provider = await self.provider_manager.get_active_provider()
        
        # Stream response
        full_response = ""
        tool_calls = None
        
        async for chunk in provider.stream_completion(
            messages=conversation.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=tools
        ):
            # Update chunk with conversation ID
            chunk.conversation_id = conversation_id
            
            # Accumulate response
            full_response += chunk.delta
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            
            yield chunk
            
            # If done, save conversation
            if chunk.done:
                assistant_message = Message(
                    role="assistant",
                    content=full_response,
                    tool_calls=tool_calls
                )
                conversation.add_message(assistant_message)
                
                # Trim and save
                max_messages = self.config.get("memory.max_messages", 100)
                conversation.trim_to_limit(max_messages)
                await self.memory.save_conversation(conversation)
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return await self.memory.get_conversation(conversation_id)
    
    async def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return await self.memory.list_conversations()
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return await self.memory.delete_conversation(conversation_id)
    
    async def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers."""
        return await self.provider_manager.check_all_providers()
    
    def set_system_message(self, message: str) -> None:
        """Set the system message for new conversations."""
        self.system_message = message