"""Message types and core conversation models."""

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class Message(BaseModel):
    """A conversation message."""
    
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(description="The message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        data = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
            
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id")
        )


class Conversation(BaseModel):
    """A conversation containing multiple messages."""
    
    messages: List[Message] = Field(default_factory=list)
    id: str = Field(description="Unique conversation ID")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Get messages in format suitable for LLM APIs."""
        llm_messages = []
        
        for msg in self.messages:
            llm_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            if msg.tool_calls:
                llm_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                llm_msg["tool_call_id"] = msg.tool_call_id
                
            llm_messages.append(llm_msg)
        
        return llm_messages
    
    def trim_to_limit(self, max_messages: int) -> None:
        """Trim conversation to maximum number of messages, keeping system messages."""
        if len(self.messages) <= max_messages:
            return
        
        # Separate system and non-system messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        other_messages = [msg for msg in self.messages if msg.role != "system"]
        
        # Keep all system messages and trim others
        max_other = max(0, max_messages - len(system_messages))
        if max_other > 0:
            other_messages = other_messages[-max_other:]
        else:
            other_messages = []
        
        # Combine back, keeping system messages at the beginning
        self.messages = system_messages + other_messages


class ChatRequest(BaseModel):
    """Request for chat completion."""
    
    message: str = Field(description="User message")
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    
    
class ChatResponse(BaseModel):
    """Response from chat completion."""
    
    message: str = Field(description="Assistant response")
    conversation_id: str = Field(description="Conversation ID")
    model: str = Field(description="Model used")
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None


class StreamingChatResponse(BaseModel):
    """Streaming response chunk from chat completion."""
    
    delta: str = Field(description="Response delta")
    conversation_id: str = Field(description="Conversation ID")
    done: bool = False
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None