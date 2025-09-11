"""Conversation memory management using JSONL files."""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..config.manager import Config
from ..core.models import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation persistence using JSONL files."""
    
    def __init__(self, config: Config):
        self.config = config
        self.file_path = Path(config.get("memory.file_path", "./data/conversations.jsonl"))
        self.auto_save = config.get("memory.auto_save", True)
        self._conversations: Dict[str, Conversation] = {}
        self._lock = asyncio.Lock()
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing conversations
        asyncio.create_task(self._load_conversations())
    
    async def _load_conversations(self) -> None:
        """Load conversations from JSONL file."""
        if not self.file_path.exists():
            return
        
        try:
            async with aiofiles.open(self.file_path, 'r') as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        conversation = self._dict_to_conversation(data)
                        self._conversations[conversation.id] = conversation
            
            logger.info(f"Loaded {len(self._conversations)} conversations from {self.file_path}")
        
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
    
    async def _save_conversations(self) -> None:
        """Save all conversations to JSONL file."""
        try:
            # Write to temporary file first
            temp_path = self.file_path.with_suffix('.tmp')
            
            async with aiofiles.open(temp_path, 'w') as f:
                for conversation in self._conversations.values():
                    data = self._conversation_to_dict(conversation)
                    await f.write(json.dumps(data) + '\n')
            
            # Atomic replace
            temp_path.replace(self.file_path)
            
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def _conversation_to_dict(self, conversation: Conversation) -> Dict:
        """Convert conversation to dictionary for JSON serialization."""
        return {
            "id": conversation.id,
            "created_at": conversation.created_at.isoformat(),
            "metadata": conversation.metadata,
            "messages": [msg.to_dict() for msg in conversation.messages]
        }
    
    def _dict_to_conversation(self, data: Dict) -> Conversation:
        """Convert dictionary to conversation object."""
        messages = [Message.from_dict(msg_data) for msg_data in data.get("messages", [])]
        
        return Conversation(
            id=data["id"],
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
            messages=messages
        )
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        async with self._lock:
            return self._conversations.get(conversation_id)
    
    async def save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation."""
        async with self._lock:
            self._conversations[conversation.id] = conversation
            
            if self.auto_save:
                await self._save_conversations()
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        async with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                
                if self.auto_save:
                    await self._save_conversations()
                
                return True
            return False
    
    async def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        async with self._lock:
            return list(self._conversations.keys())
    
    async def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        """Get a summary of a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        user_messages = [msg for msg in conversation.messages if msg.role == "user"]
        assistant_messages = [msg for msg in conversation.messages if msg.role == "assistant"]
        
        return {
            "id": conversation.id,
            "created_at": conversation.created_at.isoformat(),
            "message_count": len(conversation.messages),
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "last_message": conversation.messages[-1].content[:100] + "..." if conversation.messages else None,
            "metadata": conversation.metadata
        }
    
    async def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """Search conversations by content."""
        results = []
        query_lower = query.lower()
        
        async with self._lock:
            for conversation in self._conversations.values():
                # Simple text search in message content
                for message in conversation.messages:
                    if query_lower in message.content.lower():
                        summary = await self.get_conversation_summary(conversation.id)
                        if summary and summary not in results:
                            results.append(summary)
                            break
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def cleanup_old_conversations(self, max_age_days: int = 30) -> int:
        """Remove conversations older than specified days."""
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        async with self._lock:
            to_remove = []
            for conv_id, conversation in self._conversations.items():
                if conversation.created_at < cutoff:
                    to_remove.append(conv_id)
            
            for conv_id in to_remove:
                del self._conversations[conv_id]
                removed_count += 1
            
            if removed_count > 0 and self.auto_save:
                await self._save_conversations()
        
        logger.info(f"Cleaned up {removed_count} old conversations")
        return removed_count