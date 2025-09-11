"""Tests for SynergyX core functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from synergyx.config.manager import Config
from synergyx.core.models import Message, Conversation
from synergyx.core.engine import ChatEngine
from synergyx.memory.conversation import ConversationMemory


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = Config(load_env=False)
        assert config is not None
        assert config.get("model.name") is not None
    
    def test_config_get_with_default(self):
        """Test getting config values with defaults."""
        config = Config(load_env=False)
        
        # Existing key
        assert config.get("model.temperature", 0.5) == 0.7
        
        # Non-existing key with default
        assert config.get("non.existing.key", "default") == "default"
        
        # Non-existing key without default
        assert config.get("non.existing.key") is None


class TestMessage:
    """Test message model."""
    
    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
    
    def test_message_to_dict(self):
        """Test message serialization."""
        msg = Message(role="user", content="Hello")
        data = msg.to_dict()
        
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert "timestamp" in data
    
    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {}
        }
        
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestConversation:
    """Test conversation model."""
    
    def test_conversation_creation(self):
        """Test conversation creation."""
        conv = Conversation(id="test-123")
        assert conv.id == "test-123"
        assert len(conv.messages) == 0
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation(id="test-123")
        msg = Message(role="user", content="Hello")
        
        conv.add_message(msg)
        assert len(conv.messages) == 1
        assert conv.messages[0] == msg
    
    def test_trim_to_limit(self):
        """Test conversation trimming."""
        conv = Conversation(id="test-123")
        
        # Add system message
        system_msg = Message(role="system", content="System")
        conv.add_message(system_msg)
        
        # Add user messages
        for i in range(5):
            msg = Message(role="user", content=f"Message {i}")
            conv.add_message(msg)
        
        # Trim to 3 messages
        conv.trim_to_limit(3)
        
        # Should keep system message and last 2 user messages
        assert len(conv.messages) == 3
        assert conv.messages[0].role == "system"
        assert conv.messages[-1].content == "Message 4"


@pytest.mark.asyncio
class TestConversationMemory:
    """Test conversation memory."""
    
    async def test_memory_creation(self, tmp_path):
        """Test memory creation."""
        config = Config(load_env=False)
        config._config["memory"]["file_path"] = str(tmp_path / "test_conversations.jsonl")
        
        memory = ConversationMemory(config)
        assert memory is not None
        assert not memory._loaded
    
    async def test_save_and_get_conversation(self, tmp_path):
        """Test saving and retrieving conversations."""
        config = Config(load_env=False)
        config._config["memory"]["file_path"] = str(tmp_path / "test_conversations.jsonl")
        
        memory = ConversationMemory(config)
        
        # Create conversation
        conv = Conversation(id="test-123")
        conv.add_message(Message(role="user", content="Hello"))
        
        # Save conversation
        await memory.save_conversation(conv)
        
        # Retrieve conversation
        retrieved = await memory.get_conversation("test-123")
        assert retrieved is not None
        assert retrieved.id == "test-123"
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Hello"
    
    async def test_list_conversations(self, tmp_path):
        """Test listing conversations."""
        config = Config(load_env=False)
        config._config["memory"]["file_path"] = str(tmp_path / "test_conversations.jsonl")
        
        memory = ConversationMemory(config)
        
        # Create and save multiple conversations
        for i in range(3):
            conv = Conversation(id=f"test-{i}")
            await memory.save_conversation(conv)
        
        # List conversations
        conversations = await memory.list_conversations()
        assert len(conversations) == 3
        assert "test-0" in conversations
        assert "test-1" in conversations
        assert "test-2" in conversations


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for tests."""
    return tmp_path_factory.mktemp("synergyx_tests")


if __name__ == "__main__":
    pytest.main([__file__])