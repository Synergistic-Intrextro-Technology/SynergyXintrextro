"""Tests for Azure and HuggingFace providers."""

import pytest
import os
from unittest.mock import patch, AsyncMock

from synergyx.core.azure_provider import AzureProvider
from synergyx.core.huggingface_provider import HuggingFaceProvider
from synergyx.core.models import Message
from synergyx.config.manager import Config
from synergyx.core.manager import ProviderManager


class TestAzureProvider:
    """Test Azure OpenAI provider."""
    
    def test_azure_provider_init(self):
        """Test Azure provider initialization."""
        config = {
            "api_key": "test_key",
            "endpoint": "https://test.openai.azure.com/",
            "deployment": "gpt-35-turbo",
            "api_version": "2024-02-15-preview"
        }
        provider = AzureProvider(config)
        
        assert provider.api_key == "test_key"
        assert provider.endpoint == "https://test.openai.azure.com/"
        assert provider.deployment == "gpt-35-turbo"
        assert provider.api_version == "2024-02-15-preview"
    
    def test_azure_provider_missing_config(self):
        """Test Azure provider with missing configuration."""
        config = {}
        provider = AzureProvider(config)
        
        assert provider.api_key is None
        assert provider.endpoint is None
        assert provider.deployment is None
    
    @pytest.mark.asyncio
    async def test_azure_provider_availability_missing_config(self):
        """Test Azure provider availability with missing config."""
        config = {}
        provider = AzureProvider(config)
        
        available = await provider.is_available()
        assert not available
    
    def test_azure_provider_prepare_messages(self):
        """Test Azure provider message preparation."""
        config = {
            "api_key": "test_key",
            "endpoint": "https://test.openai.azure.com/",
            "deployment": "gpt-35-turbo"
        }
        provider = AzureProvider(config)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
        
        openai_messages = provider._prepare_messages(messages)
        
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "user"
        assert openai_messages[0]["content"] == "Hello"
        assert openai_messages[1]["role"] == "assistant"
        assert openai_messages[1]["content"] == "Hi there!"


class TestHuggingFaceProvider:
    """Test HuggingFace provider."""
    
    def test_huggingface_provider_init(self):
        """Test HuggingFace provider initialization."""
        config = {
            "api_key": "test_key",
            "model": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        provider = HuggingFaceProvider(config)
        
        assert provider.api_key == "test_key"
        assert provider.default_model == "mistralai/Mistral-7B-Instruct-v0.1"
        assert provider.base_url == "https://api-inference.huggingface.co"
    
    def test_huggingface_provider_missing_config(self):
        """Test HuggingFace provider with missing configuration."""
        config = {}
        provider = HuggingFaceProvider(config)
        
        assert provider.api_key is None
        assert provider.default_model == "mistralai/Mistral-7B-Instruct-v0.1"  # Default model
    
    @pytest.mark.asyncio
    async def test_huggingface_provider_availability_missing_config(self):
        """Test HuggingFace provider availability with missing config."""
        config = {}
        provider = HuggingFaceProvider(config)
        
        available = await provider.is_available()
        assert not available
    
    def test_huggingface_provider_prepare_prompt(self):
        """Test HuggingFace provider prompt preparation."""
        config = {
            "api_key": "test_key",
            "model": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        provider = HuggingFaceProvider(config)
        
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?")
        ]
        
        prompt = provider._prepare_prompt(messages)
        
        expected_lines = [
            "System: You are a helpful assistant.",
            "User: Hello",
            "Assistant: Hi there!",
            "User: How are you?",
            "Assistant:"
        ]
        
        assert prompt == "\n".join(expected_lines)


class TestProviderManagerIntegration:
    """Test provider manager integration with new providers."""
    
    @pytest.mark.asyncio
    async def test_provider_manager_registers_all_providers(self):
        """Test that provider manager registers all available providers."""
        # Mock environment variables
        env_vars = {
            'OPENAI_API_KEY': 'test_openai',
            'AZURE_OPENAI_API_KEY': 'test_azure',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT': 'gpt-35-turbo',
            'HUGGINGFACE_API_KEY': 'test_hf'
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            manager = ProviderManager(config)
            
            # Should have all 4 providers registered
            expected_providers = {'openai', 'azure', 'huggingface', 'ollama'}
            assert set(manager.providers.keys()) == expected_providers
    
    @pytest.mark.asyncio 
    async def test_provider_manager_only_configured_providers(self):
        """Test that only configured providers are registered."""
        # Only set OpenAI environment variable
        env_vars = {
            'OPENAI_API_KEY': 'test_openai'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            manager = ProviderManager(config)
            
            # Should have OpenAI and Ollama (Ollama is always initialized)
            expected_providers = {'openai', 'ollama'}
            assert set(manager.providers.keys()) == expected_providers
            assert 'azure' not in manager.providers
            assert 'huggingface' not in manager.providers