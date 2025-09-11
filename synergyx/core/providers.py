"""Abstract base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from ..core.models import Message, ChatResponse, StreamingChatResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config
        self._available = None
    
    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__.lower().replace("provider", "")
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate streaming chat completion."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available and properly configured."""
        pass
    
    def get_model(self, model: Optional[str] = None) -> str:
        """Get model name, using default if not specified."""
        return model or self.config.get("model", "default")
    
    def get_temperature(self, temperature: Optional[float] = None) -> float:
        """Get temperature, using default if not specified."""
        return temperature if temperature is not None else self.config.get("temperature", 0.7)
    
    def get_max_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Get max tokens, using default if not specified."""
        return max_tokens if max_tokens is not None else self.config.get("max_tokens", 1000)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderUnavailableError(ProviderError):
    """Exception raised when provider is not available."""
    pass


class ProviderConfigError(ProviderError):
    """Exception raised when provider configuration is invalid."""
    pass


class ProviderAPIError(ProviderError):
    """Exception raised when provider API returns an error."""
    pass