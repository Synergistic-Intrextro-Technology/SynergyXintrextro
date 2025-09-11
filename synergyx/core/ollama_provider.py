"""Ollama local provider."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
import logging

from ..core.providers import LLMProvider, ProviderError, ProviderAPIError, ProviderUnavailableError
from ..core.models import Message, ChatResponse, StreamingChatResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.default_model = config.get("model", "llama2")
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                self._available = response.status_code == 200
                if not self._available:
                    logger.warning(f"Ollama check failed: {response.status_code}")
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            self._available = False
        
        return self._available
    
    def _prepare_messages(self, messages: List[Message]) -> str:
        """Convert messages to Ollama prompt format."""
        # For Ollama, we'll concatenate messages into a single prompt
        # This is a simplified approach - could be enhanced for better conversation handling
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using Ollama API."""
        if not await self.is_available():
            raise ProviderUnavailableError("Ollama provider is not available")
        
        # Ollama doesn't support tools in the same way, so we'll ignore them for now
        if tools:
            logger.warning("Tool calling not yet implemented for Ollama provider")
        
        prompt = self._prepare_messages(messages)
        
        payload = {
            "model": self.get_model(model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.get_temperature(temperature),
                "num_predict": self.get_max_tokens(max_tokens)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    raise ProviderAPIError(f"Ollama API error: {response.status_code} - {response.text}")
                
                data = response.json()
                
                return ChatResponse(
                    message=data.get("response", ""),
                    conversation_id="",
                    model=data.get("model", self.get_model(model)),
                    finish_reason="stop" if data.get("done", False) else None
                )
        
        except httpx.RequestError as e:
            raise ProviderAPIError(f"Ollama request failed: {e}")
        except Exception as e:
            raise ProviderError(f"Ollama provider error: {e}")
    
    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate streaming chat completion using Ollama API."""
        if not await self.is_available():
            raise ProviderUnavailableError("Ollama provider is not available")
        
        if tools:
            logger.warning("Tool calling not yet implemented for Ollama provider")
        
        prompt = self._prepare_messages(messages)
        
        payload = {
            "model": self.get_model(model),
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.get_temperature(temperature),
                "num_predict": self.get_max_tokens(max_tokens)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        raise ProviderAPIError(f"Ollama API error: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                content = data.get("response", "")
                                done = data.get("done", False)
                                
                                yield StreamingChatResponse(
                                    delta=content,
                                    conversation_id="",
                                    done=done,
                                    finish_reason="stop" if done else None
                                )
                                
                                if done:
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
        
        except httpx.RequestError as e:
            raise ProviderAPIError(f"Ollama streaming request failed: {e}")
        except Exception as e:
            raise ProviderError(f"Ollama streaming error: {e}")