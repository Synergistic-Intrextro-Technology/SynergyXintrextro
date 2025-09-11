"""OpenAI-compatible API provider."""

import json
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
import logging

from ..core.providers import LLMProvider, ProviderError, ProviderAPIError, ProviderUnavailableError
from ..core.models import Message, ChatResponse, StreamingChatResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.default_model = config.get("model", "gpt-3.5-turbo")
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
    
    async def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        if self._available is not None:
            return self._available
        
        if not self.api_key:
            self._available = False
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                self._available = response.status_code == 200
                if not self._available:
                    logger.warning(f"OpenAI API check failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            self._available = False
        
        return self._available
    
    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            openai_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
                
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using OpenAI API."""
        if not await self.is_available():
            raise ProviderUnavailableError("OpenAI provider is not available")
        
        payload = {
            "model": self.get_model(model),
            "messages": self._prepare_messages(messages),
            "temperature": self.get_temperature(temperature),
            "max_tokens": self.get_max_tokens(max_tokens)
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                if response.status_code != 200:
                    raise ProviderAPIError(f"OpenAI API error: {response.status_code} - {response.text}")
                
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                return ChatResponse(
                    message=message.get("content", ""),
                    conversation_id="",  # Will be set by engine
                    model=data["model"],
                    usage=data.get("usage"),
                    tool_calls=message.get("tool_calls"),
                    finish_reason=choice.get("finish_reason")
                )
        
        except httpx.RequestError as e:
            raise ProviderAPIError(f"OpenAI request failed: {e}")
        except Exception as e:
            raise ProviderError(f"OpenAI provider error: {e}")
    
    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate streaming chat completion using OpenAI API."""
        if not await self.is_available():
            raise ProviderUnavailableError("OpenAI provider is not available")
        
        payload = {
            "model": self.get_model(model),
            "messages": self._prepare_messages(messages),
            "temperature": self.get_temperature(temperature),
            "max_tokens": self.get_max_tokens(max_tokens),
            "stream": True
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        raise ProviderAPIError(f"OpenAI API error: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str.strip() == "[DONE]":
                                yield StreamingChatResponse(
                                    delta="",
                                    conversation_id="",
                                    done=True
                                )
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choice = data["choices"][0]
                                delta = choice.get("delta", {})
                                
                                content = delta.get("content", "")
                                finish_reason = choice.get("finish_reason")
                                tool_calls = delta.get("tool_calls")
                                
                                yield StreamingChatResponse(
                                    delta=content,
                                    conversation_id="",
                                    done=finish_reason is not None,
                                    tool_calls=tool_calls,
                                    finish_reason=finish_reason
                                )
                                
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
        
        except httpx.RequestError as e:
            raise ProviderAPIError(f"OpenAI streaming request failed: {e}")
        except Exception as e:
            raise ProviderError(f"OpenAI streaming error: {e}")