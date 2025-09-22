"""Azure OpenAI API provider."""

import json
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
import logging

from ..core.providers import LLMProvider, ProviderError, ProviderAPIError, ProviderUnavailableError
from ..core.models import Message, ChatResponse, StreamingChatResponse

logger = logging.getLogger(__name__)


class AzureProvider(LLMProvider):
    """Azure OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint")
        self.deployment = config.get("deployment")
        self.api_version = config.get("api_version", "2024-02-15-preview")
        self.default_model = config.get("model", self.deployment or "gpt-35-turbo")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided")
        if not self.endpoint:
            logger.warning("Azure OpenAI endpoint not provided")
        if not self.deployment:
            logger.warning("Azure OpenAI deployment not provided")
    
    async def is_available(self) -> bool:
        """Check if Azure OpenAI provider is available."""
        if self._available is not None:
            return self._available
        
        if not self.api_key or not self.endpoint or not self.deployment:
            self._available = False
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions",
                    headers={
                        "api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    params={"api-version": self.api_version}
                )
                # Azure returns 405 for GET on this endpoint, but that means it's accessible
                self._available = response.status_code in [200, 405]
                if not self._available:
                    logger.warning(f"Azure OpenAI check failed: {response.status_code}")
        except Exception as e:
            logger.debug(f"Azure OpenAI availability check failed: {e}")
            self._available = False
        
        return self._available
    
    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Azure OpenAI format."""
        openai_messages = []
        for message in messages:
            openai_messages.append({
                "role": message.role,
                "content": message.content
            })
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
        """Generate chat completion using Azure OpenAI API."""
        if not await self.is_available():
            raise ProviderUnavailableError("Azure OpenAI provider is not available")
        
        payload = {
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
                    f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions",
                    headers={
                        "api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    params={"api-version": self.api_version},
                    json=payload
                )
                
                if response.status_code != 200:
                    raise ProviderAPIError(f"Azure OpenAI API error: {response.status_code} - {response.text}")
                
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                return ChatResponse(
                    message=message.get("content", ""),
                    conversation_id="",  # Will be set by engine
                    model=data.get("model", self.default_model),
                    usage=data.get("usage"),
                    tool_calls=message.get("tool_calls"),
                    finish_reason=choice.get("finish_reason")
                )
                
        except httpx.RequestError as e:
            raise ProviderAPIError(f"Azure OpenAI request failed: {e}")
        except Exception as e:
            raise ProviderError(f"Azure OpenAI error: {e}")
    
    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate streaming chat completion using Azure OpenAI API."""
        if not await self.is_available():
            raise ProviderUnavailableError("Azure OpenAI provider is not available")
        
        if tools:
            logger.warning("Tool calling in streaming mode may have limited support")
        
        payload = {
            "messages": self._prepare_messages(messages),
            "temperature": self.get_temperature(temperature),
            "max_tokens": self.get_max_tokens(max_tokens),
            "stream": True
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions",
                    headers={
                        "api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    params={"api-version": self.api_version},
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        raise ProviderAPIError(f"Azure OpenAI API error: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line and line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    choice = choices[0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    done = choice.get("finish_reason") is not None
                                    
                                    yield StreamingChatResponse(
                                        delta=content,
                                        conversation_id="",
                                        done=done,
                                        finish_reason=choice.get("finish_reason")
                                    )
                                    
                                    if done:
                                        break
                                        
                            except json.JSONDecodeError:
                                continue
        
        except httpx.RequestError as e:
            raise ProviderAPIError(f"Azure OpenAI streaming request failed: {e}")
        except Exception as e:
            raise ProviderError(f"Azure OpenAI streaming error: {e}")