"""HuggingFace Inference API provider."""

import json
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
import logging

from ..core.providers import LLMProvider, ProviderError, ProviderAPIError, ProviderUnavailableError
from ..core.models import Message, ChatResponse, StreamingChatResponse

logger = logging.getLogger(__name__)


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.default_model = config.get("model", "mistralai/Mistral-7B-Instruct-v0.1")
        self.base_url = "https://api-inference.huggingface.co"
        
        if not self.api_key:
            logger.warning("HuggingFace API key not provided")
    
    async def is_available(self) -> bool:
        """Check if HuggingFace provider is available."""
        if self._available is not None:
            return self._available
        
        if not self.api_key:
            self._available = False
            return False
        
        try:
            # Test with a simple request to the model
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.default_model}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"inputs": "test"}
                )
                # HF returns 200 for valid requests or 503 if model is loading
                self._available = response.status_code in [200, 503]
                if not self._available:
                    logger.warning(f"HuggingFace check failed: {response.status_code}")
        except Exception as e:
            logger.debug(f"HuggingFace availability check failed: {e}")
            self._available = False
        
        return self._available
    
    def _prepare_prompt(self, messages: List[Message]) -> str:
        """Convert messages to HuggingFace prompt format."""
        # For most HF models, we need to format as a single prompt
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        # Add the final Assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using HuggingFace API."""
        if not await self.is_available():
            raise ProviderUnavailableError("HuggingFace provider is not available")
        
        if tools:
            logger.warning("Tool calling not yet implemented for HuggingFace provider")
        
        prompt = self._prepare_prompt(messages)
        used_model = self.get_model(model)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.get_temperature(temperature),
                "max_new_tokens": self.get_max_tokens(max_tokens),
                "return_full_text": False
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/models/{used_model}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                if response.status_code == 503:
                    # Model is loading, retry after a delay
                    await asyncio.sleep(5)
                    response = await client.post(
                        f"{self.base_url}/models/{used_model}",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload
                    )
                
                if response.status_code != 200:
                    raise ProviderAPIError(f"HuggingFace API error: {response.status_code} - {response.text}")
                
                data = response.json()
                
                # HF returns a list of results
                if isinstance(data, list) and data:
                    generated_text = data[0].get("generated_text", "")
                else:
                    generated_text = data.get("generated_text", "")
                
                return ChatResponse(
                    message=generated_text.strip(),
                    conversation_id="",  # Will be set by engine
                    model=used_model,
                    usage=None,  # HF doesn't return usage info by default
                    tool_calls=None,
                    finish_reason="stop"
                )
                
        except httpx.RequestError as e:
            raise ProviderAPIError(f"HuggingFace request failed: {e}")
        except Exception as e:
            raise ProviderError(f"HuggingFace error: {e}")
    
    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamingChatResponse]:
        """Generate streaming chat completion using HuggingFace API."""
        if not await self.is_available():
            raise ProviderUnavailableError("HuggingFace provider is not available")
        
        if tools:
            logger.warning("Tool calling not yet implemented for HuggingFace provider")
        
        # HuggingFace Inference API doesn't support streaming by default
        # We'll simulate streaming by breaking the response into chunks
        response = await self.chat_completion(
            messages, model, temperature, max_tokens, tools, **kwargs
        )
        
        # Split response into words and yield progressively
        words = response.message.split()
        accumulated = ""
        
        for i, word in enumerate(words):
            accumulated += word
            if i < len(words) - 1:
                accumulated += " "
            
            is_done = i == len(words) - 1
            
            yield StreamingChatResponse(
                delta=word + (" " if not is_done else ""),
                conversation_id="",
                done=is_done,
                finish_reason="stop" if is_done else None
            )
            
            # Add small delay to simulate streaming
            await asyncio.sleep(0.05)