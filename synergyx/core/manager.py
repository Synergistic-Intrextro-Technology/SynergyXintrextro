"""Provider manager for handling multiple LLM providers."""

import logging
from typing import Any, Dict, List, Optional, Type
from ..config.manager import Config
from ..core.providers import LLMProvider, ProviderUnavailableError
from ..core.openai_provider import OpenAIProvider
from ..core.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class ProviderManager:
    """Manages multiple LLM providers and handles fallbacks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self._active_provider: Optional[str] = None
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all available providers."""
        provider_configs = self.config.get_provider_config()
        
        # Register provider classes
        provider_classes: Dict[str, Type[LLMProvider]] = {
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
            # TODO: Add Azure and HuggingFace providers
        }
        
        # Initialize providers that have configuration
        for name, provider_class in provider_classes.items():
            if name in provider_configs:
                try:
                    provider = provider_class(provider_configs[name])
                    self.providers[name] = provider
                    logger.info(f"Initialized {name} provider")
                except Exception as e:
                    logger.error(f"Failed to initialize {name} provider: {e}")
            else:
                # For Ollama, try to initialize even without explicit config
                if name == "ollama":
                    try:
                        provider = provider_class(provider_configs.get(name, {}))
                        self.providers[name] = provider
                        logger.info(f"Initialized {name} provider with defaults")
                    except Exception as e:
                        logger.debug(f"Failed to initialize default {name} provider: {e}")
    
    async def get_active_provider(self) -> LLMProvider:
        """Get the active provider, using fallback logic if needed."""
        if self._active_provider and self._active_provider in self.providers:
            provider = self.providers[self._active_provider]
            if await provider.is_available():
                return provider
        
        # Try providers in priority order
        priority = self.config.get("model.provider_priority", ["ollama", "openai", "azure", "huggingface"])
        
        for provider_name in priority:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                try:
                    if await provider.is_available():
                        self._active_provider = provider_name
                        logger.info(f"Selected {provider_name} as active provider")
                        return provider
                except Exception as e:
                    logger.warning(f"Provider {provider_name} check failed: {e}")
                    continue
        
        # No providers available
        raise ProviderUnavailableError("No LLM providers are available")
    
    async def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Get a specific provider by name."""
        if name not in self.providers:
            return None
        
        provider = self.providers[name]
        if await provider.is_available():
            return provider
        
        return None
    
    def list_providers(self) -> Dict[str, bool]:
        """List all providers and their availability status."""
        return {name: provider._available for name, provider in self.providers.items()}
    
    async def check_all_providers(self) -> Dict[str, bool]:
        """Check availability of all providers."""
        status = {}
        for name, provider in self.providers.items():
            try:
                status[name] = await provider.is_available()
            except Exception as e:
                logger.error(f"Error checking provider {name}: {e}")
                status[name] = False
        
        return status