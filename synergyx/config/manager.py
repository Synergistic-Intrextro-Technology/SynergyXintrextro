"""Configuration management for SynergyX."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager that loads from YAML and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None, load_env: bool = True):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. Defaults to config.yaml in project root.
            load_env: Whether to load .env file
        """
        self._config: Dict[str, Any] = {}
        
        # Load .env file if requested
        if load_env:
            load_dotenv()
            
        # Load YAML config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self._load_yaml_config(config_path)
        
    def _load_yaml_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when YAML file is not available."""
        return {
            "model": {
                "name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "streaming": True,
                "provider_priority": ["ollama", "openai", "azure", "huggingface"]
            },
            "memory": {
                "max_messages": 100,
                "file_path": "./data/conversations.jsonl",
                "auto_save": True
            },
            "tools": {
                "enabled": ["text_analysis", "code_analysis", "data_analysis", "rag", "web_fetcher"]
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "title": "SynergyX API",
                "description": "Advanced AI Chatbot and Analysis System",
                "version": "0.1.0"
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.temperature')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable."""
        return os.getenv(key, default)
    
    def get_provider_config(self) -> Dict[str, Any]:
        """Get provider configuration from environment variables."""
        providers = {}
        
        # OpenAI-compatible
        if self.get_env("OPENAI_API_KEY"):
            providers["openai"] = {
                "api_key": self.get_env("OPENAI_API_KEY"),
                "base_url": self.get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "model": self.get_env("OPENAI_MODEL", self.get("model.name", "gpt-3.5-turbo"))
            }
        
        # Azure OpenAI
        if self.get_env("AZURE_OPENAI_API_KEY"):
            providers["azure"] = {
                "api_key": self.get_env("AZURE_OPENAI_API_KEY"),
                "endpoint": self.get_env("AZURE_OPENAI_ENDPOINT"),
                "deployment": self.get_env("AZURE_OPENAI_DEPLOYMENT"),
                "api_version": self.get_env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            }
        
        # Ollama
        ollama_url = self.get_env("OLLAMA_BASE_URL", "http://localhost:11434")
        providers["ollama"] = {
            "base_url": ollama_url,
            "model": self.get_env("OLLAMA_MODEL", "llama2")
        }
        
        # HuggingFace
        if self.get_env("HUGGINGFACE_API_KEY"):
            providers["huggingface"] = {
                "api_key": self.get_env("HUGGINGFACE_API_KEY"),
                "model": self.get_env("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
            }
        
        return providers
    
    def get_data_dir(self) -> Path:
        """Get data directory path."""
        data_dir = Path(self.get_env("SYNERGYX_DATA_DIR", "./data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_reports_dir(self) -> Path:
        """Get reports directory path."""
        reports_dir = Path(self.get_env("SYNERGYX_REPORTS_DIR", "./reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        logs_dir = Path("./logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def setup_logging(config: Optional[Config] = None) -> None:
    """Setup logging based on configuration."""
    if config is None:
        config = get_config()
    
    log_level = config.get_env("SYNERGYX_LOG_LEVEL", config.get("logging.level", "INFO"))
    log_format = config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # Setup file logging if configured
    log_file = config.get("logging.file_path")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)