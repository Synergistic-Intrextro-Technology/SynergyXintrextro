"""Base classes and registry for analysis tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import logging

logger = logging.getLogger(__name__)


class AnalysisTool(ABC):
    """Base class for analysis tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema (JSON Schema format)."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    def to_function_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """Registry for managing analysis tools."""
    
    def __init__(self):
        self._tools: Dict[str, AnalysisTool] = {}
    
    def register(self, tool: AnalysisTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[AnalysisTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI function schemas for all tools."""
        return [tool.to_function_schema() for tool in self._tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        try:
            result = await tool.execute(**kwargs)
            return {
                "tool": name,
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {
                "tool": name,
                "success": False,
                "error": str(e)
            }


# Global tool registry
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry