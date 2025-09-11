"""Initialize and register all analysis tools."""

import logging
from typing import List

from .base import AnalysisTool, get_registry
from .text_analysis import TextSummarizerTool, SentimentAnalysisTool, KeywordExtractorTool
from .code_analysis import PythonASTAnalyzer, CodeLinterTool
from .data_analysis import DataAnalyzerTool, DataExplainerTool, CSVUploaderTool
from .web_fetcher import WebFetcherTool
from .rag_tools import RAGSearchTool, RAGAnswerTool, RAGIndexTool

logger = logging.getLogger(__name__)


def register_all_tools() -> List[str]:
    """Register all available tools and return list of registered tool names."""
    registry = get_registry()
    
    tools = [
        # Text analysis tools
        TextSummarizerTool(),
        SentimentAnalysisTool(),
        KeywordExtractorTool(),
        
        # Code analysis tools
        PythonASTAnalyzer(),
        CodeLinterTool(),
        
        # Data analysis tools
        DataAnalyzerTool(),
        DataExplainerTool(),
        CSVUploaderTool(),
        
        # Web tools
        WebFetcherTool(),
        
        # RAG tools
        RAGSearchTool(),
        RAGAnswerTool(),
        RAGIndexTool(),
    ]
    
    registered_names = []
    for tool in tools:
        try:
            registry.register(tool)
            registered_names.append(tool.name)
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
    
    logger.info(f"Registered {len(registered_names)} tools: {', '.join(registered_names)}")
    return registered_names


def get_available_tools() -> List[AnalysisTool]:
    """Get list of all available tools."""
    registry = get_registry()
    return list(registry._tools.values())


def get_tool_schemas() -> List[dict]:
    """Get OpenAI function schemas for all tools."""
    registry = get_registry()
    return registry.get_function_schemas()


# Auto-register tools when module is imported
register_all_tools()