"""Tests for analysis tools."""

import pytest
from unittest.mock import Mock, patch

from synergyx.tools.base import AnalysisTool, ToolRegistry
from synergyx.tools.text_analysis import TextSummarizerTool, SentimentAnalysisTool, KeywordExtractorTool
from synergyx.tools.code_analysis import PythonASTAnalyzer
from synergyx.tools.data_analysis import DataAnalyzerTool


class MockTool(AnalysisTool):
    """Mock tool for testing."""
    
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "A mock tool for testing"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"}
            },
            "required": ["input"]
        }
    
    async def execute(self, input: str, **kwargs) -> dict:
        return {"result": f"processed: {input}"}


class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_registry_creation(self):
        """Test registry creation."""
        registry = ToolRegistry()
        assert len(registry.list_tools()) == 0
    
    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        assert len(registry.list_tools()) == 1
        assert "mock_tool" in registry.list_tools()
    
    def test_get_tool(self):
        """Test getting registered tool."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        retrieved = registry.get_tool("mock_tool")
        
        assert retrieved is not None
        assert retrieved.name == "mock_tool"
    
    def test_function_schemas(self):
        """Test getting function schemas."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        schemas = registry.get_function_schemas()
        
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_tool"
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test tool execution."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        result = await registry.execute_tool("mock_tool", input="test")
        
        assert result["success"] is True
        assert result["tool"] == "mock_tool"
        assert result["result"]["result"] == "processed: test"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool."""
        registry = ToolRegistry()
        
        with pytest.raises(ValueError, match="Tool not found"):
            await registry.execute_tool("nonexistent_tool")


@pytest.mark.asyncio
class TestTextAnalysisTools:
    """Test text analysis tools."""
    
    async def test_text_summarizer(self):
        """Test text summarization."""
        tool = TextSummarizerTool()
        
        text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence."
        result = await tool.execute(text=text, max_sentences=2)
        
        assert "summary" in result
        assert result["method"] == "extractive"
        assert result["summary_sentences"] <= 2
    
    async def test_sentiment_analyzer(self):
        """Test sentiment analysis."""
        tool = SentimentAnalysisTool()
        
        # Test positive sentiment
        result = await tool.execute(text="I love this product! It's amazing and wonderful!")
        assert result["sentiment"] == "positive"
        assert result["score"] > 0
        
        # Test negative sentiment
        result = await tool.execute(text="I hate this terrible product! It's awful!")
        assert result["sentiment"] == "negative"
        assert result["score"] < 0
        
        # Test neutral sentiment
        result = await tool.execute(text="The product exists.")
        assert result["sentiment"] == "neutral"
    
    async def test_keyword_extractor(self):
        """Test keyword extraction."""
        tool = KeywordExtractorTool()
        
        text = "Python programming language is great for data science and machine learning applications."
        result = await tool.execute(text=text, max_keywords=5)
        
        assert "keywords" in result
        assert len(result["keywords"]) <= 5
        assert result["method"] == "tf_based"
        
        # Check that keywords have required fields
        if result["keywords"]:
            keyword = result["keywords"][0]
            assert "keyword" in keyword
            assert "frequency" in keyword
            assert "type" in keyword


@pytest.mark.asyncio
class TestCodeAnalysisTools:
    """Test code analysis tools."""
    
    async def test_python_ast_analyzer(self):
        """Test Python AST analysis."""
        tool = PythonASTAnalyzer()
        
        code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class TestClass:
    def method(self):
        pass
'''
        
        result = await tool.execute(code=code)
        
        assert result["valid_syntax"] is True
        assert result["function_count"] == 1
        assert result["class_count"] == 1
        assert len(result["functions"]) == 1
        assert len(result["classes"]) == 1
        
        # Check function details
        func = result["functions"][0]
        assert func["name"] == "hello_world"
        assert func["docstring"] == "Print hello world."
        
        # Check class details
        cls = result["classes"][0]
        assert cls["name"] == "TestClass"
    
    async def test_python_ast_analyzer_syntax_error(self):
        """Test AST analyzer with syntax error."""
        tool = PythonASTAnalyzer()
        
        code = "def invalid_syntax("  # Missing closing parenthesis
        result = await tool.execute(code=code)
        
        assert result["valid_syntax"] is False
        assert "error" in result


@pytest.mark.asyncio 
class TestDataAnalysisTools:
    """Test data analysis tools."""
    
    async def test_data_analyzer_csv(self):
        """Test CSV data analysis."""
        tool = DataAnalyzerTool()
        
        csv_data = """name,age,salary
John,30,50000
Jane,25,60000
Bob,35,45000"""
        
        result = await tool.execute(data=csv_data, format="csv")
        
        assert "shape" in result
        assert result["shape"] == (3, 3)  # 3 rows, 3 columns
        assert "columns" in result
        assert set(result["columns"]) == {"name", "age", "salary"}
        assert "descriptive_statistics" in result
        
        # Check numeric columns analysis
        desc_stats = result["descriptive_statistics"]
        assert "age" in desc_stats
        assert "salary" in desc_stats
    
    async def test_data_analyzer_json(self):
        """Test JSON data analysis.""" 
        tool = DataAnalyzerTool()
        
        json_data = '''[
            {"name": "John", "age": 30, "city": "NYC"},
            {"name": "Jane", "age": 25, "city": "LA"}
        ]'''
        
        result = await tool.execute(data=json_data, format="json")
        
        assert "shape" in result
        assert result["shape"] == (2, 3)
        assert "categorical_analysis" in result
        
        # Check categorical analysis
        cat_analysis = result["categorical_analysis"]
        assert "name" in cat_analysis
        assert "city" in cat_analysis
    
    async def test_data_analyzer_error(self):
        """Test data analyzer with invalid data."""
        tool = DataAnalyzerTool()
        
        result = await tool.execute(data="invalid,csv\ndata", format="csv")
        
        # Should handle gracefully but might not have expected structure
        assert "error" in result or "shape" in result


if __name__ == "__main__":
    pytest.main([__file__])