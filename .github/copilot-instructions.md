# GitHub Copilot Instructions for SynergyXintrextro

## Project Overview

SynergyXintrextro is an advanced AI chatbot and analysis system that combines interactive conversations with powerful analysis tools for text, code, data, and web content. The framework supports multiple LLM providers and provides comprehensive benchmarking capabilities.

## Code Review Focus Areas

### 1. Architecture & Design Patterns

- **Modular Design**: Ensure components follow the established pattern of providers, tools, interfaces, and memory modules
- **Provider Pattern**: New LLM providers should inherit from `LLMProvider` and implement required async methods
- **Tool Registry**: Analysis tools must inherit from `AnalysisTool` and register in the tool registry
- **Interface Separation**: Maintain clear boundaries between CLI, API, and core functionality

### 2. Python Code Quality

- **Type Hints**: All functions and methods must have proper type annotations
- **Async/Await**: Follow established async patterns for LLM calls and I/O operations
- **Error Handling**: Implement graceful error handling with meaningful error messages
- **Documentation**: Use comprehensive docstrings following Google style

### 3. Performance & Reliability

- **Circuit Breakers**: Review timeout and retry logic for external API calls
- **Memory Management**: Monitor memory usage in analysis tools and implement proper cleanup
- **Streaming Support**: Ensure streaming responses are properly handled in both CLI and API
- **Caching**: Validate efficient caching strategies for repeated operations

### 4. Security Considerations

- **API Key Management**: Ensure sensitive credentials are never logged or exposed
- **Input Validation**: Validate all user inputs, especially in analysis tools
- **Rate Limiting**: Review rate limiting implementation for API endpoints
- **Dependency Security**: Check for known vulnerabilities in dependencies

### 5. Testing & Quality Assurance

- **Test Coverage**: New features must include unit tests with good coverage
- **Integration Tests**: API endpoints and tool integrations should have integration tests
- **Benchmarking**: Performance-critical code should include benchmark tests
- **Linting Compliance**: Code must pass ruff, black, and mypy checks

### 6. API Design

- **RESTful Principles**: Follow established API patterns for new endpoints
- **OpenAPI Compatibility**: Ensure tool schemas are compatible with OpenAI function calling
- **Error Responses**: Return consistent error response formats
- **Versioning**: Maintain backward compatibility for API changes

### 7. Configuration & Environment

- **Environment Variables**: Use proper environment variable patterns with fallbacks
- **Configuration Schema**: Follow established config.yaml structure
- **Provider Configuration**: Ensure new providers follow existing configuration patterns
- **Logging**: Use appropriate log levels and structured logging

## Code Style Guidelines

### Imports
```python
# Standard library first
import asyncio
import json
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from synergyx.core.base import BaseProvider
from synergyx.tools.registry import register_tool
```

### Function Signatures
```python
async def analyze_text(
    text: str,
    provider: Optional[str] = None,
    max_length: int = 500,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze text using the specified provider.
    
    Args:
        text: Input text to analyze
        provider: LLM provider name (optional)
        max_length: Maximum response length
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        ValueError: If text is empty or invalid
        ProviderError: If provider fails to respond
    """
```

### Error Handling
```python
try:
    result = await provider.generate(prompt)
except ProviderError as e:
    logger.error(f"Provider {provider.name} failed: {e}")
    raise AnalysisError(f"Failed to analyze text: {e}") from e
except Exception as e:
    logger.exception("Unexpected error during analysis")
    raise AnalysisError("Internal analysis error") from e
```

## File Organization

- **Core Logic**: Place in `synergyx/core/`
- **Analysis Tools**: Place in `synergyx/tools/`
- **Providers**: Place in `synergyx/providers/`
- **Interfaces**: Place in `synergyx/interfaces/`
- **Tests**: Mirror structure in `tests/`
- **Documentation**: Update relevant files in `docs/`

## Common Patterns to Validate

### Tool Registration
```python
@register_tool
class NewAnalysisTool(AnalysisTool):
    @property
    def name(self) -> str:
        return "new_analysis_tool"
```

### Provider Implementation
```python
class NewProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # Implementation  
        pass
```

### API Endpoint Pattern
```python
@router.post("/v1/analyze")
async def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        tool = get_tool(request.tool_name)
        result = await tool.execute(**request.parameters)
        return AnalyzeResponse(success=True, result=result)
    except Exception as e:
        return AnalyzeResponse(success=False, error=str(e))
```

## Review Checklist

- [ ] Type hints are comprehensive and correct
- [ ] Async/await patterns are properly used
- [ ] Error handling is graceful and informative
- [ ] Tests are included for new functionality
- [ ] Documentation is updated
- [ ] Code follows project style guidelines
- [ ] Security considerations are addressed
- [ ] Performance implications are considered
- [ ] Configuration patterns are followed
- [ ] Integration points are properly tested

## Special Attention Areas

1. **Memory Usage**: Monitor large data processing in analysis tools
2. **API Compatibility**: Ensure changes don't break existing integrations
3. **Provider Fallbacks**: Validate provider selection and fallback logic
4. **Streaming**: Test streaming functionality in both CLI and API contexts
5. **Benchmarking**: Ensure performance regression testing for critical paths
6. **Dependencies**: Review impact of new dependencies on installation and deployment

## Review Comments Format

When providing feedback, please:
- Reference specific architectural patterns from this project
- Suggest concrete code improvements with examples
- Highlight potential security or performance issues
- Recommend additional test cases where appropriate
- Point out opportunities for better error handling or logging