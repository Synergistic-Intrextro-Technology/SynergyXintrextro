# SynergyX Architecture

## Overview

SynergyX is an advanced AI chatbot and analysis system built with Python. It provides both interactive chat capabilities and powerful analysis tools for text, code, data, and web content.

## Core Components

### Chat Engine
The chat engine is the central component that orchestrates conversations between users and AI providers. It manages:

- **Provider Management**: Supports multiple LLM providers (OpenAI, Azure OpenAI, Ollama, HuggingFace)
- **Conversation Memory**: Persistent conversation storage using JSONL format
- **Tool Integration**: Seamless integration with analysis tools
- **Streaming Support**: Real-time response streaming

### Provider System
The provider system offers a pluggable architecture for different LLM backends:

- **OpenAI Provider**: Compatible with OpenAI API and other OpenAI-compatible endpoints
- **Ollama Provider**: Local inference using Ollama
- **Azure OpenAI Provider**: Microsoft Azure OpenAI service integration
- **HuggingFace Provider**: HuggingFace Inference API support

### Analysis Tools
A comprehensive suite of analysis tools:

- **Text Analysis**: Summarization, sentiment analysis, keyword extraction
- **Code Analysis**: AST parsing, complexity metrics, linting
- **Data Analysis**: CSV/JSON processing, statistical analysis, outlier detection
- **RAG System**: Document search and question answering
- **Web Fetcher**: URL content extraction and analysis

### Memory System
Persistent conversation management:

- **JSONL Storage**: Efficient conversation persistence
- **Memory Limits**: Configurable conversation length limits
- **Search**: Conversation search and retrieval

### Interfaces

#### CLI Interface
Rich terminal interface with:
- Interactive chat mode
- Command support (/help, /status, /new, etc.)
- Tool invocation
- Status monitoring

#### HTTP API
RESTful API with:
- Chat endpoints (sync and streaming)
- Analysis tool endpoints
- Conversation management
- Provider status
- OpenAPI documentation

## Data Flow

1. **User Input** → CLI or API
2. **Request Processing** → Chat Engine
3. **Provider Selection** → Provider Manager
4. **Tool Analysis** (if needed) → Tool Registry
5. **LLM Generation** → Selected Provider
6. **Response** → User Interface
7. **Memory Storage** → Conversation Memory

## Configuration

The system uses a hierarchical configuration approach:
1. **YAML Configuration** (config.yaml)
2. **Environment Variables** (.env)
3. **Runtime Overrides**

## Security

- Input validation and sanitization
- Rate limiting capabilities
- Provider credential management
- Safe file operations
- Web content filtering