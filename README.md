# SynergyXintrextro

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced AI chatbot and analysis system that combines interactive conversations with powerful analysis tools for text, code, data, and web content.

## 🚀 Features

### Core Capabilities
- **Multi-Provider LLM Support**: OpenAI, Azure OpenAI, Ollama, HuggingFace Inference
- **Interactive Chat**: CLI and HTTP API interfaces with streaming support
- **Advanced Analysis Tools**: 12+ tools for comprehensive content analysis
- **RAG System**: Document search and question answering
- **Conversation Memory**: Persistent JSONL-based storage
- **Benchmarking Suite**: Comprehensive evaluation framework

### Analysis Tools
- **Text Analysis**: Summarization, sentiment analysis, keyword extraction
- **Code Analysis**: Python AST parsing, complexity metrics, linting
- **Data Analysis**: CSV/JSON processing, statistics, outlier detection
- **Web Content**: URL fetching, HTML parsing, content extraction
- **Document Search**: Vector-based retrieval with FAISS/sklearn

## 📦 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Synergistic-Intrextro-Technology/SynergyXintrextro.git
cd SynergyXintrextro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Optional: Install additional dependencies
pip install -e ".[faiss,advanced-nlp,test,lint]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

#### Provider Configuration

Set at least one provider in your `.env` file:

```bash
# OpenAI (or compatible)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# HuggingFace
HUGGINGFACE_API_KEY=your_hf_token_here
```

### Usage

#### CLI Interface

```bash
# Start interactive chat
python -m synergyx.chat

# Or use the installed command
synergyx-chat
```

**CLI Commands:**
- `/help` - Show help information
- `/status` - Check provider status
- `/new` - Start new conversation
- `/quit` - Exit application

#### HTTP API

```bash
# Start the API server
uvicorn synergyx.interfaces.api:app --host 0.0.0.0 --port 8000

# View API documentation
open http://localhost:8000/docs
```

**API Examples:**

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# Analyze text sentiment
curl -X POST http://localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "analyze_sentiment",
    "parameters": {"text": "I love this product!"}
  }'

# Search documents
curl -X POST http://localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search_documents", 
    "parameters": {"query": "What is SynergyX?"}
  }'
```

## 🔧 Analysis Tools

### Text Analysis
```python
# Summarize text
POST /v1/analyze
{
  "tool_name": "summarize_text",
  "parameters": {
    "text": "Your long text here...",
    "max_sentences": 3
  }
}

# Analyze sentiment
POST /v1/analyze  
{
  "tool_name": "analyze_sentiment",
  "parameters": {"text": "I love this!"}
}

# Extract keywords
POST /v1/analyze
{
  "tool_name": "extract_keywords", 
  "parameters": {"text": "Your text here..."}
}
```

### Code Analysis
```python
# Analyze Python code
POST /v1/analyze
{
  "tool_name": "analyze_python_code",
  "parameters": {
    "code": "def hello():\n    print('Hello, World!')",
    "include_complexity": true
  }
}

# Lint code
POST /v1/analyze
{
  "tool_name": "lint_code",
  "parameters": {
    "code": "print( 'hello' )",
    "language": "python"
  }
}
```

### Data Analysis
```python
# Analyze CSV data
POST /v1/analyze
{
  "tool_name": "analyze_data",
  "parameters": {
    "data": "name,age,city\nJohn,30,NYC\nJane,25,LA",
    "format": "csv"
  }
}

# Get data explanation
POST /v1/analyze
{
  "tool_name": "explain_data",
  "parameters": {"analysis_result": {...}}
}
```

### Web Content
```python
# Fetch web content
POST /v1/analyze
{
  "tool_name": "fetch_web_content",
  "parameters": {
    "url": "https://example.com",
    "extract_text_only": true,
    "summarize": true
  }
}
```

### Document Search (RAG)
```python
# Search documents
POST /v1/analyze
{
  "tool_name": "search_documents",
  "parameters": {
    "query": "How does the RAG system work?",
    "top_k": 5
  }
}

# Answer questions from documents  
POST /v1/analyze
{
  "tool_name": "answer_from_documents",
  "parameters": {"question": "What is SynergyX?"}
}

# Rebuild document index
POST /v1/analyze
{
  "tool_name": "rebuild_document_index",
  "parameters": {"force_rebuild": true}
}
```

## 📊 Benchmarking

### Run Benchmarks

```bash
# Quick smoke test
synergyx-bench --mode smoke

# Full benchmark suite
synergyx-bench --mode full

# Specific categories
synergyx-bench --categories capabilities,performance
```

### View Reports

Reports are generated in `reports/` directory:
- `benchmark_TIMESTAMP.json` - Machine-readable results
- `benchmark_TIMESTAMP.md` - Human-readable summary
- `benchmark_TIMESTAMP.html` - Interactive report

## 🏗️ Architecture

### Core Components

```
synergyx/
├── core/           # Chat engine and provider system
├── tools/          # Analysis tools (text, code, data, etc.)
├── memory/         # Conversation persistence
├── interfaces/     # CLI and HTTP API
├── rag/           # Document search and RAG
├── benchmarks/    # Evaluation framework
└── config/        # Configuration management
```

### Provider System
- **Pluggable Architecture**: Easy to add new LLM providers
- **Automatic Fallback**: Tries providers in priority order
- **Health Monitoring**: Real-time provider status

### Tool Registry
- **Dynamic Registration**: Tools auto-register on import
- **OpenAI Compatible**: Supports function calling schemas
- **Error Handling**: Graceful failure and error reporting

For detailed architecture information, see [docs/architecture.md](docs/architecture.md).

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=synergyx

# Run integration tests
python -m pytest tests/ -m integration

# Run benchmarks (smoke)
python -m pytest tests/ -m benchmark
```

## 🎯 Development

### Code Quality

```bash
# Format code
make format

# Lint code  
make lint

# Type checking
make typecheck

# Run all checks
make check
```

### Adding New Tools

1. Create tool class inheriting from `AnalysisTool`
2. Implement required methods (`name`, `description`, `parameters`, `execute`)
3. Register in `synergyx/tools/registry.py`

Example:
```python
from synergyx.tools.base import AnalysisTool

class MyTool(AnalysisTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property 
    def description(self) -> str:
        return "Description of what my tool does"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_text": {"type": "string", "description": "Input text"}
            },
            "required": ["input_text"]
        }
    
    async def execute(self, input_text: str, **kwargs) -> Dict[str, Any]:
        # Tool implementation
        return {"result": "processed"}
```

### Adding New Providers

1. Create provider class inheriting from `LLMProvider`
2. Implement required async methods
3. Register in `synergyx/core/manager.py`

## 📊 Configuration

### config.yaml
Default configuration settings for the system.

### Environment Variables
Override configuration via environment variables:

```bash
# Application
SYNERGYX_LOG_LEVEL=INFO
SYNERGYX_DATA_DIR=./data

# API
SYNERGYX_API_HOST=0.0.0.0  
SYNERGYX_API_PORT=8000

# Memory
SYNERGYX_MEMORY_MAX_MESSAGES=100

# RAG
SYNERGYX_RAG_DOCS_DIR=./docs
SYNERGYX_RAG_CHUNK_SIZE=500
```

## 🚧 Roadmap

- [ ] **Additional Providers**: Anthropic Claude, Google PaLM
- [ ] **Advanced Tools**: Image analysis, Audio processing
- [ ] **Web UI**: React-based interface
- [ ] **Distributed Deployment**: Docker, Kubernetes support
- [ ] **Plugin System**: Third-party tool integration
- [ ] **Advanced RAG**: Multi-modal, graph-based retrieval

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for HTTP API
- [Rich](https://rich.readthedocs.io/) for beautiful CLI interface
- [Pandas](https://pandas.pydata.org/) for data analysis
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Synergistic-Intrextro-Technology/SynergyXintrextro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Synergistic-Intrextro-Technology/SynergyXintrextro/discussions)
- **Documentation**: [docs/](docs/)

---

**SynergyXintrextro** - Empowering AI-driven analysis and conversation 🤖✨