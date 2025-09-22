# SynergyXintrextro - Unified Execution Guide

This guide explains how to execute the SynergyXintrextro system as a unified platform with multiple execution modes.

## Overview

SynergyXintrextro is an advanced AI orchestration platform that unifies multiple frameworks and modules:

- **SynergyX Chat & Analysis** - Interactive AI chatbot with analysis tools
- **Kernel Router & Cognitive OS** - Intelligent routing and cognitive processing
- **Modal Fusion & Learning Systems** - Multi-modal AI data processing  
- **Synergy Orchestrator & Discovery** - Framework composition and discovery
- **Intrextro Learning Framework** - Advanced adaptive learning system

## Quick Start

### 1. Interactive Mode (Recommended)

Start the unified system in interactive mode to choose your execution option:

```bash
# Using the unified entry point
python main.py --mode interactive

# Or use Make
make run-unified
```

This will show a menu with all available execution modes and let you select interactively.

### 2. Direct Execution Modes

Execute specific modes directly:

```bash
# CLI Chat Interface
python main.py --mode cli_chat
make run-unified-cli

# HTTP API Server
python main.py --mode api_server
make run-unified-api

# Kernel Router Service
python main.py --mode kernel_router
make run-kernel-router

# Synergy Orchestrator
python main.py --mode synergy_orchestrator
make run-synergy

# Modal Fusion System
python main.py --mode modal_fusion
make run-modal-fusion

# Benchmarks
python main.py --mode benchmark
make bench

# Intrextro Learning Framework
python main.py --mode intrextro_learning
make run-learning
```

### 3. Docker-based Execution

Run the entire system using Docker:

```bash
# Build the unified Docker image
make docker-build

# Run single container
make docker-run

# Run full multi-service setup
make docker-compose-up

# View logs
make docker-compose-logs

# Stop services
make docker-compose-down
```

## Execution Modes Detailed

### CLI Chat (`cli_chat`)
- **Purpose**: Interactive command-line chat interface
- **Features**: Multi-provider LLM support, tool integration, conversation history
- **Use case**: Development, testing, interactive AI assistance

### API Server (`api_server`)  
- **Purpose**: HTTP REST API with web interface
- **Port**: 8000 (default)
- **Features**: RESTful endpoints, OpenAPI docs, CORS support
- **Use case**: Integration with web applications, programmatic access

### Kernel Router (`kernel_router`)
- **Purpose**: Cognitive OS kernel routing service  
- **Port**: 8001 (default)
- **Features**: Intelligent request routing, kernel abstraction
- **Use case**: Distributed AI processing, service orchestration

### Synergy Orchestrator (`synergy_orchestrator`)
- **Purpose**: Multi-framework composition and synergy discovery
- **Features**: Module registry, capability analysis, runtime adaptation
- **Use case**: Research, framework integration, capability discovery

### Modal Fusion (`modal_fusion`)
- **Purpose**: Multi-modal data processing and fusion
- **Features**: Text, image, audio, numerical data integration
- **Use case**: Complex data analysis, cross-modal learning

### Benchmarks (`benchmark`)
- **Purpose**: Performance and quality benchmarking
- **Features**: Capability, performance, and quality metrics
- **Use case**: System evaluation, performance monitoring

### Intrextro Learning (`intrextro_learning`)
- **Purpose**: Advanced adaptive learning framework
- **Features**: Meta-learning, optimization, adaptive systems
- **Use case**: Advanced AI research, adaptive learning scenarios

## Configuration

### Environment Setup

1. Copy the environment template:
```bash
make setup-env
```

2. Edit `.env` with your configuration:
```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Service Configuration  
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Configuration Files

- `config.yaml` - Main system configuration
- `kernels.yaml` - Kernel routing configuration  
- `kernels.json` - Kernel metadata
- `.env` - Environment variables and API keys

## Development Workflow

### Installation

```bash
# Install in development mode
make install-dev

# Or basic installation
make install
```

### Development Commands

```bash
# Run tests
make test

# Lint code
make lint

# Format code  
make format

# Type checking
make typecheck

# Run all checks
make check
```

### Adding New Execution Modes

1. Add your mode to the `ExecutionMode` enum in `main.py`
2. Implement an `execute_your_mode()` method in `UnifiedOrchestrator`
3. Add the mode to the execution logic in the `run()` method
4. Update the CLI help and mode selection table
5. Add Make targets if needed

## Docker Services

The `docker-compose.unified.yaml` defines multiple services:

- **synergyx-orchestrator**: Main API server (port 8000)
- **kernel-router**: Kernel routing service (port 8001)  
- **synergy-orchestrator**: Framework orchestration
- **modal-fusion**: Multi-modal processing
- **benchmark-runner**: One-time benchmark execution

## Monitoring and Logs

### Log Locations
- `logs/unified_execution.log` - Main orchestrator logs
- `logs/` - Service-specific logs
- Docker logs via `docker-compose logs`

### Health Monitoring
- API health: `http://localhost:8000/health`
- Kernel status: `http://localhost:8001/status`

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports using `--port` option or environment variables
2. **Missing dependencies**: Run `make install-dev` to install all dependencies  
3. **API keys**: Ensure `.env` file contains valid API keys
4. **Module imports**: Check that all required modules are installed

### Debug Mode

```bash
# Run with debug logging
python main.py --mode interactive --log-level DEBUG
```

### Service Status

```bash
# Check what's running
make curl-health

# Test specific endpoints
make curl-chat
make curl-tools
```

## Examples

### Basic Usage Flow

1. Start interactive mode: `make run-unified`
2. Select `api_server` mode
3. Open browser to `http://localhost:8000/docs`
4. Test API endpoints or use web interface

### Multi-Service Setup

1. Start all services: `make docker-compose-up`
2. API server available at: `http://localhost:8000`
3. Kernel router at: `http://localhost:8001`
4. Monitor logs: `make docker-compose-logs`

### Development Workflow

1. Make code changes
2. Run tests: `make test`
3. Test locally: `make run-unified-cli`
4. Build Docker: `make docker-build`
5. Deploy: `make docker-compose-up`

## Contributing

To add new frameworks or execution modes:

1. Create your framework module
2. Add execution mode to `main.py`
3. Update documentation
4. Add tests
5. Update Docker configuration if needed

The unified orchestrator is designed to be extensible and can accommodate new frameworks and execution patterns.