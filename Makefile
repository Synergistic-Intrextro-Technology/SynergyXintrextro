# SynergyX Makefile

.PHONY: help install test lint format typecheck run-api run-cli bench clean docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dependencies
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[test,lint,faiss,advanced-nlp]"

test:  ## Run unit tests
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ --cov=synergyx --cov-report=html --cov-report=term

test-integration:  ## Run integration tests
	python -m pytest tests/ -m integration -v

lint:  ## Run linting checks
	python -m ruff check synergyx/
	python -m ruff check tests/

format:  ## Format code with black and ruff
	python -m ruff check --fix synergyx/
	python -m black synergyx/ tests/

typecheck:  ## Run type checking with mypy
	python -m mypy synergyx/

check: lint typecheck test  ## Run all checks

run-api:  ## Start the HTTP API server
	uvicorn synergyx.interfaces.api:app --host 0.0.0.0 --port 8000 --reload

run-cli:  ## Start the CLI interface
	python -m synergyx.chat

bench:  ## Run smoke benchmarks
	python -m synergyx.benchmarks.run --mode smoke

bench-full:  ## Run full benchmark suite
	python -m synergyx.benchmarks.run --mode full

docs:  ## Generate documentation
	@echo "Documentation is available in docs/ directory"
	@echo "API docs: http://localhost:8000/docs (when server is running)"

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development convenience targets
curl-health:  ## Test API health endpoint
	curl -s http://localhost:8000/health | python -m json.tool

curl-chat:  ## Test chat endpoint
	curl -X POST http://localhost:8000/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "Hello, how are you?"}' | python -m json.tool

curl-tools:  ## List available tools
	curl -s http://localhost:8000/v1/tools | python -m json.tool

curl-analyze:  ## Test analysis endpoint
	curl -X POST http://localhost:8000/v1/analyze \
		-H "Content-Type: application/json" \
		-d '{"tool_name": "analyze_sentiment", "parameters": {"text": "I love this!"}}' | python -m json.tool

setup-env:  ## Copy .env.example to .env
	cp .env.example .env
	@echo "Please edit .env file with your API keys"

init: setup-env install  ## Initialize project (copy env, install deps)
	@echo "Project initialized! Run 'make run-cli' or 'make run-api' to start"

# Unified execution commands
run-unified:  ## Run unified interactive mode selector
	python main.py --mode interactive

run-unified-cli:  ## Run unified CLI chat mode
	python main.py --mode cli_chat

run-unified-api:  ## Run unified API server mode
	python main.py --mode api_server --port 8000

run-kernel-router:  ## Run kernel router service
	python main.py --mode kernel_router

run-synergy:  ## Run synergy orchestrator
	python main.py --mode synergy_orchestrator

run-modal-fusion:  ## Run modal fusion system
	python main.py --mode modal_fusion

run-learning:  ## Run Intrextro learning framework
	python main.py --mode intrextro_learning

run-all-demos:  ## Run all available execution modes (demo)
	@echo "Running unified execution demos..."
	python main.py --mode benchmark

# Docker-based unified execution
docker-build:  ## Build unified Docker image
	docker build -f Dockerfile.unified -t synergyx-unified .

docker-run:  ## Run unified system in Docker
	docker run -it -p 8000:8000 -p 8001:8001 synergyx-unified

docker-compose-up:  ## Start all services with Docker Compose
	docker-compose -f docker-compose.unified.yaml up -d

docker-compose-down:  ## Stop all services
	docker-compose -f docker-compose.unified.yaml down

docker-compose-logs:  ## View logs from all services
	docker-compose -f docker-compose.unified.yaml logs -f

docker-clean:  ## Clean Docker images and containers
	docker-compose -f docker-compose.unified.yaml down -v
	docker rmi synergyx-unified 2>/dev/null || true

# Demo and examples
demo-unified:  ## Run unified execution demo
	python demo_unified.py

example-interactive:  ## Example: Run interactive mode
	@echo "Starting interactive mode - choose your execution option:"
	python main.py --mode interactive

example-api:  ## Example: Start API server  
	@echo "Starting API server on http://localhost:8000"
	@echo "API docs: http://localhost:8000/docs"
	python main.py --mode api_server --host 0.0.0.0 --port 8000
