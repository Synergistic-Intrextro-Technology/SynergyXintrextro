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
