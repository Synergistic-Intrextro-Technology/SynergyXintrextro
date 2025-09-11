# SynergyX Benchmarking Guide

## Overview

SynergyX includes a comprehensive benchmarking suite to measure system capabilities, performance, and quality across different dimensions.

## Benchmark Categories

### Capabilities Benchmarks

#### Text QA
Tests the system's ability to answer questions based on provided text:
- **Metrics**: Exact Match, F1 Score
- **Dataset**: Curated set of Q&A pairs
- **Evaluation**: Automatic scoring against ground truth

#### Summarization
Evaluates text summarization quality:
- **Metrics**: ROUGE-L, Compression Ratio
- **Dataset**: Articles with reference summaries
- **Evaluation**: Automated ROUGE scoring

#### Tool Invocation
Tests the system's ability to correctly use analysis tools:
- **Metrics**: Success Rate, Retry Count
- **Dataset**: Synthetic tasks requiring tool usage
- **Evaluation**: Task completion validation

#### RAG Retrieval
Measures document retrieval accuracy:
- **Metrics**: Precision@K, Mean Reciprocal Rank (MRR)
- **Dataset**: Query-document pairs
- **Evaluation**: Retrieval ranking assessment

### Performance Benchmarks

#### Latency Measurement
- **P50/P95 Latency**: Response time percentiles
- **Throughput**: Requests per second
- **Token Rate**: Tokens generated per second

#### Resource Usage
- **Memory Footprint**: RAM usage during operations
- **CPU Utilization**: Processing load
- **Storage**: Index and cache sizes

### Quality Benchmarks

#### Groundedness
Measures how well RAG answers are supported by retrieved context:
- **Method**: Cosine similarity between answer and context
- **Threshold**: Configurable similarity threshold
- **Output**: Percentage of grounded responses

#### LLM-as-Judge
Uses a separate LLM to evaluate response quality:
- **Dimensions**: Helpfulness, Harmlessness, Accuracy
- **Scale**: 1-5 rating scale
- **Evaluation Model**: Configurable via EVAL_MODEL

## Running Benchmarks

### Smoke Tests
Quick validation with minimal samples:
```bash
python -m synergyx.benchmarks.run --mode smoke
```

### Full Benchmark Suite
Comprehensive evaluation:
```bash
python -m synergyx.benchmarks.run --mode full
```

### Custom Benchmarks
Target specific categories:
```bash
python -m synergyx.benchmarks.run --categories capabilities,performance
```

## Configuration

### Environment Variables
```bash
# Benchmark settings
SYNERGYX_BENCH_DATASET_DIR=./data/benchmarks
SYNERGYX_BENCH_SMOKE_SAMPLES=3
SYNERGYX_BENCH_FULL_SAMPLES=50

# Evaluation model for LLM-as-judge
EVAL_MODEL=gpt-4
EVAL_PROVIDER=openai
```

### Benchmark Configuration
```yaml
benchmarks:
  reports_directory: "./reports"
  datasets_directory: "./data/benchmarks"
  smoke_samples: 3
  full_samples: 50
  warmup_runs: 1
  measurement_runs: 5
  timeout_seconds: 300
  rouge_l_threshold: 0.3
  exact_match_threshold: 0.8
  groundedness_threshold: 0.7
```

## Report Formats

### JSON Report
Machine-readable results for integration:
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "configuration": {...},
  "results": {
    "capabilities": {...},
    "performance": {...},
    "quality": {...}
  }
}
```

### Markdown Report
Human-readable summary with visualizations

### HTML Report
Interactive report with charts and drill-down capabilities

## Interpreting Results

### Capability Scores
- **Excellent**: >90%
- **Good**: 80-90%
- **Fair**: 70-80%
- **Poor**: <70%

### Performance Targets
- **Latency P95**: <2 seconds
- **Throughput**: >10 requests/second
- **Token Rate**: >50 tokens/second

### Quality Thresholds
- **Groundedness**: >70%
- **LLM-as-Judge**: >4.0/5.0

## Dataset Requirements

### Format
All benchmark datasets should follow the standard format:
```json
[
  {
    "id": "unique_id",
    "input": "input_text_or_query",
    "expected": "expected_output",
    "metadata": {...}
  }
]
```

### Custom Datasets
Add your own datasets to `data/benchmarks/` directory following the naming convention:
- `qa_custom.json` - Custom Q&A pairs
- `summarization_custom.json` - Custom summarization tasks
- `tools_custom.json` - Custom tool invocation tasks

## CI Integration

The benchmarking suite is designed for CI/CD integration:

### Pull Request Validation
- Smoke tests run on every PR
- Regression detection
- Performance comparison

### Nightly Benchmarks
- Full benchmark suite
- Trend analysis
- Report archival

### Performance Regression Alerts
- Configurable thresholds
- Automatic notifications
- Historical comparison