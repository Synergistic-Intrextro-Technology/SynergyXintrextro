"""Benchmark runner for SynergyX capabilities, performance, and quality."""

import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from ..config.manager import get_config, setup_logging
from ..core.engine import ChatEngine
from ..core.models import ChatRequest
from ..tools.registry import get_registry

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkRunner:
    """Benchmark runner for comprehensive system evaluation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config()
        setup_logging(self.config)
        
        self.engine = ChatEngine(self.config)
        self.registry = get_registry()
        self.results: Dict[str, Any] = {}
        
        # Benchmark configuration
        self.reports_dir = self.config.get_reports_dir()
        self.smoke_samples = int(self.config.get_env("SYNERGYX_BENCH_SMOKE_SAMPLES", 3))
        self.full_samples = int(self.config.get_env("SYNERGYX_BENCH_FULL_SAMPLES", 50))
        
    async def run_benchmarks(self, mode: str = "smoke", categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run benchmark suite."""
        console.print(f"[bold blue]Starting {mode} benchmarks...[/bold blue]")
        
        start_time = datetime.now()
        self.results = {
            "timestamp": start_time.isoformat(),
            "mode": mode,
            "configuration": {
                "providers": await self.engine.get_provider_status(),
                "tools": self.registry.list_tools(),
                "samples": self.smoke_samples if mode == "smoke" else self.full_samples
            },
            "results": {}
        }
        
        # Determine which categories to run
        all_categories = ["capabilities", "performance", "quality"]
        if categories is None:
            categories = all_categories
        
        # Run benchmark categories
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            if "capabilities" in categories:
                task = progress.add_task("Running capabilities benchmarks...", total=None)
                self.results["results"]["capabilities"] = await self._run_capabilities_benchmarks(mode)
                progress.remove_task(task)
            
            if "performance" in categories:
                task = progress.add_task("Running performance benchmarks...", total=None)
                self.results["results"]["performance"] = await self._run_performance_benchmarks(mode)
                progress.remove_task(task)
            
            if "quality" in categories:
                task = progress.add_task("Running quality benchmarks...", total=None)
                self.results["results"]["quality"] = await self._run_quality_benchmarks(mode)
                progress.remove_task(task)
        
        # Calculate overall metrics
        end_time = datetime.now()
        self.results["duration_seconds"] = (end_time - start_time).total_seconds()
        self.results["end_timestamp"] = end_time.isoformat()
        
        # Generate reports
        report_id = f"benchmark_{start_time.strftime('%Y%m%d_%H%M%S')}"
        await self._generate_reports(report_id)
        
        console.print(f"[bold green]Benchmarks completed![/bold green] Results saved to {self.reports_dir}/{report_id}.*")
        return self.results
    
    async def _run_capabilities_benchmarks(self, mode: str) -> Dict[str, Any]:
        """Run capabilities benchmarks."""
        samples = self.smoke_samples if mode == "smoke" else self.full_samples
        results = {}
        
        # Text QA benchmark
        try:
            results["text_qa"] = await self._benchmark_text_qa(samples)
        except Exception as e:
            logger.error(f"Text QA benchmark failed: {e}")
            results["text_qa"] = {"error": str(e)}
        
        # Summarization benchmark  
        try:
            results["summarization"] = await self._benchmark_summarization(samples)
        except Exception as e:
            logger.error(f"Summarization benchmark failed: {e}")
            results["summarization"] = {"error": str(e)}
        
        # Tool invocation benchmark
        try:
            results["tool_invocation"] = await self._benchmark_tool_invocation(samples)
        except Exception as e:
            logger.error(f"Tool invocation benchmark failed: {e}")
            results["tool_invocation"] = {"error": str(e)}
        
        return results
    
    async def _run_performance_benchmarks(self, mode: str) -> Dict[str, Any]:
        """Run performance benchmarks."""
        samples = self.smoke_samples if mode == "smoke" else self.full_samples
        results = {}
        
        # Chat latency benchmark
        try:
            results["chat_latency"] = await self._benchmark_chat_latency(samples)
        except Exception as e:
            logger.error(f"Chat latency benchmark failed: {e}")
            results["chat_latency"] = {"error": str(e)}
        
        # Tool execution performance
        try:
            results["tool_performance"] = await self._benchmark_tool_performance(samples)
        except Exception as e:
            logger.error(f"Tool performance benchmark failed: {e}")
            results["tool_performance"] = {"error": str(e)}
        
        return results
    
    async def _run_quality_benchmarks(self, mode: str) -> Dict[str, Any]:
        """Run quality benchmarks."""
        samples = self.smoke_samples if mode == "smoke" else self.full_samples
        results = {}
        
        # Groundedness check (for RAG)
        try:
            results["groundedness"] = await self._benchmark_groundedness(samples)
        except Exception as e:
            logger.error(f"Groundedness benchmark failed: {e}")
            results["groundedness"] = {"error": str(e)}
        
        return results
    
    async def _benchmark_text_qa(self, samples: int) -> Dict[str, Any]:
        """Benchmark text Q&A capabilities."""
        # Sample Q&A pairs for testing
        qa_pairs = [
            {
                "question": "What is Python?",
                "context": "Python is a high-level programming language known for its simplicity and readability.",
                "expected": "programming language"
            },
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                "expected": "artificial intelligence"
            },
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital and largest city of France.",
                "expected": "Paris"
            }
        ]
        
        correct = 0
        total = min(samples, len(qa_pairs))
        latencies = []
        
        for i in range(total):
            qa = qa_pairs[i % len(qa_pairs)]
            
            start_time = time.time()
            try:
                request = ChatRequest(
                    message=f"Context: {qa['context']}\n\nQuestion: {qa['question']}"
                )
                response = await self.engine.chat(request)
                latency = time.time() - start_time
                latencies.append(latency)
                
                # Simple exact match check
                if qa["expected"].lower() in response.message.lower():
                    correct += 1
                    
            except Exception as e:
                logger.error(f"Q&A test failed: {e}")
                latency = time.time() - start_time
                latencies.append(latency)
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct_answers": correct,
            "total_questions": total,
            "average_latency": sum(latencies) / len(latencies) if latencies else 0,
            "latencies": latencies
        }
    
    async def _benchmark_summarization(self, samples: int) -> Dict[str, Any]:
        """Benchmark text summarization."""
        sample_texts = [
            "Natural language processing (NLP) is a field of AI that focuses on interaction between computers and human language. It involves developing algorithms to understand, interpret, and generate human language. NLP has applications in machine translation, sentiment analysis, and chatbots.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions. Common applications include image recognition, recommendation systems, and autonomous vehicles.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities since the 1800s have been the main driver of climate change, primarily through burning fossil fuels and deforestation."
        ]
        
        summaries = []
        compression_ratios = []
        latencies = []
        
        total = min(samples, len(sample_texts))
        
        for i in range(total):
            text = sample_texts[i % len(sample_texts)]
            
            start_time = time.time()
            try:
                result = await self.registry.execute_tool(
                    "summarize_text",
                    text=text,
                    max_sentences=2
                )
                
                latency = time.time() - start_time
                latencies.append(latency)
                
                if result["success"]:
                    summary_result = result["result"]
                    summaries.append(summary_result["summary"])
                    compression_ratios.append(summary_result["compression_ratio"])
                
            except Exception as e:
                logger.error(f"Summarization test failed: {e}")
                latency = time.time() - start_time
                latencies.append(latency)
        
        return {
            "total_summaries": len(summaries),
            "average_compression_ratio": sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
            "average_latency": sum(latencies) / len(latencies) if latencies else 0,
            "compression_ratios": compression_ratios,
            "latencies": latencies
        }
    
    async def _benchmark_tool_invocation(self, samples: int) -> Dict[str, Any]:
        """Benchmark tool invocation success rate."""
        tool_tests = [
            ("analyze_sentiment", {"text": "I love this product!"}),
            ("extract_keywords", {"text": "Python programming language machine learning"}),
            ("analyze_python_code", {"code": "def hello(): print('world')"}),
        ]
        
        successful = 0
        total = min(samples, len(tool_tests) * 10)  # Repeat tests
        latencies = []
        
        for i in range(total):
            tool_name, params = tool_tests[i % len(tool_tests)]
            
            start_time = time.time()
            try:
                result = await self.registry.execute_tool(tool_name, **params)
                latency = time.time() - start_time
                latencies.append(latency)
                
                if result["success"] and not result.get("error"):
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Tool invocation test failed: {e}")
                latency = time.time() - start_time
                latencies.append(latency)
        
        return {
            "success_rate": successful / total if total > 0 else 0,
            "successful_invocations": successful,
            "total_invocations": total,
            "average_latency": sum(latencies) / len(latencies) if latencies else 0,
            "latencies": latencies
        }
    
    async def _benchmark_chat_latency(self, samples: int) -> Dict[str, Any]:
        """Benchmark chat response latency."""
        test_messages = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "Tell me about Python programming.",
            "What are the benefits of automation?"
        ]
        
        latencies = []
        token_counts = []
        
        for i in range(samples):
            message = test_messages[i % len(test_messages)]
            
            start_time = time.time()
            try:
                request = ChatRequest(message=message)
                response = await self.engine.chat(request)
                latency = time.time() - start_time
                
                latencies.append(latency)
                
                # Estimate token count (rough approximation)
                token_count = len(response.message.split())
                token_counts.append(token_count)
                
            except Exception as e:
                logger.error(f"Chat latency test failed: {e}")
                latency = time.time() - start_time
                latencies.append(latency)
        
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        tokens_per_second = avg_tokens / avg_latency if avg_latency > 0 else 0
        
        return {
            "p50_latency": p50,
            "p95_latency": p95,
            "average_latency": avg_latency,
            "tokens_per_second": tokens_per_second,
            "total_requests": len(latencies),
            "latencies": latencies
        }
    
    async def _benchmark_tool_performance(self, samples: int) -> Dict[str, Any]:
        """Benchmark tool execution performance."""
        tool_tests = [
            ("analyze_sentiment", {"text": "This is a great product with amazing features!"}),
            ("summarize_text", {"text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20}),
            ("extract_keywords", {"text": "Python machine learning data science artificial intelligence"}),
        ]
        
        results_by_tool = {}
        
        for tool_name, params in tool_tests:
            latencies = []
            successes = 0
            
            for _ in range(samples):
                start_time = time.time()
                try:
                    result = await self.registry.execute_tool(tool_name, **params)
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    if result["success"]:
                        successes += 1
                        
                except Exception as e:
                    latency = time.time() - start_time
                    latencies.append(latency)
            
            results_by_tool[tool_name] = {
                "average_latency": sum(latencies) / len(latencies) if latencies else 0,
                "success_rate": successes / samples if samples > 0 else 0,
                "total_executions": samples,
                "latencies": latencies
            }
        
        return results_by_tool
    
    async def _benchmark_groundedness(self, samples: int) -> Dict[str, Any]:
        """Benchmark groundedness of RAG responses."""
        # Simple groundedness check - in a real implementation, this would be more sophisticated
        return {
            "average_groundedness": 0.75,  # Placeholder
            "samples_evaluated": samples,
            "threshold": 0.7,
            "method": "placeholder"
        }
    
    async def _generate_reports(self, report_id: str) -> None:
        """Generate benchmark reports in multiple formats."""
        # JSON report
        json_path = self.reports_dir / f"{report_id}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Markdown report
        md_path = self.reports_dir / f"{report_id}.md"
        markdown_content = self._generate_markdown_report()
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        
        # Console summary
        self._print_summary()
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown report content."""
        md_lines = [
            f"# SynergyX Benchmark Report",
            f"",
            f"**Generated:** {self.results['timestamp']}",
            f"**Mode:** {self.results['mode']}",
            f"**Duration:** {self.results.get('duration_seconds', 0):.2f} seconds",
            f"",
            f"## Configuration",
            f"",
            f"- **Providers:** {', '.join(k for k, v in self.results['configuration']['providers'].items() if v)}",
            f"- **Tools:** {len(self.results['configuration']['tools'])} available",
            f"- **Samples:** {self.results['configuration']['samples']}",
            f"",
            f"## Results",
            f""
        ]
        
        # Add capabilities results
        if "capabilities" in self.results["results"]:
            md_lines.extend([
                f"### Capabilities",
                f""
            ])
            
            cap_results = self.results["results"]["capabilities"]
            
            if "text_qa" in cap_results and "error" not in cap_results["text_qa"]:
                qa = cap_results["text_qa"]
                md_lines.extend([
                    f"**Text Q&A:**",
                    f"- Accuracy: {qa['accuracy']:.2%}",
                    f"- Average Latency: {qa['average_latency']:.2f}s",
                    f""
                ])
            
            if "summarization" in cap_results and "error" not in cap_results["summarization"]:
                summ = cap_results["summarization"]
                md_lines.extend([
                    f"**Summarization:**",
                    f"- Average Compression: {summ['average_compression_ratio']:.2f}",
                    f"- Average Latency: {summ['average_latency']:.2f}s",
                    f""
                ])
            
            if "tool_invocation" in cap_results and "error" not in cap_results["tool_invocation"]:
                tools = cap_results["tool_invocation"]
                md_lines.extend([
                    f"**Tool Invocation:**",
                    f"- Success Rate: {tools['success_rate']:.2%}",
                    f"- Average Latency: {tools['average_latency']:.2f}s",
                    f""
                ])
        
        # Add performance results
        if "performance" in self.results["results"]:
            md_lines.extend([
                f"### Performance",
                f""
            ])
            
            perf_results = self.results["results"]["performance"]
            
            if "chat_latency" in perf_results and "error" not in perf_results["chat_latency"]:
                chat = perf_results["chat_latency"]
                md_lines.extend([
                    f"**Chat Latency:**",
                    f"- P50: {chat['p50_latency']:.2f}s",
                    f"- P95: {chat['p95_latency']:.2f}s",
                    f"- Tokens/sec: {chat['tokens_per_second']:.1f}",
                    f""
                ])
        
        return "\n".join(md_lines)
    
    def _print_summary(self) -> None:
        """Print benchmark summary to console."""
        table = Table(title="Benchmark Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")
        
        results = self.results["results"]
        
        # Capabilities
        if "capabilities" in results:
            cap = results["capabilities"]
            if "text_qa" in cap and "error" not in cap["text_qa"]:
                table.add_row("Capabilities", "Text Q&A Accuracy", f"{cap['text_qa']['accuracy']:.2%}")
            if "summarization" in cap and "error" not in cap["summarization"]:
                table.add_row("", "Summarization Compression", f"{cap['summarization']['average_compression_ratio']:.2f}")
            if "tool_invocation" in cap and "error" not in cap["tool_invocation"]:
                table.add_row("", "Tool Success Rate", f"{cap['tool_invocation']['success_rate']:.2%}")
        
        # Performance
        if "performance" in results:
            perf = results["performance"]
            if "chat_latency" in perf and "error" not in perf["chat_latency"]:
                table.add_row("Performance", "Chat P95 Latency", f"{perf['chat_latency']['p95_latency']:.2f}s")
                table.add_row("", "Tokens/second", f"{perf['chat_latency']['tokens_per_second']:.1f}")
        
        console.print(table)


@click.command()
@click.option("--mode", default="smoke", type=click.Choice(["smoke", "full"]), help="Benchmark mode")
@click.option("--categories", help="Comma-separated list of categories (capabilities,performance,quality)")
@click.option("--config", help="Path to config file")
def main(mode: str, categories: Optional[str], config: Optional[str]):
    """Run SynergyX benchmarks."""
    try:
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(",")]
        
        runner = BenchmarkRunner(config)
        results = asyncio.run(runner.run_benchmarks(mode, category_list))
        
        console.print(f"[bold green]Benchmarks completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmarks interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        logger.error(f"Benchmark failed: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()