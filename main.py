#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynergyXintrextro - Unified Entry Point
=======================================

A unified execution entry point for the SynergyXintrextro system that orchestrates
all modules and frameworks in a cohesive manner.

This script provides:
1. Interactive mode selection
2. Unified configuration management
3. Multi-framework orchestration
4. Service composition and execution
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from enum import Enum

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

console = Console()
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Available execution modes for the system."""
    CLI_CHAT = "cli_chat"
    API_SERVER = "api_server"
    KERNEL_ROUTER = "kernel_router"
    SYNERGY_ORCHESTRATOR = "synergy_orchestrator"
    MODAL_FUSION = "modal_fusion"
    BENCHMARK = "benchmark"
    INTREXTRO_LEARNING = "intrextro_learning"
    INTERACTIVE = "interactive"


class UnifiedOrchestrator:
    """Main orchestrator for the SynergyXintrextro system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.console = Console()
        self.running_services: Dict[str, Any] = {}
        
    def setup_logging(self, level: str = "INFO") -> None:
        """Configure logging for the system."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/unified_execution.log", mode="a")
            ]
        )
        
    def show_banner(self) -> None:
        """Display the system banner."""
        banner = """
# SynergyXintrextro - Unified AI System 🚀

An advanced AI orchestration platform that unifies multiple frameworks:
- SynergyX Chat & Analysis
- Kernel Router & Cognitive OS
- Modal Fusion & Learning Systems
- Synergy Orchestrator & Discovery
- Intrextro Learning Framework

Choose an execution mode to get started!
        """
        
        self.console.print(Panel(
            Markdown(banner),
            title="🤖 SynergyXintrextro v0.1.0",
            border_style="blue",
            padding=(1, 2)
        ))
    
    def show_execution_modes(self) -> Table:
        """Display available execution modes."""
        table = Table(title="Available Execution Modes")
        table.add_column("Mode", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Entry Point", style="yellow")
        
        modes_info = [
            ("cli_chat", "Interactive CLI chat interface", "synergyx.chat"),
            ("api_server", "HTTP API server with web interface", "synergyx.interfaces.api"),
            ("kernel_router", "Cognitive OS kernel routing service", "main (1).py"),
            ("synergy_orchestrator", "Multi-framework orchestration", "SynQAI.py"),
            ("modal_fusion", "Multi-modal AI fusion system", "Intrextro_modal_fusion.py"),
            ("benchmark", "Run performance benchmarks", "synergyx.benchmarks"),
            ("intrextro_learning", "Advanced learning framework", "intrextro_learning_framework.py"),
            ("interactive", "Interactive mode selection", "This script")
        ]
        
        for mode, desc, entry in modes_info:
            table.add_row(mode, desc, entry)
            
        return table
    
    async def execute_cli_chat(self) -> None:
        """Execute the CLI chat interface."""
        try:
            from synergyx.interfaces.cli import main as cli_main
            
            self.console.print("[green]Starting SynergyX CLI Chat...[/green]")
            await asyncio.sleep(1)  # Give user time to read
            cli_main()
            
        except Exception as e:
            self.console.print(f"[red]Error starting CLI chat: {e}[/red]")
            logger.error(f"CLI chat error: {e}")
    
    async def execute_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Execute the API server."""
        try:
            import uvicorn
            from synergyx.interfaces.api import app
            
            self.console.print(f"[green]Starting API server on {host}:{port}...[/green]")
            self.console.print(f"[blue]API docs will be available at: http://{host}:{port}/docs[/blue]")
            
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                reload=False,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.console.print(f"[red]Error starting API server: {e}[/red]")
            logger.error(f"API server error: {e}")
    
    async def execute_kernel_router(self) -> None:
        """Execute the kernel router service."""
        try:
            import uvicorn
            
            # Import the main FastAPI app from main (1).py
            sys.path.append(".")
            main_module = __import__("main (1)")
            app = main_module.app
            
            self.console.print("[green]Starting Kernel Router service...[/green]")
            
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=8001,
                reload=False,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.console.print(f"[red]Error starting kernel router: {e}[/red]")
            logger.error(f"Kernel router error: {e}")
    
    async def execute_synergy_orchestrator(self) -> None:
        """Execute the synergy orchestrator."""
        try:
            from SynQAI import SynergyOrchestrator
            
            self.console.print("[green]Starting Synergy Orchestrator...[/green]")
            
            orchestrator = SynergyOrchestrator()
            
            # Demo execution - in a real scenario, you'd load modules and execute tasks
            self.console.print("[blue]Synergy Orchestrator is ready for module registration[/blue]")
            self.console.print("[yellow]This is a framework - integrate with your specific modules[/yellow]")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.console.print(f"[red]Error starting synergy orchestrator: {e}[/red]")
            logger.error(f"Synergy orchestrator error: {e}")
    
    async def execute_modal_fusion(self) -> None:
        """Execute the modal fusion system."""
        try:
            # Check for required dependencies
            try:
                import torch
            except ImportError:
                self.console.print("[yellow]PyTorch not installed. Modal fusion requires PyTorch.[/yellow]")
                self.console.print("[blue]Install with: pip install torch torchvision[/blue]")
                return
                
            from Intrextro_modal_fusion import MultiModalFusionCore, LearningConfig
            
            self.console.print("[green]Starting Modal Fusion System...[/green]")
            
            config = LearningConfig()
            fusion_core = MultiModalFusionCore(config)
            
            self.console.print("[blue]Multi-Modal Fusion Core initialized[/blue]")
            self.console.print("[yellow]Ready for multi-modal data processing[/yellow]")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.console.print(f"[red]Error starting modal fusion: {e}[/red]")
            logger.error(f"Modal fusion error: {e}")
    
    async def execute_benchmark(self) -> None:
        """Execute the benchmark suite."""
        try:
            from synergyx.benchmarks.run import BenchmarkRunner
            
            self.console.print("[green]Starting Benchmark Suite...[/green]")
            
            # Create and run benchmark runner directly
            runner = BenchmarkRunner()
            results = await runner.run_benchmarks("smoke", None)
            
            self.console.print(f"[green]Benchmarks completed successfully![/green]")
            self.console.print(f"[blue]Results: {results}[/blue]")
            
        except Exception as e:
            self.console.print(f"[red]Error running benchmarks: {e}[/red]")
            logger.error(f"Benchmark error: {e}")
    
    async def execute_intrextro_learning(self) -> None:
        """Execute the Intrextro learning framework."""
        try:
            # Import the learning framework
            from intrextro_learning_framework import main as intrextro_main
            
            self.console.print("[green]Starting Intrextro Learning Framework...[/green]")
            
            # Execute the framework
            await intrextro_main()
            
        except Exception as e:
            self.console.print(f"[red]Error starting Intrextro learning: {e}[/red]")
            logger.error(f"Intrextro learning error: {e}")
    
    async def interactive_mode(self) -> None:
        """Run interactive mode for selecting execution options."""
        while True:
            self.console.print("\n" + "="*60)
            self.console.print(self.show_execution_modes())
            
            # Get user choice
            mode_choice = Prompt.ask(
                "\n[bold blue]Select execution mode[/bold blue]",
                choices=[mode.value for mode in ExecutionMode],
                default="interactive"
            )
            
            if mode_choice == "interactive":
                continue
                
            # Confirm choice
            if not Confirm.ask(f"Execute [bold]{mode_choice}[/bold] mode?"):
                continue
            
            try:
                # Execute the selected mode
                if mode_choice == ExecutionMode.CLI_CHAT.value:
                    await self.execute_cli_chat()
                elif mode_choice == ExecutionMode.API_SERVER.value:
                    await self.execute_api_server()
                elif mode_choice == ExecutionMode.KERNEL_ROUTER.value:
                    await self.execute_kernel_router()
                elif mode_choice == ExecutionMode.SYNERGY_ORCHESTRATOR.value:
                    await self.execute_synergy_orchestrator()
                elif mode_choice == ExecutionMode.MODAL_FUSION.value:
                    await self.execute_modal_fusion()
                elif mode_choice == ExecutionMode.BENCHMARK.value:
                    await self.execute_benchmark()
                elif mode_choice == ExecutionMode.INTREXTRO_LEARNING.value:
                    await self.execute_intrextro_learning()
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Execution interrupted[/yellow]")
                if not Confirm.ask("Return to mode selection?"):
                    break
            except Exception as e:
                self.console.print(f"[red]Execution failed: {e}[/red]")
                if not Confirm.ask("Continue with mode selection?"):
                    break
    
    async def run(self, mode: str, **kwargs) -> None:
        """Run the orchestrator in the specified mode."""
        self.setup_logging()
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        try:
            if mode == ExecutionMode.INTERACTIVE.value:
                self.show_banner()
                await self.interactive_mode()
            elif mode == ExecutionMode.CLI_CHAT.value:
                await self.execute_cli_chat()
            elif mode == ExecutionMode.API_SERVER.value:
                await self.execute_api_server(**kwargs)
            elif mode == ExecutionMode.KERNEL_ROUTER.value:
                await self.execute_kernel_router()
            elif mode == ExecutionMode.SYNERGY_ORCHESTRATOR.value:
                await self.execute_synergy_orchestrator()
            elif mode == ExecutionMode.MODAL_FUSION.value:
                await self.execute_modal_fusion()
            elif mode == ExecutionMode.BENCHMARK.value:
                await self.execute_benchmark()
            elif mode == ExecutionMode.INTREXTRO_LEARNING.value:
                await self.execute_intrextro_learning()
            else:
                self.console.print(f"[red]Unknown execution mode: {mode}[/red]")
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Fatal error: {e}[/red]")
            logger.error(f"Fatal error in orchestrator: {e}")
            sys.exit(1)
        finally:
            self.console.print("[green]SynergyXintrextro orchestrator stopped[/green]")


# CLI Interface
@click.command()
@click.option("--mode", "-m", 
              type=click.Choice([mode.value for mode in ExecutionMode]),
              default="interactive",
              help="Execution mode")
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--host", default="0.0.0.0", help="Host for server modes")
@click.option("--port", default=8000, type=int, help="Port for server modes")
@click.option("--log-level", default="INFO", help="Logging level")
def main(mode: str, config: Optional[str], host: str, port: int, log_level: str):
    """
    SynergyXintrextro Unified Entry Point
    
    Execute the system in various modes:
    - interactive: Choose mode interactively
    - cli_chat: Start CLI chat interface
    - api_server: Start HTTP API server
    - kernel_router: Start kernel routing service
    - synergy_orchestrator: Start synergy orchestration
    - modal_fusion: Start multi-modal fusion
    - benchmark: Run benchmarks
    - intrextro_learning: Start learning framework
    """
    
    try:
        orchestrator = UnifiedOrchestrator(config)
        asyncio.run(orchestrator.run(
            mode=mode,
            host=host,
            port=port,
            log_level=log_level
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to start: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()