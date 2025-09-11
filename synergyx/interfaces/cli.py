"""Command-line interface for SynergyX."""

import asyncio
import sys
import signal
from typing import Optional
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from ..config.manager import get_config, setup_logging
from ..core.engine import ChatEngine
from ..core.models import ChatRequest

console = Console()


class SynergyXCLI:
    """Interactive CLI for SynergyX."""
    
    def __init__(self):
        self.config = get_config()
        setup_logging(self.config)
        self.engine = ChatEngine(self.config)
        self.conversation_id: Optional[str] = None
        self.running = True
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.running = False
            console.print("\n[yellow]Shutting down...[/yellow]")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def show_welcome(self):
        """Show welcome message and system status."""
        welcome_text = """
# Welcome to SynergyX! 🤖

An advanced AI chatbot and analysis system.

## Available Commands:
- `/help` - Show this help message
- `/status` - Show provider status
- `/new` - Start a new conversation
- `/quit` - Exit the application

Just type your message to start chatting!
        """
        
        console.print(Panel(
            Markdown(welcome_text),
            title="SynergyX v0.1.0",
            border_style="blue"
        ))
        
        # Show provider status
        await self.show_status()
    
    async def show_status(self):
        """Show current provider status."""
        with Status("Checking providers...", console=console):
            provider_status = await self.engine.get_provider_status()
        
        status_text = "## Provider Status:\n\n"
        for provider, available in provider_status.items():
            status_icon = "✅" if available else "❌"
            status_text += f"- {provider}: {status_icon}\n"
        
        console.print(Panel(
            Markdown(status_text),
            title="System Status",
            border_style="green" if any(provider_status.values()) else "red"
        ))
    
    async def show_help(self):
        """Show help information."""
        help_text = """
## Commands:
- `/help` - Show this help message
- `/status` - Show provider status  
- `/new` - Start a new conversation
- `/history` - Show conversation history
- `/quit` or `/exit` - Exit the application

## Tips:
- Ask questions about any topic
- Request analysis of text, code, or data
- The AI can use tools to provide detailed analysis
- Conversations are automatically saved
        """
        
        console.print(Panel(
            Markdown(help_text),
            title="Help",
            border_style="cyan"
        ))
    
    async def process_command(self, command: str) -> bool:
        """Process CLI commands. Returns True if command was handled."""
        command = command.lower().strip()
        
        if command in ["/help", "/h"]:
            await self.show_help()
            return True
        
        elif command in ["/status", "/s"]:
            await self.show_status()
            return True
        
        elif command in ["/new", "/n"]:
            self.conversation_id = None
            console.print("[green]Started new conversation[/green]")
            return True
        
        elif command in ["/quit", "/exit", "/q"]:
            self.running = False
            return True
        
        elif command in ["/history"]:
            await self.show_history()
            return True
        
        return False
    
    async def show_history(self):
        """Show conversation history."""
        conversations = await self.engine.list_conversations()
        
        if not conversations:
            console.print("[yellow]No conversations found[/yellow]")
            return
        
        console.print(f"[blue]Found {len(conversations)} conversations:[/blue]")
        for conv_id in conversations[-10:]:  # Show last 10
            console.print(f"  - {conv_id}")
    
    async def chat_loop(self):
        """Main chat interaction loop."""
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask(
                    "[bold blue]You[/bold blue]",
                    default=""
                ).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self.process_command(user_input)
                    continue
                
                # Process chat message
                with Status("Thinking...", console=console):
                    request = ChatRequest(
                        message=user_input,
                        conversation_id=self.conversation_id,
                        stream=False
                    )
                    
                    response = await self.engine.chat(request)
                    self.conversation_id = response.conversation_id
                
                # Display response
                console.print(f"\n[bold green]Assistant[/bold green] ({response.model}):")
                console.print(Panel(
                    Markdown(response.message),
                    border_style="green"
                ))
                console.print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    async def run(self):
        """Run the CLI application."""
        self._setup_signal_handlers()
        
        await self.show_welcome()
        await self.chat_loop()
        
        console.print("[yellow]Goodbye! 👋[/yellow]")


@click.command()
@click.option("--config", help="Path to config file")
def main(config: Optional[str] = None):
    """SynergyX Chat CLI."""
    try:
        cli = SynergyXCLI()
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()