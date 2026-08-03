"""Interactive command-line interface for the Clinical RAG Navigator.

Loads the pipeline directly in-process (no need to run the FastAPI server
separately) — useful for quick manual testing during development.

Usage:
    python scripts/cli.py
"""
from __future__ import annotations

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from clinical_rag.pipeline import RAGPipeline
from clinical_rag.schemas import QueryResponse
from clinical_rag.utils.logging_config import setup_logging

console = Console()


def print_response(response: QueryResponse) -> None:
    console.print()
    console.print(Panel(response.answer, title="Answer", border_style="cyan"))

    if response.citations:
        table = Table(title="Sources", show_lines=True, expand=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", ratio=2)
        table.add_column("Org / Year", ratio=1)
        table.add_column("Excerpt", ratio=3)

        for i, citation in enumerate(response.citations, start=1):
            year = str(citation.year) if citation.year else "\u2014"
            table.add_row(
                str(i),
                citation.source_title,
                f"{citation.organization} ({year})",
                citation.excerpt,
            )
        console.print(table)
    else:
        console.print("[dim]No sources retrieved.[/dim]")

    if response.confidence >= 0.7:
        color = "green"
    elif response.confidence >= 0.4:
        color = "yellow"
    else:
        color = "red"
    console.print(f"\nConfidence: [{color}]{response.confidence:.1%}[/{color}]")


def main() -> None:
    load_dotenv()
    setup_logging()

    console.print(
        Panel.fit(
            "[bold cyan]Clinical RAG Navigator[/bold cyan]\n"
            "Ask a clinical question. Type 'exit' or 'quit' to leave.",
            border_style="cyan",
        )
    )

    with console.status("[bold cyan]Loading pipeline (embedding model, vector index)...", spinner="dots"):
        try:
            pipeline = RAGPipeline()
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            return

    console.print("[green]Ready.[/green]\n")

    while True:
        try:
            question = console.input("[bold]> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                response = pipeline.query(question)
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
                continue

        print_response(response)


if __name__ == "__main__":
    main()
