import uuid

from rich.console import Console
from rich.markdown import Markdown

from agents import build_graph
from cli.display import render_markdown
from config.clients import llm
from memory import MemoryManager


def main():
    console = Console()
    memory = MemoryManager(llm)

    # Show past applications
    applied = memory.get_applied_jobs()
    if applied:
        console.print(f"[yellow]You have {len(applied)} previously applied jobs.[/yellow]")

    # Run search
    graph = build_graph()
    initial_state = {
        "queries": [],
        "raw_results": [],
        "verified_jobs": [],
        "final_report": None,
    }

    output = graph.invoke(initial_state)
    report = output["final_report"]

    # Render
    console.print(Markdown(render_markdown(report)))

    # Memory: session tracking
    session_id = str(uuid.uuid4())[:8]
    memory.add_search_session(session_id, "Job search")

    # Annotate with memory info
    applied_urls = {j.get("id") for j in memory.get_applied_jobs()}
    applied_companies = memory.get_applied_companies()

    for i, job in enumerate(report.jobs, 1):
        notes = []
        if str(job.url) in applied_urls:
            notes.append("[red]Already applied[/red]")
        elif job.company in applied_companies:
            notes.append(f"[yellow]Previously applied at {job.company}[/yellow]")
        if notes:
            console.print(f"  {i}. {job.title} - {', '.join(notes)}")

    choice = input("\nWhich jobs to apply to? (numbers, 'all', 'none'): ")
    if choice.strip().lower() == "all":
        selected = [j.model_dump() for j in report.jobs]
    elif choice.strip().lower() != "none":
        indices = [int(x.strip()) for x in choice.split(",")]
        selected = [report.jobs[i - 1].model_dump() for i in indices]
    else:
        selected = []

    if selected:
        memory.mark_applied(session_id, selected)
        console.print(f"[green]Saved {len(selected)} applied jobs to memory.[/green]")


if __name__ == "__main__":
    main()
