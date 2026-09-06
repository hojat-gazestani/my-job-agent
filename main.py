from rich.console import Console
from rich.markdown import Markdown

from agents import build_graph


def main():
    graph = build_graph()
    initial_state = {
        "queries": [],
        "raw_results": [],
        "verified_jobs": [],
        "final_report": "",
    }

    output = graph.invoke(initial_state)
    console = Console()
    print("FINAL JOB SEARCH REPORT")
    console.print(Markdown(output["final_report"]))


if __name__ == "__main__":
    main()
