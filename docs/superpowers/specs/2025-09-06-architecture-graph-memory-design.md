# Agentic Job App — Architecture & Graph Memory Design

**Date:** 2025-09-06
**Status:** Approved

## Overview

Refactor the monolithic `main.py` into a modular architecture and add a graph memory layer for persisting job search history and applied jobs across sessions.

## Architecture

### Module Structure

```
agentic-job-app/
  main.py              # Entry point, CLI dispatch
  config/
    __init__.py        # Public: get_llm(), get_tavily(), get_profile()
    settings.py        # Env vars, paths
    clients.py         # LLM, Tavily client factories
    profile.py         # Candidate profile loading
  agents/
    __init__.py        # Public: build_graph(), run_search()
    state.py           # AgentState TypedDict
    nodes.py           # planner, search, verifier, scorer nodes
    pipeline.py        # StateGraph compilation
    schemas.py         # GeneratedQueries pydantic model
  memory/
    __init__.py        # Public: MemoryManager
    graph.py           # KnowledgeGraph model, extract/update/save/load
    manager.py         # MemoryManager: session tracking, applied jobs
  cli/
    __init__.py        # Public: run_search_session(), run_chat_session()
    display.py         # Rich rendering, report formatting
    interactive.py     # User prompts, job selection
  models/
    __init__.py
    models.py          # JobMatch, JobReport (unchanged)
```

### Module Responsibilities

| Module | Purpose |
|---|---|
| `config` | Environment setup, client initialization, profile loading |
| `agents` | LangGraph pipeline: state, nodes, graph compilation |
| `memory` | Knowledge graph: extract, update, persist, query |
| `cli` | User interaction: rendering, prompts, selection |
| `models` | Pydantic schemas (unchanged) |

## Graph Memory

### Graph Schema

- **Node types:** `search_session`, `job`, `company`, `applied_job`, `preference`
- **Edge types:** `searched`, `belongs_to`, `applied`, `matched`

### Persistence Flow

1. **Extraction** — `extract_knowledge()`: send conversation text to LLM with structured output prompt, return `KnowledgeGraph(nodes + edges)`
2. **Update** — `update_graph(nx_graph, kg)`: idempotently merge extracted nodes/edges into existing `nx.DiGraph`, preserve memory status on existing nodes
3. **Persistence** — `save_graph_to_disk(graph)`: serialize via `nx.node_link_data()`, write to `graph_memory.json`

### Integration Points

- **After each search:** `extract_knowledge() -> update_graph() -> save_graph_to_disk()`
- **On startup:** `st.session_state.graph = load_graph_from_disk()`

### MemoryManager API

```python
class MemoryManager:
    def load_graph() -> DiGraph
    def save_graph() -> None
    def get_applied_jobs() -> list[dict]
    def mark_applied(session_id: str, jobs: list[dict]) -> None
    def add_search_session(session_id: str, query_summary: str) -> None
```

## CLI Flow

### Startup

1. Load graph from disk, check for previously applied jobs
2. If applied jobs exist: "You have X previously applied jobs. Choose:"
   - `(1) New search`
   - `(2) Review past applications`
   - `(3) New search (skip review)`

### New Search

1. Run LangGraph pipeline (planner -> search -> verify -> score)
2. Render report with Rich
3. **Before prompt:** check each job against graph memory:
   - Job URL already applied: mark "Already applied"
   - Company has prior application: show "Previously applied to another role at {company}"
   - Pre-select these but allow user to deselect
4. Prompt: "Which jobs would you like to apply to? Enter numbers, 'all', or 'none':"
5. User selects jobs
6. Extract knowledge, update graph, save to disk
7. Print confirmation

### Review Past

1. Query graph for `applied_job` nodes
2. Display applied jobs with company, date, status

## Dependencies

New:
- `networkx` — graph data structure and persistence

Existing (unchanged):
- `langchain`, `langchain-openai`, `langgraph`, `tavily-python`, `rich`, `python-dotenv`, `requests`, `pydantic`
