"""Utility functions for the deep research agent."""

import os
from pathlib import Path

from exa_py import Exa
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from deep_research.state import ResearchBrief


class TodoList(BaseModel):
    """The todo list for the research agent."""

    tasks: list[str] = Field(description="The full updated list of research tasks.")
    completed_tasks: list[str] = Field(default_factory=list)


# Resolve RESEARCH_ROOT relative to this file to avoid hardcoded absolute paths.
# utils.py lives at src/deep_research/utils.py, so .parent.parent.parent = project root.
RESEARCH_ROOT = Path(__file__).parent.parent.parent / "vfs"


# --- Todo List Helpers ---


def write_todo_list(todo: TodoList, path: Path) -> None:
    """Write a TodoList to disk as JSON.

    This is a plain helper called programmatically by supervisor_tools.
    The caller is responsible for ensuring `path` is valid and inside the VFS.

    Args:
        todo: The TodoList object to persist.
        path: Absolute path to the JSON file (e.g., run_root / "todo_list.json").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(todo.model_dump_json(indent=2))


# --- Worker File Management Tools ---


def get_worker_filesystem_tools(worker_vfs_dir: str) -> list[BaseTool]:
    """Get the file management tools scoped to a worker's VFS directory.

    Args:
        worker_vfs_dir: Absolute path to the worker's dedicated VFS directory.

    Returns:
        A list of tools with a protected write operation to preserve raw logs.
    """
    toolkit = FileManagementToolkit(
        root_dir=worker_vfs_dir,
        selected_tools=["write_file", "read_file", "list_directory"],
    )
    all_tools = toolkit.get_tools()

    final_tools = []
    for t in all_tools:
        if t.name == "write_file":
            # Wrap the write tool to protect raw_content.md
            original_run = t._run
            original_arun = t._arun

            def protected_run(*args, **kwargs):
                file_path = kwargs.get("file_path") or (args[0] if args else None)
                if file_path and (file_path == "raw_content.md" or file_path.endswith("/raw_content.md")):
                    return "ERROR: 'raw_content.md' is managed automatically by 'exa_search'. Manual writes are forbidden to prevent data loss. Please write your synthesis to 'compressed_summary.md' instead."
                return original_run(*args, **kwargs)

            async def protected_arun(*args, **kwargs):
                file_path = kwargs.get("file_path") or (args[0] if args else None)
                if file_path and (file_path == "raw_content.md" or file_path.endswith("/raw_content.md")):
                    return "ERROR: 'raw_content.md' is managed automatically by 'exa_search'. Manual writes are forbidden to prevent data loss. Please write your synthesis to 'compressed_summary.md' instead."
                return await original_arun(*args, **kwargs)

            t._run = protected_run
            t._arun = protected_arun
            t.description = "Write a file to the VFS. NOTE: 'raw_content.md' is PROTECTED and cannot be written manually."

        final_tools.append(t)

    return final_tools


# --- Search Tools ---


@tool
def exa_search(
    query: str,
    num_results: int = 8,
    search_type: str = "auto",
    livecrawl: str = "fallback",
    category: str | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    max_characters: int = 8000,
    vfs_path: str | None = None,
) -> str:
    """Search the web using Exa's neural search engine.

    Use this tool to find information on any topic. Choose parameters
    strategically to get the most relevant, up-to-date results.

    Args:
        query: Natural language search query. Be specific and descriptive.
        num_results: Number of results to return (1-20). Default 8.
        search_type: Search strategy ('neural', 'keyword', 'auto').
            Neural is best for semantics; keyword is best for exact names/terms.
        livecrawl: Fetch the latest page content ('always', 'fallback', 'never').
            Use 'always' for very recent news or earnings reports.
        category: Filter by source type. Options: 'company', 'news', 'research paper',
            'pdf', 'github', 'tweet', 'blog post', 'personal site', 'social media'.
        start_published_date: Filter results published AFTER this date (YYYY-MM-DD).
        end_published_date: Filter results published BEFORE this date (YYYY-MM-DD).
        include_domains: List of domains to LIMIT the search to (e.g., ['nature.com']).
        exclude_domains: List of domains to REMOVE from the results (e.g., ['reddit.com']).
        max_characters: Max chars per result (1000-15000). Default 8000.
        vfs_path: (Internal) Path to automatically record raw results.

    Returns:
        Formatted string of search results with titles, URLs, and summaries.
    """
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))

    results = exa.search_and_contents(
        query,
        num_results=num_results,
        type=search_type,
        livecrawl=livecrawl,
        category=category,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        text={"max_characters": max_characters},
        highlights={"query": query, "num_sentences": 5},
        summary={"query": query},
    )

    if not results.results:
        return "No results found."

    # 1. Generate the detailed version for disk (includes Full Text)
    disk_parts = []
    for i, r in enumerate(results.results, 1):
        lines = [f"## Result {i}: {r.title or 'Untitled'}"]
        lines.append(f"**URL**: {r.url}")
        if r.published_date:
            lines.append(f"**Published**: {r.published_date}")
        if r.summary:
            lines.append(f"\n### Summary\n{r.summary}")
        if r.highlights:
            lines.append("\n### Key Highlights")
            for h in r.highlights:
                lines.append(f"- {h.strip()}")
        if r.text:
            lines.append(f"\n### Full Text Excerpt\n{r.text}")
        disk_parts.append("\n".join(lines))

    # 2. Generate the lightweight version for the LLM (NO Full Text)
    llm_parts = []
    for i, r in enumerate(results.results, 1):
        lines = [f"## Result {i}: {r.title or 'Untitled'}"]
        lines.append(f"**URL**: {r.url}")
        if r.published_date:
            lines.append(f"**Published**: {r.published_date}")
        if r.summary:
            lines.append(f"\n### Summary\n{r.summary}")
        if r.highlights:
            lines.append("\n### Key Highlights")
            for h in r.highlights:
                lines.append(f"- {h.strip()}")
        llm_parts.append("\n".join(lines))

    # Record to disk if path is provided (Comprehensive)
    if vfs_path:
        path = Path(vfs_path) / "raw_content.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n# Search: {query}\n\n" + "\n\n---\n\n".join(disk_parts))

    # Return to LLM (Lightweight)
    return "\n\n---\n\n".join(llm_parts)


def get_search_tools() -> list[BaseTool]:
    """Get the Exa search tool for the research worker.

    Returns:
        A list containing the custom exa_search tool.
    """
    return [exa_search]


# --- Prompt Formatting Helpers ---


def brief_to_prompt_vars(brief: ResearchBrief) -> dict:
    """Format the ResearchBrief fields for injection into the supervisor prompt.

    Args:
        brief: The approved ResearchBrief from state.

    Returns:
        A dict with keys matching the SUPERVISOR_PROMPT template placeholders.
    """
    return {
        "topic": brief.topic,
        "main_objective": brief.main_objective,
        "scope": brief.scope,
        "sub_objectives": "\n".join([f"  - {obj}" for obj in brief.sub_objectives]),
    }


def todo_to_string(todo: TodoList) -> str:
    """Format the current TodoList as a readable checklist for the supervisor prompt.

    Args:
        todo: The current TodoList object.

    Returns:
        A string with pending tasks marked [ ] and completed tasks marked [x].
    """
    lines = []
    for task in todo.tasks:
        lines.append(f"[ ] {task}")
    for task in todo.completed_tasks:
        lines.append(f"[x] {task}")
    return "\n".join(lines) if lines else "No tasks in list."


def get_findings_summary(vfs_root: Path) -> str:
    """Scan the VFS for completed worker compressed summaries.

    Used to give the supervisor a high-level view of what's been researched.
    Note: The supervisor primarily sees findings via ToolMessages in its message
    history. This helper provides a supplementary directory listing.

    Args:
        vfs_root: The root VFS directory to scan.

    Returns:
        A formatted string listing completed research topics, or a placeholder
        if no findings exist yet.
    """
    if not vfs_root.exists():
        return "No finding files yet."

    summaries = []
    for summary_path in vfs_root.glob("**/compressed_summary.md"):
        topic_name = summary_path.parent.name
        summaries.append(f"  - {topic_name}")

    return "\n".join(summaries) if summaries else "No finding files yet."
