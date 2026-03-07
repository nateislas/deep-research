"""Utility functions for the deep research agent."""

import os
from pathlib import Path

from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import BaseTool, tool
from langchain_exa import ExaSearchResults
from pydantic import BaseModel, Field

from deep_research.state import ResearchBrief


class TodoList(BaseModel):
    """The todo list for the research agent."""

    tasks: list[str] = Field(description="The full updated list of research tasks.")
    completed_tasks: list[str] = Field(default_factory=list)


# Resolve RESEARCH_ROOT relative to this file to avoid hardcoded absolute paths.
# utils.py lives at src/deep_research/utils.py, so .parent.parent.parent = project root.
RESEARCH_ROOT = Path(__file__).parent.parent.parent / "vfs"


# --- Todo List Tool ---


@tool
def update_todo_list(todo_data: TodoList, todo_path_str: str) -> str:
    """Update the entire research todo list in the VFS.

    Use this to plan next steps or check off finished items.

    Args:
        todo_data: The updated TodoList object.
        todo_path_str: The path to the todo list file in the VFS.

    Returns:
        A confirmation message with the path to the updated file.

    Raises:
        ValueError: If the path is outside the RESEARCH_ROOT (path traversal attempt).
    """
    # Sanitize and resolve the path
    target_path = Path(todo_path_str).resolve()
    root_path = RESEARCH_ROOT.resolve()

    # Security check: Ensure we are inside the RESEARCH_ROOT
    if not str(target_path).startswith(str(root_path)):
        # If it's a relative path, try joining with RESEARCH_ROOT
        target_path = (RESEARCH_ROOT / todo_path_str).resolve()
        if not str(target_path).startswith(str(root_path)):
            raise ValueError(f"Path traversal attempt blocked: {todo_path_str}")

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w") as f:
        f.write(todo_data.model_dump_json(indent=2))
    return f"Todo list updated successfully at {target_path}."


# --- Worker File Management Tools ---


def get_worker_filesystem_tools(worker_vfs_dir: str) -> list[BaseTool]:
    """Get the file management tools scoped to a worker's VFS directory.

    The worker LLM will call write_file and read_file as tools during its
    search loop. Scoping to the worker's directory prevents cross-worker
    interference and enforces the VFS security boundary.

    Args:
        worker_vfs_dir: Absolute path to the worker's dedicated VFS directory
            (e.g., '/path/to/vfs/floating_offshore_wind_lcoe').

    Returns:
        A list of LangChain file management tools scoped to the worker's directory.
    """
    return FileManagementToolkit(
        root_dir=worker_vfs_dir,
        # Only expose the tools the worker actually needs.
        # Workers write findings and read them back to compress — no delete/move needed.
        selected_tools=["write_file", "read_file", "list_directory"],
    ).get_tools()


# --- Search Tools ---


def get_search_tools() -> list[BaseTool]:
    """Get the Exa search tools for the research worker.

    Returns:
        A list containing the ExaSearchResults tool.
    """
    exa_search_results = ExaSearchResults(api_key=os.getenv("EXA_API_KEY"))
    return [exa_search_results]


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
