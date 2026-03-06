"""Utility functions for the deep research agent."""

import os
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_exa import ExaSearchResults, ExaFindSimilarResults
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from pathlib import Path


class TodoList(BaseModel):
    """The todo list for the research agent."""

    tasks: list[str] = Field(description="The full updated list of research tasks.")
    completed_tasks: list[str] = Field(default_factory=list)


# Constants for VFS management
RESEARCH_ROOT = Path("/Users/nathanielislas/CursorProjects/deep-research/vfs")


@tool
def update_todo_list(todo_data: TodoList, todo_path_str: str):
    """Updates the entire research todo list in the VFS.
    Use this to plan next steps or check off finished items.
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


# --- File Management Tools ---
def get_filesystem_tools(root_dir: str) -> list[BaseTool]:
    """Get the file management tools for the deep research agent."""
    file_tools = FileManagementToolkit(root_dir=root_dir).get_tools()
    return file_tools


# --- Search Tools ---
def get_search_tools() -> list[BaseTool]:
    """Get the search tools for the deep research agent."""
    # Will only use ExaSearchResults for now
    exa_search_results = ExaSearchResults(api_key=os.getenv("EXA_API_KEY"))
    return [exa_search_results]
