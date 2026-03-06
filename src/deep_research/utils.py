"""Utility functions for the deep research agent."""

import os
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_exa import ExaSearchResults, ExaFindSimilarResults
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class TodoList(BaseModel):
    """The todo list for the research agent."""

    tasks: list[str] = Field(description="The full updated list of research tasks.")
    completed_tasks: list[str] = Field(default_factory=list)


@tool
def update_todo_list(todo_data: TodoList, todo_path: str):
    """Updates the entire research todo list in the VFS.
    Use this to plan next steps or check off finished items.
    """

    # Logic to write todo_data.json() to todo_path
    with open(todo_path, "w") as f:
        f.write(todo_data.model_dump_json())
    return "Todo list updated successfully."


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
