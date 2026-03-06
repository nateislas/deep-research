"""This module defines the state schemas for the deep research agent's LangGraph.

It contains TypedDict definitions for the different phases of the research process,
including brief generation, supervisor coordination, and worker search loops.
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class BriefState(TypedDict):
    """State representing the initial research brief generation and approval process."""

    # Using AnyMessage b/c it allows us to be more flexible with the messages
    # add_messages is a "reducer" that appends new messages to the history.
    messages: Annotated[list[AnyMessage], add_messages]

    brief: str

    # Control the flow of the brief generation process, once marked approved, the brief generation process is complete
    # Options: "pending", "approved", "rejected"
    brief_status: Literal["pending", "approved", "rejected"]

    brief_path: str


# Don't need to use BaseModel here b/c we don't need strict typing and validation
# Also better b/c we don't need to update the entire object
class SupervisorState(TypedDict):
    """State management for the main research supervisor coordinating sub-tasks."""

    # Renamed to supervisor_messages to prevent collision with child or parent states
    supervisor_messages: Annotated[list[AnyMessage], add_messages]

    brief_path: str  # The path to the brief in the VFS

    # TODO: Will need to figure out what we want this to look like
    # Will it simply be rewriting the to-do list? Can we figure out a way to check items off?
    # Will probably use a json
    todo_list_path: str

    # Need this to track which workers are active
    active_tasks: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    """Isolated state for a research sub-agent performing specific search tasks."""

    brief_path: str
    worker_todo_list_path: str  # Local task list for the specific sub-topic

    # Every worker needs its own message history for its search loop
    # Renamed to researcher_messages so it can be isolated from the supervisor_messages
    researcher_messages: Annotated[list[AnyMessage], add_messages]
