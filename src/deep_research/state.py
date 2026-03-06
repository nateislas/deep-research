"""State schemas for the deep research agent's LangGraph.

This module defines the TypedDict structures used for the parent graph,
research supervisor, and specialized research workers.
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


# Decided to use BaseModel b/c we need strict typing and validation, we also need
# to describe to the LLM what is absolutely required in the state, such as the brief
# also tool inspiration to add a "research_topic" as an attribute
# The GlobalState is the source of truth for the entire graph.
class GlobalState(TypedDict):
    """The root state for the entire deep research process."""

    # Core message history for user interaction
    messages: Annotated[list[AnyMessage], add_messages]

    # Shared research context for subgraphs
    # Including supervisor_messages here allows the parent to initialize the supervisor
    supervisor_messages: Annotated[list[AnyMessage], add_messages]

    # Brief generation status and outputs
    research_topic: str
    brief: str
    brief_status: Literal["pending", "approved", "rejected"]
    brief_path: str  # Path to the .md brief in VFS

    # Supervisor coordination
    todo_list_path: str  # Path to the .json todo list in VFS
    active_tasks: Annotated[list[str], operator.add]

    # Final outputs
    research_findings: list[str]
    final_report: str


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
