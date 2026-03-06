"""State schemas for the deep research agent's LangGraph.

This module defines the TypedDict structures used for the parent graph,
research supervisor, and specialized research workers.
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    """The research brief for the deep research agent."""

    topic: str = Field(
        description="A concise headline and summary of the research subject."
    )
    main_objective: str = Field(
        description="The primary goal of the research. Define if the output should be exploratory (broad discovery) or confirmatory (validating a specific hypothesis)."
    )
    scope: str = Field(
        description="The boundaries of the search. Define what is explicitly included and what is excluded (e.g., timeline, geography, or specific industries)."
    )
    sub_objectives: list[str] = Field(
        description="A list of 5-15 granular, independent research questions. Each should be distinct enough to be handled by a separate sub-agent without overlapping with others."
    )

    brief_status: Literal["pending", "approved", "proposed"] = Field(default="pending")


# Using TypedDict for state because it allows for flexible updates without
# requiring strict Pydantic validation at every node transition.
# The GlobalState is the source of truth for the entire graph.
class GlobalState(TypedDict):
    """The root state for the entire deep research process."""

    # Core message history for user interaction
    messages: Annotated[list[AnyMessage], add_messages]

    # Shared research context for subgraphs
    # Including supervisor_messages here allows the parent to initialize the supervisor
    supervisor_messages: Annotated[list[AnyMessage], add_messages]

    # Brief generation status and outputs
    brief: ResearchBrief

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

    brief: ResearchBrief

    # TODO: Will need to figure out what we want this to look like
    # Will it simply be rewriting the to-do list? Can we figure out a way to check items off?
    # Will probably use a json
    todo_list_path: str

    # Need this to track which workers are active
    active_tasks: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    """Isolated state for a research sub-agent performing specific search tasks."""

    brief: ResearchBrief
    worker_todo_list_path: str  # Local task list for the specific sub-topic

    # Every worker needs its own message history for its search loop
    # Renamed to researcher_messages so it can be isolated from the supervisor_messages
    researcher_messages: Annotated[list[AnyMessage], add_messages]
