"""State schemas for the deep research agent's LangGraph.

This module defines the TypedDict structures used for the parent graph,
research supervisor, and specialized research workers.
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


# Structured data
class ConductResearch(BaseModel):
    """Dispatch a research worker to investigate a specific sub-topic.

    The worker will autonomously decide which Exa search queries to run
    and how many iterations to perform. You provide the what, not the how.
    """

    sub_topic: str = Field(
        description="The specific sub-topic or research question for the worker to investigate. Be precise and focused."
    )
    context: str = Field(
        default="",
        description=(
            "Optional strategic context for the worker. E.g., 'A previous worker found X — "
            "dig deeper into Y' or 'Focus only on post-2022 data'. Leave empty if no special "
            "direction is needed."
        ),
    )
    output_dirname: str = Field(
        description=(
            "The VFS directory name where the worker saves its findings. Use snake_case and be "
            "descriptive (e.g., 'floating_offshore_wind_lcoe'). Worker will create "
            "raw_content.md and compressed_summary.md inside this directory."
        )
    )


class AddSubTopic(BaseModel):
    """Add a new sub-topic to the research todo list.

    Use this when worker findings reveal an important angle that the original
    ResearchBrief did not anticipate.
    """

    new_sub_topic: str = Field(
        description="The new sub-topic to add to the research plan."
    )
    rationale: str = Field(
        description="Why this sub-topic is worth investigating. Cite the worker finding that surfaced it."
    )


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
        description="A list of 5-15 granular, independent research questions. Each should be distinct enough to be handled by a separate sub-agent without overlapping with others.",
        min_items=5,
        max_items=15,
    )

    brief_status: Literal["pending", "approved", "proposed"] = Field(default="pending")


# States


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
    brief: NotRequired[ResearchBrief]

    # Supervisor coordination
    todo_list_path: NotRequired[str]  # Path to the .json todo list in VFS
    active_tasks: NotRequired[Annotated[list[str], operator.add]]

    # Track VFS directories containing worker findings
    findings_paths: NotRequired[Annotated[list[str], operator.add]]

    # Final outputs
    research_findings: NotRequired[list[str]]
    final_report: NotRequired[str]


# Don't need to use BaseModel here b/c we don't need strict typing and validation
# Also better b/c we don't need to update the entire object
class SupervisorState(TypedDict):
    """State management for the main research supervisor coordinating sub-tasks."""

    # Renamed to supervisor_messages to prevent collision with child or parent states
    supervisor_messages: Annotated[list[AnyMessage], add_messages]

    brief: ResearchBrief

    # Path to the .json todo list in VFS.
    # NotRequired because it's None on the first supervisor iteration before the LLM creates one.
    todo_list_path: NotRequired[str]

    # Need this to track which workers are active
    active_tasks: Annotated[list[str], operator.add]

    # Track VFS directories containing worker findings
    findings_paths: NotRequired[Annotated[list[str], operator.add]]

    # Counter for research iterations to prevent infinite loops
    iteration_count: int


class WorkerState(TypedDict):
    """Isolated state for a research sub-agent performing specific search tasks."""

    brief: ResearchBrief
    output_dirname: str  # The directory name for this worker's findings
    run_root: str        # The base VFS path for this thread

    # Every worker needs its own message history for its search loop
    # Renamed to researcher_messages so it can be isolated from the supervisor_messages
    researcher_messages: Annotated[list[AnyMessage], add_messages]
