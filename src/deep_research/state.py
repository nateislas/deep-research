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
class ResearchTask(BaseModel):
    """An individual research task to be performed by a worker."""

    task_id: int = Field(
        description="The integer ID of the specific pending task from the Todo List."
    )
    context: str = Field(
        default="",
        description=(
            "Optional strategic context or constraints to guide the worker's search "
            "and synthesis based on prior findings or specific requirements."
        ),
    )
    output_dirname: str = Field(
        description=(
            "The VFS directory name where the worker saves its findings. Use snake_case."
        )
    )


class ConductResearchBatch(BaseModel):
    """Dispatch MULTIPLE research workers simultaneously to investigate sub-topics.

    The workers will autonomously decide which Exa search queries to run.
    You MUST include every pending task from the Todo List up to the max parallel limit.
    """

    tasks: list[ResearchTask] = Field(
        description="A list of pending research tasks to dispatch to workers."
    )


class NewSubTopic(BaseModel):
    """A new sub-topic to be added to the research plan."""

    new_sub_topic: str = Field(
        description="A clear, human-readable description of the new research direction or thematic component."
    )
    rationale: str = Field(
        description="Why this sub-topic is worth investigating. Cite the worker finding that surfaced it."
    )


class AddSubTopicBatch(BaseModel):
    """Add MULTIPLE new sub-topics to the research todo list simultaneously.

    Use this when worker findings reveal important angles that the original
    ResearchBrief did not anticipate.
    """

    topics: list[NewSubTopic] = Field(
        description="A list of new sub-topics to add to the research plan."
    )


class ResearchBrief(BaseModel):
    """The research brief for the deep research agent."""

    topic: str = Field(
        description="A clear, multi-sentence summary of the research subject. Must be at least 2-3 sentences providing context on why this topic matters."
    )
    main_objective: str = Field(
        description="The primary goal of the research. What specific overarching question are we trying to answer?"
    )
    scope: str = Field(
        description="The formal search boundaries. Define the specific parameters for inclusion and exclusion."
    )
    sub_objectives: list[str] = Field(
        description="A list of 5-10 foundational, actionable search directives. These must be empirical and searchable categories of information. Do not include methodological instructions.",
        min_items=5,
        max_items=10,
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
    todo_list_path: NotRequired[str]  # Path to the .json todo list in VFS

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

    # Track VFS directories containing worker findings

    # Track VFS directories containing worker findings
    findings_paths: NotRequired[Annotated[list[str], operator.add]]

    # Counter for research iterations to prevent infinite loops
    iteration_count: int


class WorkerState(TypedDict):
    """Isolated state for a research sub-agent performing specific search tasks."""

    brief: ResearchBrief
    output_dirname: str  # The directory name for this worker's findings
    run_root: str  # The base VFS path for this thread

    # Every worker needs its own message history for its search loop
    # Renamed to researcher_messages so it can be isolated from the supervisor_messages
    researcher_messages: Annotated[list[AnyMessage], add_messages]
