"""Main graph definition for the deep research agent.

This module assembles the LangGraph by defining nodes for brief generation,
supervision, and research tasks, and connecting them with appropriate routing.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_research.state import GlobalState, SupervisorState, WorkerState


# --- Nodes ---


async def generate_brief(
    state: GlobalState,
) -> Command[Literal["supervisor", "__end__"]]:
    """Generate and refine the research brief.

    When approved, this node triggers the research phase by ensuring the supervisor
    has access to the VFS paths for the brief and the todo list.
    """
    # TODO: Implement LLM logic for brief generation
    if state.get("brief_status") == "approved":
        return Command(
            goto="supervisor",
            update={
                "brief_path": state.get("brief_path"),
                "todo_list_path": state.get("todo_list_path"),
            },
        )
    return Command(goto=END)


async def supervisor(
    state: SupervisorState,
) -> Command[Literal["supervisor_tools", "__end__"]]:
    """Plan research strategy using tools to read the brief and todo list.

    The supervisor acts as a manager that pulls context from the VFS as needed.
    """
    # TODO: Implement Supervisor reasoning logic (GPT-1o-mini?)
    return Command(goto="supervisor_tools")


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor"]]:
    """Execute research tasks by invoking the researcher_subgraph in parallel."""
    # TODO: Implement logic to:
    # 1. Parse 'ConductResearch' tool calls from the supervisor
    # 2. Call researcher_subgraph.ainvoke() for each task in parallel
    # 3. Save findings to VFS and update the supervisor state
    return Command(goto="supervisor")


# --- Supervisor Subgraph Construction ---


supervisor_builder = StateGraph(SupervisorState)

supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.add_edge(START, "supervisor")
# Routing is now handled inside the nodes via Command

supervisor_subgraph = supervisor_builder.compile()


# --- Worker Subgraph ---


async def worker(state: WorkerState) -> Command[Literal["worker_tools", "__end__"]]:
    """Perform specialized research using search tools."""
    # TODO: Implement Worker search loop logic (GPT-5-nano?)
    return Command(goto="worker_tools")


async def worker_tools(state: WorkerState) -> Command[Literal["worker"]]:
    """Execute search tools and write raw findings to VFS."""
    # TODO: Implement Tavily/Exa tool calls
    return Command(goto="worker")


worker_builder = StateGraph(WorkerState)

worker_builder.add_node("worker", worker)
worker_builder.add_node("worker_tools", worker_tools)

worker_builder.add_edge(START, "worker")
# Routing is now handled inside the nodes via Command

researcher_subgraph = worker_builder.compile()


async def generate_final_report(state: GlobalState) -> Command[Literal["__end__"]]:
    """Synthesize all compressed findings into a final report."""
    # TODO: Implement report synthesis based on VFS content
    return Command(goto=END, update={"final_report": "Report content..."})


# --- Main Graph Construction ---


deep_research_builder = StateGraph(GlobalState)

deep_research_builder.add_node("generate_brief", generate_brief)
deep_research_builder.add_node("supervisor", supervisor_subgraph)
deep_research_builder.add_node("generate_final_report", generate_final_report)

deep_research_builder.add_edge(START, "generate_brief")
# Note: If a node returns a Command with a 'goto', it overrides these edges.
# But we still define the entry point with START.

# Compile the graph
deep_research_graph = deep_research_builder.compile(name="Deep Research Graph")
