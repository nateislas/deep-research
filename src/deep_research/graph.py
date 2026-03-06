"""Main graph definition for the deep research agent.

This module assembles the LangGraph by defining nodes for brief generation,
supervision, and research tasks, and connecting them with appropriate routing.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langchain.chat_models import init_chat_model

from deep_research.state import (
    GlobalState,
    SupervisorState,
    WorkerState,
    ResearchBrief,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)

from deep_research.prompts import RESEARCH_INTAKE_PROMPT


# --- Nodes ---


async def research_intake(
    state: GlobalState,
) -> Command[Literal["supervisor", "__end__"]]:
    """Handles the intake conversation and brief finalization."""
    messages = state["messages"]
    last_user_msg = ""
    # Extract last user message content safely (handle lists for multimodal models)
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            if isinstance(m.content, str):
                last_user_msg = m.content.lower()
            elif isinstance(m.content, list):
                # Join text parts if it's a list (multimodal format)
                last_user_msg = " ".join(
                    [part["text"] for part in m.content if isinstance(part, dict) and part.get("type") == "text"]
                ).lower()
            break

    # 1. The brief is approved
    if ("approve" in last_user_msg or "approved" in last_user_msg) and state.get(
        "brief"
    ):
        # update the brief status to approved
        updated_brief = state["brief"].model_copy(update={"brief_status": "approved"})
        return Command(
            goto="supervisor",
            update={
                "brief": updated_brief,
                "todo_list_path": "research/todo.json",
                "supervisor_messages": [],  # Explicitly initialize supervisor history
                "active_tasks": [],  # Explicitly initialize task list
            },
        )
    # We need to clarify or propose a brief
    # initialize the model and bind the ResearchBrief as a structured tool
    model = init_chat_model(
        model="gpt-5-nano", model_provider="openai", temperature=0.1
    )

    # want to use bind_tools instead of bind_structured_tool
    # bind_structured_tool would require the model to respond with a ResearchBrief
    llm_with_brief_tool = model.bind_tools([ResearchBrief])

    # Prepend the system prompt to the history
    system_message = SystemMessage(content=RESEARCH_INTAKE_PROMPT)
    # Filter messages to avoid context overflow if conversation gets very long
    history = [system_message] + messages
    response = await llm_with_brief_tool.ainvoke(history)

    # 3. Process the Output
    # If the model called the ResearchBrief tool to propose a plan
    if response.tool_calls:
        brief_args = response.tool_calls[0]["args"].copy()
        # Ensure we don't have duplicate status arguments
        brief_args.pop("brief_status", None)

        # create a new ResearchBrief object with status "proposed"
        new_brief = ResearchBrief(**brief_args, brief_status="proposed")

        # Extract the assistant's thinking/text if it exists, otherwise use a default
        instruction = "I've drafted a research plan. Please review the objectives and scope below. Type **'Approve'** to start research, or let me know if you'd like to adjust it."

        # If the LLM included text (thinking/rationale) in the response content,
        # we keep it so the user sees the reasoning.
        if (
            response.content
            and isinstance(response.content, str)
            and len(response.content.strip()) > 0
        ):
            ui_message = AIMessage(content=f"{response.content}\n\n{instruction}")
        else:
            ui_message = AIMessage(content=instruction)

        return Command(
            update={
                "messages": [
                    response,  # Keep the tool call for history
                    ui_message,  # Present the thinking + instruction
                ],
                "brief": new_brief,
            }
        )

    # 4. Standard Text Response (Clarification Question)
    return Command(update={"messages": [response]})


async def supervisor(
    state: SupervisorState,
) -> Command[Literal["supervisor_tools", "__end__"]]:
    """Plan research strategy using tools to read the brief and todo list.

    The supervisor acts as a manager that pulls context from the VFS as needed.
    """
    # TODO: Implement Supervisor reasoning logic (GPT-1o-mini?)
    # TODO: Add exit check for when brief is fully addressed or max iterations reached
    # if research_is_complete(state):
    #     return Command(goto=END)

    return Command(goto="supervisor_tools")


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor"]]:
    """Execute research tasks by invoking the researcher_subgraph in parallel."""
    # TODO: Implement logic to:
    # 1. Parse 'ConductResearch' tool calls from the supervisor
    # 2. Call researcher_subgraph.ainvoke() for each task in parallel.
    #    This must build a full WorkerState (brief_path, worker_todo_list_path, researcher_messages).
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
    # TODO: Add exit check for when sub-topic is fully researched
    # if task_completed(state):
    #     return Command(goto=END)

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

deep_research_builder.add_node("research_intake", research_intake)
deep_research_builder.add_node("supervisor", supervisor_subgraph)
deep_research_builder.add_node("generate_final_report", generate_final_report)

deep_research_builder.add_edge(START, "research_intake")
# Note: If a node returns a Command with a 'goto', it overrides these edges.
# But we still define the entry point with START.

# Compile the graph
graph = deep_research_builder.compile(name="Deep Research Graph")
