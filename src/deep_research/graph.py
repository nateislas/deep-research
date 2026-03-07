"""Main graph definition for the deep research agent.

This module assembles the LangGraph by defining nodes for brief generation,
supervision, and research tasks, and connecting them with appropriate routing.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_research.prompts import (
    RESEARCH_INTAKE_PROMPT,
    SUPERVISOR_PROMPT,
)
from deep_research.state import (
    AddSubTopic,
    ConductResearch,
    GlobalState,
    ResearchBrief,
    SupervisorState,
    WorkerState,
)
from deep_research.utils import (
    RESEARCH_ROOT,
    TodoList,
    brief_to_prompt_vars,
    get_findings_summary,
    todo_to_string,
    write_todo_list,
)

# Hard limit on supervisor iterations to prevent infinite research loops.
# If research isn't done in 10 rounds, something is wrong with the prompt or model.
MAX_ITERATIONS = 10

# How many workers the supervisor can spawn in a single round.
# This controls parallelism vs. cost. 3 is a safe default.
MAX_CONCURRENT_WORKERS = 3

# --- Nodes ---


async def research_intake(
    state: GlobalState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", END]]:
    """Handle the intake conversation and brief finalization."""
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
                    [
                        part["text"]
                        for part in m.content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                ).lower()
            break

    # 1. The brief is approved
    # Regex for whole-word positive phrases, avoiding simple substrings.
    approval_pattern = re.compile(
        r"\b(approve|approved|i approve|agree|yes|correct|looks good)\b"
    )
    # Basic negation check to ensure "don't approve" isn't caught.
    negation_pattern = re.compile(r"\b(don't|dont|not|never|no|stop|reject)\b")

    is_approved = bool(approval_pattern.search(last_user_msg)) and not bool(
        negation_pattern.search(last_user_msg)
    )

    if is_approved and state.get("brief"):
        # update the brief status to approved
        updated_brief = state["brief"].model_copy(update={"brief_status": "approved"})
        return Command(
            goto="supervisor",
            update={
                "brief": updated_brief,
                "todo_list_path": state.get("todo_list_path"),
                "supervisor_messages": [],  # Explicitly initialize supervisor history
                "active_tasks": [],  # Explicitly initialize task list
                "iteration_count": 0,  # Initialize loop counter
            },
        )
    # We need to clarify or propose a brief
    # initialize the model and bind the ResearchBrief as a structured tool
    model = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature=0.1,
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
                    ui_message,  # CodeRabbit: Do NOT persist the unresolved tool call 'response'
                ],
                "brief": new_brief,
            }
        )

    # 4. Standard Text Response (Clarification Question)
    return Command(update={"messages": [response]})


async def supervisor(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools", END]]:
    """Plan research strategy using tools to read the brief and todo list.

    The supervisor acts as a manager that pulls context from the VFS as needed.
    """
    # Max iterations to prevent infinite loops
    iter_count = state.get("iteration_count", 0)

    # If we've reached the max iterations, end the graph
    if iter_count >= MAX_ITERATIONS:
        return Command(
            goto=END,
            update={
                "supervisor_messages": [
                    AIMessage(
                        content="Maximum research iterations reached. Proceeding to report generation."
                    )
                ]
            },
        )

    # Initialize or load the todo list
    brief = state["brief"]

    # Extract the thread-specific run root
    thread_id = config["configurable"].get("thread_id", "default")
    run_root = RESEARCH_ROOT / thread_id
    run_root.mkdir(parents=True, exist_ok=True)

    # On the first iteration, we need to check if a todo list exists
    # If not we need to create one
    todo_path = state.get("todo_list_path")

    # If it's the first run, todo_path might be None.
    # Even if it exists from a previous run on this thread, we check disk.
    todo_list_exists = todo_path is not None and Path(todo_path).exists()

    if todo_list_exists:
        # Load the todo list so we can display its status in the prompt
        todo = TodoList.model_validate_json(Path(todo_path).read_text())
        todo_status_str = todo_to_string(todo)
    else:
        # todo list does not exist — need to have the LLM create one
        todo_status_str = (
            "No todo list created yet. You must create one using update_todo_list."
        )

    # Build the message history
    brief = state["brief"]

    # If there is not a message history with the supervisor yet, we're going to
    # inject the necessary context
    if not state["supervisor_messages"]:
        prompt_vars = brief_to_prompt_vars(brief)
        system_message = SystemMessage(
            content=SUPERVISOR_PROMPT.format(  # fill the prompt with the context
                **prompt_vars,
                todo_status=todo_status_str,
                findings_summary=get_findings_summary(run_root),
                max_concurrent_workers=MAX_CONCURRENT_WORKERS,
            )
        )
        messages = [system_message]
    else:
        messages = state["supervisor_messages"]

    # First iteration and need to create the list and spawn workers
    # The LLM sees update_todo_list as a tool schema, but we handle execution ourselves.
    if not todo_list_exists:
        tools = [TodoList, ConductResearch, AddSubTopic]
    else:
        tools = [ConductResearch, AddSubTopic]

    # invoke the LLM
    model = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    llm_with_tools = model.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)

    # route based on response
    if response.tool_calls:
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "iteration_count": iter_count + 1,
            },
        )
    else:
        # No tool calls = LLM determined research is complete
        return Command(
            goto=END,
            update={"supervisor_messages": [response]},
        )


# --- Supervisor Tools Helpers ---
# These keep supervisor_tools() readable by extracting each responsibility.


def _handle_todo_updates(
    todo_update_calls: list[dict],
    run_root: Path,
    todo_path: str | None,
) -> tuple[str | None, list[ToolMessage]]:
    """Process update_todo_list tool calls: parse, write JSON, return messages.

    Args:
        todo_update_calls: List of tool call dicts with name == "TodoList".
        run_root: Thread-scoped VFS root directory.
        todo_path: Current todo_list_path from state (may be None on first call).

    Returns:
        Tuple of (updated_todo_path, list_of_ToolMessages).
    """
    messages: list[ToolMessage] = []
    for tc in todo_update_calls:
        todo_data = TodoList(**tc["args"])
        actual_path = run_root / "todo_list.json"
        write_todo_list(todo_data, actual_path)
        todo_path = str(actual_path)
        messages.append(
            ToolMessage(
                content=f"Todo list updated at {actual_path}.",
                tool_call_id=tc["id"],
            )
        )
    return todo_path, messages


def _handle_subtopic_additions(
    add_subtopic_calls: list[dict],
    todo_path: str | None,
) -> list[ToolMessage]:
    """Append new sub-topics to the existing TodoList on disk.

    Args:
        add_subtopic_calls: List of tool call dicts with name == "AddSubTopic".
        todo_path: Path to the current todo_list.json.

    Returns:
        List of ToolMessages confirming each addition.
    """
    messages: list[ToolMessage] = []
    if not add_subtopic_calls or not todo_path or not Path(todo_path).exists():
        return messages

    todo = TodoList.model_validate_json(Path(todo_path).read_text())
    for tc in add_subtopic_calls:
        args = tc["args"]
        todo.tasks.append(args["new_sub_topic"])
        messages.append(
            ToolMessage(
                content=f"Added sub-topic: '{args['new_sub_topic']}' — Reason: {args['rationale']}",
                tool_call_id=tc["id"],
            )
        )
    Path(todo_path).write_text(todo.model_dump_json(indent=2))
    return messages


def _mark_tasks_completed(
    capped_tasks: list[dict],
    todo_path: str | None,
) -> None:
    """Mark completed worker tasks in the TodoList JSON.

    Uses simple case-insensitive comparison. The LLM writes both the todo task
    and the ConductResearch sub_topic in the same inference pass, so they are
    nearly always identical.

    Args:
        capped_tasks: The ConductResearch tool calls that were actually dispatched.
        todo_path: Path to the current todo_list.json.
    """
    if not capped_tasks or not todo_path or not Path(todo_path).exists():
        return

    todo = TodoList.model_validate_json(Path(todo_path).read_text())
    for tc in capped_tasks:
        completed = tc["args"]["sub_topic"].strip().lower()
        for task in todo.tasks:
            if task.strip().lower() == completed:
                todo.tasks.remove(task)
                todo.completed_tasks.append(task)
                break  # One match per completed topic
    Path(todo_path).write_text(todo.model_dump_json(indent=2))


# --- Supervisor Tools Node ---


async def supervisor_tools(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor"]]:
    """Execute research tasks by invoking the worker_subgraph in parallel."""
    # extract the tool calls from the LLM's last message
    last_ai_message = state["supervisor_messages"][-1]
    tool_calls = last_ai_message.tool_calls

    # Bin the tool calls by type so we can handle each category differently
    conduct_research_calls = [
        tc for tc in tool_calls if tc["name"] == "ConductResearch"
    ]
    add_subtopic_calls = [tc for tc in tool_calls if tc["name"] == "AddSubTopic"]
    todo_update_calls = [tc for tc in tool_calls if tc["name"] == "TodoList"]

    # Extract the thread-specific run root
    thread_id = config["configurable"].get("thread_id", "default")
    run_root = RESEARCH_ROOT / thread_id

    # Track which VFS directories workers write to
    findings_paths = list(state.get("findings_paths") or [])
    todo_path = state.get("todo_list_path")

    # 1. Handle TodoList updates (must be first for the initial iteration)
    todo_path, todo_msgs = _handle_todo_updates(todo_update_calls, run_root, todo_path)

    # 2. Handle AddSubTopic calls
    subtopic_msgs = _handle_subtopic_additions(add_subtopic_calls, todo_path)

    # 3. Dispatch workers
    # Limit to MAX_CONCURRENT_WORKERS; defer remaining calls to next round
    capped_tasks = conduct_research_calls[:MAX_CONCURRENT_WORKERS]
    deferred_tasks = conduct_research_calls[MAX_CONCURRENT_WORKERS:]

    # Set up VFS directories for capped tasks
    for tc in capped_tasks:
        vfs_dir = run_root / tc["args"]["output_dirname"]
        vfs_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of awaitables — call .ainvoke() directly like open-deep-research
    research_coroutines = [
        worker_subgraph.ainvoke(
            {
                "brief": state["brief"],
                "worker_todo_list_path": str(
                    run_root / tc["args"]["output_dirname"] / "worker_todo.json"
                ),
                "researcher_messages": [
                    SystemMessage(
                        content="You are a research worker. Search for information on your assigned topic and write findings to the VFS."
                    ),
                    HumanMessage(
                        content=(
                            f"Sub-topic to research: {tc['args']['sub_topic']}\n"
                            f"Additional context: {tc['args'].get('context') or 'None'}\n\n"
                            f"Save your findings to these files in your VFS directory:\n"
                            f"  - raw_content.md      (full search results)\n"
                            f"  - compressed_summary.md (your distilled key findings)"
                        )
                    ),
                ],
            }
        )
        for tc in capped_tasks
    ]

    # Run them all concurrently. gather() preserves the order!
    results = await asyncio.gather(*research_coroutines, return_exceptions=True)

    # 4. Process results
    worker_msgs: list[ToolMessage] = []
    for tc, result in zip(capped_tasks, results):
        args = tc["args"]
        if isinstance(result, Exception):
            worker_msgs.append(
                ToolMessage(
                    content=f"Worker failed on '{args['sub_topic']}': {result!s}",
                    tool_call_id=tc["id"],
                )
            )
            continue
        vfs_dir = run_root / args["output_dirname"]
        summary_path = vfs_dir / "compressed_summary.md"
        if summary_path.exists():
            snippet = summary_path.read_text()[:800]
            content = (
                f"Worker completed: '{args['sub_topic']}'\n\nKey Findings:\n{snippet}"
            )
        else:
            content = (
                f"Worker completed: '{args['sub_topic']}' — no summary file found."
            )
        worker_msgs.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        findings_paths.append(str(vfs_dir))

    # 5. Deferred task messages
    deferred_msgs = [
        ToolMessage(
            content=f"Deferred: '{tc['args']['sub_topic']}' — worker limit reached, will dispatch next iteration.",
            tool_call_id=tc["id"],
        )
        for tc in deferred_tasks
    ]

    # 6. Mark completed tasks in the TodoList
    _mark_tasks_completed(capped_tasks, todo_path)

    # Combine all tool messages in order
    all_tool_messages = todo_msgs + subtopic_msgs + worker_msgs + deferred_msgs

    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": all_tool_messages,
            "findings_paths": findings_paths,
            "todo_list_path": todo_path,
        },
    )


# --- Supervisor Subgraph Construction ---


supervisor_builder = StateGraph(SupervisorState)

supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.add_edge(START, "supervisor")
# Routing is now handled inside the nodes via Command

supervisor_subgraph = supervisor_builder.compile()


# --- Worker Subgraph ---


async def worker(
    state: WorkerState,
    config: RunnableConfig,
) -> Command[Literal["worker_tools", END]]:
    """Perform specialized research using search tools."""
    # TODO: Implement the real worker logic (search + file writing)
    raise NotImplementedError("worker node logic not implemented")


async def worker_tools(
    state: WorkerState,
    config: RunnableConfig,
) -> Command[Literal["worker"]]:
    """Execute search tools and write raw findings to VFS."""
    # TODO: Implement Tavily/Exa tool calls
    return Command(goto="worker")


worker_builder = StateGraph(WorkerState)

worker_builder.add_node("worker", worker)
worker_builder.add_node("worker_tools", worker_tools)

worker_builder.add_edge(START, "worker")
# Routing is now handled inside the nodes via Command

worker_subgraph = worker_builder.compile()


async def generate_final_report(
    state: GlobalState,
    config: RunnableConfig,
) -> Command[Literal[END]]:
    """Synthesize all compressed findings into a final report."""
    # TODO: Implement report synthesis based on VFS content
    return Command(goto=END, update={"final_report": "Report content..."})


# --- Main Graph Construction ---


deep_research_builder = StateGraph(GlobalState)

deep_research_builder.add_node("research_intake", research_intake)
deep_research_builder.add_node("supervisor", supervisor_subgraph)
deep_research_builder.add_node("generate_final_report", generate_final_report)

deep_research_builder.add_edge(START, "research_intake")
deep_research_builder.add_edge("supervisor", "generate_final_report")
# Note: If a node returns a Command with a 'goto', it overrides these edges.
# But we still define the entry point with START.

# Compile the graph
graph = deep_research_builder.compile(name="Deep Research Graph")
