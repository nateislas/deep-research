"""Main graph definition for the deep research agent.

This module assembles the LangGraph by defining nodes for brief generation,
supervision, and research tasks, and connecting them with appropriate routing.
"""

from __future__ import annotations

import asyncio
import difflib
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
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from deep_research.prompts import (
    RESEARCH_INTAKE_PROMPT,
    SUPERVISOR_PROMPT,
)
from deep_research.state import (
    AddSubTopic,  # Used to add subtopics to the brief
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
    update_todo_list,
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
        # Load the todo list so we can display it's status in the prompt
        todo = TodoList.model_validate_json(Path(todo_path).read_text())
        todo_status_str = todo_to_string(todo)
    else:
        # todo list does not exists and need to have the LLM create one
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
    if not todo_list_exists:
        tools = [update_todo_list, ConductResearch, AddSubTopic]
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
    todo_update_calls = [tc for tc in tool_calls if tc["name"] == "update_todo_list"]

    tool_messages = []

    # Extract the thread-specific run root
    thread_id = config["configurable"].get("thread_id", "default")
    run_root = RESEARCH_ROOT / thread_id

    # Track which VFS directories workers write to, so generate_final_report
    # knows where to look. We start from whatever was accumulated in prior iterations.
    findings_paths = list(state.get("findings_paths") or [])
    todo_path = state.get("todo_list_path")

    # Handle the update_todo_list calls
    # This should be done first to handle the first iteration
    for tc in todo_update_calls:
        todo_data = TodoList(**tc["args"]["todo_data"])
        actual_path = str(run_root / "todo_list.json")
        run_root.mkdir(parents=True, exist_ok=True)

        # write the file via the utility tool
        result = update_todo_list.invoke(
            {
                "todo_data": todo_data,
                "todo_path_str": actual_path,
            }
        )

        todo_path = actual_path
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Handle AddSubTopic Calls
    if add_subtopic_calls and todo_path and Path(todo_path).exists():
        todo = TodoList.model_validate_json(Path(todo_path).read_text())
        from deep_research.utils import Task
        for tc in add_subtopic_calls:
            args = tc["args"]
            todo.tasks.append(
                Task(id=args["task_id"], description=args["new_sub_topic"])
            )
            tool_messages.append(
                ToolMessage(
                    content=f"Added sub-topic ID: '{args['task_id']}' — Reason: {args['rationale']}",
                    tool_call_id=tc["id"],
                )
            )
        Path(todo_path).write_text(todo.model_dump_json(indent=2))

    # Handle ConductResearch Calls
    # Limit to MAX_CONCURRENT_WORKERS; defer remaining calls to next round
    capped_tasks = conduct_research_calls[:MAX_CONCURRENT_WORKERS]
    deferred_tasks = conduct_research_calls[MAX_CONCURRENT_WORKERS:]

    # First, set up the VFS directories for all capped tasks
    for tc in capped_tasks:
        vfs_dir = run_root / tc["args"]["output_dirname"]
        vfs_dir.mkdir(parents=True, exist_ok=True)

    # build the list of awaitables
    # call .ainvoke() directly like open-deep-research
    research_tasks = [
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

    # run them all concurrently. gather() preserves the order!
    results = await asyncio.gather(*research_tasks)

    # TODO: START HERE TOMORROW

    # Process exactly like LangChain does: zip the results with the original calls
    for tc, _worker_result in zip(capped_tasks, results):
        args = tc["args"]
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
        tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        findings_paths.append(str(vfs_dir))
    # Mark deferred tasks
    for tc in deferred_tasks:
        tool_messages.append(
            ToolMessage(
                content=f"Deferred: '{tc['args']['sub_topic']}' — worker limit reached, will dispatch next iteration.",
                tool_call_id=tc["id"],
            )
        )
    # -----------------------------------------------------------------
    # Update TodoList
    # -----------------------------------------------------------------
    if capped_tasks and todo_path and Path(todo_path).exists():
        todo = TodoList.model_validate_json(Path(todo_path).read_text())
        # Create a normalized index of current tasks: {normalized_text: original_text}
        task_index = {re.sub(r"[^\w\s]", "", t.lower()).strip(): t for t in todo.tasks}

        for tc in capped_tasks:
            completed_topic = tc["args"]["sub_topic"]
            normalized_completed = re.sub(
                r"[^\w\s]", "", completed_topic.lower()
            ).strip()

            # Try exact match first, then fuzzy match
            if normalized_completed in task_index:
                matched_task = task_index[normalized_completed]
            else:
                matches = difflib.get_close_matches(
                    normalized_completed, list(task_index.keys()), n=1, cutoff=0.7
                )
                matched_task = task_index[matches[0]] if matches else None

            if matched_task and matched_task in todo.tasks:
                todo.tasks.remove(matched_task)
                todo.completed_tasks.append(matched_task)

        Path(todo_path).write_text(todo.model_dump_json(indent=2))
    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": tool_messages,
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
    # Temporarily returning END to avoid infinite loop before worker is implemented
    return Command(goto=END)


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
