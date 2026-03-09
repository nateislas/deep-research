"""Main graph definition for the deep research agent.

This module assembles the LangGraph by defining nodes for brief generation,
supervision, and research tasks, and connecting them with appropriate routing.
"""

from __future__ import annotations

import asyncio
import datetime
import re
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from deep_research.prompts import (
    REPORT_SYNTHESIS_PROMPT,
    RESEARCH_INTAKE_PROMPT,
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
)
from deep_research.state import (
    AddSubTopicBatch,
    ConductResearchBatch,
    GlobalState,
    ResearchBrief,
    SupervisorState,
    WorkerState,
)
from deep_research.tools import (
    get_search_tools,
    get_worker_filesystem_tools,
)
from deep_research.utils import (
    RESEARCH_ROOT,
    TodoList,
    brief_to_prompt_vars,
    get_findings_summary,
    handle_subtopic_additions,
    handle_todo_updates,
    mark_tasks_completed,
    todo_to_string,
)

# Hard limit on supervisor iterations to prevent infinite research loops.
# If research isn't done in 15 rounds, something is wrong with the prompt or model.
MAX_ITERATIONS = 15

# How many workers the supervisor can spawn in a single round.
# 10 allows for high throughput during complex research tasks.
MAX_CONCURRENT_WORKERS = 10

# --- Nodes ---


async def research_intake(
    state: GlobalState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "__end__"]]:
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
                "messages": [
                    AIMessage(
                        content=f"Research Brief Approved. Starting deep research phase for: *{updated_brief.topic}*."
                    )
                ],
                "supervisor_messages": [],  # Explicitly initialize supervisor history
                "iteration_count": 0,  # Initialize loop counter
            },
        )
    # We need to clarify or propose a brief
    # initialize the model and bind the ResearchBrief as a structured tool
    # gpt-4.1-mini: sharp enough to infer intent and propose a ResearchBrief;
    # fast and cheap for the conversational intake loop.
    model = init_chat_model(
        model="gpt-5-nano", model_provider="openai", reasoning_effort="low"
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
) -> Command[Literal["supervisor", "supervisor_tools", "__end__"]]:
    """Plan research strategy using tools to read the brief and todo list.

    The supervisor acts as a manager that pulls context from the VFS as needed.
    """
    # Max iterations to prevent infinite loops
    iter_count = state.get("iteration_count", 0)

    # If we've reached the max iterations, end the graph
    if iter_count >= MAX_ITERATIONS:
        return Command(
            goto="__end__",
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
        assert todo_path is not None
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

    # Always regenerate the system message on every iteration so it sees
    # the LATEST todo_status and findings_summary
    prompt_vars = brief_to_prompt_vars(brief)
    system_message = SystemMessage(
        content=SUPERVISOR_PROMPT.format(
            **prompt_vars,
            todo_status=todo_status_str,
            findings_summary=get_findings_summary(run_root),
            max_concurrent_workers=MAX_CONCURRENT_WORKERS,
        )
    )

    if not state["supervisor_messages"]:
        messages: list[AnyMessage] = [system_message]
        # Also notify the user in the main chat
        main_chat_update = [
            AIMessage(
                content="Supervisor is analyzing the research brief and preparing the initial task list..."
            )
        ]
    else:
        # Update the existing memory by replacing the old system message (first element)
        # with the freshly formatted one containing new state.
        messages = list(state["supervisor_messages"])
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = system_message
        else:
            messages.insert(0, system_message)
        main_chat_update = []

    # First iteration and need to create the list and spawn workers
    # The LLM sees update_todo_list as a tool schema, but we handle execution ourselves.
    if not todo_list_exists:
        tools = [TodoList, ConductResearchBatch, AddSubTopicBatch]
    else:
        tools = [ConductResearchBatch, AddSubTopicBatch]

    # invoke the LLM
    # o3-mini: strongest available planner for decomposing the ResearchBrief
    # and orchestrating workers. Called only ~5-10 times per run, so cost is fine.
    model = init_chat_model(
        model="o3-mini", model_provider="openai", reasoning_effort="high"
    )
    llm_with_tools = model.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)

    # route based on response
    if response.tool_calls:
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "messages": main_chat_update,
                "iteration_count": iter_count + 1,
            },
        )
    else:
        # No tool calls — check if there are still pending tasks before allowing exit.
        # This prevents small models from quitting prematurely after a TodoList write.
        if todo_path and Path(todo_path).exists():
            todo = TodoList.model_validate_json(Path(todo_path).read_text())
            if todo.tasks:  # still has pending items
                # Inject a corrective nudge and loop back to supervisor
                nudge = HumanMessage(
                    content=(
                        f"You have {len(todo.tasks)} pending research tasks remaining. "
                        "You must call ConductResearch for each one before declaring done. "
                        "Do NOT stop until all tasks are completed."
                    )
                )
                return Command(
                    goto="supervisor",
                    update={
                        "supervisor_messages": [response, nudge],
                        "iteration_count": iter_count + 1,
                    },
                )

        # All tasks complete — LLM correctly determined research is done
        return Command(
            goto="__end__",
            update={
                "supervisor_messages": [response],
                "messages": [
                    AIMessage(
                        content="Research complete. I have gathered all necessary data from the workers. Now, I am synthesizing everything into a detailed final report..."
                    )
                ],
            },
        )


# --- Supervisor Tools Node ---


async def supervisor_tools(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor"]]:
    """Execute research tasks by invoking the worker_subgraph in parallel."""
    # extract the tool calls from the LLM's last message
    last_ai_message = state["supervisor_messages"][-1]
    if not isinstance(last_ai_message, AIMessage):
        tool_calls = getattr(last_ai_message, "tool_calls", [])
    else:
        tool_calls = last_ai_message.tool_calls

    # Bin the tool calls by type so we can handle each category differently
    # Bin the tool calls by type
    batch_research_calls = [
        tc for tc in tool_calls if tc["name"] == "ConductResearchBatch"
    ]
    add_subtopic_calls = [tc for tc in tool_calls if tc["name"] == "AddSubTopicBatch"]
    todo_update_calls = [tc for tc in tool_calls if tc["name"] == "TodoList"]

    # Extract the thread-specific run root
    thread_id = config["configurable"].get("thread_id", "default")
    run_root = RESEARCH_ROOT / thread_id

    # Track VFS directories workers write to
    new_findings_paths = []
    todo_path = state.get("todo_list_path")

    # 1. Handle TodoList updates
    todo_path, todo_msgs = handle_todo_updates(todo_update_calls, run_root, todo_path)

    # 2. Handle AddSubTopic calls
    subtopic_msgs = handle_subtopic_additions(add_subtopic_calls, todo_path)

    # 3. Flatten and Dispatch Workers from ConductResearchBatch
    # We collect ALL tasks from all batch calls, then cap them.
    all_task_items = []
    for bc in batch_research_calls:
        for t_args in bc["args"]["tasks"]:
            all_task_items.append({"args": t_args, "batch_id": bc["id"]})

    capped_tasks = all_task_items[:MAX_CONCURRENT_WORKERS]

    # Setup variables for workers
    worker_inputs = []
    if todo_path and Path(todo_path).exists():
        todo = TodoList.model_validate_json(Path(todo_path).read_text())
    else:
        todo = None

    for item in capped_tasks:
        args = item["args"]
        vfs_dir = run_root / args["output_dirname"]
        vfs_dir.mkdir(parents=True, exist_ok=True)

        task_id = args["task_id"]
        sub_topic = f"Task ID {task_id}"
        if todo:
            for t in todo.tasks:
                if t.id == task_id:
                    sub_topic = t.task
                    break

        worker_inputs.append(
            {
                "task_id": task_id,
                "sub_topic": sub_topic,
                "args": args,
                "batch_id": item["batch_id"],
            }
        )

    # Build the list of awaitables
    research_coroutines = [
        worker_subgraph.ainvoke(
            {
                "brief": state["brief"],
                "run_root": str(run_root),
                "output_dirname": item["args"]["output_dirname"],
                "researcher_messages": [
                    SystemMessage(content=RESEARCHER_PROMPT),
                    HumanMessage(
                        content=(
                            f"## OVERARCHING PROJECT CONTEXT\n"
                            f"**Project Topic:** {state['brief'].topic}\n"
                            f"**Main Objective:** {state['brief'].main_objective}\n\n"
                            f"--- \n\n"
                            f"## YOUR SPECIFIC TASK\n"
                            f"**Sub-topic to research:** {item['sub_topic']}\n"
                            f"**Additional context:** {item['args'].get('context') or 'None'}\n\n"
                            f"1. Run 2-3 targeted searches.\n"
                            f"2. Write your distilled findings to 'compressed_summary.md' using the write_file tool.\n"
                            f"Do NOT finish until you have written your summary file."
                        )
                    ),
                ],
            },  # type: ignore
            config=config,
        )
        for item in worker_inputs
    ]

    # Run them all concurrently
    results = await asyncio.gather(*research_coroutines, return_exceptions=True)

    # 4. Process results & Mark completed topics
    # We aggregate outcomes by batch_id
    batch_outcomes: dict[str, list[str]] = {bc["id"]: [] for bc in batch_research_calls}
    completed_task_ids: list[int] = []

    for item, result in zip(worker_inputs, results):
        task_id = item["task_id"]
        sub_topic = item["sub_topic"]
        batch_id = item["batch_id"]

        if isinstance(result, Exception):
            batch_outcomes[batch_id].append(f"Task {task_id} FAILED: {result!s}")
            continue

        worker_output_dir = run_root / item["args"]["output_dirname"]
        summary_file = worker_output_dir / "compressed_summary.md"
        if not summary_file.exists():
            batch_outcomes[batch_id].append(
                f"Task {task_id} FAILED: Missing compressed_summary.md from worker."
            )
            continue

        batch_outcomes[batch_id].append(
            f"Task {task_id} COMPLETED: {sub_topic[:40]}..."
        )
        completed_task_ids.append(task_id)
        new_findings_paths.append(str(run_root / item["args"]["output_dirname"]))

    # Build ToolMessages for the batches
    batch_msgs = [
        ToolMessage(
            content="\n".join(outcomes) or "No tasks processed in this batch.",
            tool_call_id=bid,
        )
        for bid, outcomes in batch_outcomes.items()
    ]

    # 6. Mark successful tasks in the TodoList
    mark_tasks_completed(completed_task_ids, todo_path)

    # Combine all tool messages in order
    all_tool_messages = todo_msgs + subtopic_msgs + batch_msgs

    # Catch any unhandled tool calls to prevent OpenAI BadRequestError
    handled_ids = {m.tool_call_id for m in all_tool_messages}
    unhandled_msgs = [
        ToolMessage(
            content=f"Error: Unknown tool or unhandled tool call {tc['name']}",
            tool_call_id=tc["id"],
        )
        for tc in tool_calls
        if tc["id"] not in handled_ids
    ]
    all_tool_messages += unhandled_msgs

    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": all_tool_messages,
            "findings_paths": new_findings_paths,
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
) -> Command[Literal["worker_tools", "__end__"]]:
    """Perform specialized research using search tools."""
    # 1. Derive the worker's working directory
    worker_dir = str(Path(state["run_root"]) / state["output_dirname"])

    # 2. Initialize tools — no vfs_path binding needed here.
    # The actual injection happens in worker_tools when tools are invoked.
    search_tools = get_search_tools()
    fs_tools = get_worker_filesystem_tools(worker_dir)
    all_tools = search_tools + fs_tools

    # 3. Initialize Model & Bind Tools
    # gpt-5-nano w/ low reasoning_effort: nearly free ($0.05/1M in), fast,
    # and capable enough for targeted search + synthesis tasks.
    model = init_chat_model(
        model="gpt-5-nano",
        model_provider="openai",
        reasoning_effort="low",
    )
    # We bind all tools so the worker can search OR write files
    llm_with_tools = model.bind_tools(all_tools)

    # 4. Invoke LLM
    # Note: The initial prompt is already in state["researcher_messages"]
    # from the supervisor's dispatch.
    messages = state["researcher_messages"]
    response = await llm_with_tools.ainvoke(messages)

    # 5. Routing Logic
    if response.tool_calls:
        return Command(goto="worker_tools", update={"researcher_messages": [response]})

    # No tool calls means the worker is satisfied with its findings
    return Command(goto="__end__")


async def worker_tools(
    state: WorkerState,
    config: RunnableConfig,
) -> Command[Literal["worker"]]:
    """Execute search tools and write raw findings to VFS."""
    last_msg = state["researcher_messages"][-1]
    if not isinstance(last_msg, AIMessage):
        tool_calls = getattr(last_msg, "tool_calls", [])
    else:
        tool_calls = last_msg.tool_calls

    worker_dir = str(Path(state["run_root"]) / state["output_dirname"])

    tools_map = {
        t.name: t
        for t in (get_search_tools() + get_worker_filesystem_tools(worker_dir))
    }

    tool_messages = []
    for tc in tool_calls:
        tool = tools_map.get(tc["name"])
        if tool:
            args = dict(tc["args"])
            # Inject vfs_path into exa_search calls so raw content is auto-logged.
            # We do it here (at invocation time) because functools.partial doesn't
            # work with StructuredTool's _arun dispatch path.
            if tc["name"] == "exa_search":
                args["vfs_path"] = worker_dir

            # Retry transient failures (e.g. rate limits, timeouts) up to 2 times.
            max_attempts = 3
            last_error: Exception | None = None
            result = None
            for attempt in range(max_attempts):
                try:
                    result = await tool.ainvoke(args)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(3 * (attempt + 1))  # 3s, then 6s

            if last_error is not None:
                result = f"Tool error after {max_attempts} attempts: {last_error!s}"

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tc['name']} not found.", tool_call_id=tc["id"]
                )
            )

    return Command(goto="worker", update={"researcher_messages": tool_messages})


worker_builder = StateGraph(WorkerState)

worker_builder.add_node("worker", worker)
worker_builder.add_node("worker_tools", worker_tools)

worker_builder.add_edge(START, "worker")
# Routing is now handled inside the nodes via Command

worker_subgraph = worker_builder.compile()


# --- Report Synthesis ---


async def generate_final_report(
    state: GlobalState,
    config: RunnableConfig,
) -> Command[Literal["__end__"]]:
    """Synthesize all compressed findings into a final report."""
    # 1. Prepare data
    thread_id = config["configurable"].get("thread_id", "default")
    run_root = RESEARCH_ROOT / thread_id
    brief = state["brief"]
    findings_summary = get_findings_summary(run_root)

    # 1a. Load the todo list for extra context
    todo_path = run_root / "todo_list.json"
    completed_tasks_str = "No tasks completed yet."
    if todo_path.exists():
        try:
            todo = TodoList.model_validate_json(todo_path.read_text())
            completed_tasks_str = "\n".join(
                [f"- [x] {t.task}" for t in todo.completed_tasks]
            )
        except Exception:
            pass

    # 1b. Format user messages (language matching)
    # Filter out short boilerplate messages like "approve", "yes", etc.
    approval_keywords = {"approve", "approved", "yes", "correct", "looks good", "agree"}
    user_inputs = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            content = msg.content.strip().lower()
            # Only include if it's not a simple approval or if it's longer than a few words
            if content not in approval_keywords or len(content.split()) > 3:
                user_inputs.append(msg.content)

    user_messages_str = (
        "\n".join(user_inputs) if user_inputs else "(No initial context provided.)"
    )

    # 2. Format the system prompt
    system_prompt = REPORT_SYNTHESIS_PROMPT.format(
        date=datetime.date.today().strftime("%B %d, %Y"),
        user_messages=user_messages_str,
        topic=brief.topic,
        main_objective=brief.main_objective,
        scope=brief.scope,
        sub_objectives="\n".join([f"- {obj}" for obj in brief.sub_objectives]),
        completed_tasks=completed_tasks_str,
        findings_summary=findings_summary,
    )

    # 3. Initialize Model
    model = init_chat_model(
        model="o3-mini",
        model_provider="openai",
        reasoning_effort="high",
    )

    # 4. Invoke LLM with just the system prompt and a human message to kick it off
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content="Begin writing the comprehensive final report now based on the provided research findings. Do not provide a conversational opening."
        ),
    ]
    response = await model.ainvoke(messages)

    report_text = ""
    if isinstance(response.content, str):
        report_text = response.content
    elif isinstance(response.content, list):
        report_text = " ".join(
            [
                part["text"]
                for part in response.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
        )

    # Save the final report to disk in the run directory
    try:
        report_file = run_root / "final_report.md"
        report_file.write_text(report_text, encoding="utf-8")
    except Exception as e:
        import logging

        logging.exception(f"Failed to write final_report.md: {e}")
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=f"Failed to save final report: {e}")]
            },
        )

    return Command(
        goto="__end__",
        update={
            "final_report": report_text,
            "messages": [
                AIMessage(
                    content="Final Research Report Generation Complete. You can read it below."
                )
            ],
        },
    )


# --- Main Graph Construction ---


deep_research_builder = StateGraph(GlobalState)

# Need a node for research_intake to call the research_brief tool
# This also performs question clarification
deep_research_builder.add_node("research_intake", research_intake)

# The supervisor recieves the full ResearchBrief, and decomposes it
# into actionable sub-steps
deep_research_builder.add_node("supervisor", supervisor_subgraph)


deep_research_builder.add_node("generate_final_report", generate_final_report)

deep_research_builder.add_edge(START, "research_intake")
deep_research_builder.add_edge("supervisor", "generate_final_report")
# Note: If a node returns a Command with a 'goto', it overrides these edges.
# But we still define the entry point with START.

# Compile the graph
graph = deep_research_builder.compile(name="Deep Research Graph")
