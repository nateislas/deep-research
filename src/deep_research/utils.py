"""Utility functions for the deep research agent."""

import asyncio
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from deep_research.prompts import RESEARCHER_PROMPT
from deep_research.state import ResearchBrief, TaskItem, TodoList

# Resolve RESEARCH_ROOT, allowing override from environment variable.
# Default is project root / "vfs"
DEFAULT_VFS_PATH = Path(__file__).parent.parent.parent / "vfs"
RESEARCH_ROOT = Path(os.getenv("RESEARCH_VFS_PATH", str(DEFAULT_VFS_PATH)))


# --- Todo List Helpers ---


def write_todo_list(todo: TodoList, path: Path) -> None:
    """Write a TodoList to disk as JSON.

    This is a plain helper called programmatically by supervisor_tools.
    The caller is responsible for ensuring `path` is valid and inside the VFS.

    Args:
        todo: The TodoList object to persist.
        path: Absolute path to the JSON file (e.g., run_root / "todo_list.json").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(todo.model_dump_json(indent=2))


# --- Supervisor Tools Helpers ---


def handle_todo_updates(
    todo_update_calls: list[dict[str, Any]],
    run_root: Path,
    todo_path: str | None,
) -> tuple[str | None, list[ToolMessage]]:
    """Process update_todo_list tool calls: parse, write JSON, return messages."""
    messages: list[ToolMessage] = []
    for tc in todo_update_calls:
        todo_data = TodoList(tasks=tc["args"]["tasks"])
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


def handle_batch_task_updates(
    update_batch_calls: list[dict[str, Any]],
    todo_path: str | None,
) -> list[ToolMessage]:
    """Append new tasks to the existing TodoList on disk."""
    messages: list[ToolMessage] = []
    if not update_batch_calls:
        return messages

    if not todo_path or not Path(todo_path).exists():
        for bc in update_batch_calls:
            messages.append(
                ToolMessage(
                    content="ERROR: Cannot add new tasks because no TodoList exists yet. Please create one using TodoList first.",
                    tool_call_id=bc["id"],
                )
            )
        return messages

    todo = TodoList.model_validate_json(Path(todo_path).read_text())

    # Build a set of all existing tasks (pending and completed) for fast deduplication
    existing_tasks_normalized = {
        t.task.strip().lower() for t in todo.tasks + todo.completed_tasks
    }

    # Generate sequential IDs starting from the max existing ID
    max_id = max([t.id for t in todo.tasks + todo.completed_tasks], default=0)

    for bc in update_batch_calls:
        batch_outcomes: list[str] = []
        for task_args in bc["args"]["new_tasks"]:
            new_task_str = task_args["new_task"]
            new_task_normalized = new_task_str.strip().lower()

            # Deduplication check
            if new_task_normalized in existing_tasks_normalized:
                batch_outcomes.append(
                    f"Skipped adding task: '{new_task_str}' — Reason: Already exists in the todo list."
                )
                continue

            max_id += 1
            new_task_obj = TaskItem(id=max_id, task=new_task_str)
            todo.tasks.append(new_task_obj)
            # Add to set to prevent deduplication within the same tool call or across calls
            existing_tasks_normalized.add(new_task_normalized)
            batch_outcomes.append(
                f"Added task: '{new_task_str}' — Reason: {task_args['rationale']} (ID {max_id})"
            )

        messages.append(
            ToolMessage(
                content="\n".join(batch_outcomes) or "No tasks added in this batch.",
                tool_call_id=bc["id"],
            )
        )

    # Save the updated todo list
    Path(todo_path).write_text(todo.model_dump_json(indent=2))

    return messages


def mark_tasks_completed(
    completed_task_ids: list[int],
    todo_path: str | None,
) -> None:
    """Mark completed worker tasks in the TodoList JSON based on their integer IDs."""
    if not completed_task_ids or not todo_path or not Path(todo_path).exists():
        return

    todo = TodoList.model_validate_json(Path(todo_path).read_text())

    # We will iterate through pending tasks, and if the ID matches, move it to completed
    tasks_to_keep = []

    for task_item in todo.tasks:
        if task_item.id in completed_task_ids:
            todo.completed_tasks.append(task_item)
        else:
            tasks_to_keep.append(task_item)

    todo.tasks = tasks_to_keep
    Path(todo_path).write_text(todo.model_dump_json(indent=2))


async def dispatch_workers_concurrently(
    batch_research_calls: list[dict],
    run_root: Path,
    todo_path: str | None,
    brief: ResearchBrief,
    config: dict,
    worker_subgraph: callable,
    max_concurrent_workers: int,
) -> tuple[list[int], list[ToolMessage], list[str]]:
    """Dispatch and Aggregate parallel workers.

    Returns:
        completed_task_ids: IDs of successfully executed tasks to be crossed off the Todo list.
        batch_msgs: Parsed ToolMessages detailing success/failure rates.
        new_findings_paths: VFS paths written to by workers.
    """
    all_task_items = []
    for bc in batch_research_calls:
        for t_args in bc["args"]["tasks"]:
            all_task_items.append({"args": t_args, "batch_id": bc["id"]})

    capped_tasks = all_task_items[:max_concurrent_workers]

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

    research_coroutines = [
        worker_subgraph.ainvoke(
            {
                "brief": brief,
                "run_root": str(run_root),
                "output_dirname": item["args"]["output_dirname"],
                "researcher_messages": [
                    SystemMessage(content=RESEARCHER_PROMPT),
                    HumanMessage(
                        content=(
                            f"## OVERARCHING PROJECT CONTEXT\n"
                            f"**Project Topic:** {brief.topic}\n"
                            f"**Main Objective:** {brief.main_objective}\n\n"
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

    results = await asyncio.gather(*research_coroutines, return_exceptions=True)

    batch_outcomes: dict[str, list[str]] = {bc["id"]: [] for bc in batch_research_calls}
    completed_task_ids: list[int] = []
    new_findings_paths: list[str] = []

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
        new_findings_paths.append(str(worker_output_dir))

    batch_msgs = [
        ToolMessage(
            content="\n".join(outcomes) or "No tasks processed in this batch.",
            tool_call_id=bid,
        )
        for bid, outcomes in batch_outcomes.items()
    ]

    return completed_task_ids, batch_msgs, new_findings_paths


async def execute_tools_concurrently(
    tool_calls: list[dict],
    tools_map: dict[str, BaseTool],
    max_attempts: int = 3,
) -> list[ToolMessage]:
    """Execute multiple LLM tool calls concurrently with built-in retry logic.

    Args:
        tool_calls: List of dictionary tool calls from the LLM message.
        tools_map: Dictionary mapping tool names to BaseTool objects.
        max_attempts: Number of times to retry a failed tool.

    Returns:
        List of formatted ToolMessages containing the results or errors.
    """

    async def execute_single_tool(tc: dict) -> ToolMessage:
        tool = tools_map.get(tc["name"])
        if not tool:
            return ToolMessage(
                content=f"Tool {tc['name']} not found.", tool_call_id=tc["id"]
            )

        args = dict(tc["args"])
        last_error = None
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

        return ToolMessage(content=str(result), tool_call_id=tc["id"])

    # Launch all tool executions concurrently
    coroutines = [execute_single_tool(tc) for tc in tool_calls]
    results = await asyncio.gather(*coroutines)
    return list(results)


# --- Prompt Formatting Helpers ---


def brief_to_prompt_vars(brief: ResearchBrief) -> dict:
    """Format the ResearchBrief fields for injection into the supervisor prompt.

    Args:
        brief: The approved ResearchBrief from state.

    Returns:
        A dict with keys matching the SUPERVISOR_PROMPT template placeholders.
    """
    return {
        "topic": brief.topic,
        "main_objective": brief.main_objective,
        "scope": brief.scope,
        "sub_objectives": "\n".join([f"  - {obj}" for obj in brief.sub_objectives]),
    }


def todo_to_string(todo: TodoList) -> str:
    """Format the current TodoList as a readable checklist for the supervisor prompt.

    Args:
        todo: The current TodoList object.

    Returns:
        A string with pending tasks marked [ ] and completed tasks marked [x].
    """
    lines = []
    for t in todo.tasks:
        lines.append(f"[ ] ID {t.id}: {t.task}")
    for t in todo.completed_tasks:
        lines.append(f"[x] ID {t.id}: {t.task}")
    return "\n".join(lines) if lines else "No tasks in list."


def get_findings_summary(vfs_root: Path) -> str:
    """Read all findings and aggregate their compressed_summary.md contents into a single research summary."""
    if not vfs_root.exists():
        return "No findings yet."

    summaries = []
    all_leads = []

    # Identify subdirectories (worker assignments) that have a summary
    for summary_path in sorted(vfs_root.glob("**/compressed_summary.md")):
        try:
            content = summary_path.read_text(encoding="utf-8").strip()

            # Extract Promising Leads for the consolidated section
            # Look for headers like "Promising Leads", "Follow-ups", etc.
            lower_content = content.lower()
            lead_markers = [
                "## promising leads",
                "## follow-up",
                "## leads for expansion",
            ]

            found_marker = None
            for marker in lead_markers:
                if marker in lower_content:
                    found_marker = marker
                    break

            if found_marker:
                # Find start and end of section
                start_idx = lower_content.find(found_marker)
                # Split original content to preserve formatting
                leads_part = content[start_idx:]
                # Header is everything on the first line
                lines = leads_part.split("\n")
                if len(lines) > 1:
                    # Body is everything after the header line until the next header
                    body_lines = []
                    for line in lines[1:]:
                        if line.startswith("##"):
                            break
                        body_lines.append(line)

                    leads_raw = "\n".join(body_lines).strip()
                    if leads_raw:
                        topic_name = summary_path.parent.name.replace("_", " ").title()
                        all_leads.append(f"### Leads from {topic_name}:\n{leads_raw}")

            # Truncation logic has been intentionally removed because LangGraph/LLMs
            # now natively handle very large context windows dynamically.

            # Format as a clean section for the summary
            topic_dir = summary_path.parent.name
            topic_title = topic_dir.replace("_", " ").title()
            summaries.append(
                f"### Finding: {topic_title}\n**DirectoryID: {topic_dir}**\n{content}\n"
            )
        except Exception as e:
            summaries.append(
                f"### Finding: {summary_path.parent.name}\n(Error reading summary: {e})\n"
            )

    if not summaries:
        return "No research results found in VFS yet."

    main_findings = "\n---\n".join(summaries)

    if all_leads:
        leads_summary = "\n\n".join(all_leads)
        return (
            f"{main_findings}\n\n"
            "--- \n\n"
            "## CONSOLIDATED LEADS FOR EXPANSION\n"
            "The following leads were proposed by workers for further investigation. "
            "Evaluate these against the Main Objective and use AddSubTopic to follow the most promising ones:\n\n"
            f"{leads_summary}"
        )

    return main_findings
