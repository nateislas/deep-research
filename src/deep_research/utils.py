"""Utility functions for the deep research agent."""

import os
from pathlib import Path

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from deep_research.state import ResearchBrief


class TaskItem(BaseModel):
    """A single research task in the todo list."""

    id: int = Field(description="The unique integer ID of the research task.")
    task: str = Field(description="The description of the research task.")


class TodoList(BaseModel):
    """The todo list for the research agent."""

    tasks: list[TaskItem] = Field(
        description="The full updated list of pending research tasks."
    )
    completed_tasks: list[TaskItem] = Field(default_factory=list)


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
    todo_update_calls: list[dict],
    run_root: Path,
    todo_path: str | None,
) -> tuple[str | None, list[ToolMessage]]:
    """Process update_todo_list tool calls: parse, write JSON, return messages."""
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


def handle_subtopic_additions(
    add_subtopic_calls: list[dict],
    todo_path: str | None,
) -> list[ToolMessage]:
    """Append new sub-topics to the existing TodoList on disk."""
    messages: list[ToolMessage] = []
    if not add_subtopic_calls or not todo_path or not Path(todo_path).exists():
        return messages

    todo = TodoList.model_validate_json(Path(todo_path).read_text())

    # Build a set of all existing tasks (pending and completed) for fast deduplication
    existing_tasks_normalized = {
        t.task.strip().lower() for t in todo.tasks + todo.completed_tasks
    }

    # Generate sequential IDs starting from the max existing ID
    max_id = max([t.id for t in todo.tasks + todo.completed_tasks], default=0)

    for bc in add_subtopic_calls:
        batch_outcomes: list[str] = []
        for topic_args in bc["args"]["topics"]:
            new_topic = topic_args["new_sub_topic"]
            new_topic_normalized = new_topic.strip().lower()

            # Deduplication check
            if new_topic_normalized in existing_tasks_normalized:
                batch_outcomes.append(
                    f"Skipped adding sub-topic: '{new_topic}' — Reason: Already exists in the todo list."
                )
                continue

            max_id += 1
            new_task = TaskItem(id=max_id, task=new_topic)
            todo.tasks.append(new_task)
            # Add to set to prevent deduplication within the same tool call or across calls
            existing_tasks_normalized.add(new_topic_normalized)
            batch_outcomes.append(
                f"Added sub-topic: '{new_topic}' — Reason: {topic_args['rationale']} (ID {max_id})"
            )

        messages.append(
            ToolMessage(
                content="\n".join(batch_outcomes)
                or "No sub-topics added in this batch.",
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

            # Character limit guard to prevent prompt overflow
            # if len(content) > 4000:
            #    content = content[:4000] + "... [Truncated]"

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
