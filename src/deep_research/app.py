"""Streamlit app for visualizing deep research execution."""

import asyncio
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load env variables before importing deep_research components
load_dotenv()

# We need to set these before loading the graph to ensure tracing works
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "deep_research_streamlit"

from deep_research.graph import graph  # noqa: E402
from deep_research.utils import TodoList  # noqa: E402

st.set_page_config(page_title="Deep Research Agent", layout="wide")

# -----------------
# State Management
# -----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# We keep the raw LangGraph state dictionary here
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}

# We keep a separate UI messages list to render the chat cleanly
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# Persistent logs for the status execution
if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []

if "is_running" not in st.session_state:
    st.session_state.is_running = False


# -----------------
# Helper Functions
# -----------------
def get_todo_list() -> TodoList | None:
    """Retrieve and parse the current TodoList from the VFS."""
    todo_path = st.session_state.graph_state.get("todo_list_path")
    if todo_path and Path(todo_path).exists():
        try:
            return TodoList.model_validate_json(Path(todo_path).read_text())
        except Exception:
            return None
    return None


def render_sidebar():
    """Render the Research Brief and TodoList in the sidebar."""
    with st.sidebar:
        st.title("Research Control")

        if st.session_state.is_running:
            st.error("⚠️ execution in progress")
            if st.button("🛑 STOP RESEARCH", type="primary", use_container_width=True):
                st.session_state.is_running = False
                st.stop()

        brief_placeholder = st.empty()
        st.divider()
        todo_placeholder = st.empty()
    return brief_placeholder, todo_placeholder


def draw_brief(placeholder) -> None:
    """Render the Research Brief in the given Streamlit placeholder."""
    with placeholder.container():
        brief = st.session_state.graph_state.get("brief")
        if brief:
            st.subheader("Research Brief")
            st.write(f"**Topic:** {brief.topic}")
            st.write(f"**Objective:** {brief.main_objective}")
            st.write(f"**Status:** {brief.brief_status.upper()}")
            with st.expander("Scope Details"):
                st.write(brief.scope)
            with st.expander("Sub-objectives"):
                for obj in brief.sub_objectives:
                    st.write(f"- {obj}")
        else:
            st.info("No brief proposed yet. Enter a topic in the main chat to begin.")


def draw_todo(placeholder) -> None:
    """Render the Todo List in the given Streamlit placeholder."""
    with placeholder.container():
        todo = get_todo_list()
        if todo:
            st.subheader("Todo List")
            pending = len(todo.tasks)
            completed = len(todo.completed_tasks)
            st.write(f"**Tasks:** {pending} Pending | {completed} Completed")
            with st.expander("Research Tasks", expanded=True):
                for t in todo.completed_tasks:
                    st.markdown(f"- [x] {t}")
                for t in todo.tasks:
                    if t not in todo.completed_tasks:
                        st.markdown(f"- [ ] {t}")
        else:
            st.info("Todo list will appear once the supervisor starts planning.")


async def process_user_input(user_input: str, brief_placeholder, todo_placeholder):
    """Run the graph asynchronously and capture updates."""
    # Build configuration
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Check if this is an "Approve" message or an initial prompt
    is_approve = user_input.lower().strip() in ["approve", "yes", "looks good"]
    has_brief = bool(st.session_state.graph_state.get("brief"))

    # If it's not an approval and no brief exists, use the instant-brief prompt injection
    if not is_approve and not has_brief:
        prompt = f"{user_input} Please generate the formal ResearchBrief immediately without asking any clarifying questions."
        msg = HumanMessage(content=prompt)
    else:
        msg = HumanMessage(content=user_input)

    # Append message to graph state
    current_state = st.session_state.graph_state.copy()
    current_messages = list(current_state.get("messages", []))
    current_messages.append(msg)
    current_state["messages"] = current_messages

    # Create a status container for real-time updates
    status_container = st.status("Agent Executing...", expanded=True)
    st.session_state.is_running = True
    st.session_state.execution_logs = []  # Reset for new run

    final_state_snapshot = current_state

    try:
        async for chunk in graph.astream(
            current_state, config, stream_mode="updates", subgraphs=True
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                namespace, event = chunk
            else:
                event = chunk

            for node, update in event.items():
                log_entry = f"**Node:** `{node}`"
                status_container.write(log_entry)
                st.session_state.execution_logs.append(log_entry)

                # Update our final state snapshot with the diffs
                for key, val in update.items():
                    if key in [
                        "messages",
                        "supervisor_messages",
                        "active_tasks",
                        "findings_paths",
                        "researcher_messages",
                    ]:
                        existing = final_state_snapshot.get(key, [])
                        if isinstance(val, list):
                            final_state_snapshot[key] = existing + val
                        else:
                            final_state_snapshot[key] = existing + [val]
                    else:
                        final_state_snapshot[key] = val

                # Update state so helpers can read it live
                st.session_state.graph_state = final_state_snapshot

                # Display specific meaningful events
                if "brief" in update:
                    log_brief = "✅ Proposed Research Brief!"
                    status_container.write(log_brief)
                    st.session_state.execution_logs.append(log_brief)
                    brief_placeholder.empty()
                    draw_brief(brief_placeholder)

                if "supervisor_messages" in update:
                    for sm in update["supervisor_messages"]:
                        if hasattr(sm, "tool_calls") and sm.tool_calls:
                            for tc in sm.tool_calls:
                                log_tool = f"🛠 Supervisor called tool: `{tc['name']}`"
                                status_container.write(log_tool)
                                st.session_state.execution_logs.append(log_tool)
                                with status_container.expander(
                                    f"Parameters for {tc['name']}"
                                ):
                                    st.json(tc["args"])

                if "researcher_messages" in update:
                    for rm in update["researcher_messages"]:
                        if hasattr(rm, "tool_calls") and rm.tool_calls:
                            for tc in rm.tool_calls:
                                log_worker = f"🧑‍🔬 Worker called tool: `{tc['name']}`"
                                status_container.write(log_worker)
                                st.session_state.execution_logs.append(log_worker)
                                with status_container.expander(
                                    f"Parameters for {tc['name']}"
                                ):
                                    st.json(tc["args"])

                if "todo_list_path" in update or node == "supervisor_tools":
                    log_todo = "📝 Updated Todo List."
                    status_container.write(log_todo)
                    st.session_state.execution_logs.append(log_todo)
                    todo_placeholder.empty()
                    draw_todo(todo_placeholder)

                if "final_report" in update:
                    log_report = "🎉 Final Report generated!"
                    status_container.write(log_report)
                    st.session_state.execution_logs.append(log_report)

        status_container.update(
            label="Execution Complete", state="complete", expanded=False
        )

    except Exception as e:
        status_container.update(label=f"Error: {e}", state="error", expanded=True)
        st.error(f"Execution failed: {e}")
    finally:
        st.session_state.is_running = False

    # Save the updated state back to session state
    st.session_state.graph_state = final_state_snapshot

    # Extract the last AI response from the updated messages
    updated_messages = final_state_snapshot.get("messages", [])
    if updated_messages:
        last_msg = updated_messages[-1]
        if hasattr(last_msg, "content") and getattr(last_msg, "type", "") == "ai":
            # Add it to UI messages
            st.session_state.ui_messages.append(
                {"role": "assistant", "content": last_msg.content}
            )


# -----------------
# Main UI Layout
# -----------------
brief_placeholder, todo_placeholder = render_sidebar()
draw_brief(brief_placeholder)
draw_todo(todo_placeholder)

st.title("Deep Research Agent Visualization")
st.markdown(
    "Enter a topic to generate a Research Brief, then type **'Approve'** to begin the multi-agent research loop.\n\n"
    "🚨 **To stop or terminate execution at any time**, click the square **Stop** icon in the chat input box at the bottom, or in the top right corner of the page!"
)

# Render chat history
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Render persistent execution logs if there are any
if st.session_state.execution_logs:
    with st.expander("📝 Last Run Execution Logs", expanded=False):
        for log in st.session_state.execution_logs:
            st.markdown(log)

# Final Report Display (pinned at the bottom before chat input if exists)
if st.session_state.graph_state.get("final_report"):
    st.divider()
    st.header("Final Research Report")
    st.markdown(st.session_state.graph_state["final_report"])

# Chat Input
if prompt := st.chat_input("What would you like to research?"):
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run graph
    asyncio.run(process_user_input(prompt, brief_placeholder, todo_placeholder))

    # Rerun to update sidebar and chat
    st.rerun()
