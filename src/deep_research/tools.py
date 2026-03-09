"""Tools for the deep research agent."""

import os
from pathlib import Path

from exa_py import Exa
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import BaseTool, StructuredTool, tool

# --- Worker File Management Tools ---


def get_worker_filesystem_tools(worker_vfs_dir: str) -> list[BaseTool]:
    """Get the file management tools scoped to a worker's VFS directory.

    Args:
        worker_vfs_dir: Absolute path to the worker's dedicated VFS directory.

    Returns:
        A list of tools with a protected write operation to preserve raw logs.
    """
    toolkit = FileManagementToolkit(
        root_dir=worker_vfs_dir,
        selected_tools=["write_file", "read_file", "list_directory"],
    )
    all_tools = toolkit.get_tools()

    final_tools = []
    for t in all_tools:
        if t.name == "write_file":
            original_tool = t

            def protected_write(file_path: str, text: str, append: bool = False) -> str:
                """Write a file to the VFS. NOTE: 'raw_content.md' is PROTECTED and cannot be written manually."""
                if file_path and (
                    file_path == "raw_content.md"
                    or file_path.endswith("/raw_content.md")
                ):
                    return "ERROR: 'raw_content.md' is managed automatically by 'exa_search'. Manual writes are forbidden to prevent data loss. Please write your synthesis to 'compressed_summary.md' instead."
                return original_tool.invoke(
                    {"file_path": file_path, "text": text, "append": append}
                )

            async def protected_awrite(
                file_path: str, text: str, append: bool = False
            ) -> str:
                """Write a file to the VFS. NOTE: 'raw_content.md' is PROTECTED and cannot be written manually."""
                if file_path and (
                    file_path == "raw_content.md"
                    or file_path.endswith("/raw_content.md")
                ):
                    return "ERROR: 'raw_content.md' is managed automatically by 'exa_search'. Manual writes are forbidden to prevent data loss. Please write your synthesis to 'compressed_summary.md' instead."
                return await original_tool.ainvoke(
                    {"file_path": file_path, "text": text, "append": append}
                )

            t = StructuredTool.from_function(
                func=protected_write,
                coroutine=protected_awrite,
                name="write_file",
                description="Write a file to the VFS. NOTE: 'raw_content.md' is PROTECTED and cannot be written manually.",
                args_schema=original_tool.args_schema,
            )

        final_tools.append(t)

    return final_tools


# --- Search Tools ---


@tool
def exa_search(
    query: str,
    search_type: str = "auto",
    livecrawl: str = "fallback",
    category: str | None = None,
    start_published_date: str | None = None,
    vfs_path: str | None = None,
) -> str:
    """Search the web using Exa's neural search engine.

    Use this tool to find information on any topic. Choose parameters
    strategically to get the most relevant, up-to-date results.

    Args:
        query: Natural language search query. Be specific and descriptive.
        search_type: Search strategy ('neural', 'keyword', 'auto').
            - 'neural': best for concepts, intent, and semantic discovery.
            - 'keyword': best for exact names, model numbers, or rare technical terms.
            - 'auto': default; let Exa decide.
        livecrawl: Whether to fetch fresh page content ('always', 'fallback', 'never').
            Use 'always' for breaking news or very recent events.
        category: Filter by source type. Options: 'news', 'research paper', 'company',
            'pdf', 'github', 'tweet', 'blog post', 'personal site', 'social media'.
        start_published_date: Only return results published AFTER this date (YYYY-MM-DD).
            Use this to exclude outdated information in fast-moving fields.
        vfs_path: (Internal) Path to automatically record raw results. Do not set this.

    Returns:
        Formatted string of search results with titles, URLs, and summaries.
    """
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))

    num_results = int(os.getenv("EXA_NUM_RESULTS", "8"))
    max_chars = int(os.getenv("EXA_MAX_CHARACTERS", "8000"))
    num_sentences = int(os.getenv("EXA_NUM_SENTENCES", "5"))

    results = exa.search_and_contents(
        query,
        num_results=num_results,
        type=search_type,
        livecrawl=livecrawl,
        category=category,
        start_published_date=start_published_date,
        text={"max_characters": max_chars},  # type: ignore
        highlights={"query": query, "num_sentences": num_sentences},  # type: ignore
        summary={"query": query},  # type: ignore
    )

    if not results.results:
        return "No results found."

    # 1. Generate the detailed version for disk (includes Full Text)
    disk_parts = []
    for i, r in enumerate(results.results, 1):
        lines = [f"## Result {i}: {r.title or 'Untitled'}"]
        lines.append(f"**URL**: {r.url}")
        if r.published_date:
            lines.append(f"**Published**: {r.published_date}")
        if r.summary:
            lines.append(f"\n### Summary\n{r.summary}")
        if r.highlights:
            lines.append("\n### Key Highlights")
            for h in r.highlights:
                lines.append(f"- {h.strip()}")
        if r.text:
            lines.append(f"\n### Full Text Excerpt\n{r.text}")
        disk_parts.append("\n".join(lines))

    # 2. Generate the lightweight version for the LLM (NO Full Text)
    llm_parts = []
    for i, r in enumerate(results.results, 1):
        lines = [f"## Result {i}: {r.title or 'Untitled'}"]
        lines.append(f"**URL**: {r.url}")
        if r.published_date:
            lines.append(f"**Published**: {r.published_date}")
        if r.summary:
            lines.append(f"\n### Summary\n{r.summary}")
        if r.highlights:
            lines.append("\n### Key Highlights")
            for h in r.highlights:
                lines.append(f"- {h.strip()}")
        llm_parts.append("\n".join(lines))

    # Record to disk if path is provided (Comprehensive)
    if vfs_path:
        path = Path(vfs_path) / "raw_content.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n# Search: {query}\n\n" + "\n\n---\n\n".join(disk_parts))

    # Return to LLM (Lightweight)
    return "\n\n---\n\n".join(llm_parts)


def get_search_tools() -> list[BaseTool]:
    """Get the Exa search tool for the research worker.

    Returns:
        A list containing the custom exa_search tool.
    """
    return [exa_search]
