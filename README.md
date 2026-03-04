# Deep Research Agent

A simple deep research agent implemented with [LangGraph](https://github.com/langchain-ai/langgraph), designed to accept a query and return a report grounded in web search results.

## Overview

This project implements a multi-step research workflow:

1. **Scoping**: Analyzes the initial query to identify key research topics.
2. **Researching**: Executes parallelized web searches using tools like Tavily or Exa.
3. **Synthesis**: Compiles findings into a comprehensive report with citations.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed for package management.
- Python 3.10+ (managed via `uv`).

### Setup

1. **Clone the repository** (if not already in the project root).

2. **Initialize Environment**:

   ```bash
   uv sync
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory (based on `.env.example`):

   ```bash
   cp .env.example .env
   ```

   Add the following keys:
   - `LANGSMITH_API_KEY`: For tracing and visualization in LangGraph Studio.
   - `TAVILY_API_KEY`: (or your preferred search API).
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: For the LLM backend.

4. **Launch Agent Server**:
   Run the development server to use LangGraph Studio or the local API:

   ```bash
   uv run langgraph dev
   ```

## Project Structure

- `src/agent/graph.py`: Core LangGraph implementation.
- `docs/ASSIGNMENT.md`: Original project requirements.
- `docs/DESIGN.md`: Detailed engineering process and design decisions.

## Agent Graph

*(Screenshot of the agent graph from LangSmith Studio will be placed here)*

## Development

- Use `uv add <package>` to install new dependencies.
- Run tests (once implemented) via `uv run pytest`.

For more information, refer to the [LangGraph documentation](https://langchain-ai.github.io/langgraph/).
