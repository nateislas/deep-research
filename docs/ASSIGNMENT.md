# LangChain, Applied AI Takehome - Deep Research

# Overview

- Your task is to implement a simple deep research agent implemented with LangGraph.
- You may use the Python, or TypeScript libraries.
- This should take ~4 hours to complete.

## Task

Implement a simple deep research agent which can accept a query from a user, and return a report grounded in web search results.

## Guidelines

- It should have a `messages` field in state.
  - Here is where users will input their request, and the final report should be passed as an assistant messages.
  - Feel free to include the internally generated messages (e.g. tool calls for executing searches, search results, etc.) here as well, or you may store them in a separate field (up to you!)
- Users should be able to pass in a query, and be returned a report grounded in context from the web
- It should use web search APIs to gather context for the report.
  - See [here](https://python.langchain.com/docs/integrations/tools/) for a list of LangChain tools you may use in Python, and [here](https://js.langchain.com/docs/integrations/tools/) for TypeScript
  - I recommend one of Tavily, Exa, or SerpAPI (feel free to use one of these, or one of your choosing!)

## Ideas for Getting Started

- We’ve published open source deep research agents in the past (<https://github.com/langchain-ai/open_deep_research>, <https://github.com/langchain-ai/local-deep-researcher>) you may use these as inspiration, but do not copy/fork for your submission.
- Use the big lab’s (OpenAI, Anthropic, Grok…) deep research agents to see how they act, how their reports are structured, what types of sources they use, etc.

## Evaluation Criteria

- Code quality
- Report structure & generation quality
- Communication of ideas — we encourage you to talk through any questions or ideas you have for performance improvements

## Submission

- A public GitHub repository containing the code.
- The repository should include a README.md with instructions on how to setup, and run locally.
  - (Not required but encouraged) If you are able to run your agent with LangSmith Studio your README should include a screenshot of your agent graph.
- Your repository should include a [DESIGN.md](http://DESIGN.md) with a thorough overview of your thought process in designing your agent. Treat this as a brain dump of your engineering process. Some ideas of what to include (feel free to deviate from these). We ask that you do not use AI to write this design document, this should be done manually.
  - What did you try, what worked well, and what didn’t work well. How did this influence changes you made. Specific examples are good!
  - What are known shortcomings that you didn’t have time to address. How would you fix them if you had more time?
  - What are future features you would add if you had more time?

## Take it further (optional)

- Make the agent configurable! Below are some examples of ways you can make the agent configurable:
  - Number of searches
  - Model used
  - System prompts
  - Report structure
  - Search APIs
  - …
- Write evals!
