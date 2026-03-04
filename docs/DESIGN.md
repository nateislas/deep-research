
## Assignment

* Implement a simple deep research agent implemented with LangGraph
* Accepts a query -> returns a report grounded in web search results
* Should use search APIs to gather context for the report

Use: Tavily, Exa, SerpAPI or others

## Notes

* I wonder if we can try using multiple search APIs? Need to look in to if open-deep research only used one search API
* Need to look in to if there are any significant differences between the search APIs, know that Tavily is build for LLMs

## Background

### Google's Deep Research

* Uses a multi-agent architecture
* Essentially uses a supervisor + subagent architecture
* Breaks down the initual query into sub-queries -> specicialized agents work on sub-queries with targeted search
* adapts based on what it finds and refines the queries as it goes along
* Synthesizes gathered information into final report w/ in-line citations
* Untilizes gemini's massive context window (1M tokens)

### Open Deep Research

* Three core steps: scope, research, write
* scope gathers additional context from the users and asks clarifying follow-up questions -> generates a comprehenisive research breif
* Always looking at the research brief to guide us and refer back to it throughout the research
* Also uses a supervisor with sub-agent architecture
* Supervisor dynimaically spawns and assigns sub-topics to the sub-agents. Each sub-agent is responsible for only the sub-topic **these are parellelized**
* Each sub-agents runs as a tool-calling loop (**need to check when it stops looping**)
* When a sub-agent finishes running, it write a detailed answer to the sub-topic, helps to reduce context bloat -> returns this to supervisor
* supervisor reasons (need to use a reasoning model) about whether the sufficiently address the scope of the brief (will probably do something similar, with the addition of reasoning if the subagent sufficiently addressed the assigned sub-topic)
* adaptively addresses research gaps and spawns sub-agents to address gaps
* when the supervisor determines that brief is sufficeitnly addressed, trigger report writing (**produces the final report in a single shot, with the brief and research finding as context**)

## Notes

* It would be cool to try the figuring out a way to make the queries adaptive as it finds better and more relevant information (similar to Google's Deep Research). Also, it would be good to switch approaches if the initial queries are not yeilding good results.
* One of the things that I'll have to look in to is how we can best handle large web pages that might exceed the context window of the LLM (unlikely but still need to look in to). Will also have to see how Google, OpeAI, etc handle information that might be inside of documents. For example if we wanted an academic literature review of sorts, most of the information would be present in PDFs, which are require additional page depth + actually processing the PDF
* Instead of processing an entire page by simply feeding it in to the LLM, we can maybe use RAG, although this would add additional latency
