# Deep Research Agent

## Assignment

* Implement a simple deep research agent implemented with LangGraph
* Accepts a query -> returns a report grounded in web search results
* Should use search APIs to gather context for the report
* Encouraged to use LangSmith Studio to visualize the agent graph
* Should have a messages field in state (users will input their request, and the final report should be passed as an assistant messages)

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
* Utilizes gemini's huge context window (1M tokens)
* Also use RL-trained agents for researching the web better (out-of-scope for this)

### Open Deep Research

* Three core steps: scope, research, write
* scope gathers additional context from the users and asks clarifying follow-up questions -> generates a comprehenisive research breif
* Always looking at the research brief to guide us and refer back to it throughout the research
* Also uses a supervisor with sub-agent architecture
* Supervisor dynimaically spawns and assigns sub-topics to the sub-agents. Each sub-agent is responsible for only the sub-topic **these are parellelized**
* Each sub-agents runs as a tool-calling loop (**need to check when it stops looping**) (uses a ReACT loop)
* When a sub-agent finishes running, it write a detailed answer to the sub-topic, helps to reduce context bloat -> returns this to supervisor
* supervisor reasons (need to use a reasoning model) about whether the sufficiently address the scope of the brief (will probably do something similar, with the addition of reasoning if the subagent sufficiently addressed the assigned sub-topic)
* adaptively addresses research gaps and spawns sub-agents to address gaps
* when the supervisor determines that brief is sufficeitnly addressed, trigger report writing (**produces the final report in a single shot, with the brief and research finding as context**)

### OpenAI Deep Research

* trained using end-to-end RL for difficult browing and reasoning tasks
* learned to plan and execute multi-step trajectories
* "reacts to real-time information" -> adaptively plans
* uses a fine-tuned reasoning model (now o3)
* Clarification -> planning and decomposition -> iterative search -> synthesis and report

## Notes

* It would be cool to try the figuring out a way to make the queries adaptive as it finds better and more relevant information (similar to Google's Deep Research). Also, it would be good to switch approaches if the initial queries are not yeilding good results.
* One of the things that I'll have to look in to is how we can best handle large web pages that might exceed the context window of the LLM (unlikely but still need to look in to). Will also have to see how Google, OpeAI, etc handle information that might be inside of documents. For example if we wanted an academic literature review of sorts, most of the information would be present in PDFs, which are require additional page depth + actually processing the PDF
* Instead of processing an entire page by simply feeding it in to the LLM, we can maybe use RAG, although this would add additional latency
* All approaches use the question refinement/detailed brief approach
* Maybe have two types of sub-agents, one for broad searches, another for more precise searches. This will be controlled via tavily's search depth parameter and the number of results we pull

## High-level Initial Plan and Design

Since all approaches use the question refinement/detailed brief approach, we will as well. It makes sense since user's queries are usually vague. It will also allow the user to adjust the proposed research brief if it disagress with some parts, or there was a misunderstanding.

We will also use the supervisor + sub-agent architecture, with parellelized sub-agents. The major deviation and addition from Open Deep Research is the use of a dedicated to-do list tool that the supervisor manages (will also possibly have the sub-agents use this as well). I also plan to use 2 different search APIs. Tavily for broader searcher, and then Exa for more precise searches or to find possible gaps that the worker might have missed. I'll have to decide a strategy for when to use which API.

As of right now, I also want to utilize a virtual file system to store intermediate search results/findings (ie raw content), then we can simply use an LLM/Tool call to summarize that content and return it back to the supervisor. This will help with context bloat.

The major consideration that I need to figure out is how a worker decides when it is done researching its assigned sub-topic. Is it when it stops finding new information that contributes to it's understanding? Or maybe that's up to the supervisor to decide once the worker returns its findings.

I also want the supervisor and workers to be able to dynamically adjust the research brief as needed when it finds new information that might be relevant to the research brief.

For the supervisor, I will use a more powerful model and workers will use a cheaper model. This is because the supervisor will be doing more reasoning and planning, while the workers will be doing more information gathering.

I am also planning on making everything configureable (e.g. search depth, model, etc.)

User -> Question Refinement/Brief Generation -> Supervisor + Sub-agent loop -> Report Generation

```mermaid
graph LR
    User[User] <--> Brief[Question Refinement / Brief Generation]
    Brief --> Supervisor[Supervisor]
    Supervisor --> Report[Report Generation]
    Supervisor <--> SubAgents[Sub-agents<br/>(parallelized)]
```

### Tools we'll need

#### Search APIs

##### Tavily

* Optimized for LLMs
* Summarizes information from multiple sources
* Allows us to control search depth, number of results, etc
* Good for quick, broad searches that performs the function of RAG essentially

<https://docs.langchain.com/oss/python/integrations/tools/tavily_search>

##### Exa

* uses a "neural" embedding search engine, essentially just tries to understand the query and find relevant results, even w/o keyword matches
* high token effiency bc it grabs the most relevant parts of a web page

<https://docs.langchain.com/oss/python/integrations/tools/exa_search>

#### Process Managment

##### To-Do List

* Will probably want to use a to-do list tool that the supervisor manages, and maybe for the sub-agents as well, this could help the supervisor keep track of addressed comoponents of the research brief, especially if the research tasks is very complex

#### Virtual File System

* The state context for sub-agents could become quite large
* Could possibly actually right to files instead of storing everything in memory. This addes additional tools calls and maybe a bit of latency, but it would allow the sub-agents to store intermediate search results/findings (ie raw content), then we can simply use an LLM/Tool call to summarize that content and return it back to the supervisor.
* Maybe we only use this for the supervisor, and the sub-agents just store their findings in their state (which will be passed to the supervisor)

<https://docs.langchain.com/oss/python/integrations/tools/filesystem>
