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

Model Contenders:

* gpt-4.1-mini (used by Open Deep Research)     input: $0.40, output: $1.60
* gpt-4.1 (used by Open Deep Research)          input: $2.00, output: $8.00
* gpt-5-mini                                    input: $0.25, output: $2.00
* gpt-5-nano                                    input: $0.05, output: $0.40
* gpt-4o-mini                                   input: $1.10, output: $4.40
* gpt-4o-mini-deep-research                     input: $2.00, output: $8.00
* gpt-3o-mini                                   input: $1.10, output: $4.40
* gpt-1o-mini                                   input: $1.10, output: $4.40

The models have advanced quite a bit since gpt-4.1. GPT-5-nano is better at reasoning (based on initial findings) and the cost is roughly comparable to gpt-4.1. We'll use reasoning_effort = "low" to descrease latency though.
We can use gpt-1o-mini for the supervisor and gpt-5-nano for the workers. This is because the supervisor will be doing more reasoning and planning, while the workers will be doing more information gathering.

I am also planning on making everything configureable (e.g. search depth, model, etc.)

User <-> Question Refinement/Brief Generation -> Supervisor + Sub-agent loop -> Report Generation

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
* ~~will need to split this up in to write_todo, read_todo, and update_todo~~
* Actually, it better to just have an update_todo tool (instead of write_todo, update_todo, read_todo) that takes the todo list and updates the entire thing. Makes it easier for the agent to decide on the tool.

#### Virtual File System

* The state context for sub-agents could become quite large
* Could possibly actually right to files instead of storing everything in memory. This addes additional tools calls and maybe a bit of latency, but it would allow the sub-agents to store intermediate search results/findings (ie raw content), then we can simply use an LLM/Tool call to summarize that content and return it back to the supervisor.
* Maybe we only use this for the supervisor, and the sub-agents just store their findings in their state (which will be passed to the supervisor)

<https://docs.langchain.com/oss/python/integrations/tools/filesystem>

### Defining the Graph and States

We need to keep track of:

* Global State
* Supervisor State
* Sub-agent State

I essentially want to have a separate node for the brief generation process, which will then route to the supervisor once the user is satisfied with the brief.

Instead of storing all information in the state, we're simply going to use a VFS, and store the paths to the files in the state. I think this is ideal because we don't have to feed in the entire state to the LLM, and it also could allow for extendability where multiple LLMs could collaborate on the same task.

 Combined question clarification and brief generation

Essentially want clarification to be part of the brief generation process.After clarification we essentially still need to "clarify" with the user if this is what they're looking for. However, it might be more clean to use separate and distinct steps for this, but we'll test it out.

After a lot of research (plus back and forth with Gemini), I decided to go with the approach of having supervisor/worker tool calls as **tool nodes**. I was convinced because:

* allows for the separation of reasoning vs acting (ie if tool calls were directly in the node and it failed, we'd have to dissect whether the reasoning or the tool itself broke. With separate tool nodes, we can see failures easily and it makes debugging way cleaner).
* allows the "thinking" state to be saved (ie if a tool crashes, we don't lose the LLM's plan and can just pick up exactly where we left off since the state was checkpointed after the reasoning node).

One thought that I do have is right now, the supervisor has to wait on all workers to return. The result is additional latency because the fastest running worker has to wait on the longest running worker. I would like to make the workers independent of eachother (i.e not blocking) by allowing them to return whenever they are finished. The main drawback of this approach is that there could be race conditions (2 workers report back at the same time). Another major drawback is that this might not be the most effient way to spawn workers to cover independent sub-topics.

ToDo List tool:

* At first I was just doing to rely on the VFS to manage the to-do list, but I think it's better to have a dedicated tool for this. Especially because we want the ability to keep track of what's been checked off/completed.
* It might also be easier to track the supervisors thinking if we're able to see when and why the to-do list was updated.
* Also, by defining a todo list schema, we can ensure that we maintain a consistent data structure for the to-do list.

The Research Breif:

* At first I was going to have the research brief stored in a file using the VFS, however, I think it's better to store the research breif in the state, since that is serving as our "north star". If I would have used the VFS, the supervisor would constantly have to check the file to reference it.

The ResearchBrief Prompt:

* At first I had a prompt that tried to have the user give as many details as possible if the model felt the query was to ambiguous. This caused it to keep looping because it kept believing it did not have enough info
* I changed the prompt to ask clarifiying questions, but also rely on some of it's internal knowledge of the world to infer what the user might mean and think about possible research avenues based on that internal knowledge.
* This worked fairly well, and it generates very detailed research briefs. I think this is a good trade-off because the user could easily tell the model to change aspects about the brief if they felt the agent misunderstood their intent

* Decided to essentially hardcode the todolist because we don't need the LLM to decide to create it and where to create it becasue we always need it in the same spot

### Worker

The supervisor creates a detailed ConductResearch object which seeds the worker with a subtopic, additional context, and a directory name. Each worker saves information in a folder assigned to it. There are two files in this folder: raw_content.md and compressed_summary.md. When a worker makes a tool call to exa_search(), the search tool directly appends the ENTIRE search result information (summary, full text excerpts, etc) to the raw_content.md file. The tool returns only a 'summary' and 'highlights' back to the worker to keep the context window small. Upon completion, the worker **must call the write_file tool to save its compressed_summary.md synthesis, which it creates from its own message history.

### Supervisor + Worker loop

Use a single source of truth in the compressed_summary.md files to keep track of the findings.

Upon worker completion, we inject ALL compressed summaries into the supervisors prompt. I decided to go with this approach rather than storing it in the state because I wanted to keep the state with only neccessary informtion that actually needs to be checkpointed and passed between nodes.
