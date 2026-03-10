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

### Model Contenders

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

### Inital Architecture

User <-> Question Refinement/Brief Generation -> Supervisor + Sub-agent loop -> Report Generation

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

### Workers

The supervisor creates a detailed ConductResearch object which seeds the worker with a subtopic, additional context, and a directory name. Each worker saves information in a folder assigned to it. There are two files in this folder: raw_content.md and compressed_summary.md. When a worker makes a tool call to exa_search(), the search tool directly appends the ENTIRE search result information (summary, full text excerpts, etc) to the raw_content.md file. The tool returns only a 'summary' and 'highlights' back to the worker to keep the context window small. Upon completion, the worker **must call the write_file tool to save its compressed_summary.md synthesis, which it creates from its own message history.

### Supervisor + Worker loop

Use a single source of truth in the compressed_summary.md files to keep track of the findings.

Upon worker completion, we inject ALL compressed summaries into the supervisors prompt. I decided to go with this approach rather than storing it in the state because I wanted to keep the state with only neccessary informtion that actually needs to be checkpointed and passed between nodes.

Initally, the supervisor had access to ConductResearch tool, which would spawn an individual worker for the sub-topic. Howver, I noticed that the supervisor kept only spawning a couple at a time, even when there were many tasks waiting to be completed in the to-do list. Because of this, I changed the tool it had access to ConductResearchBatch, which more consistency spun up many workers at once.

### Adaptive search and to-do list updates

Similar to the supervisor failing to spawn multiple sub-topics at the same time, the supervisor also failed to update the to-do list multiple times at the same time. I went with a batch approach here as well.

### Current ResearchBrief generation process

**IMPORTANT, come back to this**

Right now, we're having the first node generate the research brief, and then pass it to the supervisor node. It creates 5-15 sub-objectives for the research brief, and then passes it to the supervisor node. This works well, but it could be introducing bias from the model which is reliant upon (usually) outdated information. However, maybe what we should be doing instead is keep it to about 4-6 broad sub-objectives, and then let the workers do the heavy lifting of breaking down the sub-objectives into more specific tasks based on what information was found. This reflects the actual process of research, where we start with a broad overview and then break down the sub-objectives into more specific tasks based on what information was found.

I initally wanted the VFS because I wanted to be able to refer to the raw_content.md files to generate the report if the final generation model felt like it needed additional information. This got very complex because the raw_content.md files are very large bc they contain the entire search results and full text excerpts.

I ended up just feeding the compressed_summary.md files to the final generation model which seems to work well.

## Final Design

### Architecture

As I mentioned previously, I decide to go with a supervisor-woker architecture. There are three main components of the workflow:

1. **Research Intake** This is a conversational loop. The user first prompts the agent with a query. If the agent feel like it has enough information, the LLM will attempt to decompose the initial query into a ResearchBrief consisting of a main topic, objective, and 4-6 broad sub-topics that will serve as the foundation of the research. The agent then returns the proposed ResearchBrief, and the user can either refine the brief and provide further clarification, or approve. Upon approval, we then transion to the Supervisor.

2. **Research Loop** Upon initialization, the supervisor first analyzes the ResearchBrief, breaks it down into actionable tasks, and adds those to the to-do list. The supervisor can generate multiple workers in parallel up to max_concurrent workers. Each of the workers is handed a unique task id, a general topic that it should research, aditional context for why it's researching that sub-topic, and a file sub-directory where it will store the full search results and the compressed summary. The workers can execute multiple queries. During research, the workers saved the raw results to a raw_content.md file. After it's executed the queries it came up with, it analyzes the full raw_content.md file and creates a compressed_summary.md file. One of the most important aspects of the compressed_summary.md is a section called **Promising Leads & Follow-ups**. The worker is told to indentify any possible information that would further out understanding of the research topic. This is used by the supervisor to create addition tasks in the to-do list to explore more deeply into aspects of the topic that should be covered.
3. **Final Report Generation** Once the supervisor determines that all tasks on the to-do list are complete (or we hit the hard iteration limit), we execute the generate_final_report node. This node aggregates the core state such as the ResearchBrief, initial user conversation, the completed TodoList, and compiles every compressed_summary.md generated by the workers and injects it into the prompt. It uses a high-reasoning model to reason over the findings into a cohesive and in-depth final report with in-text citation.

### Core data structures and state

* **GlobalState**: This is the root state and the primary source of truth for the entire graph. It holds the conversation messages and the finalized brief and references to the todo_list_path in the VFS and the final report. I included supervisor_messages here so the intake node can seed the supervisor memory right before the transition from ResearchIntake to Supervisor.
* **SupervisorState**: This manages the supervisor_messages and the iteration_count. I added the iteration_count to enforce a hard stop usually around 10 to 15 loops. This ensures we don't fall into an infinite research loop even though the supervisor actually tended to under explore.
* **WorkerState**: This is the isolated state for individual sub agents. It tracks researcher_messages and the output_dirname. I isolated researcher_messages from the rest of the graph so the supervisor isn't bogged down reading raw search results or tool errors. This helped prevent massive context window growth.

The most critical data structure is the ResearchBrief. The intake process builds it and once it is formalized it gets injected into the prompt context of every supervisor iteration and worker thread. It grounds the model and prevents it from going off on tangents.

We also have Pydantic schemas that define TaskItem and TodoList along with batching commands like ConductResearchBatch and UpdateTodoListBatch. By forcing the model to use these strictly typed schemas for its tool calls we guarantee that when the LLM decides to update the plan or spawn parallel workers it provides the exact arguments we need to update our JSON files.

### Tools

I decided to implement my own to-do list tool. It's simply a JSON file that contains a list of tasks and completed tasks. When workers successfully complete a task, the task is moved from the pending list to the completed list.

I created my own search too using Exa's SDK because I wanted the workers to be able to dynamically control search parameters like search depth and category (eg news, research paper, company, etc)

I used the off the shelf file management tools for the workers.

### Notes on design choices based on experience

* Difficult to get the supervisor to go deep based on the information that the workers discover
* I think that having the research_intake node decompose the research goal into broad "foundational" sub-topics is a good approach. At first I relied on this stage to generate the ENTIRE ResearchBrief and only add a few new topics occassionally. However, I wanted this deep research agent to reflect how people actually perform research (i.e. you start with a few broad search based on prior knowledge and expand based on the new information you find)
* At every stage, I use the ResearchBrief as the guiding values for the agent. Everything that the model does, it should be in service at understanding the goal and answering the questions laid out.
* I noticed early on that even with a long to-do list, the supervisor would only spawn one or two workers at a time. Switching to ConductResearchBatch and UpdateTodoListBatch was the only way to get it to actually utilize parallelism and update the plan efficiently without wasting dozen of loops.

#### Future Improvements

* Make the model choice configureable
* Make the final report generation process more dynamic
* Need to explore further on how detailed the information extraction from search results should be. Right now, we're probably saving too much information in the raw_content.md files
* I've noticed that the research_intake makes too many assumptions. There's a real trade-off here though. Do we bombard the user with questions to ensure we have the most accurate information, or do we make assumptions to speed up the process?

## Architecture Diagram

<p align="center">
  <img src="figures/fig1.png" alt="Deep Research Architecture" width="800px" />
</p>
