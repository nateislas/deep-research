import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from deep_research.graph import graph

# Load environment variables (API keys)
load_dotenv()

async def test_proactive_intake():
    print("Testing Proactive Research Intake Flow...")
    
    # Starting with a broad topic, let's see if the agent infers and suggests
    initial_state = {
        "messages": [HumanMessage(content="I want to research current AI energy use in the US.")],
        "supervisor_messages": [],
        "active_tasks": [],
    }
    
    print("\n--- NEW TRIAL: PROACTIVE SCOPING ---")
    config = {"configurable": {"thread_id": "proactive-test"}}
    
    async for event in graph.astream(initial_state, config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if not isinstance(last_msg, HumanMessage):
                print(f"\n[Agent Response Type]: {type(last_msg).__name__}")
                print(f"[Agent Response Content]:\n{last_msg.content}")
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                   print(f"\n[Tool Call]: {last_msg.tool_calls[0].get('name')}")
                   print(f"[Tool Call Args]: {last_msg.tool_calls[0].get('args')}")
        if "brief" in event and event["brief"]:
            print(f"\n[Brief]: {event['brief'].model_dump_json(indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_proactive_intake())
