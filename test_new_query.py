import asyncio
import uuid
import os
from pprint import pprint

from langchain_core.messages import HumanMessage
from deep_research.graph import graph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "deep_research_new_query_test"

async def main():
    # We will simulate a conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\n--- [1] Sending initial prompt ---")
    # GLP-1 drugs: hard clinical + financial data, good stress test for raw_content depth
    query = (
        "Research the competitive landscape of GLP-1 receptor agonist drugs for obesity "
        "and diabetes treatment as of 2024/2025. Focus on semaglutide (Ozempic/Wegovy), "
        "tirzepatide (Mounjaro/Zepbound), and pipeline drugs like retatrutide. "
        "I need specific clinical trial results (weight loss %, HbA1c reduction), "
        "FDA approval status and dates, annual revenue figures, peak revenue projections, "
        "and competitive positioning of Novo Nordisk vs. Eli Lilly vs. emerging players."
    )

    state = {"messages": [HumanMessage(content=f"{query} Please generate the formal ResearchBrief immediately without asking any clarifying questions.")]}

    
    # We'll run the graph and accumulate the state updates
    final_state_after_turn_1 = state.copy()
    async for event in graph.astream(state, config, stream_mode="values"):
        final_state_after_turn_1 = event
        
    if final_state_after_turn_1.get('brief'):
        print(f"Generated Brief Topic: {final_state_after_turn_1.get('brief').topic}")
        print(f"Sub-objectives counts: {len(final_state_after_turn_1.get('brief').sub_objectives)}")
    else:
        print(f"No brief generated. Model said:\n{final_state_after_turn_1.get('messages')[-1].content}")
        return

    print("\n--- [2] Approving the brief ---")
    # Take the exact state from step 1, append the "Approve" message, and run again
    new_messages = list(final_state_after_turn_1.get("messages", []))
    new_messages.append(HumanMessage(content="Approve"))
    final_state_after_turn_1["messages"] = new_messages
    
    # Run the graph again with the "Approve" message
    # This will trigger the intake approval -> supervisor -> workers
    # We stream "updates" to see node transitions
    async for event in graph.astream(final_state_after_turn_1, config, stream_mode="updates"):
        print("\n=== Event ===")
        for node, update in event.items():
            print(f"Node: {node}")
            if node == "research_intake":
                # Intake should move to supervisor
                pass
            
            if "supervisor_messages" in update:
                for msg in update["supervisor_messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"  Supervisor called tools: {[tc['name'] for tc in msg.tool_calls]}")
                    elif isinstance(msg, HumanMessage):
                        pass # Ignore user messages
                    else:
                        content_snippet = str(msg.content)[:100].replace('\n', ' ')
                        print(f"  Supervisor message: {content_snippet}...")
            
            if "findings_paths" in update:
                print(f"  Findings paths updated: {update['findings_paths']}")
            
            if "todo_list_path" in update:
                print(f"  Todo list path created/updated: {update['todo_list_path']}")
                
            if "final_report" in update:
                print("\n--- [FINAL REPORT] ---")
                print(update["final_report"])

    print("\n--- [3] Done ---")

if __name__ == "__main__":
    asyncio.run(main())
