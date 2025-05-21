from langgraph.graph import START, END, StateGraph, MessagesState
from chatbot.graph_nodes import query_or_respond, generate
from chatbot.retriever import retrieve
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

def build_graph():
    tools = ToolNode([retrieve])
    memory = MemorySaver()
    graph_builder = StateGraph(MessagesState)

    # Only include tool and generate nodes
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    # Set the entry point directly to tools
    graph_builder.set_entry_point("tools")

    # Define flow: tools → generate → END
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    config = {"configurable": {"thread_id": "abc123"}}
    return graph_builder.compile(checkpointer=memory), config

    
