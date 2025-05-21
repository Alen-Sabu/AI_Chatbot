from langchain_core.messages import SystemMessage, AIMessage
from chatbot.llm import model
from chatbot.retriever import retrieve
from langgraph.graph import MessagesState

def query_or_respond(state: MessagesState):
    """
    This function decides whether the LLM needs 
    to call an external retrieval tool to fetch context
    from the memory
    """
    llm_with_tools = model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate(state: MessagesState):
    """
    This function handles the final answer generation.
    It takes the output of the retrieval tool (retrieved documents) and 
    constructs a prompt to generate the final, concise LLM response.
    """
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "Use three sentences maximum and keep the answer concise. "
        "If you are unable to answer from the given content, give an answer on your own."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        msg for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]
    prompt = [SystemMessage(system_prompt)] + conversation_messages
    response = model.invoke(prompt)
    return {"messages": [response]}
