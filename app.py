from chatbot.graph_setup import build_graph
from chatbot_ui import run_chatbot


if __name__ == "__main__":
    graph, config = build_graph()
    run_chatbot(graph, config)
