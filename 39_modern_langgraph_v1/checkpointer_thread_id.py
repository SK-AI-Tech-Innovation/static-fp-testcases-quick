# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangGraph 1.0 — .compile(checkpointer=InMemorySaver()) + thread_id config
# WHY-CORRECT: Persistence/memory in LangGraph is enabled by compiling with a checkpointer and
#   then invoking with `config={"configurable": {"thread_id": ...}}`. The same thread_id across
#   calls continues a conversation; the checkpointer restores prior state. This is the correct
#   short-term-memory pattern, not a missing-context bug.
# EXPECTED-WRONG: engine may claim the second invoke "loses history", flag InMemorySaver as
#   non-persistent/incorrect, or say thread_id is not a recognized config key.
# CORRECT-VERDICT: no findings
"""Persist conversation state across invokes using a checkpointer and a stable thread_id."""
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def echo(state: ChatState) -> dict:
    last = state["messages"][-1]
    text = last["content"] if isinstance(last, dict) else last.content
    return {"messages": [{"role": "assistant", "content": f"echo: {text}"}]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}
    graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config=config)
    state = graph.invoke({"messages": [{"role": "user", "content": "again"}]}, config=config)
    print(len(state["messages"]))
