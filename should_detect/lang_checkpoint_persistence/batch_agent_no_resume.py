# ACE-EXPECT: detect
# CATEGORY: should_detect/lang_checkpoint_persistence
# LANGUAGE: python
# ISSUE: Long batch agent run over many items has no checkpoint, so a crash restarts from scratch
# EXPECTED-FINDING: The graph is compiled without a checkpointer and invoked with no thread_id per item, so a mid-batch failure can't resume
# EXPECTED-FIX: Compile with a persistent checkpointer and invoke each item with a stable thread_id to enable resume
# SEVERITY-HINT: warning
"""Overnight batch classifier looping thousands of records through a checkpointer-less graph."""

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

llm = ChatOpenAI(model="gpt-4o")


class ItemState(TypedDict):
    text: str
    category: str


def classify(s: ItemState) -> dict:
    return {"category": llm.invoke(f"Categorize: {s['text']}").content}


_g = StateGraph(ItemState)
_g.add_node("classify", classify)
_g.add_edge(START, "classify")
_g.add_edge("classify", END)
pipeline = _g.compile()  # no checkpointer -> no durable progress


def run_batch(records: list[str]) -> list[str]:
    results = []
    for text in records:
        # invoked with no thread_id/config; if this dies at record 9000 we start over at 0
        out = pipeline.invoke({"text": text, "category": ""})
        results.append(out["category"])
    return results
