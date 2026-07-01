# ACE-EXPECT: detect
# CATEGORY: should_detect/unbounded_recursion_limit
# LANGUAGE: python
# SOURCE: SK ATIT project feedback — CSwind (Notion: https://app.notion.com/p/sk-atit/CSwind-d1d1772c74d5839db2c281ee42cd0346)
# ISSUE: A LangGraph invoke sets recursion_limit=50000, removing the practical guard against runaway loops
# EXPECTED-FINDING: recursion_limit is set to an extreme value (50000), so a cyclic or misbehaving graph can loop tens of thousands of times — burning cost/latency and masking a real termination bug instead of failing fast
# EXPECTED-FIX: Use a sane bound (tens, occasionally low hundreds) sized to the expected number of node steps; fix the underlying loop/termination condition rather than raising the ceiling
# SEVERITY-HINT: warning
"""LangGraph runner that disables the recursion guard with an absurd limit.

recursion_limit exists to catch graphs that never reach END. Setting it to 50000
means an accidental cycle runs effectively forever (huge cost/latency) before the
guard ever trips — the symptom of papering over a termination bug.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SchedulerState(TypedDict):
    pending: list[str]
    assigned: list[str]


def assign_node(state: SchedulerState) -> dict:
    # Imagine a routing bug here that sometimes fails to drain `pending`,
    # so the conditional edge keeps looping back instead of going to END.
    return {"assigned": state["assigned"]}


def route(state: SchedulerState) -> str:
    return "assign" if state["pending"] else "done"


def build_scheduler() -> object:
    graph = StateGraph(SchedulerState)
    graph.add_node("assign", assign_node)
    graph.add_edge(START, "assign")
    graph.add_conditional_edges("assign", route, {"assign": "assign", "done": END})
    return graph.compile()


def run(pending: list[str]) -> dict:
    scheduler_graph = build_scheduler()
    # recursion_limit=50000 effectively removes the runaway-loop guard.
    result = scheduler_graph.invoke(
        {"pending": pending, "assigned": []},
        {"recursion_limit": 50000},
    )
    return result
