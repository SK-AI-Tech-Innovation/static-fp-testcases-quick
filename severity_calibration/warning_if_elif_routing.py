# ACE-EXPECT: detect
# CATEGORY: should_detect/lang_state_graph_workflows
# LANGUAGE: python
# ISSUE: IT-ticket triage routing implemented as a classify-then-if/elif/else dispatch ladder instead of a LangGraph StateGraph
# EXPECTED-FINDING: Agent orchestration uses hand-rolled if/elif/else branching to route to per-category handlers, not explicit nodes/edges/conditional routing
# EXPECTED-FIX: Model the router as a LangGraph StateGraph with a conditional edge dispatching to handler nodes
# SEVERITY-HINT: warning
"""IT-helpdesk triage bot whose category dispatch is a fragile if/elif/else ladder."""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")


def classify_ticket(text: str) -> str:
    out = llm.invoke(f"Classify into password|network|hardware|access|other:\n{text}")
    return out.content.strip().lower()


def triage(text: str) -> str:
    category = classify_ticket(text)
    # routing as branching logic rather than a StateGraph with conditional edges
    if category == "password":
        return llm.invoke(f"Draft password-reset steps for: {text}").content
    elif category == "network":
        return llm.invoke(f"Draft network-troubleshooting steps for: {text}").content
    elif category == "hardware":
        return llm.invoke(f"Draft a hardware-RMA response for: {text}").content
    elif category == "access":
        return llm.invoke(f"Draft an access-request response for: {text}").content
    else:
        return llm.invoke(f"Draft a generic helpdesk response for: {text}").content


if __name__ == "__main__":
    print(triage("I can't log in, my password keeps getting rejected"))
