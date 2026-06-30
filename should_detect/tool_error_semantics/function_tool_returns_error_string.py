# ACE-EXPECT: detect
# CATEGORY: should_detect/tool_error_semantics
# LANGUAGE: python
# ISSUE: An OpenAI function-tool returns a bare "ERROR: ..." string on failure instead of a structured error object
# EXPECTED-FINDING: The model receives an unstructured error string with no error_code/recoverable/suggestion, so it cannot reliably distinguish a retryable failure from a permanent one
# EXPECTED-FIX: Return a structured payload like {"status": "error", "error_code": "FILE_TOO_LARGE", "recoverable": False, "suggestion": ...} the LLM can branch on
# SEVERITY-HINT: warning
"""A read_document tool exposed to an OpenAI tool-calling agent."""

import json
import os

MAX_BYTES = 1_000_000


def read_document(path: str) -> str:
    """Read a text document for the agent. Returns the file contents as a string."""
    if not os.path.exists(path):
        return f"ERROR: file not found at {path}"
    if os.path.getsize(path) > MAX_BYTES:
        return "ERROR: file too large to read"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return "ERROR: could not decode file as utf-8"


# The agent feeds whatever string comes back to the model as the tool result.
def dispatch_tool_call(name: str, arguments: str) -> str:
    args = json.loads(arguments)
    if name == "read_document":
        return read_document(args["path"])
    return f"ERROR: unknown tool {name}"
