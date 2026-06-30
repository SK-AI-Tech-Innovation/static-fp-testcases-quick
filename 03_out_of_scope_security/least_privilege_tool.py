# ACE-FP-EXPECT: scoped-out
# CATEGORY: 03_out_of_scope_security
# SOURCE: an agent tool whose executor is scoped to an allowlist of read-only paths (least privilege)
# WHY-CORRECT: least-privilege / permission scoping is a SECURITY concern owned by the Sentinel engine.
#              Static must stay silent on authorization and access scope. The code is also security-correct:
#              the file-read tool resolves paths and rejects anything outside an allowlisted directory.
#              The Anthropic tool-use wiring itself (schema, tool_use dispatch) is standard and correct.
# EXPECTED-WRONG: an authorization/privilege finding ("restrict tool permissions", "path traversal risk",
#                 "scope tool access") — security is out-of-scope; static must defer to Sentinel.
# CORRECT-VERDICT: no findings (least-privilege/authz is Sentinel's domain; static stays silent)
"""A read-only file tool scoped to an allowlisted dir, exposed to Claude (security-correct; defer to Sentinel)."""
from __future__ import annotations

import os

import anthropic

client = anthropic.Anthropic()

_ALLOWED_ROOT = os.path.realpath("/srv/agent_data")

TOOLS = [
    {
        "name": "read_doc",
        "description": "Read a UTF-8 text document from the agent's allowlisted data directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]


def read_doc(path: str) -> str:
    resolved = os.path.realpath(os.path.join(_ALLOWED_ROOT, path))
    # Enforce least privilege: never escape the allowlisted root.
    if not resolved.startswith(_ALLOWED_ROOT + os.sep):
        return "access denied"
    with open(resolved, encoding="utf-8") as fh:
        return fh.read()


def run(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        return "".join(b.text for b in response.content if b.type == "text").strip()
    results = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "read_doc":
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": read_doc(block.input["path"]),
                }
            )
    messages.append({"role": "user", "content": results})
    final = client.messages.create(model="claude-opus-4-5", max_tokens=1024, tools=TOOLS, messages=messages)
    return "".join(b.text for b in final.content if b.type == "text").strip()


if __name__ == "__main__":
    print(run("Read notes.txt and tell me the first line."))
