# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: AutoGen (ag2) ConversableAgent + GroupChat + GroupChatManager round-robin
# WHY-CORRECT: This is the canonical AutoGen multi-agent pattern: a UserProxyAgent with
#              human_input_mode set, several AssistantAgents each with a system_message and
#              an llm_config, wired into a GroupChat managed by a GroupChatManager. The
#              termination is handled by is_termination_msg; the LLM transport is configured
#              via llm_config (config_list), not by raw client calls.
# EXPECTED-WRONG: structured-output / prompt checks may flag "no output schema" and the
#                 lambda is_termination_msg may look like ad-hoc parsing; human_input_mode
#                 "NEVER" can be misread as an unsafe autonomous-agent smell.
# CORRECT-VERDICT: no findings
"""Idiomatic AutoGen (ag2): conversable agents in a managed group chat."""
from __future__ import annotations

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "ENV"}],
    "temperature": 0,
    "cache_seed": 42,
}


def _is_done(message: dict) -> bool:
    """Terminate the chat when an agent signals completion."""
    content = (message.get("content") or "").strip()
    return content.endswith("TERMINATE")


user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=4,
    is_termination_msg=_is_done,
    code_execution_config=False,
)

planner = AssistantAgent(
    name="planner",
    system_message="Break the task into concrete steps. Append TERMINATE when satisfied.",
    llm_config=llm_config,
)

critic = AssistantAgent(
    name="critic",
    system_message="Critique the plan for gaps and risks before it is accepted.",
    llm_config=llm_config,
)


def build_manager() -> GroupChatManager:
    """Assemble a round-robin group chat over the planner and critic."""
    group_chat = GroupChat(
        agents=[user_proxy, planner, critic],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin",
    )
    return GroupChatManager(groupchat=group_chat, llm_config=llm_config)


def solve(task: str) -> None:
    """Start the multi-agent conversation for the given task."""
    user_proxy.initiate_chat(build_manager(), message=task)
