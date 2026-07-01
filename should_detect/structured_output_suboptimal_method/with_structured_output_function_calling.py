# ACE-EXPECT: detect
# CATEGORY: should_detect/structured_output_suboptimal_method
# LANGUAGE: python
# SOURCE: SK ATIT project feedback — Talent-AX (Notion: https://app.notion.com/p/sk-atit/Talent-AX-9e81772c74d583c3ae85819c9873985b)
# ISSUE: `with_structured_output(..., method="function_calling")` is pinned on an Azure OpenAI GPT model that natively supports structured outputs, forcing the slower/older tool-calling path
# EXPECTED-FINDING: The chain explicitly forces method="function_calling" for structured output even though the configured model supports native structured outputs (json_schema), adding unnecessary tool-call tokens and leaving the more reliable native path unused
# EXPECTED-FIX: Drop the explicit method="function_calling" (let LangChain pick the native json_schema path) or set method="json_schema" for models that support it; reserve function_calling only for models without native structured output
# SEVERITY-HINT: suggestion
"""Structured output pinned to the function_calling method on a native-capable model.

This is not *wrong* — function_calling produces valid structured output — but on an
Azure OpenAI GPT deployment that natively supports structured outputs it is the
sub-optimal choice: extra tool-call overhead and tokens for no benefit. The fix is a
one-line change, hence a suggestion rather than a warning.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field


class SectionPriority(BaseModel):
    section: str = Field(description="섹션명")
    priority: int = Field(ge=1, le=5, description="우선순위 (1=highest)")


class AllSectionsPriorityResponse(BaseModel):
    priorities: list[SectionPriority]


# A current Azure GPT deployment that supports native structured outputs.
llm = AzureChatOpenAI(azure_deployment="gpt-4o", api_version="2024-10-21", temperature=0)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Assign a priority to each section."),
        ("human", "{sections}"),
    ]
)

# method="function_calling" forces the tool-calling path even though this model
# supports the native json_schema structured-output path.
chain = prompt_template | llm.with_structured_output(
    AllSectionsPriorityResponse, method="function_calling"
)


def prioritize(sections: str) -> AllSectionsPriorityResponse:
    return chain.invoke({"sections": sections})
