# ACE-EXPECT: detect
# CATEGORY: should_detect/nondeterministic_float_scoring
# LANGUAGE: python
# SOURCE: SK ATIT project feedback — Talent-AX (Notion: https://app.notion.com/p/sk-atit/Talent-AX-9e81772c74d583c3ae85819c9873985b)
# ISSUE: The LLM is asked to emit a free continuous float score (0.0–1.0) used to rank items, which is poorly reproducible across runs
# EXPECTED-FINDING: A ranking decision depends on a model-generated continuous `score: float` in [0.0, 1.0]; small run-to-run variation in the float reorders results, so the same input can yield different rankings with no determinism control
# EXPECTED-FIX: Make scoring reproducible — use a small discrete scale (int / Literal bucket), reduce the value range, set a seed where supported, or average over a small batch — instead of a single free-form float
# SEVERITY-HINT: suggestion
"""Priority ranking driven by a model-produced continuous float score.

Because the score is a free 0.0–1.0 float chosen by the LLM, two runs on identical
input can produce slightly different scores and therefore a different ranking. For a
decision that feeds scheduling/prioritization, that non-reproducibility is a real risk.
"""
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class SectionScore(BaseModel):
    section: str = Field(description="섹션명")
    score: float = Field(ge=0.0, le=1.0)  # continuous, model-chosen -> low reproducibility
    reason: str = Field(description="점수 판단 근거")


class PriorityScores(BaseModel):
    scores: List[SectionScore]


llm = ChatOpenAI(model="gpt-4o")


def rank_sections(prompt: str) -> List[SectionScore]:
    structured_llm = llm.with_structured_output(PriorityScores)
    response: PriorityScores = structured_llm.invoke(prompt)
    # The final order depends on a free float the model picked — not reproducible.
    return sorted(response.scores, key=lambda s: s.score, reverse=True)
