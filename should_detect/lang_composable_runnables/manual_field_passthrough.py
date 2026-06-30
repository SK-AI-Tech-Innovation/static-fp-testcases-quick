# ACE-EXPECT: detect
# CATEGORY: should_detect/lang_composable_runnables
# LANGUAGE: python
# ISSUE: Input fields are threaded through by hand into a dict instead of using RunnablePassthrough
# EXPECTED-FINDING: Manual dict reassembly to carry the original question alongside the answer reimplements RunnablePassthrough
# EXPECTED-FIX: Use RunnablePassthrough.assign / RunnableParallel to keep the question while computing the answer
# SEVERITY-HINT: suggestion
"""QA helper that manually carries the question alongside the generated answer."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Answer concisely: {question}")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


def answer_with_context(question: str) -> dict:
    messages = prompt.invoke({"question": question})
    response = model.invoke(messages)
    answer = parser.invoke(response)
    result = {}
    result["question"] = question
    result["answer"] = answer
    return result
