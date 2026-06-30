# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: arize-phoenix / openinference-instrumentation-openai
# WHY-CORRECT: OpenInference auto-instrumentation patches the OpenAI client so every call emits an OpenTelemetry span with prompts, model, latency, and token usage exported to Phoenix. Observability (tracing + token/cost capture) is fully handled at the instrumentation layer.
# EXPECTED-WRONG: missing observability / no logging or token tracking on the LLM call
# CORRECT-VERDICT: no findings
"""Auto-instrument the OpenAI client with OpenInference and export to Phoenix."""

from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

# register() wires up the OTel tracer provider and exporter to Phoenix.
tracer_provider = register(project_name="chat-service", auto_instrument=True)

# Patches the OpenAI SDK so each request becomes a fully-attributed span.
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = OpenAI()


def chat(prompt: str) -> str:
    """Every call here is traced to Phoenix with tokens and latency captured."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
