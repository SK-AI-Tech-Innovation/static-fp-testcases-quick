# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: Anthropic "Contextual Retrieval" — prepend LLM-generated chunk context before embedding
# WHY-CORRECT: Contextual Retrieval improves recall by having an LLM write a short situating
#   context for each chunk (given the whole document) and PREPENDING it to the chunk before
#   embedding/indexing. The full document is sent with cache_control so the per-chunk calls reuse
#   the prompt cache. Embedding the context-augmented chunk is the entire point — not a bug.
# EXPECTED-WRONG: engine may flag "embedding modified/augmented text instead of the raw chunk",
#   claim prepending generated context corrupts the index, or that cache_control is misused.
# CORRECT-VERDICT: no findings
"""Anthropic Contextual Retrieval: situate each chunk with an LLM, then embed the augmented text."""
import anthropic
from openai import OpenAI

llm = anthropic.Anthropic()
embedder = OpenAI()

CONTEXT_PROMPT = (
    "Here is a chunk we want to situate within the whole document:\n<chunk>{chunk}</chunk>\n"
    "Give a short context to situate this chunk for search. Answer only with the context."
)


def situate_chunk(document: str, chunk: str) -> str:
    resp = llm.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>{document}</document>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": CONTEXT_PROMPT.format(chunk=chunk)},
                ],
            }
        ],
    )
    return resp.content[0].text


def embed_contextual_chunk(document: str, chunk: str) -> list[float]:
    context = situate_chunk(document, chunk)
    augmented = f"{context}\n\n{chunk}"
    out = embedder.embeddings.create(model="text-embedding-3-large", input=augmented)
    return out.data[0].embedding


if __name__ == "__main__":
    doc = "Acme Corp Q3 report. Revenue grew on cloud demand."
    vec = embed_contextual_chunk(doc, "Revenue grew on cloud demand.")
    print(len(vec))
