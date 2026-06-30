# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: text-embedding-ada-002 + openai-python
# WHY-CORRECT: text-embedding-ada-002 is an older but fully valid embeddings model; the embeddings.create call shape is correct.
# EXPECTED-WRONG: engine flags "use text-embedding-3-small/large instead" as a finding, but embedding model choice is not a best-practice defect.
# CORRECT-VERDICT: no findings
"""Generate a document embedding with text-embedding-ada-002."""

from openai import OpenAI

client = OpenAI()


def embed(text: str) -> list[float]:
    """Return the embedding vector for a single document."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return response.data[0].embedding


if __name__ == "__main__":
    vec = embed("Vector databases store high-dimensional embeddings.")
    print(f"dim={len(vec)}")
