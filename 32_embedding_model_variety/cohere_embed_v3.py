# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: Cohere Python SDK (`cohere`) `client.embed` v3
# WHY-CORRECT: Cohere embed v3 requires an `input_type` ("search_document" vs "search_query") so
#              document and query embeddings land in compatible spaces. Both are set correctly for
#              their respective sides. This is the documented, complete v3 usage.
# EXPECTED-WRONG: dated skill pack (trained on embed v2, which had no input_type) flags
#                 `input_type` as an "unknown argument" or claims it is unnecessary.
# CORRECT-VERDICT: no findings
"""Embed documents and queries with Cohere embed-english-v3.0 using input_type."""
import cohere

client = cohere.Client()
MODEL = "embed-english-v3.0"


def embed_documents(docs: list[str]) -> list[list[float]]:
    response = client.embed(texts=docs, model=MODEL, input_type="search_document")
    return response.embeddings


def embed_query(query: str) -> list[float]:
    response = client.embed(texts=[query], model=MODEL, input_type="search_query")
    return response.embeddings[0]


if __name__ == "__main__":
    docs = embed_documents(["the cat sat", "the dog ran"])
    q = embed_query("where is the cat")
    print(len(docs), len(q))
