# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: ChromaDB (`chromadb`) PersistentClient + OpenAI chat
# WHY-CORRECT: textbook RAG flow — create collection, add documents (Chroma embeds them with
#              its default embedding function), query top-k by text, inject the retrieved
#              passages into the prompt, then answer. Embed -> store -> retrieve -> inject -> answer.
# EXPECTED-WRONG: engine invents "add reranking", "tune chunk overlap", or "cache embeddings"
# CORRECT-VERDICT: no findings
"""Answer a question over a Chroma collection by injecting retrieved passages."""
import chromadb
from openai import OpenAI

llm = OpenAI()
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="docs")


def index(documents: list[str]) -> None:
    collection.add(
        ids=[f"doc-{i}" for i in range(len(documents))],
        documents=documents,
    )


def answer(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])
    prompt = f"Use the context to answer.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    index(["Paris is the capital of France.", "Berlin is the capital of Germany."])
    print(answer("What is the capital of France?"))
