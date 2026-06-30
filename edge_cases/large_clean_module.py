# ACE-FP-EXPECT: clean
# CATEGORY: edge_cases/large_file
# LANGUAGE: python
# SOURCE: synthetic; mirrors a production RAG + chat assistant module
# WHY-CORRECT: every LLM call uses schema-enforced output or grounding, bounded retries with backoff, and structured logging; no untrusted sink, no missing fallback
# EXPECTED-WRONG: flagging "many functions in one file" as a god object, or claiming missing retries/observability that are actually present
# CORRECT-VERDICT: no findings
"""A large but correct AI assistant module.

Provides:
  - bounded, observable chat completions (Anthropic SDK)
  - retrieval-augmented answering with explicit grounding instructions
  - schema-enforced structured extraction (OpenAI SDK, json_schema)
  - a small embedding-backed retriever

This file is deliberately long to exercise the static analyzer against a
sizeable, fully-correct input. There should be no findings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import anthropic
from openai import OpenAI

logger = logging.getLogger("ai.assistant")

ANTHROPIC_MODEL = "claude-opus-4-20250514"
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0


@dataclass(frozen=True)
class Document:
    """A retrievable chunk with its source metadata."""

    doc_id: str
    text: str
    source_uri: str
    score: float = 0.0


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class RetrievalResult:
    documents: list[Document] = field(default_factory=list)

    def as_grounding_block(self) -> str:
        """Render retrieved documents into a citation-friendly context block."""
        lines: list[str] = []
        for doc in self.documents:
            lines.append(f"[{doc.doc_id}] (source={doc.source_uri})\n{doc.text}")
        return "\n\n".join(lines)


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with a hard ceiling, used by all retry loops."""
    delay = min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
    logger.info("backing off", extra={"attempt": attempt, "delay_seconds": delay})
    time.sleep(delay)


class AnthropicChatClient:
    """Thin wrapper around the Anthropic Messages API with retries + logging."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()

    def complete(
        self,
        system: str,
        messages: Sequence[ChatTurn],
        max_tokens: int = 1024,
    ) -> str:
        """Run a chat completion with bounded retries and structured logs.

        Raises the last error after MAX_RETRIES exhausted so callers can
        surface a degraded-mode response rather than silently swallowing it.
        """
        payload = [{"role": t.role, "content": t.content} for t in messages]
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    "anthropic.complete.start",
                    extra={"attempt": attempt, "model": ANTHROPIC_MODEL},
                )
                response = self._client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=payload,
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                logger.info(
                    "anthropic.complete.ok",
                    extra={
                        "attempt": attempt,
                        "output_tokens": response.usage.output_tokens,
                    },
                )
                return text
            except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
                last_error = exc
                logger.warning(
                    "anthropic.complete.retryable",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                _sleep_backoff(attempt)

        logger.error(
            "anthropic.complete.exhausted",
            extra={"retries": MAX_RETRIES, "error": str(last_error)},
        )
        assert last_error is not None
        raise last_error


class EmbeddingRetriever:
    """A minimal cosine-similarity retriever over an in-memory corpus."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI()
        self._corpus: list[Document] = []
        self._vectors: list[list[float]] = []

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.embeddings.create(
                    model=OPENAI_EMBED_MODEL,
                    input=list(texts),
                )
                return [item.embedding for item in resp.data]
            except Exception as exc:  # narrow retry: embeddings are idempotent
                last_error = exc
                logger.warning(
                    "embed.retryable",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                _sleep_backoff(attempt)
        assert last_error is not None
        raise last_error

    def index(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        if not docs:
            logger.info("retriever.index.empty")
            return
        self._vectors.extend(self._embed([d.text for d in docs]))
        self._corpus.extend(docs)
        logger.info("retriever.index.ok", extra={"count": len(docs)})

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 4) -> RetrievalResult:
        if not self._corpus:
            logger.info("retriever.search.empty_corpus")
            return RetrievalResult(documents=[])
        query_vec = self._embed([query])[0]
        scored: list[Document] = []
        for doc, vec in zip(self._corpus, self._vectors):
            score = self._cosine(query_vec, vec)
            scored.append(
                Document(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    source_uri=doc.source_uri,
                    score=score,
                )
            )
        scored.sort(key=lambda d: d.score, reverse=True)
        top = scored[:top_k]
        logger.info(
            "retriever.search.ok",
            extra={"query_len": len(query), "returned": len(top)},
        )
        return RetrievalResult(documents=top)


GROUNDING_SYSTEM = (
    "You are a careful assistant. Answer ONLY using the provided context. "
    "Cite the document id in square brackets for every claim. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information to answer that.\""
)


def answer_with_grounding(
    chat: AnthropicChatClient,
    retriever: EmbeddingRetriever,
    question: str,
) -> str:
    """RAG answer: retrieve, ground the prompt, then complete.

    The model is explicitly instructed to refuse when context is missing,
    so the output is grounded and there is no hallucination sink.
    """
    result = retriever.search(question)
    if not result.documents:
        logger.info("rag.no_context", extra={"question_len": len(question)})
        return "I don't have enough information to answer that."

    context_block = result.as_grounding_block()
    user_content = f"Context:\n{context_block}\n\nQuestion: {question}"
    return chat.complete(
        system=GROUNDING_SYSTEM,
        messages=[ChatTurn(role="user", content=user_content)],
        max_tokens=800,
    )


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "company": {"type": "string"},
        "intent": {
            "type": "string",
            "enum": ["sales", "support", "billing", "other"],
        },
        "priority": {"type": "integer"},
    },
    "required": ["name", "email", "company", "intent", "priority"],
    "additionalProperties": False,
}


class StructuredExtractor:
    """Schema-enforced extraction via OpenAI structured outputs.

    Uses json_schema with strict=True so the response is guaranteed to
    validate. No json.loads on free-form model text — the SDK parses it.
    """

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI()

    def extract_contact(self, raw_text: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info("extract.start", extra={"attempt": attempt})
                resp = self._client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract the contact record from the message.",
                        },
                        {"role": "user", "content": raw_text},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "contact_record",
                            "strict": True,
                            "schema": EXTRACTION_SCHEMA,
                        },
                    },
                )
                parsed = resp.choices[0].message.parsed
                logger.info("extract.ok", extra={"attempt": attempt})
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "extract.retryable",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                _sleep_backoff(attempt)
        logger.error("extract.exhausted", extra={"error": str(last_error)})
        assert last_error is not None
        raise last_error


def build_default_pipeline() -> tuple[AnthropicChatClient, EmbeddingRetriever, StructuredExtractor]:
    """Wire the components together with their default SDK clients."""
    chat = AnthropicChatClient()
    retriever = EmbeddingRetriever()
    extractor = StructuredExtractor()
    logger.info("pipeline.ready")
    return chat, retriever, extractor


def run_demo(question: str, corpus: Sequence[Document]) -> str:
    """End-to-end demo helper used by integration tests."""
    chat, retriever, _ = build_default_pipeline()
    retriever.index(corpus)
    return answer_with_grounding(chat, retriever, question)
