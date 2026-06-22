from __future__ import annotations

import logging
from typing import Any

from .embedder import TextEmbedder, tokenize, topic_entity_text
from .storage import MemoryStorage
from .vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


class HybridRetriever:
    """SQLite keyword recall + Chroma semantic recall, returned as a timeline."""

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: ChromaVectorStore,
        embedder: TextEmbedder,
        keyword_weight: float = 0.45,
        semantic_weight: float = 0.55,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.embedder = embedder
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

    def recall(self, topic: str, core_entity: str, intent: str | None = None, entities: list[str] | None = None, top_k: int = 5) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_text = topic_entity_text(topic, core_entity, entities or intent)
        query_tokens = set(tokenize(query_text))
        semantic_candidates = self.vector_store.query(
            query_embedding=self.embedder.embed(query_text),
            memory_type="qa",
            top_k=max(top_k * 4, 20),
        )
        semantic_scores = {item["metadata"]["memory_id"]: item["similarity"] for item in semantic_candidates}
        scored = []
        for qa in self.storage.list_qas():
            keyword_score = self._keyword_score(query_tokens, qa)
            if intent and qa["intent"] == intent:
                keyword_score = min(1.0, keyword_score + 0.15)
            semantic_score = semantic_scores.get(qa["qa_id"], 0.0)
            score = self.keyword_weight * keyword_score + self.semantic_weight * semantic_score
            if score > 0:
                scored.append({"qa": qa, "score": score, "keyword_score": keyword_score, "semantic_score": semantic_score})
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
        results = sorted(ranked, key=lambda item: item["qa"]["timestamp"])
        logger.info("混合召回完成 semantic_candidates=%s candidates=%s returned=%s", len(semantic_candidates), len(scored), len(results))
        return {"query": {"topic": topic, "core_entity": core_entity, "intent": intent, "entities": entities or []}, "results": results}

    def _keyword_score(self, query_tokens: set[str], qa: dict[str, Any]) -> float:
        memory_text = " ".join([qa.get("topic", ""), qa.get("core_entity", ""), qa.get("intent", ""), " ".join(qa.get("entities") or []), qa.get("user_input", ""), qa.get("assistant_output", "")])
        memory_tokens = set(tokenize(memory_text))
        return len(query_tokens & memory_tokens) / len(query_tokens) if query_tokens and memory_tokens else 0.0


class StructuredRetriever(HybridRetriever):
    """Backward-compatible alias."""