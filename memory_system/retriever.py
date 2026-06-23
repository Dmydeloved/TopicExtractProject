from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from .embedder import (
    TextEmbedder,
    build_experience_embedding_text,
    build_qa_embedding_text,
    build_segment_embedding_text,
    tokenize,
)
from .storage import MemoryStorage
from .vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default.copy() if hasattr(default, "copy") else default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default.copy() if hasattr(default, "copy") else default


def normalize_entity(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def jaccard_score(left: list[str] | None, right: list[str] | None) -> float:
    left_set = {normalize_entity(item) for item in left or [] if normalize_entity(item)}
    right_set = {normalize_entity(item) for item in right or [] if normalize_entity(item)}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def keyword_score(query_tokens: set[str], text: str) -> float:
    memory_tokens = set(tokenize(text))
    if not query_tokens or not memory_tokens:
        return 0.0
    return len(query_tokens & memory_tokens) / len(query_tokens)


def recency_score(value: str | None, half_life_days: float = 180.0) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = max(
            0.0,
            (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
            / 86400.0,
        )
        return clamp01(math.exp(-math.log(2) * age_days / half_life_days))
    except (TypeError, ValueError, OverflowError):
        return 0.5


def parse_summary_json(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        parsed = safe_json_loads(stripped, None)
        if parsed is None:
            return "" if stripped[:1] in "[{" else stripped
    else:
        parsed = value
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        if isinstance(parsed.get("summary"), str):
            return parsed["summary"].strip()
        for key in ("long", "short"):
            if isinstance(parsed.get(key), str) and parsed[key].strip():
                return parsed[key].strip()
    return ""


def _core_entity_score(query: str, candidate: str) -> float:
    query_value = normalize_entity(query)
    candidate_value = normalize_entity(candidate)
    if not query_value or not candidate_value:
        return 0.0
    if query_value == candidate_value:
        return 1.0
    aliases = {
        normalize_entity(part)
        for part in re.split(r"[/|,，、;；()]", str(candidate))
        if normalize_entity(part)
    }
    if query_value in aliases:
        return 0.8
    if query_value in candidate_value or candidate_value in query_value:
        return 0.5
    return 0.5 if jaccard_score(tokenize(query), tokenize(candidate)) > 0 else 0.0


def parse_state_json(value: Any) -> dict[str, Any]:
    parsed = safe_json_loads(value, {})
    return parsed if isinstance(parsed, dict) else {}


def parse_intents_json(value: Any) -> list[str]:
    parsed = safe_json_loads(value, [])
    return [str(item) for item in parsed if str(item).strip()] if isinstance(parsed, list) else []


def parse_entities_json(value: Any) -> list[str]:
    parsed = safe_json_loads(value, [])
    return [str(item) for item in parsed if str(item).strip()] if isinstance(parsed, list) else []


def build_query_text(
    topic: str,
    core_entity: str,
    intent: str | None = None,
    entities: list[str] | None = None,
) -> str:
    return "\n".join(
        [
            f"主题：{topic}",
            f"核心实体：{core_entity}",
            f"用户意图：{intent or ''}",
            f"相关实体：{'、'.join(entities or [])}",
        ]
    )


def build_context_text(
    experiences: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    qas: list[dict[str, Any]],
) -> str:
    lines = ["【长期记忆 Experience】"]
    if not experiences:
        lines.append("未找到相关长期记忆。")
    for index, experience in enumerate(experiences, 1):
        prefix = f"{index}. " if len(experiences) > 1 else ""
        lines.extend(
            [
                f"{prefix}主题：{experience['topic']}",
                f"核心实体：{experience['core_entity']}",
                f"相关意图：{'、'.join(experience['intents'])}",
                f"摘要：{experience['summary']}",
                "",
            ]
        )
    lines.append("【相关阶段 Segment】")
    if not segments:
        lines.append("未找到相关阶段。")
    for index, segment in enumerate(segments, 1):
        lines.extend(
            [
                f"{index}. 意图：{segment['intent']}",
                f"   摘要：{segment['summary']}",
                "",
            ]
        )
    lines.append("【原始问答 QA】")
    if not qas:
        lines.append("未找到相关原始问答。")
    for index, qa in enumerate(qas, 1):
        lines.extend(
            [
                f"{index}. 时间：{qa['timestamp']}",
                f"用户：{qa['user_input']}",
                f"助手：{qa['assistant_output']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


class HybridRetriever:
    """Hierarchical structured-first Experience -> Segment -> QA retriever."""

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: ChromaVectorStore | None = None,
        embedder: TextEmbedder | None = None,
        **_: Any,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.embedder = embedder

    def recall(
        self,
        topic: str,
        core_entity: str,
        intent: str | None = None,
        entities: list[str] | None = None,
        top_experience: int = 3,
        top_segment: int = 5,
        top_qa: int = 8,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if top_k is not None:
            top_qa = top_k
        if min(top_experience, top_segment, top_qa) <= 0:
            raise ValueError("top_experience, top_segment and top_qa must be positive")

        query_entities = [str(item) for item in entities or [] if str(item).strip()]
        query_text = build_query_text(topic, core_entity, intent, query_entities)
        query_tokens = set(tokenize(query_text))
        experience_vectors = self._vector_scores("experience", query_text, max(20, top_experience * 8))
        experiences, experience_count = self._recall_experiences(
            topic, core_entity, intent, query_text, experience_vectors, top_experience
        )
        segment_vectors = self._vector_scores("segment", query_text, max(20, top_segment * 8))
        segments, segment_count = self._recall_segments(
            experiences, intent, query_text, query_tokens, segment_vectors, top_segment
        )
        qa_vectors = self._vector_scores("qa", query_text, max(20, top_qa * 8))
        qas, qa_count = self._recall_qas(
            segments, intent, query_entities, query_text, query_tokens, qa_vectors, top_qa
        )

        debug = {
            "experience_candidates": experience_count,
            "segment_candidates": segment_count,
            "qa_candidates": qa_count,
            "vector_experience_candidates": len(experience_vectors),
            "vector_segment_candidates": len(segment_vectors),
            "vector_qa_candidates": len(qa_vectors),
        }
        logger.info(
            "分层召回完成 experience=%s/%s segment=%s/%s qa=%s/%s vectors=%s/%s/%s",
            len(experiences), experience_count, len(segments), segment_count,
            len(qas), qa_count, len(experience_vectors), len(segment_vectors), len(qa_vectors),
        )
        return {
            "query": {
                "topic": topic,
                "core_entity": core_entity,
                "intent": intent or "",
                "entities": query_entities,
            },
            "experiences": experiences,
            "segments": segments,
            "qas": qas,
            "context_text": build_context_text(experiences, segments, qas),
            "debug": debug,
            "results": [{"qa": qa, "score": qa["score"]} for qa in qas],
        }

    def _vector_scores(
        self, memory_type: str, query_text: str, top_k: int
    ) -> dict[str, float]:
        if self.vector_store is None or self.embedder is None:
            return {}
        try:
            items = self.vector_store.query(
                query_embedding=self.embedder.embed(query_text),
                memory_type=memory_type,
                top_k=top_k,
            )
        except Exception:
            logger.exception("%s 向量召回失败，继续使用结构化检索", memory_type)
            return {}
        scores: dict[str, float] = {}
        for item in items:
            metadata = item.get("metadata") or {}
            memory_id = metadata.get("memory_id") or metadata.get(f"{memory_type}_id")
            if memory_id:
                scores[str(memory_id)] = clamp01(float(item.get("similarity", 0.0)))
        return scores

    def _recall_experiences(
        self,
        topic: str,
        core_entity: str,
        intent: str | None,
        query_text: str,
        semantic_scores: dict[str, float],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        exact = self.storage.find_experiences(topic, core_entity, max(limit * 4, limit))
        candidates = list(exact)
        if not candidates:
            candidates = self.storage.list_experiences_by_topic(topic, max(limit * 8, 20))
            known = {item["experience_id"] for item in candidates}
            vector_rows = self.storage.get_experiences(list(semantic_scores))
            candidates.extend(
                item for item in vector_rows if item["experience_id"] not in known
            )

        scored: list[dict[str, Any]] = []
        for item in candidates:
            intents = parse_intents_json(item.get("intents_link", item.get("intents_link_json")))
            summary = parse_summary_json(item.get("summary", item.get("summary_json")))
            state = parse_state_json(item.get("state", item.get("state_json")))
            topic_value = 1.0 if item.get("topic") == topic else 0.0
            core_value = _core_entity_score(core_entity, item.get("core_entity", ""))
            intent_value = 1.0 if intent and intent in intents else 0.0
            semantic_value = semantic_scores.get(item["experience_id"], 0.0)
            score = (
                0.45 * topic_value
                + 0.30 * core_value
                + 0.10 * intent_value
                + 0.10 * semantic_value
                + 0.05 * recency_score(item.get("updated_at"))
            )
            scored.append({
                "experience_id": item["experience_id"],
                "topic": item.get("topic", ""),
                "core_entity": item.get("core_entity", ""),
                "intents": intents,
                "summary": summary,
                "state": state,
                "score": score,
                "updated_at": item.get("updated_at", ""),
            })
        scored.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
        logger.info("Experience 候选数量=%s 向量候选数量=%s", len(candidates), len(semantic_scores))
        return scored[:limit], len(candidates)

    def _recall_segments(
        self,
        experiences: list[dict[str, Any]],
        intent: str | None,
        query_text: str,
        query_tokens: set[str],
        semantic_scores: dict[str, float],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        experience_ids = [item["experience_id"] for item in experiences]
        candidates = self.storage.list_segments_by_experience_ids(experience_ids)
        scored: list[dict[str, Any]] = []
        for item in candidates:
            segment_intent = str(item.get("intent") or "")
            if not intent:
                intent_value = 0.2
            elif intent == segment_intent:
                intent_value = 1.0
            elif keyword_score(set(tokenize(intent)), segment_intent) > 0:
                intent_value = 0.6
            else:
                intent_value = 0.2
            summary = str(item.get("summary") or "")
            summary_value = keyword_score(query_tokens, summary)
            memory_text = " ".join(
                str(item.get(key) or "")
                for key in ("topic", "core_entity", "intent", "summary")
            )
            keyword_value = keyword_score(query_tokens, memory_text)
            semantic_value = semantic_scores.get(item["segment_id"], 0.0)
            score = (
                0.35 * intent_value
                + 0.30 * summary_value
                + 0.20 * semantic_value
                + 0.10 * keyword_value
                + 0.05 * recency_score(item.get("updated_at"))
            )
            scored.append({
                "segment_id": item["segment_id"],
                "experience_id": item["experience_id"],
                "topic": item.get("topic", ""),
                "core_entity": item.get("core_entity", ""),
                "intent": segment_intent,
                "status": item.get("status", ""),
                "summary": summary,
                "score": score,
                "updated_at": item.get("updated_at", ""),
            })
        scored.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
        logger.info("Segment 候选数量=%s 向量候选数量=%s", len(candidates), len(semantic_scores))
        return scored[:limit], len(candidates)

    def _recall_qas(
        self,
        segments: list[dict[str, Any]],
        intent: str | None,
        entities: list[str],
        query_text: str,
        query_tokens: set[str],
        semantic_scores: dict[str, float],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        segment_ids = [item["segment_id"] for item in segments]
        candidates = self.storage.list_qas_by_segment_ids(segment_ids)
        scored: list[dict[str, Any]] = []
        for item in candidates:
            qa_entities = parse_entities_json(item.get("entities", item.get("entities_json")))
            memory_text = " ".join([
                str(item.get("topic") or ""),
                str(item.get("core_entity") or ""),
                str(item.get("intent") or ""),
                " ".join(qa_entities),
                str(item.get("user_input") or ""),
                str(item.get("assistant_output") or ""),
            ])
            semantic_value = semantic_scores.get(item["qa_id"], 0.0)
            keyword_value = keyword_score(query_tokens, memory_text)
            intent_value = 1.0 if intent and item.get("intent") == intent else 0.0
            entity_value = jaccard_score(entities, qa_entities)
            confidence = clamp01(float(item.get("confidence") or 0.0))
            score = (
                0.25 * semantic_value
                + 0.25 * keyword_value
                + 0.15 * intent_value
                + 0.15 * entity_value
                + 0.10 * recency_score(item.get("timestamp"))
                + 0.10 * confidence
            )
            scored.append({
                "qa_id": item["qa_id"],
                "segment_id": item["segment_id"],
                "timestamp": item.get("timestamp", ""),
                "user_input": item.get("user_input", ""),
                "assistant_output": item.get("assistant_output", ""),
                "topic": item.get("topic", ""),
                "core_entity": item.get("core_entity", ""),
                "intent": item.get("intent", ""),
                "entities": qa_entities,
                "confidence": confidence,
                "score": score,
                "reasoning": item.get("reasoning", ""),
            })
        ranked = sorted(
            scored, key=lambda item: (item["score"], item["timestamp"]), reverse=True
        )[:limit]
        ranked.sort(key=lambda item: item["timestamp"])
        logger.info("QA 候选数量=%s 向量候选数量=%s", len(candidates), len(semantic_scores))
        return ranked, len(candidates)


class StructuredRetriever(HybridRetriever):
    """Backward-compatible alias."""
