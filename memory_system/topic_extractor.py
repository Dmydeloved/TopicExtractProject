from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from .topic_prompts import common_extractor_prompt


RESULT_FIELDS = (
    "topic",
    "core_entity",
    "intent",
    "entities",
    "confidence",
    "reasoning",
)
REQUIRED_RESULT_FIELDS = set(RESULT_FIELDS)
DEFAULT_TOPIC_EXTRACT_MODEL = "qwen-plus"
DEFAULT_TOPIC_EXTRACT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TopicRecord = dict[str, Any]
TopicResult = TopicRecord | list[TopicRecord]


def strip_markdown_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_topic_response(content: str) -> TopicResult:
    payload = json.loads(strip_markdown_code_fence(content))

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Topic response array must not be empty.")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(
                "Every topic record in the response array must be a JSON object."
            )
        return payload[0] if len(payload) == 1 else payload

    if not isinstance(payload, dict):
        raise ValueError("Topic response must be a JSON object or a JSON array.")

    return payload


def validate_topic_record(value: Any) -> TopicRecord:
    if not isinstance(value, dict):
        raise ValueError("Topic result must be a JSON object.")
    missing = REQUIRED_RESULT_FIELDS - value.keys()
    if missing:
        raise ValueError(f"Topic result is missing fields: {sorted(missing)}")

    result = {key: value[key] for key in RESULT_FIELDS}
    for key in ("topic", "core_entity", "intent", "reasoning"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
        result[key] = result[key].strip()

    entities = result["entities"]
    if not isinstance(entities, list) or not entities:
        raise ValueError("entities must be a non-empty list.")
    result["entities"] = [str(entity).strip() for entity in entities if str(entity).strip()]
    if not result["entities"]:
        raise ValueError("entities must contain at least one non-empty value.")

    confidence = result["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be numeric.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")
    result["confidence"] = float(confidence)
    return result


def validate_topic_result(value: Any) -> TopicResult:
    if isinstance(value, list):
        if not value:
            raise ValueError("Topic result list must not be empty.")
        return [validate_topic_record(item) for item in value]
    return validate_topic_record(value)


def add_result_metadata(topic_result: TopicResult, user_input: str) -> TopicResult:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(topic_result, list):
        return [
            {
                **item,
                "user_input": user_input,
                "timestamp": timestamp,
            }
            for item in topic_result
        ]

    return {
        **topic_result,
        "user_input": user_input,
        "timestamp": timestamp,
    }


class BailianTopicExtractor:
    """OpenAI-compatible topic extractor backed by DashScope/Bailian."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_TOPIC_EXTRACT_MODEL,
        base_url: str = DEFAULT_TOPIC_EXTRACT_BASE_URL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        from openai import OpenAI

        api_key = api_key or os.getenv("TOPIC_EXTRACT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set TOPIC_EXTRACT_API_KEY or DASHSCOPE_API_KEY before creating BailianTopicExtractor."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def extract(
        self,
        user_input: str,
        context: str = "",
        domain_knowledge: str = "",
    ) -> TopicResult:
        prompt = common_extractor_prompt(user_input, context, domain_knowledge)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (response.choices[0].message.content or "").strip()
                return validate_topic_result(parse_topic_response(content))
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"Topic extraction failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


def run_entity_extract(
    user_input: str,
    ctx: str = "",
    kg: str = "",
    *,
    api_key: str | None = None,
    model: str = DEFAULT_TOPIC_EXTRACT_MODEL,
    base_url: str = DEFAULT_TOPIC_EXTRACT_BASE_URL,
) -> TopicResult:
    extractor = BailianTopicExtractor(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    return extractor.extract(user_input=user_input, context=ctx, domain_knowledge=kg)


__all__ = [
    "BailianTopicExtractor",
    "TopicRecord",
    "TopicResult",
    "add_result_metadata",
    "parse_topic_response",
    "run_entity_extract",
    "strip_markdown_code_fence",
    "validate_topic_record",
    "validate_topic_result",
]
