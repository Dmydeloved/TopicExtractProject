from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")


class TextEmbedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Encode one text into a vector."""


class BailianEmbedder:
    """Alibaba Bailian embedding client using its OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-v4",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        from openai import OpenAI

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Set DASHSCOPE_API_KEY before creating BailianEmbedder.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text cannot be empty.")
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class HashingEmbedder:
    """Deterministic offline embedder used only by unit tests."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += (
                1.0 if digest[4] % 2 == 0 else -1.0
            )
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


def topic_entity_text(topic: str, core_entity: str, extra: object = None) -> str:
    parts = [topic, core_entity]
    if isinstance(extra, list):
        parts.extend(str(item) for item in extra)
    elif extra:
        parts.append(str(extra))
    return " ".join(part for part in parts if part)