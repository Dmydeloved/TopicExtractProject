from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import chromadb


class ChromaVectorStore:
    """Persistent Chroma collection for all memory-layer vectors."""

    def __init__(
        self,
        persist_path: str | Path = "data/memory/chroma",
        collection_name: str = "topic_memory",
        ephemeral: bool = False,
    ) -> None:
        self.persist_path = Path(persist_path)
        if not ephemeral:
            self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = (
            chromadb.EphemeralClient()
            if ephemeral
            else chromadb.PersistentClient(path=str(self.persist_path))
        )
        if ephemeral:
            collection_name = f"{collection_name}_{uuid.uuid4().hex}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, memory_type: str, memory_id: str, text: str, embedding: list[float], updated_at: str) -> None:
        self.collection.upsert(
            ids=[f"{memory_type}:{memory_id}"],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"memory_type": memory_type, "memory_id": memory_id, "updated_at": updated_at}],
        )

    def query(self, query_embedding: list[float], memory_type: str = "qa", top_k: int = 20) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where={"memory_type": memory_type},
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for index, chroma_id in enumerate(result["ids"][0]):
            distance = float(result["distances"][0][index])
            items.append({
                "chroma_id": chroma_id,
                "document": result["documents"][0][index],
                "metadata": result["metadatas"][0][index],
                "distance": distance,
                "similarity": max(0.0, min(1.0, 1.0 - distance)),
            })
        return items

    def count(self) -> int:
        return self.collection.count()