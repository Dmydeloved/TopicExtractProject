from __future__ import annotations

import json
from typing import Any

from .storage import MemoryStorage


class StructuredRetriever:
    """Basic structured recall for the MVP memory system."""

    def __init__(self, storage: MemoryStorage) -> None:
        self.storage = storage

    def recall(self, topic: str, core_entity: str, intent: str | None = None) -> dict[str, Any]:
        experience = self.storage.find_experience(topic, core_entity)
        if not experience:
            return {"experience": None, "segments": [], "qas": []}

        segments = [
            self.storage.get_segment(segment_id)
            for segment_id in experience.get("segment_ids", [])
        ]
        segments = [segment for segment in segments if segment]
        if intent:
            segments = [segment for segment in segments if segment["intent"] == intent]

        qa_ids = [qa_id for segment in segments for qa_id in segment.get("qa_ids", [])]
        qas = []
        for qa_id in qa_ids:
            row = self.storage.connection.execute(
                "SELECT * FROM qa_memory WHERE qa_id = ?",
                (qa_id,),
            ).fetchone()
            qa = self.storage._row_to_dict(row)
            if qa:
                qas.append(qa)
        return {"experience": experience, "segments": segments, "qas": qas}
