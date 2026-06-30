#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_system import BailianEmbedder, ChromaVectorStore, MemoryStorage, StructuredRetriever


MEMORY_DB = PROJECT_ROOT / "data/memory/topic_memory.sqlite3"
CHROMA_DIR = PROJECT_ROOT / "data/memory/chroma"


def main() -> int:
    query_text = "我想找一家位于市中心、价格偏高的餐厅，可以告诉我推荐餐厅的电话号码吗？"

    try:
        embedder = BailianEmbedder(api_key='sk-29434e233c3e437281144cd9e2a1f04f')
    except ValueError as error:
        print(f"初始化 BailianEmbedder 失败: {error}")
        return 1

    storage = MemoryStorage(MEMORY_DB)
    try:
        retriever = StructuredRetriever(
            storage=storage,
            vector_store=ChromaVectorStore(persist_path=CHROMA_DIR),
            embedder=embedder,
            retrieval_model= 'gpt-5.5',
            retrieval_base_url= 'https://api.gpt.ge/v1/',
            retrieval_api_key='sk-RFUNAF0b6zfJWCVz9dA1Aa244aEa43Dc974693370b8d4338'
        )
        result = retriever.recall(
            topic="餐厅推荐",
            core_entity="餐厅",
            intent="查询",
            entities=["餐厅", "市中心", "高价位", "电话号码"],
            top_k=3,
        )
    finally:
        storage.close()

    print(f"query_text: {query_text}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())