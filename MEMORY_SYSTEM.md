# 结构化主题记忆库

该模块实现 QA、Segment、Experience 三层结构化记忆、运行时状态，以及 SQLite + Chroma 混合检索。

## 向量架构

    主题结构化数据
      -> SQLite：保存 QA / Segment / Experience / runtime_state
      -> 阿里百炼 text-embedding-v4：编码 topic + core_entity + entities/intent
      -> Chroma：持久化 QA / Segment / Experience 向量

API Key 不写入代码，通过环境变量配置：

    $env:DASHSCOPE_API_KEY = "your-api-key"

安装依赖：

    pip install -r requirements-memory.txt

## 运行导入

    python scripts/build_memory_from_multiwoz_topics.py --limit 20

默认存储位置：

    data/memory/topic_memory.sqlite3
    data/memory/chroma/

## 代码写入

    from memory_system import (
        BailianEmbedder,
        ChromaVectorStore,
        MemoryManager,
        MemoryStorage,
    )

    storage = MemoryStorage("data/memory/topic_memory.sqlite3")
    embedder = BailianEmbedder()
    vector_store = ChromaVectorStore("data/memory/chroma")
    manager = MemoryManager(storage, vector_store, embedder)

    manager.add_qa(
        topic_result={
            "topic": "餐厅推荐",
            "core_entity": "餐厅",
            "intent": "查询",
            "entities": ["餐厅", "电话"],
            "confidence": 0.9,
            "reasoning": "承接餐厅推荐主题，请求电话，意图为查询",
        },
        user_input="Could I get the phone number?",
        assistant_output="The phone number is ...",
    )

## 混合检索

关键词候选来自 SQLite，语义候选来自 Chroma：

    score = 0.45 * keyword_score + 0.55 * semantic_score

先按混合分取得 top_k，再按 QA 时间戳升序返回。

    from memory_system import HybridRetriever

    retriever = HybridRetriever(storage, vector_store, embedder)
    result = retriever.recall(
        topic="餐厅推荐",
        core_entity="餐厅",
        intent="查询",
        top_k=5,
    )

    for item in result["results"]:
        print(
            item["qa"]["timestamp"],
            item["qa"]["user_input"],
            item["score"],
            item["keyword_score"],
            item["semantic_score"],
        )

## 聚合规则

    topic + core_entity 相同 -> 同一个 Experience
    intent 相同             -> 追加当前 Segment
    intent 不同             -> 总结旧 Segment，新建 Segment
    Segment 满 5 条 QA       -> 更新 Segment 总结
    Experience 新增 5 段     -> 更新 Experience 总结