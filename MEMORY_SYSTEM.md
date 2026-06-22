# Memory System

这个项目现在只保留一条主线：`memory_system` 负责主题提取、MultiWOZ 主题增强、结构化记忆写入、向量存储和混合检索；旧实验内容统一放在 `experiment/`。

## 目录

```text
memory_system/
  topic_prompts.py                 # 主题提取 prompt
  topic_extractor.py               # 主题提取与结果校验
  storage.py                       # SQLite 结构化存储
  vector_store.py                  # Chroma 向量存储
  manager.py                       # QA / Segment / Experience 写入
  retriever.py                     # SQLite + Chroma 混合检索
  summarizer.py                    # Segment / Experience 总结

scripts/
  process_multiwoz_2_2.py          # 原始 MultiWOZ -> scene/dialogue JSONL
  multiwoz_topic_pipeline.py       # 为 user turn 增加 topic_extraction
  build_memory_from_multiwoz_topics.py
                                  # topic_extraction -> memory_system

tests/
  test_topic_extraction.py
  test_multiwoz_topic_extraction.py
  test_memory_system.py
  test_retriever.py

experiment/
  topic_extraction_experiment.py   # 历史试验脚本
  sample_topics.py                 # 历史试验样本
  first_result.json
  second_result.json
  third_result.json
```

## 安装

```bash
pip install -r requirements-memory.txt
```

环境变量：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

说明：

- `BailianEmbedder` 默认读取 `DASHSCOPE_API_KEY`
- `BailianTopicExtractor` 优先读取 `TOPIC_EXTRACT_API_KEY`，没有时回退到 `DASHSCOPE_API_KEY`

## 数据流

```text
MultiWOZ 2.2 原始数据
  -> scripts/process_multiwoz_2_2.py
  -> data/processed/*.jsonl
  -> scripts/multiwoz_topic_pipeline.py
  -> data/topic_extracted/*.jsonl
  -> scripts/build_memory_from_multiwoz_topics.py
  -> SQLite + Chroma
  -> StructuredRetriever / HybridRetriever
```

### 1. 提取精简对话

```bash
python scripts/process_multiwoz_2_2.py --overwrite
```

输出格式：

```json
{
  "scene": ["restaurant", "hotel"],
  "dialogue": [
    {"role": "user", "content": "i need a place to dine..."},
    {"role": "system", "content": "there are several restaurants..."}
  ]
}
```

### 2. 提取主题

```bash
python scripts/multiwoz_topic_pipeline.py --progress-every 1
```

写回 `user` 轮次的字段：

```json
"topic_extraction": {
  "topic": "餐厅推荐",
  "core_entity": "餐厅",
  "intent": "查询",
  "entities": ["餐厅", "电话号码"],
  "confidence": 0.88,
  "reasoning": "承接历史餐厅推荐主题；请求电话号码体现查询意图"
}
```

多主题时，`topic_extraction` 为数组。

### 3. 建立记忆库

```bash
python scripts/build_memory_from_multiwoz_topics.py --limit 20
```

默认输出：

```text
data/memory/topic_memory.sqlite3
data/memory/chroma/
```

## 主题提取

核心在 `memory_system/topic_extractor.py`：

- `parse_topic_response`：兼容普通 JSON、代码块包裹 JSON、单条/多条结果
- `validate_topic_record` / `validate_topic_result`：强校验 `topic/core_entity/intent/entities/confidence/reasoning`
- `BailianTopicExtractor`：调用阿里百炼兼容接口完成提取

MultiWOZ 管线在 `scripts/multiwoz_topic_pipeline.py`：

- 每段对话独立维护一个 `SemanticContext`
- `current_topic_state` 保存当前语义状态
- `recent_semantic_history` 保留最近 5 条语义历史
- 支持 checkpoint、失败记录和断点续跑

## 结构化记忆

SQLite 表：

- `qa_memory`
- `segment_memory`
- `experience_memory`
- `runtime_state`

聚合规则：

```text
topic + core_entity 相同 -> 同一个 Experience
intent 相同             -> 追加当前 Segment
intent 不同             -> 总结旧 Segment，新建 Segment
Segment 满 5 条 QA       -> 更新 Segment 总结
Experience 新增 5 段     -> 更新 Experience 总结
```

写入入口：

```python
from memory_system import BailianEmbedder, ChromaVectorStore, MemoryManager, MemoryStorage

storage = MemoryStorage("data/memory/topic_memory.sqlite3")
embedder = BailianEmbedder()
vector_store = ChromaVectorStore("data/memory/chroma")
manager = MemoryManager(storage, vector_store, embedder)
```

## 向量存储

向量文本构造规则：

- QA：`topic + core_entity + entities`
- Segment：`topic + core_entity + intent`
- Experience：`topic + core_entity`

Chroma 持久化目录分两层：

- `data/memory/chroma/chroma.sqlite3`
  保存 collection、segment、metadata、embeddings_queue 等元数据
- `data/memory/chroma/<uuid>/`
  保存 HNSW 向量索引二进制文件

## 检索

`StructuredRetriever` 是 `HybridRetriever` 的兼容别名。

检索流程：

```text
query
  -> topic_entity_text(topic, core_entity, entities/intent)
  -> embedder.embed(query_text)
  -> Chroma 召回语义候选
  -> SQLite 全量 QA 算关键词重叠分
  -> 混合打分
```

打分公式：

```text
score = 0.45 * keyword_score + 0.55 * semantic_score
```

先按混合分取 `top_k`，再按 QA 时间戳升序返回。

## 测试

运行单项测试：

```bash
python -m unittest tests.test_topic_extraction
python -m unittest tests.test_multiwoz_topic_extraction
python -m unittest tests.test_memory_system
python tests/test_retriever.py
```

说明：

- `tests/test_retriever.py` 是直接执行脚本，使用现有 memory 数据做一次真实召回并打印结果
- 若要跑主题提取和检索里的真实百炼调用，需要本机可访问 DashScope 接口
