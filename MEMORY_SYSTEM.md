# 结构化主题记忆库

该模块实现 QA、Segment、Experience 三层结构化记忆，以及运行时状态 `runtime_state`。

## 写入规则

每次输入一条主题结构化数据：

```json
{
  "topic": "",
  "core_entity": "",
  "intent": "",
  "entities": [],
  "confidence": 0.0,
  "reasoning": ""
}
```

写入流程：

```text
读取 runtime_state
  -> 判断 current Experience 的 topic + core_entity 是否一致
      -> 一致：继续判断 current Segment
      -> 不一致：总结当前 Segment / Experience，再查找或创建目标 Experience
  -> 判断 Segment 的 intent 是否一致
      -> 一致：追加 QA
      -> 不一致：总结旧 Segment，新建 Segment
  -> 创建 QA
  -> 更新 Segment / Experience / runtime_state
  -> Segment 满 5 条 QA 触发总结
  -> Experience 新增 5 个 Segment 触发总结
```

QA 层已去除 `conversation_id`、`turn_index`、`scene` 字段。

## 运行导入

从已经完成主题提取的 MultiWOZ JSONL 构建记忆库：

```powershell
python scripts/build_memory_from_multiwoz_topics.py --limit 20
```

默认数据库：

```text
data/memory/topic_memory.sqlite3
```

## 代码入口

```python
from memory_system import MemoryManager, MemoryStorage

storage = MemoryStorage("data/memory/topic_memory.sqlite3")
manager = MemoryManager(storage)

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
```
