# MultiWOZ 2.2 用户轮次主题提取

该模块按照 `topic_extraction.py` 中 third 模式的语义状态方法，对每个用户轮次进行主题提取：

- 每段对话创建独立上下文。
- `current_topic_state` 保存上一轮结构化主题结果；若单轮包含多个主题，则保存该轮的结果数组。
- `recent_semantic_history` 保存最近 5 条结构化主题结果；多主题轮次按整轮结果写入历史。
- 系统回复不进行提取，也不加入语义历史。
- 不将 MultiWOZ `scene` 作为领域知识传入模型。
- 直接复用 `topic_extraction.run_entity_extract` 中配置的模型、API Key 和 URL。

## 首次测试

默认最多处理 1000 个用户轮次：

```powershell
python multiwoz_topic_extraction.py
```

输出文件：

```text
data/topic_extracted/multiwoz_2.2_with_topics.jsonl
data/topic_extracted/multiwoz_2.2_checkpoint.json
data/topic_extracted/multiwoz_2.2_failures.jsonl
```

脚本默认从 checkpoint 继续执行，因此再次运行会继续处理后续 1000 个用户轮次。

## 数据流程

```text
multiwoz_2.2_scene_dialogue_10000.jsonl
        │
        ├─ 逐行读取一段原始对话
        │
        ├─ 为该对话初始化 SemanticContext
        │      ├─ current_topic_state
        │      └─ recent_semantic_history（最多 5 条）
        │
        ├─ 按原顺序遍历 dialogue
        │      ├─ system：保持原样，不调用模型
        │      └─ user：
        │             1. 读取 content
        │             2. 将 SemanticContext 转为历史上下文 JSON
        │             3. 调用 run_entity_extract(user_input, ctx, kg="")
        │             4. 校验单条或多条 topic/core_entity/intent/entities/confidence/reasoning
        │             5. 将结果写入当前 user 轮次的 topic_extraction
        │             6. 用结果更新 SemanticContext
        │             7. 保存 checkpoint
        │
        ├─ 对话处理完成后写入增强后的 JSONL
        │
        └─ 达到处理上限时保存部分对话，下次从断点原位继续
```

数据边界：

- 每段对话开始时重置语义上下文，不会污染下一段对话。
- 模型仅看到当前 user 输入与结构化语义历史。
- `scene` 和 system 内容保留在输出中，但不会传入模型。
- 输出仍是一行一段对话，仅为 user 轮次增加 `topic_extraction`。
- 若模型识别到多个独立主题，则 `topic_extraction` 为数组；单主题时保持对象格式。

## 控制台日志

默认每成功处理 10 个 user 轮次打印一次进度：

```text
进度 run=20/1000 total=20/68380(0.03%) dialogue=2 turn=8 topic=餐厅服务 core_entity=restaurant elapsed=32.4s
```

日志还会显示启动配置、断点位置、模型重试、处理失败、部分对话保存和结束汇总。

每个 user 轮次都打印进度：

```powershell
python multiwoz_topic_extraction.py --progress-every 1
```

同时打印对话级调试日志：

```powershell
python multiwoz_topic_extraction.py --verbose
```

## 输出结构

输出保留原有 MultiWOZ 精简 JSON 格式，仅在每个 `user` 轮次中增加
`topic_extraction`。`system` 轮次保持不变；多主题时该字段会是数组：

```json
{
  "scene": ["restaurant", "hotel"],
  "dialogue": [
    {
      "role": "user",
      "content": "i need a place to dine in the center",
      "topic_extraction": {
        "topic": "餐厅服务",
        "core_entity": "restaurant",
        "intent": "query",
        "entities": ["restaurant"],
        "confidence": 0.9,
        "reasoning": "根据当前输入和语义历史判断"
      }
    },
    {
      "role": "system",
      "content": "What kind of food would you prefer?"
    }
  ]
}
```

当处理数量限制发生在一段对话中间时，输出会保存部分增强后的原格式对话；下次续跑时会继续补充该对话，并替换部分结果，不会产生重复对话。

## 常用参数

从头重新执行并删除已有提取输出：

```powershell
python multiwoz_topic_extraction.py --no-resume
```

不限制本次处理轮数：

```powershell
python multiwoz_topic_extraction.py --max-user-turns 0
```

遇到单轮错误时记录失败并继续：

```powershell
python multiwoz_topic_extraction.py --continue-on-error
```
