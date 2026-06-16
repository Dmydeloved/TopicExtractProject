# MultiWOZ 2.2 场景与对话提取

## 目录

```text
data/
  raw/multiwoz_2.2/                         # 官方原始 MultiWOZ 2.2
  processed/multiwoz_2.2_scene_dialogue_10000.jsonl
scripts/
  process_multiwoz_2_2.py
```

原始数据保持官方格式，不做修改。处理结果每行是一段对话，仅保留：

```json
{
  "scene": ["restaurant", "hotel"],
  "dialogue": [
    {"role": "user", "content": "i need a place to dine..."},
    {"role": "system", "content": "there are several restaurants..."}
  ]
}
```

已去除对话 ID、数据集划分、状态、意图、槽位、动作、跨度标注等内容。

## 运行

默认提取 10000 段对话：

```powershell
python scripts/process_multiwoz_2_2.py
```

重新生成：

```powershell
python scripts/process_multiwoz_2_2.py --overwrite
```

提取全部对话：

```powershell
python scripts/process_multiwoz_2_2.py --limit 0 --output data/processed/multiwoz_2.2_scene_dialogue_all.jsonl
```
