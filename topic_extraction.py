from openai import OpenAI
import json
from prompts_storage import common_extractor_prompt
from experiment_data import get_single_topic, get_increase_topic, get_shift_topic
from datetime import datetime

class FixedArray:
    def __init__(self, max_size):
        # 初始化固定容量
        self.max_size = max_size
        self.array = []  # 实际存储的列表

    def add(self, item):
        # 添加元素
        self.array.append(item)

        # 如果超过容量，删除第一个（最开始的）
        if len(self.array) > self.max_size:
            self.array.pop(0)

    def get_all(self):
        # 获取所有元素
        return self.array

    def to_json_string(self):
        # 新增：把所有元素 转成 格式化的 JSON 字符串
        return json.dumps(
            self.array,
            ensure_ascii=False,  # 中文正常显示
            indent=2  # 格式化缩进，好看
        )
    def to_list(self):
        # 转成普通列表，用于序列化
        return self.array.copy()

def context_to_json(context):
    # 把自定义类转成普通列表
    data = {
        "current_topic_state": context["current_topic_state"],
        "recent_semantic_history": context["recent_semantic_history"].to_list()
    }
    # 转成 JSON 字符串（中文不乱码）
    return json.dumps(data, ensure_ascii=False, indent=2)

def run_entity_extract(user_input: str, ctx: str = "", kg: str = "") -> dict:
    """调用LLM完成实体主题提取"""
    # 初始化客户端，替换为自己的key与base_url
    client = OpenAI(
        api_key="sk-MQHPgxtvAKdBQxiE90A4C6A68fAc4eBe9d4b899f107fB9Fb",
        base_url="https://api.gpt.ge/v1/"
    )
    prompt = common_extractor_prompt(user_input, ctx, kg)
    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    content = resp.choices[0].message.content.strip()
    return json.loads(content)

def single_topic_extract():
    first_result = {
        "single_topic_data": [],
        "increase_topic_data": [],
        "shift_topic_data": []
    }
    single_topic_data = get_single_topic()
    increase_topic_data = get_increase_topic()
    shift_topic_data = get_shift_topic()

    for index, message in single_topic_data.items():
        temp_result = run_entity_extract(user_input=message)
        first_result["single_topic_data"].append(temp_result)
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    for index, message in increase_topic_data.items():
        temp_result = run_entity_extract(user_input=message)
        first_result["increase_topic_data"].append(temp_result)
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    for index, message in shift_topic_data.items():
        temp_result = run_entity_extract(user_input=message)
        first_result["shift_topic_data"].append(temp_result)
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    # 1. 美化打印到控制台
    print("提取结果：")
    print(json.dumps(first_result, ensure_ascii=False, indent=4))

    # 2. 保存到 result.json 文件（最推荐）
    with open("first_result.json", "w", encoding="utf-8") as f:
        json.dump(
            first_result,
            f,
            ensure_ascii=False,  # 显示中文不乱码
            indent=4  # 格式化缩进，方便阅读
        )

    print("\n✅ 结果已保存到 first_result.json")

def original_topic_extract():
    second_result = {
        "single_topic_data":[],
        "increase_topic_data":[],
        "shift_topic_data":[]
    }
    single_topic_data = get_single_topic()
    increase_topic_data = get_increase_topic()
    shift_topic_data = get_shift_topic()

    conversation_context = FixedArray(5)

    for index, message in single_topic_data.items():
        temp_result = run_entity_extract(user_input=message, ctx=conversation_context.to_json_string())
        conversation_context.add(message)
        second_result["single_topic_data"].append(temp_result)
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    for index, message in increase_topic_data.items():
        temp_result = run_entity_extract(user_input=message, ctx=conversation_context.to_json_string())
        conversation_context.add(message)
        second_result["increase_topic_data"].append(temp_result)
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    for index, message in shift_topic_data.items():
        temp_result = run_entity_extract(user_input=message, ctx=conversation_context.to_json_string())
        conversation_context.add(message)
        second_result["shift_topic_data"].append(temp_result)
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    # 1. 美化打印到控制台
    print("提取结果：")
    print(json.dumps(second_result, ensure_ascii=False, indent=4))

    # 2. 保存到 result.json 文件（最推荐）
    with open("second_result.json", "w", encoding="utf-8") as f:
        json.dump(
            second_result,
            f,
            ensure_ascii=False,  # 显示中文不乱码
            indent=4             # 格式化缩进，方便阅读
        )

    print("\n✅ 结果已保存到 second_result.json")

if __name__ == "__main__":
    single_topic_extract()
    original_topic_extract()
    third_result = {
        "single_topic_data": [],
        "increase_topic_data": [],
        "shift_topic_data": []
    }
    single_topic_data = get_single_topic()
    increase_topic_data = get_increase_topic()
    shift_topic_data = get_shift_topic()

    conversation_context = {
        "current_topic_state":{},
        "recent_semantic_history": FixedArray(5)
    }

    for index, message in single_topic_data.items():
        temp_result = run_entity_extract(user_input=message, ctx=context_to_json(conversation_context))
        third_result["single_topic_data"].append(temp_result)
        temp_result['user_input'] = message
        temp_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_context["recent_semantic_history"].add(temp_result)
        conversation_context['current_topic_state'] = temp_result
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    for index, message in increase_topic_data.items():
        temp_result = run_entity_extract(user_input=message, ctx=context_to_json(conversation_context))
        third_result["increase_topic_data"].append(temp_result)
        temp_result['user_input'] = message
        temp_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_context["recent_semantic_history"].add(temp_result)
        conversation_context['current_topic_state'] = temp_result
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    for index, message in shift_topic_data.items():
        temp_result = run_entity_extract(user_input=message, ctx=context_to_json(conversation_context))
        third_result["shift_topic_data"].append(temp_result)
        temp_result['user_input'] = message
        temp_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_context["recent_semantic_history"].add(temp_result)
        conversation_context['current_topic_state'] = temp_result
        print(f"index = {index}\n{json.dumps(temp_result, ensure_ascii=False, indent=4)}\n")

    # 1. 美化打印到控制台
    print("提取结果：")
    print(json.dumps(third_result, ensure_ascii=False, indent=4))
    print(context_to_json(conversation_context))

    # 2. 保存到 result.json 文件（最推荐）
    with open("third_result.json", "w", encoding="utf-8") as f:
        json.dump(
            third_result,
            f,
            ensure_ascii=False,  # 显示中文不乱码
            indent=4  # 格式化缩进，方便阅读
        )

    print("\n✅ 结果已保存到 third_result.json")