from memory_system.topic_extractor import run_entity_extract
import json

def main() -> int:
    query_text = "我想找一家位于市中心、价格偏高的餐厅，可以告诉我推荐餐厅的电话号码吗？"

    try:
        result = run_entity_extract(
            user_input=query_text,
            api_key='sk-RFUNAF0b6zfJWCVz9dA1Aa244aEa43Dc974693370b8d4338',
            model='gpt-5.5',
            base_url='https://api.gpt.ge/v1/'
        )
    except RuntimeError as err:
        print(str(err))

    print(f"query_text: {query_text}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())