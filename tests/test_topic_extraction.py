import unittest

from topic_extraction import parse_topic_response


class TopicExtractionResponseParsingTests(unittest.TestCase):
    def test_parse_single_item_array_returns_object(self):
        payload = parse_topic_response(
            '[{"topic":"餐厅推荐","core_entity":"餐厅","intent":"推荐","entities":["餐厅"],"confidence":0.9,"reasoning":"test"}]'
        )
        self.assertEqual("餐厅推荐", payload["topic"])
        self.assertEqual("餐厅", payload["core_entity"])

    def test_parse_code_fenced_json_returns_object(self):
        payload = parse_topic_response(
            '```json\n{"topic":"酒店预订","core_entity":"酒店","intent":"预订","entities":["酒店"],"confidence":0.8,"reasoning":"test"}\n```'
        )
        self.assertEqual("酒店预订", payload["topic"])

    def test_parse_multi_item_array_returns_list(self):
        payload = parse_topic_response(
            '[{"topic":"餐厅推荐","core_entity":"餐厅","intent":"推荐","entities":["餐厅"],"confidence":0.9,"reasoning":"a"},'
            '{"topic":"火车出行","core_entity":"火车","intent":"查询","entities":["火车"],"confidence":0.9,"reasoning":"b"}]'
        )
        self.assertEqual(2, len(payload))
        self.assertEqual("餐厅推荐", payload[0]["topic"])
        self.assertEqual("火车出行", payload[1]["topic"])


if __name__ == "__main__":
    unittest.main()
