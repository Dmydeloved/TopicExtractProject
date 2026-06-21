import tempfile
import unittest
from pathlib import Path

from memory_system import MemoryManager, MemoryStorage


def topic(topic_name="餐厅推荐", entity="餐厅", intent="查询"):
    return {
        "topic": topic_name,
        "core_entity": entity,
        "intent": intent,
        "entities": [entity],
        "confidence": 0.9,
        "reasoning": "测试主题提取结果",
    }


class MemorySystemTests(unittest.TestCase):
    def make_manager(self):
        directory = tempfile.TemporaryDirectory()
        storage = MemoryStorage(Path(directory.name) / "memory.sqlite3")
        manager = MemoryManager(storage)
        return directory, storage, manager

    def test_same_experience_and_intent_appends_same_segment(self):
        directory, storage, manager = self.make_manager()
        try:
            first = manager.add_qa(topic(), "user 1", "assistant 1")
            second = manager.add_qa(topic(), "user 2", "assistant 2")
            self.assertEqual(first["experience_id"], second["experience_id"])
            self.assertEqual(first["segment_id"], second["segment_id"])
            self.assertEqual(2, storage.count_rows("qa_memory"))
            self.assertEqual(1, storage.count_rows("segment_memory"))
            self.assertEqual(1, storage.count_rows("experience_memory"))
        finally:
            storage.close()
            directory.cleanup()

    def test_intent_change_creates_new_segment(self):
        directory, storage, manager = self.make_manager()
        try:
            first = manager.add_qa(topic(intent="查询"), "user 1")
            second = manager.add_qa(topic(intent="预订"), "user 2")
            self.assertEqual(first["experience_id"], second["experience_id"])
            self.assertNotEqual(first["segment_id"], second["segment_id"])
            old_segment = storage.get_segment(first["segment_id"])
            self.assertTrue(old_segment["summary"])
        finally:
            storage.close()
            directory.cleanup()

    def test_experience_switch_and_return_uses_existing_experience(self):
        directory, storage, manager = self.make_manager()
        try:
            first = manager.add_qa(topic(topic_name="餐厅推荐", entity="餐厅"), "user 1")
            hotel = manager.add_qa(topic(topic_name="酒店推荐", entity="酒店"), "user 2")
            returned = manager.add_qa(topic(topic_name="餐厅推荐", entity="餐厅"), "user 3")
            self.assertNotEqual(first["experience_id"], hotel["experience_id"])
            self.assertEqual(first["experience_id"], returned["experience_id"])
            self.assertEqual(2, storage.count_rows("experience_memory"))
        finally:
            storage.close()
            directory.cleanup()

    def test_segment_summary_triggered_every_five_qas(self):
        directory, storage, manager = self.make_manager()
        try:
            result = None
            for index in range(5):
                result = manager.add_qa(topic(), f"user {index}")
            segment = storage.get_segment(result["segment_id"])
            self.assertEqual(5, segment["last_summarized_qa_count"])
            self.assertTrue(segment["summary"])
        finally:
            storage.close()
            directory.cleanup()

    def test_experience_summary_triggered_every_five_segments(self):
        directory, storage, manager = self.make_manager()
        try:
            result = None
            for index in range(5):
                result = manager.add_qa(topic(intent=f"意图{index}"), f"user {index}")
            experience = storage.get_experience(result["experience_id"])
            self.assertEqual(5, experience["last_summarized_segment_count"])
            self.assertTrue(experience["summary"]["short"])
        finally:
            storage.close()
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
