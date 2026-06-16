import json
import tempfile
import unittest
from pathlib import Path

from multiwoz_topic_extraction import (
    SemanticContext,
    input_stats,
    process_dialogues,
    validate_topic_result,
)


class FakeExtractor:
    def __init__(self) -> None:
        self.contexts = []

    def extract(self, user_input, context):
        self.contexts.append(json.loads(context))
        return {
            "topic": "餐厅服务",
            "core_entity": "restaurant",
            "intent": "query",
            "entities": ["restaurant"],
            "confidence": 0.9,
            "reasoning": "根据当前输入归纳主题并判断查询意图",
        }


class MultiWOZTopicExtractionTests(unittest.TestCase):
    def test_input_stats_counts_dialogues_and_user_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "scene": [],
                        "dialogue": [
                            {"role": "user", "content": "first"},
                            {"role": "system", "content": "reply"},
                            {"role": "user", "content": "second"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual((1, 2), input_stats(path))

    def test_semantic_context_keeps_last_five_results(self):
        context = SemanticContext(history_size=5)
        for index in range(7):
            context.update({"topic": str(index)})
        self.assertEqual(["2", "3", "4", "5", "6"], [x["topic"] for x in context.recent_semantic_history])
        self.assertEqual("6", context.current_topic_state["topic"])

    def test_result_validation_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            validate_topic_result(
                {
                    "topic": "a",
                    "core_entity": "b",
                    "intent": "query",
                    "entities": ["b"],
                    "confidence": 2,
                    "reasoning": "test",
                }
            )

    def test_pipeline_preserves_original_shape_and_adds_results_to_user_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            dialogues = [
                {
                    "scene": ["restaurant"],
                    "dialogue": [
                        {"role": "user", "content": "first"},
                        {"role": "system", "content": "reply"},
                        {"role": "user", "content": "second"},
                    ],
                },
                {
                    "scene": ["hotel"],
                    "dialogue": [{"role": "user", "content": "third"}],
                },
            ]
            input_path.write_text(
                "".join(json.dumps(item) + "\n" for item in dialogues),
                encoding="utf-8",
            )
            extractor = FakeExtractor()
            processed = process_dialogues(
                extractor=extractor,
                input_path=input_path,
                output_path=root / "output.jsonl",
                checkpoint_path=root / "checkpoint.json",
                failures_path=root / "failures.jsonl",
                max_user_turns=3,
                resume=False,
            )

            self.assertEqual(3, processed)
            self.assertEqual({}, extractor.contexts[0]["current_topic_state"])
            self.assertEqual("餐厅服务", extractor.contexts[1]["current_topic_state"]["topic"])
            self.assertEqual({}, extractor.contexts[2]["current_topic_state"])

            output = [
                json.loads(line)
                for line in (root / "output.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(dialogues[0]["scene"], output[0]["scene"])
            self.assertNotIn("topic_extraction", output[0]["dialogue"][1])
            self.assertEqual("餐厅服务", output[0]["dialogue"][0]["topic_extraction"]["topic"])

    def test_exact_limit_writes_partial_dialogue_and_resume_replaces_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "scene": ["restaurant"],
                        "dialogue": [
                            {"role": "user", "content": "first"},
                            {"role": "system", "content": "reply"},
                            {"role": "user", "content": "second"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            paths = {
                "output_path": root / "output.jsonl",
                "checkpoint_path": root / "checkpoint.json",
                "failures_path": root / "failures.jsonl",
            }
            self.assertEqual(
                1,
                process_dialogues(FakeExtractor(), input_path, max_user_turns=1, resume=False, **paths),
            )
            partial = json.loads(paths["output_path"].read_text(encoding="utf-8"))
            self.assertIn("topic_extraction", partial["dialogue"][0])
            self.assertNotIn("topic_extraction", partial["dialogue"][2])

            self.assertEqual(
                1,
                process_dialogues(FakeExtractor(), input_path, max_user_turns=1, resume=True, **paths),
            )
            lines = paths["output_path"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            completed = json.loads(lines[0])
            self.assertIn("topic_extraction", completed["dialogue"][0])
            self.assertIn("topic_extraction", completed["dialogue"][2])


if __name__ == "__main__":
    unittest.main()
