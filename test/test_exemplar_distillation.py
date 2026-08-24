import unittest
from unittest import mock

from pyopl.genai import exemplar_distillation


class TestExemplarDistillation(unittest.TestCase):
    def test_prompt_contains_three_style_examples_and_current_inputs(self):
        examples = [
            {"description": f"description {index}", "model": f"model {index}", "data": f"data {index}"} for index in range(3)
        ]

        prompt = exemplar_distillation.build_distillation_prompt(
            model="current model",
            data="current data",
            source_session="current session",
            examples=examples,
        )

        self.assertEqual(prompt.count('<example index="'), 3)
        self.assertIn("style references only", prompt)
        self.assertIn("<current_model>\ncurrent model", prompt)
        self.assertIn("<current_data>\ncurrent data", prompt)
        self.assertIn("<current_session>\ncurrent session", prompt)

    def test_distillation_retrieves_three_examples_and_normalizes_response(self):
        examples = [{"description": "d", "model": "m", "data": "x"}] * 3
        with (
            mock.patch.object(exemplar_distillation._STRATEGY, "gather_few_shots", return_value=examples) as gather,
            mock.patch.object(
                exemplar_distillation._STRATEGY,
                "llm_generate_text",
                return_value="```text\nDescription: Assign crews while minimizing cost.\n```",
            ) as generate,
        ):
            result = exemplar_distillation.distill_exemplar_description(
                model="model",
                data="data",
                source_session="session",
                provider="openai",
                model_name="test-model",
            )

        self.assertEqual(result, "Assign crews while minimizing cost.")
        self.assertEqual(gather.call_args.kwargs["k"], 3)
        self.assertIn("<current_session>\nsession", generate.call_args.kwargs["input_text"])


if __name__ == "__main__":
    unittest.main()
